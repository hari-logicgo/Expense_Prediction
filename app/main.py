# app.py
import calendar
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.collection import Collection

load_dotenv()

app = FastAPI(title="Expense Prediction API", version="1.0.0")

# ---------- Configurable constants ----------
MAX_HISTORY_MONTHS = int(os.getenv("MAX_HISTORY_MONTHS", "36"))  # months to fetch for detection/tuning
SEASONALITY_PERIOD = int(os.getenv("SEASONALITY_PERIOD", "12"))  # monthly seasonality (12 months)
SEASONALITY_AMPLITUDE_THRESHOLD = float(os.getenv("SEASONALITY_AMPLITUDE_THRESHOLD", "0.18"))
# grid-search limits (keeps tuning light)
ALPHA_GRID = [0.3, 0.5, 0.7]
BETA_GRID = [0.1, 0.3, 0.5]
GAMMA_GRID = [0.1, 0.3, 0.5]
MAX_GRID_SEARCH_COMBINATIONS = 30  # safety cap
# ------------------------------------------------

class MonthlyExpense(BaseModel):
    year: int
    month: int
    total: float = Field(..., description="Total expenses recorded for the month")


class CategoryPrediction(BaseModel):
    headCategoryId: str
    title: str
    history: List[MonthlyExpense]
    predictionMonth: MonthlyExpense


class PredictionResponse(BaseModel):
    userId: str
    categories: List[CategoryPrediction]


class MongoConnection:
    def __init__(self) -> None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("MONGO_URI is not configured in the environment")

        self._client = MongoClient(mongo_uri, tz_aware=True)
        self._database = self._client.get_default_database()
        self.transactions: Collection = self._database["transactions"]
        self.headcategories: Collection = self._database["headcategories"]


mongo = MongoConnection()

# ----------------- Date helpers -----------------
def _first_day_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _shift_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, last_day)
    return dt.replace(year=year, month=month, day=day)


def month_to_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def index_to_month(idx: int) -> Tuple[int, int]:
    year = idx // 12
    month = (idx % 12) + 1
    return year, month
# ------------------------------------------------

# ----------------- Time series utilities -----------------
def build_continuous_series(history: List[MonthlyExpense]) -> Tuple[List[float], List[Tuple[int, int]]]:
    """
    Given sparse monthly history items (year, month, total), build a continuous series
    covering from earliest to latest month in history. Missing months are represented by None.
    Returns (values_list_with_none, list_of_(year,month)_corresponding).
    """
    if not history:
        return [], []

    # sort history
    history_sorted = sorted(history, key=lambda h: (h.year, h.month))
    start_idx = month_to_index(history_sorted[0].year, history_sorted[0].month)
    end_idx = month_to_index(history_sorted[-1].year, history_sorted[-1].month)
    length = end_idx - start_idx + 1

    idx_to_val = {}
    for h in history_sorted:
        idx = month_to_index(h.year, h.month)
        idx_to_val[idx] = h.total

    series = []
    months = []
    for i in range(start_idx, end_idx + 1):
        months.append(index_to_month(i))
        series.append(idx_to_val.get(i, None))

    return series, months


def impute_missing(series: List[Optional[float]]) -> List[float]:
    """
    Fill missing values (None) by linear interpolation. If leading/trailing Nones remain,
    forward/backfill with nearest value or 0 if no data.
    """
    n = len(series)
    if n == 0:
        return []

    arr = [None if v is None else float(v) for v in series]

    # collect indices of non-None
    known = [i for i, v in enumerate(arr) if v is not None]

    if not known:
        # all missing -> return zeros
        return [0.0] * n

    # linear interpolation between known points
    for i in range(len(known) - 1):
        a = known[i]
        b = known[i + 1]
        va = arr[a]
        vb = arr[b]
        step = (vb - va) / (b - a)
        for j in range(a + 1, b):
            arr[j] = va + step * (j - a)

    # fill leading
    first = known[0]
    for i in range(0, first):
        arr[i] = arr[first]

    # fill trailing
    last = known[-1]
    for i in range(last + 1, n):
        arr[i] = arr[last]

    return [float(x) for x in arr]


def seasonal_strength(series: List[float], period: int = SEASONALITY_PERIOD) -> float:
    """
    Estimate seasonality strength for monthly data.
    Returns amplitude_ratio = (max_month_mean - min_month_mean) / overall_mean
    Higher value => stronger seasonality.
    Requires at least 2 * period data points for a reliable estimate.
    """
    n = len(series)
    if n < 2 * period:
        return 0.0

    # compute month-of-year means
    month_buckets = [[] for _ in range(period)]
    for idx, val in enumerate(series):
        month = idx % period
        month_buckets[month].append(val)

    month_means = [ (sum(b)/len(b)) if b else 0.0 for b in month_buckets ]
    overall_mean = sum(series) / len(series) if series else 0.0
    if overall_mean == 0:
        return 0.0
    amplitude = max(month_means) - min(month_means)
    return amplitude / overall_mean


# ----------------- Forecasting algorithms -----------------
def holt_double_forecast(series: List[float], alpha: float, beta: float, n_forecast: int = 1) -> List[float]:
    """
    Holt's linear method (double exponential smoothing).
    Returns list of length n_forecast (forecast ahead).
    """
    n = len(series)
    if n == 0:
        return [0.0] * n_forecast
    if n == 1:
        return [series[-1]] * n_forecast

    level = series[0]
    trend = series[1] - series[0]

    for t in range(1, n):
        value = series[t]
        prev_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    # forecast h steps ahead
    forecasts = [level + (i + 1) * trend for i in range(n_forecast)]
    return [max(0.0, f) for f in forecasts]


def holt_winters_additive(series: List[float], season_length: int, alpha: float, beta: float, gamma: float, n_forecast: int = 1) -> List[float]:
    """
    Additive Holt-Winters seasonal method.
    series: list of floats (no missing) where season_length is known (e.g., 12)
    """
    n = len(series)
    if n == 0:
        return [0.0] * n_forecast
    if n < season_length * 2:
        # not enough data to initialize seasonals reliably -> fallback to holt_double
        return holt_double_forecast(series, alpha, beta, n_forecast)

    # initialize level, trend, seasonals
    seasonals = _initial_seasonal_components(series, season_length)
    level = sum(series[:season_length]) / season_length
    trend = (sum(series[season_length:2*season_length]) - sum(series[:season_length])) / (season_length * season_length)

    result = []
    for i in range(n + n_forecast):
        if i < n:
            val = series[i]
            last_level = level
            level = alpha * (val - seasonals[i % season_length]) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonals[i % season_length] = gamma * (val - level) + (1 - gamma) * seasonals[i % season_length]
            # in-sample prediction (not used)
        else:
            # forecast
            m = i - n + 1
            forecast = level + m * trend + seasonals[i % season_length]
            result.append(max(0.0, forecast))

    # ensure length matches n_forecast
    return result[:n_forecast]


def _initial_seasonal_components(series: List[float], season_length: int) -> List[float]:
    """
    Initialize seasonality components by averaging.
    """
    seasonals = [0.0] * season_length
    n_seasons = len(series) // season_length
    if n_seasons == 0:
        return seasonals
    season_averages = []
    for j in range(n_seasons):
        start = j * season_length
        season_avg = sum(series[start:start + season_length]) / season_length
        season_averages.append(season_avg)
    for i in range(season_length):
        s = 0.0
        for j in range(n_seasons):
            s += series[j * season_length + i] - season_averages[j]
        seasonals[i] = s / n_seasons
    return seasonals

# ----------------- Dynamic WMA -----------------
def dynamic_wma(series: List[float], max_len: int = 6) -> float:
    """
    Compute a dynamic WMA using up to max_len most recent months.
    The weights adapt based on volatility: higher volatility -> smoother (older months get more weight).
    """
    n = len(series)
    if n == 0:
        return 0.0
    take = min(n, max_len)
    recent = series[-take:]
    # compute month-to-month relative changes
    if len(recent) >= 2:
        changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        vol = sum(changes) / len(changes) if changes else 0.0
    else:
        vol = 0.0

    # base weights favor recent months
    base_weights = [ (i + 1) for i in range(take) ]  # 1..take
    base_weights = list(reversed(base_weights))  # newest highest
    total = sum(base_weights)
    base_weights = [w/total for w in base_weights]

    # adaptation factor: more vol -> flatten weights
    # vol_ratio normalized roughly w.r.t average magnitude
    avg = sum(recent) / len(recent) if recent else 1.0
    vol_ratio = (vol / avg) if avg else 0.0
    # clamp vol_ratio
    vol_ratio = max(0.0, min(vol_ratio, 1.0))

    # blend between base_weights and equal weights
    equal_weights = [1.0 / take] * take
    blend = min(0.7, vol_ratio)  # limit blend to avoid extreme flattening
    weights = [(1 - blend) * bw + blend * ew for bw, ew in zip(base_weights, equal_weights)]
    # compute prediction
    prediction = sum(w * v for w, v in zip(weights, reversed(recent)))  # reversed so weights map newest->oldest
    return max(0.0, prediction)

# ----------------- Parameter tuning (lightweight) -----------------
def walk_forward_cv_mse(series: List[float], forecast_func, params: dict, min_train_size: int = 6) -> float:
    """
    Perform walk-forward validation computing MSE. forecast_func must accept (train_series, params) and return a single-step forecast.
    """
    n = len(series)
    if n < min_train_size + 1:
        # not enough data to validate -> return large error so tuner avoids complex models
        return float("inf")

    errors = []
    # iterate rolling window
    for split in range(min_train_size, n):
        train = series[:split]
        actual = series[split]
        try:
            pred = forecast_func(train, params)
        except Exception:
            return float("inf")
        if pred is None:
            return float("inf")
        errors.append((pred - actual) ** 2)
    return sum(errors) / len(errors) if errors else float("inf")


def forecast_wrapper_holt(train: List[float], params: dict) -> float:
    alpha = params.get("alpha", 0.5)
    beta = params.get("beta", 0.3)
    return holt_double_forecast(train, alpha, beta, n_forecast=1)[0]


def forecast_wrapper_hw(train: List[float], params: dict) -> float:
    alpha = params.get("alpha", 0.5)
    beta = params.get("beta", 0.3)
    gamma = params.get("gamma", 0.2)
    season_length = params.get("season_length", SEASONALITY_PERIOD)
    return holt_winters_additive(train, season_length, alpha, beta, gamma, n_forecast=1)[0]


def tune_parameters(series: List[float], seasonal: bool, season_length: int = SEASONALITY_PERIOD) -> dict:
    """
    Lightweight grid search for (alpha, beta, gamma) returning best params.
    Uses walk-forward CV to score parameter combinations.
    """
    best = None
    best_score = float("inf")
    combos_tested = 0

    if seasonal:
        grid = []
        for a in ALPHA_GRID:
            for b in BETA_GRID:
                for g in GAMMA_GRID:
                    grid.append({"alpha": a, "beta": b, "gamma": g, "season_length": season_length})
    else:
        grid = [{"alpha": a, "beta": b} for a in ALPHA_GRID for b in BETA_GRID]

    # cap combos
    if len(grid) > MAX_GRID_SEARCH_COMBINATIONS:
        grid = grid[:MAX_GRID_SEARCH_COMBINATIONS]

    for params in grid:
        combos_tested += 1
        if seasonal:
            score = walk_forward_cv_mse(series, forecast_wrapper_hw, params, min_train_size=max(6, season_length))
        else:
            score = walk_forward_cv_mse(series, forecast_wrapper_holt, params, min_train_size=6)
        if score < best_score:
            best_score = score
            best = params

    if best is None:
        # fallback default
        if seasonal:
            return {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "season_length": season_length}
        else:
            return {"alpha": 0.5, "beta": 0.3}

    return best

# ----------------- Top-level predictor combining everything -----------------
def _predict_next_month(history: List[MonthlyExpense]) -> float:
    """
    Comprehensive predictor:
    - builds continuous series and imputes missing months
    - auto-detects seasonality
    - tunes parameters (lightweight) per series
    - uses Holt-Winters if seasonal, else Holt
    - fallback to dynamic WMA for very short/noisy series
    """
    if not history:
        return 0.0

    # limit history length to MAX_HISTORY_MONTHS (use most recent months)
    history_sorted = sorted(history, key=lambda h: (h.year, h.month))
    if len(history_sorted) > MAX_HISTORY_MONTHS:
        history_sorted = history_sorted[-MAX_HISTORY_MONTHS:]

    # Build continuous series (may contain Nones for missing months)
    series_with_none, months = build_continuous_series(history_sorted)
    series = impute_missing(series_with_none)

    # if after imputation all zeros, return 0
    if all(v == 0.0 for v in series):
        return 0.0

    n = len(series)

    # If very short history (<=2) use simple rules / dynamic WMA
    if n <= 2:
        return round(dynamic_wma(series, max_len=2), 2)

    # Seasonality detection: needs at least 2 * season_length samples for reliability
    season_strength = seasonal_strength(series, period=SEASONALITY_PERIOD)
    is_seasonal = season_strength >= SEASONALITY_AMPLITUDE_THRESHOLD and n >= 2 * SEASONALITY_PERIOD

    # If not much data but still some seasonality signal present and we have at least season_length points,
    # we can still attempt seasonal HW but with care.
    season_length_used = SEASONALITY_PERIOD if is_seasonal else None

    # Tuning: per-series personalized coefficients
    try:
        tuned = tune_parameters(series, seasonal=is_seasonal, season_length=season_length_used or SEASONALITY_PERIOD)
    except Exception:
        tuned = None

    # If tuning failed or not enough data, fallback defaults
    if tuned is None:
        if is_seasonal:
            tuned = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "season_length": SEASONALITY_PERIOD}
        else:
            tuned = {"alpha": 0.5, "beta": 0.3}

    # Edge case: if the series is extremely volatile compared to mean, prefer dynamic WMA (more robust)
    mean_val = sum(series) / len(series) if series else 0.0
    diffs = [abs(series[i] - series[i - 1]) for i in range(1, len(series))] if len(series) >= 2 else [0.0]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    volatility_ratio = (avg_diff / mean_val) if mean_val else 0.0

    if volatility_ratio > 1.0 and n < 6:
        # extremely volatile and short history -> WMA is safer
        pred = dynamic_wma(series, max_len=min(6, n))
        return round(pred, 2)

    # Choose model
    if is_seasonal:
        alpha = tuned.get("alpha", 0.5)
        beta = tuned.get("beta", 0.3)
        gamma = tuned.get("gamma", 0.2)
        season_length = tuned.get("season_length", SEASONALITY_PERIOD)
        pred = holt_winters_additive(series, season_length, alpha, beta, gamma, n_forecast=1)[0]
    else:
        alpha = tuned.get("alpha", 0.5)
        beta = tuned.get("beta", 0.3)
        pred = holt_double_forecast(series, alpha, beta, n_forecast=1)[0]

    # final safety clamps
    if math.isnan(pred) or pred is None or pred < 0:
        # fallback to recent avg
        pred = sum(series[-3:]) / min(3, len(series))

    return round(float(pred), 2)


# ----------------- API endpoint -----------------
@app.get("/users/{user_id}/expense-prediction", response_model=PredictionResponse)
def predict_expense(user_id: str) -> PredictionResponse:
    try:
        user_object_id = ObjectId(user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid user id") from exc

    now = datetime.now(timezone.utc)
    # fetch up to MAX_HISTORY_MONTHS of history
    start_period = _shift_months(_first_day_of_month(now), -MAX_HISTORY_MONTHS + 1)
    prediction_month = _shift_months(_first_day_of_month(now), 1)

    pipeline = [
        {
            "$match": {
                "user": user_object_id,
                "type": "EXPENSE",
                "headCategory": {"$ne": None},
                "date": {"$gte": start_period},
            }
        },
        {
            "$project": {
                "amount": 1,
                "headCategory": 1,
                "year": {"$year": "$date"},
                "month": {"$month": "$date"},
            }
        },
        {
            "$group": {
                "_id": {
                    "headCategory": "$headCategory",
                    "year": "$year",
                    "month": "$month",
                },
                "total": {"$sum": "$amount"},
            }
        },
        {
            "$lookup": {
                "from": "headcategories",
                "localField": "_id.headCategory",
                "foreignField": "_id",
                "as": "headCategoryDoc",
            }
        },
        {"$unwind": "$headCategoryDoc"},
        {"$sort": {"_id.headCategory": 1, "_id.year": 1, "_id.month": 1}},
    ]

    results = list(mongo.transactions.aggregate(pipeline))

    grouped: Dict[ObjectId, Dict[str, List[MonthlyExpense]]] = defaultdict(lambda: {"history": []})

    for item in results:
        head_category_id: ObjectId = item["_id"]["headCategory"]
        category_record = grouped[head_category_id]
        category_record["title"] = item["headCategoryDoc"].get("title", "Unknown")
        category_record["history"].append(
            MonthlyExpense(
                year=item["_id"]["year"],
                month=item["_id"]["month"],
                total=float(item["total"]),
            )
        )

    categories: List[CategoryPrediction] = []
    for head_category_id, record in grouped.items():
        history = sorted(record["history"], key=lambda doc: (doc.year, doc.month))
        predicted_total = _predict_next_month(history)

        categories.append(
            CategoryPrediction(
                headCategoryId=str(head_category_id),
                title=record.get("title", "Unknown"),
                history=history,
                predictionMonth=MonthlyExpense(
                    year=prediction_month.year,
                    month=prediction_month.month,
                    total=predicted_total,
                ),
            )
        )

    return PredictionResponse(userId=user_id, categories=categories)


# Optional: health check
@app.get("/health")
def health():
    return {"status": "healthy"}











# import calendar
# import os
# from collections import defaultdict
# from datetime import datetime, timezone
# from typing import Dict, List

# from bson import ObjectId
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from pymongo import MongoClient
# from pymongo.collection import Collection

# load_dotenv()

# app = FastAPI(title="Expense Prediction API", version="1.0.0")


# class MonthlyExpense(BaseModel):
#     year: int
#     month: int
#     total: float = Field(..., description="Total expenses recorded for the month")


# class CategoryPrediction(BaseModel):
#     headCategoryId: str
#     title: str
#     history: List[MonthlyExpense]
#     predictionMonth: MonthlyExpense


# class PredictionResponse(BaseModel):
#     userId: str
#     categories: List[CategoryPrediction]


# class MongoConnection:
#     def __init__(self) -> None:
#         mongo_uri = os.getenv("MONGO_URI")
#         if not mongo_uri:
#             raise RuntimeError("MONGO_URI is not configured in the environment")

#         self._client = MongoClient(mongo_uri, tz_aware=True)
#         self._database = self._client.get_default_database()
#         self.transactions: Collection = self._database["transactions"]
#         self.headcategories: Collection = self._database["headcategories"]


# mongo = MongoConnection()


# def _first_day_of_month(dt: datetime) -> datetime:
#     return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# def _shift_months(dt: datetime, months: int) -> datetime:
#     month_index = dt.month - 1 + months
#     year = dt.year + month_index // 12
#     month = month_index % 12 + 1
#     last_day = calendar.monthrange(year, month)[1]
#     day = min(dt.day, last_day)
#     return dt.replace(year=year, month=month, day=day)


# # -----------------------------------------------------------
# # NEW: Weighted Moving Average-based prediction function
# # -----------------------------------------------------------

# def _predict_next_month(history: List[MonthlyExpense]) -> float:
#     """Predict next month's expense using Weighted Moving Average (WMA)."""
#     totals = [h.total for h in history]

#     # Only one month → Just repeat last month
#     if len(totals) == 1:
#         return round(totals[-1], 2)

#     # Two months → Slight smoothing
#     if len(totals) == 2:
#         last, prev = totals[-1], totals[-2]
#         prediction = last * 0.7 + prev * 0.3
#         return round(prediction, 2)

#     # Three or more months → Use 3-month WMA (0.5, 0.3, 0.2)
#     last3 = totals[-3:]
#     weights = [0.2, 0.3, 0.5]  # oldest → newest
#     prediction = sum(v * w for v, w in zip(last3, weights))

#     return round(prediction, 2)


# # -----------------------------------------------------------
# # EXPENSE PREDICTION ENDPOINT
# # -----------------------------------------------------------

# @app.get("/users/{user_id}/expense-prediction", response_model=PredictionResponse)
# def predict_expense(user_id: str) -> PredictionResponse:
#     try:
#         user_object_id = ObjectId(user_id)
#     except Exception as exc:
#         raise HTTPException(status_code=400, detail="Invalid user id") from exc

#     now = datetime.now(timezone.utc)
#     start_period = _shift_months(_first_day_of_month(now), -2)
#     prediction_month = _shift_months(_first_day_of_month(now), 1)

#     pipeline = [
#         {
#             "$match": {
#                 "user": user_object_id,
#                 "type": "EXPENSE",
#                 "headCategory": {"$ne": None},
#                 "date": {"$gte": start_period},
#             }
#         },
#         {
#             "$project": {
#                 "amount": 1,
#                 "headCategory": 1,
#                 "year": {"$year": "$date"},
#                 "month": {"$month": "$date"},
#             }
#         },
#         {
#             "$group": {
#                 "_id": {
#                     "headCategory": "$headCategory",
#                     "year": "$year",
#                     "month": "$month",
#                 },
#                 "total": {"$sum": "$amount"},
#             }
#         },
#         {
#             "$lookup": {
#                 "from": "headcategories",
#                 "localField": "_id.headCategory",
#                 "foreignField": "_id",
#                 "as": "headCategoryDoc",
#             }
#         },
#         {"$unwind": "$headCategoryDoc"},
#         {"$sort": {"_id.headCategory": 1, "_id.year": 1, "_id.month": 1}},
#     ]

#     results = list(mongo.transactions.aggregate(pipeline))

#     grouped: Dict[ObjectId, Dict[str, List[MonthlyExpense]]] = defaultdict(
#         lambda: {"history": []}
#     )

#     for item in results:
#         head_category_id: ObjectId = item["_id"]["headCategory"]
#         category_record = grouped[head_category_id]
#         category_record["title"] = item["headCategoryDoc"].get("title", "Unknown")
#         category_record["history"].append(
#             MonthlyExpense(
#                 year=item["_id"]["year"],
#                 month=item["_id"]["month"],
#                 total=float(item["total"]),
#             )
#         )

#     categories: List[CategoryPrediction] = []
#     for head_category_id, record in grouped.items():
#         history = sorted(record["history"], key=lambda doc: (doc.year, doc.month))
#         predicted_total = _predict_next_month(history)

#         categories.append(
#             CategoryPrediction(
#                 headCategoryId=str(head_category_id),
#                 title=record.get("title", "Unknown"),
#                 history=history,
#                 predictionMonth=MonthlyExpense(
#                     year=prediction_month.year,
#                     month=prediction_month.month,
#                     total=predicted_total,
#                 ),
#             )
#         )

#     return PredictionResponse(userId=user_id, categories=categories)
