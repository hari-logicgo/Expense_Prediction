import calendar
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.collection import Collection

load_dotenv()

app = FastAPI(title="Expense Prediction API", version="1.0.0")


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


def _first_day_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _shift_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, last_day)
    return dt.replace(year=year, month=month, day=day)


def _predict_next_month(history: List[MonthlyExpense]) -> float:
    """Project next month using a simple trend extrapolation."""
    if len(history) >= 2:
        last = history[-1].total
        prev = history[-2].total
        change = last - prev
        projected = last + 0.5 * change  # dampened trend for stability
        return max(projected, 0.0)
    if history:
        return max(history[-1].total, 0.0)
    return 0.0


@app.get("/users/{user_id}/expense-prediction", response_model=PredictionResponse)
def predict_expense(user_id: str) -> PredictionResponse:
    try:
        user_object_id = ObjectId(user_id)
    except Exception as exc:  # pragma: no cover - fastapi handles validation
        raise HTTPException(status_code=400, detail="Invalid user id") from exc

    now = datetime.now(timezone.utc)
    start_period = _shift_months(_first_day_of_month(now), -2)
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
        predicted_total = round(_predict_next_month(history), 2)
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
