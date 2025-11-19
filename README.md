# Expense Prediction API

FastAPI service that reads transactions from MongoDB and predicts a user's next-month expense based on the last two months of **EXPENSE** transactions.

## Prerequisites

- Python 3.11+
- MongoDB instance containing an `expenses.transactions` collection
- `.env` file with `MONGO_URI` (already included)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app.main:app --reload
```

## Endpoint

`GET /users/{user_id}/expense-prediction`

Example response:

```json
{
  "userId": "68a834c3f4694b11efedacd2",
  "categories": [
    {
      "headCategoryId": "68a834c3f4694b11efedacd6",
      "title": "Food & Drinks",
      "predictionMonth": {
        "year": 2025,
        "month": 12,
        "total": 450.25
      },
      "history": [
        {"year": 2025, "month": 10, "total": 400.5},
        {"year": 2025, "month": 11, "total": 500.0}
      ]
    }
  ]
}
```

- `history` lists totals for each of the last two months (max) that contain **EXPENSE** entries for that head category.
- `predictionMonth.total` applies a simple trend-based forecast (dampened extrapolation of the last two months; falls back to the most recent total).

The service only performs read operations on MongoDB.
