"""
Recommendation routes for the Car Price API
===========================================
Endpoints
---------
GET /recommend          best-value cars for a budget (+ optional filters)
GET /similar/{car_id}   cars similar to a given listing

Needs one file next to this one (or set RECOMMENDER_PATH):
    recommender.pkl    dict with keys "listings", "feat_matrix", "nn"

Create it once, in your training script:
    with open("recommender.pkl", "wb") as f:
        pickle.dump({"listings": listings, "feat_matrix": feat_matrix, "nn": nn}, f)

Load it only in the same scikit-learn version you trained with, and only
files you created yourself (pickle can run code when loaded).
"""

import json
import os
import pickle
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

_HERE = Path(__file__).resolve().parent

# First file that exists wins. Set RECOMMENDER_PATH to use any other name/location.
_CANDIDATES = [
    Path(p) for p in [os.getenv("RECOMMENDER_PATH", "")] if p
] + [
    _HERE / "recommender.pkl",
    _HERE / "recomendation.pkl",
    _HERE / "recommendation.pkl",
]


def _find_pickle():
    for path in _CANDIDATES:
        if path.exists():
            return path
    return None

# Ignore "too good to be true" ads (typos / scams) when ranking deals.
MAX_DEAL_SCORE = 0.4

OUTPUT_COLS = [
    "nam", "Year", "Price", "Predicted_Price",
    "Deal_Score", "Millage_km", "Province",
]
SIMILAR_COLS = ["nam", "Year", "Price", "Millage_km", "Province"]

router = APIRouter(tags=["recommendations"])

_state = {"listings": None, "nn": None, "feat_matrix": None}


def _get_state():
    """Load the saved recommender on first use."""
    if _state["listings"] is not None:
        return _state

    path = _find_pickle()
    if path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Recommender pickle not found. Put recommender.pkl (or "
                f"recomendation.pkl) in {_HERE}."
            ),
        )

    with open(path, "rb") as f:
        saved = pickle.load(f)

    # car_id == row position (feat_matrix rows line up with listings rows)
    _state.update(
        listings=saved["listings"].reset_index(drop=True),
        nn=saved["nn"],
        feat_matrix=saved["feat_matrix"],
    )
    return _state


def _to_records(df):
    # to_json turns NaN into null, which plain dict conversion would not
    return json.loads(df.to_json(orient="records"))


@router.get("/recommend", summary="Recommend best-value cars")
def recommend(
    budget: float = Query(..., gt=0, description="Maximum price in PKR"),
    make: Optional[str] = Query(None, description="Text to match in the car name"),
    body_type: Optional[str] = Query(None, description="Exact value, e.g. Sedan"),
    fuel: Optional[str] = Query(None, description="Exact value, e.g. Petrol"),
    province: Optional[str] = Query(None, description="Exact value, e.g. Punjab"),
    min_year: Optional[int] = Query(None, ge=1900, le=2100),
    max_km: Optional[float] = Query(None, ge=0),
    top_n: int = Query(10, ge=1, le=50),
):
    df = _get_state()["listings"]

    df = df[df["Price"] <= budget]
    if make:
        df = df[df["nam"].str.contains(make, case=False, na=False)]
    if body_type:
        df = df[df["Body Type"] == body_type]  # dataset column has a space
    if fuel:
        df = df[df["Fuel"] == fuel]
    if province:
        df = df[df["Province"] == province]
    if min_year:
        df = df[df["Year"] >= min_year]
    if max_km is not None:
        df = df[df["Millage_km"] <= max_km]

    df = df[df["Deal_Score"].between(0, MAX_DEAL_SCORE)]
    df = df.sort_values("Deal_Score", ascending=False).head(top_n)

    out = df[OUTPUT_COLS].copy()
    out.insert(0, "car_id", out.index)
    return {"count": len(out), "results": _to_records(out)}


@router.get("/similar/{car_id}", summary="Find cars similar to a listing")
def similar(car_id: int, top_n: int = Query(5, ge=1, le=20)):
    state = _get_state()
    listings = state["listings"]

    if not 0 <= car_id < len(listings):
        raise HTTPException(status_code=404, detail="car_id not found")

    _, idx = state["nn"].kneighbors(
        state["feat_matrix"][car_id:car_id + 1], n_neighbors=top_n + 1
    )
    neighbours = [i for i in idx[0] if i != car_id][:top_n]

    out = listings.iloc[neighbours][SIMILAR_COLS].copy()
    out.insert(0, "car_id", out.index)
    return {"count": len(out), "results": _to_records(out)}