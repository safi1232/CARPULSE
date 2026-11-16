import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
from scipy import sparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter, Query
from fastapi.responses import Response

from schemas.user_input import UserInput
from schemas.prediction_response import PredictionResponse
from model.predict import predict_output, model, MODEL_VERSION
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CARIMAGES_API_KEY = os.getenv("CARIMAGES_API_KEY", "").strip()
CARIMAGES_API_URL = "https://carimagesapi.com/image"

HERE = Path(__file__).resolve().parent

MAX_DEAL_SCORE = 0.4

OUTPUT_COLS = [
    "nam",
    "Year",
    "Price",
    "Predicted_Price",
    "Deal_Score",
    "Millage_km",
    "Province",
]

SIMILAR_COLS = [
    "nam",
    "Year",
    "Price",
    "Millage_km",
    "Province",
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Car Price Prediction API",
    description="API for used car price prediction and recommendations",
    version=MODEL_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cars-45n.pages.dev",
        "http://localhost:5173",
        "https://carpulse-fxa430z2b-acme-4a68.vercel.app",
        "https://carpulse-67gq9o9bh-acme-4a68.vercel.app",
        "https://carpulse-ep6slj84g-acme-4a68.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# RECOMMENDER CONFIGURATION
# ============================================================

recommender_env = os.getenv("RECOMMENDER_PATH", "").strip()

CANDIDATES = []

if recommender_env:
    recommender_path = Path(recommender_env)

    if not recommender_path.is_absolute():
        recommender_path = HERE / recommender_path

    CANDIDATES.append(recommender_path)

CANDIDATES.extend([
    HERE / "recommender.pkl",
    HERE / "recomendation.pkl",
    HERE / "recommendation.pkl",
])

_state = {
    "listings": None,
    "nn": None,
    "feat_matrix": None,
    "body_type_col": None,
    "pickle_path": None,
}


def _find_pickle():
    for path in CANDIDATES:
        if path.exists() and path.is_file():
            return path

    return None


def _get_state():
    """
    Load and validate recommender pickle once.
    """

    if _state["listings"] is not None:
        return _state

    path = _find_pickle()

    if path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Recommender pickle not found. "
                f"Expected one of: {[str(p) for p in CANDIDATES]}"
            ),
        )

    try:
        with open(path, "rb") as file:
            saved = pickle.load(file)

        if not isinstance(saved, dict):
            raise ValueError("Recommender pickle must contain a dictionary.")

        required_keys = ["listings", "nn", "feat_matrix"]

        for key in required_keys:
            if key not in saved:
                raise ValueError(
                    f"Missing '{key}' in recommender pickle."
                )

        listings = saved["listings"]

        if not isinstance(listings, pd.DataFrame):
            raise ValueError("'listings' must be a pandas DataFrame.")

        listings = listings.reset_index(drop=True)

        # Detect the actual body type column.
        if "Body Type" in listings.columns:
            body_type_col = "Body Type"
        elif "Body_Type" in listings.columns:
            body_type_col = "Body_Type"
        else:
            raise ValueError(
                "Neither 'Body Type' nor 'Body_Type' exists in listings."
            )

        required_columns = set(
            OUTPUT_COLS + ["Fuel", body_type_col]
        )

        missing_columns = required_columns - set(listings.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in listings: {sorted(missing_columns)}"
            )

        nn = saved["nn"]
        feat_matrix = saved["feat_matrix"]

        if not hasattr(feat_matrix, "shape"):
            raise ValueError("Feature matrix must have a shape attribute.")

        # Sparse matrices don't support len(); convert to CSR so that
        # row slicing in /similar works, and use shape[0] for row count.
        if sparse.issparse(feat_matrix):
            feat_matrix = feat_matrix.tocsr()

        if feat_matrix.shape[0] != len(listings):
            raise ValueError(
                "Feature matrix and listings have different row counts."
            )

        if not hasattr(nn, "kneighbors"):
            raise ValueError(
                "Saved nearest-neighbor model is invalid."
            )

        if getattr(nn, "n_samples_fit_", None) != len(listings):
            raise ValueError(
                "Nearest-neighbor model and listings have different row counts."
            )

        _state.update({
            "listings": listings,
            "nn": nn,
            "feat_matrix": feat_matrix,
            "body_type_col": body_type_col,
            "pickle_path": str(path),
        })

        logger.info(
            "Recommender loaded successfully from %s",
            path
        )

        return _state

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Recommender loading failed")

        raise HTTPException(
            status_code=503,
            detail=f"Could not load recommender: {str(e)}",
        )


def _to_records(df):
    """
    Convert DataFrame into JSON-safe records.
    NaN and infinite values become null.
    """

    safe_df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return json.loads(
        safe_df.to_json(orient="records")
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Car Price Prediction API is running",
        "model_version": MODEL_VERSION,
        "endpoints": {
            "health": "/health",
            "prediction": "/predict",
            "recommendations": "/recommend",
            "similar_cars": "/similar/{car_id}",
            "car_posters": "/posters",
            "documentation": "/docs",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    recommender_status = (
        "loaded"
        if _state["listings"] is not None
        else "not_loaded"
    )

    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_version": MODEL_VERSION,
        "prediction_model": model is not None,
        "recommender": recommender_status,
        "recommender_path": _state["pickle_path"],
    }


# ============================================================
# CAR PRICE PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(data: UserInput):

    try:
        user_data = {
            "nam": data.nam,
            "Year": data.Year,
            "Fuel": data.Fuel,
            "Transmission": data.Transmission,
            "Province": data.Province,
            "Color": data.Color,
            "Assembly": data.Assembly,
            "Body Type": data.Body_Type,
            "Features": ", ".join(data.Features),
            "Millage_km": data.Millage_km,
            "Engine_Value": data.Engine_Value,
            "Engine_Unit": data.Engine_Unit,
        }

        prediction = predict_output(user_data)

        if prediction is None:
            raise ValueError("Prediction returned None.")

        prediction = float(prediction)

        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError(
                f"Invalid predicted price: {prediction}"
            )

        return {
            "predicted_price": round(prediction, 2),
            "model_version": MODEL_VERSION,
            "currency": "PKR",
            "message": "Car price predicted successfully",
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ============================================================
# CAR POSTER / IMAGE
# ============================================================

@app.get(
    "/posters",
    summary="Get Car Poster",
    description="Fetch a car poster using make, model and year.",
)
def get_car_poster(
    make: str = Query(
        ...,
        min_length=1,
        description="Car manufacturer",
    ),
    model_name: str = Query(
        ...,
        alias="model",
        min_length=1,
        description="Car model",
    ),
    year: int = Query(
        ...,
        ge=1900,
        le=2100,
        description="Manufacturing year",
    ),
    width: int = Query(
        800,
        ge=200,
        le=2000,
        description="Image width",
    ),
):

    if not CARIMAGES_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="CarImages API key is missing.",
        )

    params = {
        "make": make,
        "model": model_name,
        "year": year,
        "width": width,
        "key": CARIMAGES_API_KEY,
    }

    try:
        response = requests.get(
            CARIMAGES_API_URL,
            params=params,
            timeout=(10, 60),
            allow_redirects=True,
        )

        if response.status_code == 401:
            raise HTTPException(
                status_code=502,
                detail="CarImages API key is invalid or inactive.",
            )

        if response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="Car poster not found.",
            )

        response.raise_for_status()

        content_type = (
            response.headers.get("Content-Type", "")
            .split(";")[0]
            .strip()
            .lower()
        )

        if content_type.startswith("image/"):
            return Response(
                content=response.content,
                media_type=content_type,
                headers={
                    "Content-Disposition": (
                        'inline; filename="car_poster.webp"'
                    ),
                    "Cache-Control": "public, max-age=3600",
                },
            )

        raise HTTPException(
            status_code=502,
            detail="Unexpected response from CarImages API.",
        )

    except HTTPException:
        raise

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="CarImages API request timed out.",
        )

    except requests.exceptions.RequestException as e:
        logger.exception("Car image request failed")

        raise HTTPException(
            status_code=502,
            detail=f"Image provider connection failed: {str(e)}",
        )


# ============================================================
# CAR RECOMMENDATIONS ROUTER
# ============================================================

router = APIRouter(
    tags=["Car Recommendations"]
)


# ============================================================
# BEST VALUE CAR RECOMMENDATIONS
# ============================================================

@router.get(
    "/recommend",
    summary="Recommend best-value cars",
)
def recommend(
    budget: float = Query(
        ...,
        gt=0,
        description="Maximum budget in PKR",
    ),
    make: Optional[str] = Query(
        None,
        description="Car manufacturer or name",
    ),
    body_type: Optional[str] = Query(
        None,
        description="Body type, e.g. Sedan",
    ),
    fuel: Optional[str] = Query(
        None,
        description="Fuel type, e.g. Petrol",
    ),
    province: Optional[str] = Query(
        None,
        description="Province, e.g. Punjab",
    ),
    min_year: Optional[int] = Query(
        None,
        ge=1900,
        le=2100,
    ),
    max_km: Optional[float] = Query(
        None,
        ge=0,
    ),
    top_n: int = Query(
        10,
        ge=1,
        le=50,
    ),
):

    state = _get_state()

    listings = state["listings"]
    body_type_col = state["body_type_col"]

    # Preserve original listing IDs.
    df = listings.copy()
    df.insert(0, "car_id", df.index)

    # Ensure numeric columns are usable.
    numeric_columns = [
        "Price",
        "Predicted_Price",
        "Deal_Score",
        "Year",
        "Millage_km",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove rows with invalid essential values.
    df = df.dropna(
        subset=[
            "Price",
            "Predicted_Price",
            "Deal_Score",
            "Year",
            "Millage_km",
        ]
    )

    df = df[
        (df["Price"] > 0)
        & (df["Predicted_Price"] > 0)
    ]

    # Budget.
    df = df[df["Price"] <= budget]

    # Manufacturer or car name.
    if make:
        df = df[
            df["nam"].astype(str).str.contains(
                make.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    # Body type.
    if body_type:
        df = df[
            df[body_type_col].astype(str).str.strip().str.lower()
            == body_type.strip().lower()
        ]

    # Fuel.
    if fuel:
        df = df[
            df["Fuel"].astype(str).str.strip().str.lower()
            == fuel.strip().lower()
        ]

    # Province.
    if province:
        df = df[
            df["Province"].astype(str).str.strip().str.lower()
            == province.strip().lower()
        ]

    # Minimum manufacturing year.
    if min_year is not None:
        df = df[df["Year"] >= min_year]

    # Maximum mileage.
    if max_km is not None:
        df = df[df["Millage_km"] <= max_km]

    # Keep only reasonable positive deal scores.
    df = df[
        df["Deal_Score"].between(
            0,
            MAX_DEAL_SCORE,
            inclusive="both",
        )
    ]

    # Highest deal score first.
    df = df.sort_values(
        by="Deal_Score",
        ascending=False,
    ).head(top_n)

    output_columns = ["car_id"] + OUTPUT_COLS

    out = df[output_columns].copy()

    return {
        "count": len(out),
        "results": _to_records(out),
        "message": (
            "Recommendations found successfully."
            if not out.empty
            else "No cars found matching your filters."
        ),
    }


# ============================================================
# SIMILAR CARS
# ============================================================

@router.get(
    "/similar/{car_id}",
    summary="Find cars similar to a listing",
)
def similar(
    car_id: int,
    top_n: int = Query(
        5,
        ge=1,
        le=20,
    ),
):

    state = _get_state()

    listings = state["listings"]
    nn = state["nn"]
    feat_matrix = state["feat_matrix"]

    if not 0 <= car_id < len(listings):
        raise HTTPException(
            status_code=404,
            detail="car_id not found.",
        )

    available_cars = len(listings) - 1

    if available_cars == 0:
        return {
            "count": 0,
            "results": [],
            "message": "No similar cars available.",
        }

    actual_top_n = min(
        top_n,
        available_cars,
    )

    try:
        _, indices = nn.kneighbors(
            feat_matrix[car_id:car_id + 1],
            n_neighbors=actual_top_n + 1,
        )

        neighbours = [
            int(index)
            for index in indices[0]
            if index != car_id
        ][:actual_top_n]

        out = listings.iloc[neighbours][SIMILAR_COLS].copy()

        out.insert(
            0,
            "car_id",
            neighbours,
        )

        return {
            "count": len(out),
            "results": _to_records(out),
            "message": "Similar cars retrieved successfully.",
        }

    except Exception as e:
        logger.exception("Similar car recommendation failed")

        raise HTTPException(
            status_code=500,
            detail=f"Similar car recommendation failed: {str(e)}",
        )


# ============================================================
# REGISTER ROUTER
# ============================================================

app.include_router(router)