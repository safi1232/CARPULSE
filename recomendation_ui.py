"""
Recommendation UI for the Car Price Predictor Streamlit frontend.

Talks to two FastAPI endpoints:
    GET /recommend            -> best-value listings for a budget + filters
    GET /similar/{car_id}     -> listings similar to one recommended car

Every input is a dropdown. Recommended cars are shown with their poster
(fetched through your existing fetch_poster -> FastAPI /posters).

Usage in frontend.py:

    from recommendations_ui import render_recommendations

    render_recommendations(
        api_url,
        car_options=CAR_OPTIONS,
        fuel_options=FUEL_OPTIONS,
        province_options=PROVINCE_OPTIONS,
        body_type_options=BODY_TYPE_OPTIONS,
        split_car_name=split_car_name,
        fetch_poster=fetch_poster,
    )
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import pandas as pd
import requests
import streamlit as st

ANY = "Any"
NO_LIMIT = "No limit"
CUSTOM_BUDGET = "Custom amount…"

# The backend only returns Deal_Score values between 0 and MAX_DEAL_SCORE (0.4).
MAX_DEAL_SCORE_PCT = 40

REQUEST_TIMEOUT = 30

# Dropdown choices
BUDGET_PRESETS = [
    1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 3_500_000,
    4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000, 10_000_000,
    12_000_000, 15_000_000, 20_000_000, 25_000_000, 30_000_000, 40_000_000,
    50_000_000, 75_000_000, 100_000_000, 200_000_000,
]
DEFAULT_BUDGET = 3_000_000
MILEAGE_PRESETS = [10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 300_000]
RESULT_COUNTS = [5, 10, 15, 20, 30, 50]
SIMILAR_COUNTS = [3, 5, 10, 15, 20]
POSTER_COUNTS = [0, 3, 6, 9, 12]  # 0 = off
MIN_YEARS = list(range(2026, 1989, -1))

# Poster gallery
POSTER_WIDTH = 800
POSTER_COLUMNS = 3
POSTER_WORKERS = 4
POSTER_CACHE_LIMIT = 150

FetchPoster = Callable[..., tuple]  # (api_url, make, model, year, width) -> (img, ctype, err)
SplitCarName = Callable[[str], tuple]  # (name) -> (make, model, year)


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def _error_message(exc: requests.exceptions.RequestException) -> str:
    """Turn a requests exception into a short, readable message."""
    if isinstance(exc, requests.exceptions.Timeout):
        return "The request timed out. Please try again."
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Could not connect to the API. Make sure the FastAPI server is running."
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        return f"API error {status}: {detail}" if detail else f"API error {status}."
    return f"Request failed: {exc}"


def _get_json(url: str, params: Optional[dict] = None):
    """GET a JSON payload. Returns (data, error_message)."""
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, _error_message(exc)
    except ValueError:
        return None, "The API returned an invalid response."


def fetch_recommendations(
    api_url: str,
    *,
    budget: float,
    make: Optional[str] = None,
    body_type: Optional[str] = None,
    fuel: Optional[str] = None,
    province: Optional[str] = None,
    min_year: Optional[int] = None,
    max_km: Optional[float] = None,
    top_n: int = 10,
):
    """Call GET /recommend. Optional filters that are None/empty are not sent."""
    params = {"budget": budget, "top_n": top_n}
    optional = {
        "make": make,
        "body_type": body_type,
        "fuel": fuel,
        "province": province,
        "min_year": min_year,
        "max_km": max_km,
    }
    params.update({k: v for k, v in optional.items() if v is not None and v != ""})
    return _get_json(f"{api_url}/recommend", params)


def fetch_similar(api_url: str, car_id: int, top_n: int = 5):
    """Call GET /similar/{car_id}."""
    return _get_json(f"{api_url}/similar/{int(car_id)}", {"top_n": top_n})


# --------------------------------------------------------------------------
# Make / model dropdown data (built from your CAR_OPTIONS list)
# --------------------------------------------------------------------------
def _common_prefix(token_lists: list) -> list:
    """Longest shared run of leading words, e.g. [['Land','Cruiser','ZX'], ['Land','Cruiser','AX']]."""
    prefix = token_lists[0]
    for tokens in token_lists[1:]:
        n = 0
        while n < len(prefix) and n < len(tokens) and prefix[n] == tokens[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _model_choices(models: set) -> dict:
    """
    Build {dropdown label: text sent to the API} for one make.

    Models that start with the same word are grouped, and each group of two or
    more gets an "<family> (all)" entry (e.g. "Corolla (all)") before its
    individual variants.
    """
    groups: dict = {}
    for model in models:
        groups.setdefault(model.split()[0], []).append(model)

    choices = {}
    for group in groups.values():
        if len(group) > 1:
            family = " ".join(_common_prefix([m.split() for m in group]))
            choices[f"{family} (all)"] = family
            group = [m for m in group if m != family]  # covered by "(all)"
        for model in group:
            choices[model] = model
    return dict(sorted(choices.items(), key=lambda item: item[0].lower()))


@st.cache_data(show_spinner=False)
def _build_catalog(car_options: tuple, _split_car_name: SplitCarName) -> dict:
    """{make: {model label: search text}} for every make found in car_options."""
    models_by_make: dict = {}
    for name in car_options:
        make, model, _year = _split_car_name(name)
        # Skip junk labels such as a stray lowercase "a Raize Z 2020".
        if not make or not (make[0].isupper() or make[0].isdigit()):
            continue
        models = models_by_make.setdefault(make, set())
        if model:
            models.add(model)
    return {make: _model_choices(models) for make, models in models_by_make.items()}


# --------------------------------------------------------------------------
# Dropdown label helpers
# --------------------------------------------------------------------------
def _budget_label(value) -> str:
    if isinstance(value, str):
        return value
    label = f"PKR {value:,}"
    if value >= 10_000_000:
        return f"{label} ({value / 10_000_000:g} crore)"
    return f"{label} ({value / 100_000:g} lakh)"


def _mileage_label(value) -> str:
    return value if isinstance(value, str) else f"Up to {value:,} km"


def _poster_count_label(value: int) -> str:
    return "Off" if value == 0 else f"Top {value} cars"


# --------------------------------------------------------------------------
# Table helpers
# --------------------------------------------------------------------------
def _money(value) -> str:
    return f"{value:,.0f}" if pd.notna(value) else "—"


def _to_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _recommendation_frame(results: list) -> pd.DataFrame:
    """Raw numeric frame (used for metrics and cards) from the /recommend results."""
    df = pd.DataFrame(results)
    df = _to_numeric(
        df, ["car_id", "Year", "Price", "Predicted_Price", "Deal_Score", "Millage_km"]
    )
    df["Savings"] = df["Predicted_Price"] - df["Price"]
    return df


def _recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Display-ready table for /recommend results."""
    return pd.DataFrame(
        {
            "ID": df["car_id"].astype("Int64"),
            "Car": df["nam"],
            "Year": df["Year"].astype("Int64"),
            "Price (PKR)": df["Price"].map(_money),
            "Predicted (PKR)": df["Predicted_Price"].map(_money),
            "Savings (PKR)": df["Savings"].map(_money),
            "Deal score (%)": df["Deal_Score"] * 100,
            "Mileage (km)": df["Millage_km"].map(_money),
            "Province": df["Province"],
        }
    )


def _similar_table(results: list) -> pd.DataFrame:
    """Display-ready table for /similar results."""
    df = _to_numeric(pd.DataFrame(results), ["car_id", "Year", "Price", "Millage_km"])
    return pd.DataFrame(
        {
            "ID": df["car_id"].astype("Int64"),
            "Car": df["nam"],
            "Year": df["Year"].astype("Int64"),
            "Price (PKR)": df["Price"].map(_money),
            "Mileage (km)": df["Millage_km"].map(_money),
            "Province": df["Province"],
        }
    )


_ID_YEAR_CONFIG = {
    "ID": st.column_config.NumberColumn("ID", format="%d"),
    "Year": st.column_config.NumberColumn("Year", format="%d"),
}


# --------------------------------------------------------------------------
# Posters
# --------------------------------------------------------------------------
def _poster_key(api_url: str, spec: tuple) -> tuple:
    make, model, year = spec
    return (api_url, make, model, year, POSTER_WIDTH)


def _get_posters(api_url: str, specs: list, fetch_poster: FetchPoster) -> list:
    """
    Return [(img, content_type, error)] for each (make, model, year) spec.

    Missing posters are fetched in parallel and kept in session_state, so
    reruns (changing a dropdown, picking a similar car) don't hit the API again.
    """
    cache = st.session_state.setdefault("rec_poster_cache", {})
    keys = [_poster_key(api_url, spec) for spec in specs]
    todo = [key for key in dict.fromkeys(keys) if key not in cache]

    def load(key):
        try:
            return fetch_poster(*key)
        except Exception as exc:  # never let one bad poster break the page
            return None, None, f"Poster request failed: {exc}"

    if todo:
        with st.spinner(f"Loading {len(todo)} poster(s)..."):
            with ThreadPoolExecutor(max_workers=POSTER_WORKERS) as pool:
                for key, outcome in zip(todo, pool.map(load, todo)):
                    cache[key] = outcome

    # Keep the cache bounded: drop the oldest entries we aren't showing now.
    for old_key in list(cache):
        if len(cache) <= POSTER_CACHE_LIMIT:
            break
        if old_key not in keys:
            del cache[old_key]

    return [cache[key] for key in keys]


def _render_gallery(
    api_url: str,
    df: pd.DataFrame,
    count: int,
    split_car_name: SplitCarName,
    fetch_poster: FetchPoster,
) -> None:
    """Cards with poster + deal details for the top `count` recommended cars."""
    rows = list(df.head(count).itertuples())

    specs = []
    for row in rows:
        make, model, parsed_year = split_car_name(row.nam)
        year = int(row.Year) if pd.notna(row.Year) else parsed_year
        specs.append((make, model, year))

    st.markdown("#### 🖼️ Top picks")
    posters = _get_posters(api_url, specs, fetch_poster)

    for start in range(0, len(rows), POSTER_COLUMNS):
        columns = st.columns(POSTER_COLUMNS)
        for offset, column in enumerate(columns):
            index = start + offset
            if index >= len(rows):
                break
            row, (img, _content_type, err) = rows[index], posters[index]
            with column, st.container(border=True):
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.caption(f"🖼️ Poster unavailable: {err}")
                st.markdown(f"**#{index + 1} {row.nam}**")
                st.markdown(f"PKR {_money(row.Price)}")
                st.caption(
                    f"Market estimate PKR {_money(row.Predicted_Price)} · "
                    f"saves PKR {_money(row.Savings)} "
                    f"({row.Deal_Score * 100:.1f}% deal score)"
                )
                st.caption(
                    f"{_money(row.Millage_km)} km · {row.Province} · ID {int(row.car_id)}"
                )

    failed = [
        _poster_key(api_url, spec)
        for spec, (img, _c, _e) in zip(specs, posters)
        if img is None
    ]
    if failed and st.button("🔄 Retry failed posters", key="rec_poster_retry"):
        cache = st.session_state.get("rec_poster_cache", {})
        for key in failed:
            cache.pop(key, None)
        st.rerun()


# --------------------------------------------------------------------------
# Results rendering
# --------------------------------------------------------------------------
def _render_similar(api_url: str, df: pd.DataFrame) -> None:
    """'Find similar cars' picker + results for the current recommendations."""
    st.markdown("#### 🔍 Find similar cars")

    labels = {
        int(row.car_id): f"{row.nam} — PKR {_money(row.Price)}"
        for row in df.itertuples()
    }
    car_id = st.selectbox(
        "Pick a recommended car",
        options=list(labels),
        format_func=lambda cid: labels[cid],
        key="rec_similar_pick",
    )
    top_n = st.selectbox(
        "Number of similar cars",
        SIMILAR_COUNTS,
        index=SIMILAR_COUNTS.index(5),
        key="rec_similar_n",
    )

    if st.button("🔍 Show similar cars", use_container_width=True, key="rec_similar_btn"):
        with st.spinner("Finding similar cars..."):
            data, err = fetch_similar(api_url, car_id, top_n)
        st.session_state["rec_similar"] = {"car_id": car_id, "data": data, "error": err}

    similar = st.session_state.get("rec_similar")
    if not similar or similar["car_id"] not in labels:
        return

    st.markdown(f"**Cars similar to:** {labels[similar['car_id']]}")
    if similar["error"]:
        st.error(similar["error"])
        return

    rows = similar["data"].get("results", [])
    if not rows:
        st.info(similar["data"].get("message", "No similar cars available."))
        return

    st.dataframe(
        _similar_table(rows),
        hide_index=True,
        use_container_width=True,
        column_config=_ID_YEAR_CONFIG,
    )


def _render_results(
    api_url: str,
    result: dict,
    poster_count: int,
    split_car_name: SplitCarName,
    fetch_poster: FetchPoster,
) -> None:
    if result["error"]:
        st.error(result["error"])
        return

    rows = result["data"].get("results", [])
    if not rows:
        st.info("No cars match these filters. Try a higher budget or fewer filters.")
        return

    df = _recommendation_frame(rows)

    m1, m2, m3 = st.columns(3)
    m1.metric("Cars found", len(df))
    m2.metric("Best deal score", f"{df['Deal_Score'].iloc[0] * 100:.1f}%")
    m3.metric("Avg. savings vs. market", f"PKR {_money(df['Savings'].mean())}")

    if poster_count > 0:
        _render_gallery(api_url, df, poster_count, split_car_name, fetch_poster)

    st.markdown("#### 📋 All recommendations")
    st.dataframe(
        _recommendation_table(df),
        hide_index=True,
        use_container_width=True,
        column_config={
            **_ID_YEAR_CONFIG,
            "Deal score (%)": st.column_config.ProgressColumn(
                "Deal score (%)",
                min_value=0,
                max_value=MAX_DEAL_SCORE_PCT,
                format="%.1f",
            ),
        },
    )
    st.caption(
        "Deal score shows how far a listing's price sits below the model's "
        "predicted market price (higher is better). The backend only returns "
        f"scores between 0% and {MAX_DEAL_SCORE_PCT}%."
    )

    _render_similar(api_url, df)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def render_recommendations(
    api_url: str,
    *,
    car_options: Sequence[str],
    fuel_options: Sequence[str],
    province_options: Sequence[str],
    body_type_options: Sequence[str],
    split_car_name: SplitCarName,
    fetch_poster: FetchPoster,
) -> None:
    """Render the full recommendation section (filters, results, posters, similar cars)."""
    st.title("🎯 Best-Value Recommendations")
    st.write(
        "Choose a budget and optional filters to see listings priced below "
        "their estimated market value."
    )

    catalog = _build_catalog(tuple(car_options), split_car_name)
    makes = sorted(catalog, key=str.lower)

    # These are plain widgets (not an st.form) so the Model list can update
    # as soon as a Make is picked.
    col1, col2 = st.columns(2)
    with col1:
        budget_choice = st.selectbox(
            "Maximum budget",
            [*BUDGET_PRESETS, CUSTOM_BUDGET],
            index=BUDGET_PRESETS.index(DEFAULT_BUDGET),
            format_func=_budget_label,
            key="rec_budget_choice",
        )
        if budget_choice == CUSTOM_BUDGET:
            budget = st.number_input(
                "Custom budget (PKR)",
                min_value=100_000,
                max_value=1_000_000_000,
                value=DEFAULT_BUDGET,
                step=100_000,
                key="rec_budget_custom",
            )
        else:
            budget = budget_choice

        make = st.selectbox("Make", [ANY, *makes], key="rec_make")
        if make == ANY:
            model = st.selectbox(
                "Model",
                [ANY],
                disabled=True,
                help="Pick a make first.",
                key="rec_model_none",
            )
        else:
            model_choices = catalog[make]
            model = st.selectbox(
                "Model",
                [ANY, *model_choices],
                key=f"rec_model_{make}",  # per-make key: switching make resets Model
            )
        fuel = st.selectbox("Fuel type", [ANY, *fuel_options], key="rec_fuel")

    with col2:
        body_type = st.selectbox(
            "Body type", [ANY, *body_type_options], key="rec_body_type"
        )
        province = st.selectbox(
            "Province", [ANY, *province_options], key="rec_province"
        )
        min_year = st.selectbox(
            "Minimum year", [ANY, *MIN_YEARS], key="rec_min_year"
        )
        max_km = st.selectbox(
            "Maximum mileage",
            [NO_LIMIT, *MILEAGE_PRESETS],
            format_func=_mileage_label,
            key="rec_max_km",
        )

    col3, col4 = st.columns(2)
    with col3:
        top_n = st.selectbox(
            "Number of results",
            RESULT_COUNTS,
            index=RESULT_COUNTS.index(10),
            key="rec_top_n",
        )
    with col4:
        poster_count = st.selectbox(
            "Show posters for",
            POSTER_COUNTS,
            index=POSTER_COUNTS.index(6),
            format_func=_poster_count_label,
            key="rec_poster_count",
        )

    if st.button("🎯 Find best deals", use_container_width=True, key="rec_search"):
        if make == ANY:
            make_query = None
        elif model == ANY:
            make_query = make
        else:
            make_query = f"{make} {catalog[make][model]}"

        with st.spinner("Searching listings..."):
            data, err = fetch_recommendations(
                api_url,
                budget=float(budget),
                make=make_query,
                body_type=None if body_type == ANY else body_type,
                fuel=None if fuel == ANY else fuel,
                province=None if province == ANY else province,
                min_year=None if min_year == ANY else int(min_year),
                max_km=None if max_km == NO_LIMIT else float(max_km),
                top_n=top_n,
            )
        st.session_state["rec_result"] = {"data": data, "error": err}
        st.session_state.pop("rec_similar", None)  # drop stale similar-car results

    result = st.session_state.get("rec_result")
    if result:
        _render_results(api_url, result, poster_count, split_car_name, fetch_poster)