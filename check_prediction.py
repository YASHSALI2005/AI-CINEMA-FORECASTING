"""
Check whether PREDICTED sales vs REAL (actual) sales match.
Edit only the CONFIG block at the top: movie, cinema_id, date, and 3 scenario params.
"""

import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from datetime import datetime

# =============================================================================
#  EDIT ONLY THIS BLOCK — change these 6 values and run:  python check_prediction.py
# =============================================================================

MOVIE_NAME   = "ANIMAL (HINDI)"      # Must exist in movie_features_safe.csv (original_name)
CINEMA_ID    = 412                   # Cinema ID (e.g. 412, 890, 1049)
DATE         = "2023-12-02"          # Show date, YYYY-MM-DD

# 3 scenario parameters (used by the model)
SCENARIO_COMPETITORS = 2             # How many other movies on screen? (0–10)
SCENARIO_MOVIE_HYPE  = 380.0        # Movie hype / trend last 7d (0–500, 380 = blockbuster)
SCENARIO_CINEMA      = 110.0        # Cinema status last 7d (0–500, 100 = normal)

# Optional: show hour (24h). Used to pick closest real show and for prediction. Default 21 = 9 PM.
SHOW_HOUR = 21

# =============================================================================
#  DO NOT EDIT BELOW
# =============================================================================

from holiday_utils import get_holiday_weight

def _normalize_date(s):
    d = pd.to_datetime(s)
    return d.strftime("%Y-%m-%d")

def run():
    print(f"\n[*] Loading model & data for: {MOVIE_NAME} @ Cinema {CINEMA_ID} on {DATE} ...")

    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json")
    encoder = joblib.load("cinema_encoder_v4.pkl")
    movies_df = pd.read_csv("movie_features_safe.csv")
    history_df = pd.read_csv("final_training_data_v3.csv")

    # --- Movie metadata ---
    match = movies_df[movies_df["original_name"] == MOVIE_NAME]
    if match.empty:
        print(f"[X] Movie '{MOVIE_NAME}' not in movie_features_safe.csv")
        return
    movie_data = match.iloc[0]

    # --- Actual sales from history (match movie + cinema; prefer original_name or movie_name) ---
    history_df["show_time"] = pd.to_datetime(history_df["show_time"])
    date_norm = _normalize_date(DATE)
    target_ts = pd.to_datetime(f"{date_norm} {SHOW_HOUR}:00:00")

    mask = (
        ((history_df["movie_name"] == MOVIE_NAME) | (history_df["original_name"] == MOVIE_NAME))
        & (history_df["cinema_id"].astype(str) == str(CINEMA_ID))
    )
    candidates = history_df[mask].copy()
    candidates["time_diff"] = (candidates["show_time"] - target_ts).abs()
    closest = candidates.nsmallest(1, "time_diff")

    if not closest.empty and closest.iloc[0]["time_diff"].total_seconds() < 7200:
        actual_sales = int(closest.iloc[0]["sold_tickets"])
        real_show = closest.iloc[0]["show_time"]
        print(f"[OK] Found actual: {real_show} -> sold_tickets = {actual_sales}")
    else:
        actual_sales = None
        print("[!] No matching show in final_training_data_v3.csv for this movie/cinema/date. Actual = N/A.")

    # --- Features for prediction ---
    date_obj = pd.to_datetime(DATE)
    day_of_week = date_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    rel = pd.to_datetime(movie_data["release_date"], errors="coerce")
    sel = pd.to_datetime(DATE)
    days_since = max((sel.normalize() - rel.normalize()).days, 0) if pd.notna(rel) else 0
    log_days = np.log1p(days_since)
    hw = get_holiday_weight(date_norm)

    try:
        cinema_enc = encoder.transform([str(CINEMA_ID)])[0]
    except Exception:
        cinema_enc = 0
        print(f"[!] Cinema ID {CINEMA_ID} not in encoder; using 0.")

    X = pd.DataFrame({
        "budget": [movie_data["budget"]],
        "runtime": [movie_data["runtime"]],
        "popularity": [movie_data["popularity"]],
        "vote_average": [movie_data["vote_average"]],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "hour": [SHOW_HOUR],
        "holiday_weight": [hw],
        "competitors_on_screen": [SCENARIO_COMPETITORS],
        "log_days_since_release": [log_days],
        "cinema_id_encoded": [cinema_enc],
        "movie_trend_7d": [SCENARIO_MOVIE_HYPE],
        "cinema_trend_7d": [SCENARIO_CINEMA],
    })

    predicted = max(int(model.predict(X)[0]), 0)

    # --- Report ---
    print()
    print("=" * 55)
    print(f"  PREDICTED vs ACTUAL - {MOVIE_NAME}")
    print("=" * 55)
    print(f"  Cinema: {CINEMA_ID}  |  Date: {DATE}  |  Hour: {SHOW_HOUR}:00")
    print(f"  Scenario: Competitors={SCENARIO_COMPETITORS}, Hype={SCENARIO_MOVIE_HYPE}, Cinema={SCENARIO_CINEMA}")
    print("-" * 55)
    print(f"  PREDICTED:  {predicted} tickets")
    print(f"  ACTUAL:     {actual_sales if actual_sales is not None else 'N/A'}")
    print("-" * 55)

    if actual_sales is not None:
        diff = predicted - actual_sales
        err = abs(diff) / actual_sales * 100
        acc = (1 - abs(diff) / actual_sales) * 100
        print(f"  Difference:  {diff:+d}  |  Error: {err:.1f}%  |  Accuracy: {acc:.1f}%")
        if err <= 15:
            verdict = "MATCH (within 15%)"
        elif err <= 30:
            verdict = "CLOSE (15-30% off)"
        else:
            verdict = "OFF (>30% off)"
        print(f"  Verdict: {verdict}")
    else:
        print("  Verdict: (no actual data to compare)")
    print("=" * 55)

if __name__ == "__main__":
    run()
