import json
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


DATA_FILE = r"d:\Forecasting\data.csv.xlsx"
MODELS_DIR = r"d:\Forecasting\models"

UNIFIED_MODEL_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_model.json")
UNIFIED_CINEMA_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_cinema_encoder.pkl")
UNIFIED_MOVIE_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_encoder.pkl")
UNIFIED_STATS_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_stats.json")

AVG_TICKET_PRICE_RS = 200
TIME_SLOTS = ["09:00", "12:00", "15:00", "18:00", "21:00"]


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    cleaned = re.sub(r"\(.*?\)", "", name)
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned)
    return cleaned.strip().lower()


def extract_cinema_id(row):
    for col in ["cinema_id", "fk_cinema_id"]:
        if col in row and pd.notna(row[col]):
            return str(int(float(row[col])))

    for col in ["PGroup_strName", "PGroup_strCode", "PGroup_strCode.1"]:
        if col in row and pd.notna(row[col]):
            txt = str(row[col])
            nums = re.findall(r"\b\d{3,5}\b", txt)
            if nums:
                return nums[0]

    return "890"


def safe_encode(encoder, value):
    value = str(value)
    classes = set(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    return 0


def load_resources():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    required = [
        UNIFIED_MODEL_FILE,
        UNIFIED_CINEMA_ENCODER_FILE,
        UNIFIED_MOVIE_ENCODER_FILE,
        UNIFIED_STATS_FILE,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing unified artifacts: {missing}")

    df = pd.read_excel(DATA_FILE)
    model = xgb.XGBRegressor()
    model.load_model(UNIFIED_MODEL_FILE)
    cinema_encoder = joblib.load(UNIFIED_CINEMA_ENCODER_FILE)
    movie_encoder = joblib.load(UNIFIED_MOVIE_ENCODER_FILE)

    with open(UNIFIED_STATS_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)

    features = stats.get("features", [
        "day_of_week", "is_weekend", "hour",
        "log_days_since_release",
        "cinema_id_encoded", "movie_id_encoded",
        "competitors_on_screen",
        "cinema_avg_daily", "cinema_hour_avg",
        "budget", "runtime", "popularity", "vote_average",
        "movie_trend_7d", "cinema_trend_7d",
    ])

    return df, model, cinema_encoder, movie_encoder, features


def build_prediction_dataframe(df):
    work = df.copy()
    work["show_time"] = pd.to_datetime(work.get("Session_dtmShowing"), errors="coerce")
    work = work[work["show_time"].notna()].copy()

    work["movie_name"] = work.get("Film_strTitle", "").astype(str)
    work["cinema_id"] = work.apply(extract_cinema_id, axis=1)

    work["day_of_week"] = work["show_time"].dt.dayofweek
    work["is_weekend"] = (work["day_of_week"] >= 4).astype(int)
    work["hour"] = work["show_time"].dt.hour

    work["runtime"] = pd.to_numeric(work.get("Film_intDuration", 0), errors="coerce").fillna(120)
    work["budget"] = 50000000
    work["popularity"] = 10
    work["vote_average"] = 7.0
    work["competitors_on_screen"] = 5
    work["movie_trend_7d"] = 0
    work["cinema_trend_7d"] = 0
    work["cinema_avg_daily"] = 500
    work["cinema_hour_avg"] = 50
    work["log_days_since_release"] = 2.0

    return work


def predict_sold_tickets(work_df, model, cinema_encoder, movie_encoder, features):
    predict_df = work_df.copy()
    predict_df["cinema_id_encoded"] = predict_df["cinema_id"].astype(str).apply(
        lambda x: safe_encode(cinema_encoder, x)
    )

    predict_df["movie_id_encoded"] = predict_df["movie_name"].astype(str).apply(
        lambda x: safe_encode(movie_encoder, x)
    )

    for f in features:
        if f not in predict_df.columns:
            predict_df[f] = 0
        predict_df[f] = pd.to_numeric(predict_df[f], errors="coerce").fillna(0)

    raw_pred = np.maximum(model.predict(predict_df[features]), 0)
    predict_df["sold_tickets"] = np.round(raw_pred, 0).astype(int)
    predict_df["estimated_revenue_rs"] = predict_df["sold_tickets"] * AVG_TICKET_PRICE_RS
    return predict_df


def save_enriched_data(pred_df):
    out_file = "data_with_sold_tickets.xlsx"
    pred_df.to_excel(out_file, index=False)
    print(f"Saved enriched data: {out_file}")


def generate_schedule_excel(pred_df):
    unique_dates = sorted(pred_df["show_time"].dt.date.unique())
    movies = sorted(pred_df["movie_name"].dropna().unique().tolist())
    cinema_ids = sorted(pred_df["cinema_id"].dropna().unique().tolist())

    for cid in cinema_ids:
        for d in unique_dates:
            rows = []
            date_mask = pred_df["show_time"].dt.date == d
            cinema_day = pred_df[(pred_df["cinema_id"] == cid) & date_mask].copy()
            if cinema_day.empty:
                continue

            screens = sorted(cinema_day.get("Screen_bytNum", pd.Series([1, 2, 3, 4, 5])).dropna().unique().tolist())
            screens = [int(s) for s in screens if int(s) > 0] or [1, 2, 3, 4, 5]

            for slot in TIME_SLOTS:
                slot_dt = pd.to_datetime(f"{d} {slot}")
                slot_scores = []
                for mv in movies:
                    m = cinema_day[cinema_day["movie_name"] == mv]
                    if m.empty:
                        continue

                    same_hour = m[m["show_time"].dt.hour == slot_dt.hour]
                    base = same_hour if not same_hour.empty else m
                    pred = float(base["sold_tickets"].mean())
                    slot_scores.append({"name": mv, "sales": pred})

                slot_scores.sort(key=lambda x: x["sales"], reverse=True)
                row = {"Date": d.strftime("%Y-%m-%d"), "Time": slot}

                for i, screen in enumerate(screens):
                    if not slot_scores:
                        row[f"Audi {screen}"] = "No Movie"
                        continue

                    winner = slot_scores[i % len(slot_scores)]
                    runner = slot_scores[(i + 1) % len(slot_scores)]
                    diff = winner["sales"] - runner["sales"]
                    rev_diff = int(round(diff * AVG_TICKET_PRICE_RS))
                    row[f"Audi {screen}"] = (
                        f"{winner['name']}  |  Expected +{rev_diff} Rs vs {runner['name']} "
                        f"[Key Factor: Unified Cinepolis model]"
                    )

                rows.append(row)

            if rows:
                out = pd.DataFrame(rows).set_index(["Date", "Time"])
                out_file = f"Proposed_Schedule_Cinepolis_{cid}_{d.strftime('%Y-%m-%d')}.xlsx"
                out.to_excel(out_file)
                print(f"Saved schedule: {out_file}")


def main():
    print("Loading Cinepolis unified resources...")
    df, model, cinema_encoder, movie_encoder, features = load_resources()
    print(f"Loaded {len(df)} rows from data file")

    work = build_prediction_dataframe(df)
    pred_df = predict_sold_tickets(work, model, cinema_encoder, movie_encoder, features)
    save_enriched_data(pred_df)
    generate_schedule_excel(pred_df)
    print("Done.")


if __name__ == "__main__":
    main()
