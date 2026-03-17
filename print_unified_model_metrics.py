import json
import os
from datetime import datetime


STATS_PATH = os.path.join("models", "movie_cinema_unified_stats.json")


def main() -> None:
    if not os.path.exists(STATS_PATH):
        print(f"Stats file not found at {STATS_PATH}")
        return

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    created_at = stats.get("created_at")
    try:
        created_dt = datetime.fromisoformat(created_at) if created_at else None
    except Exception:
        created_dt = None

    print("=" * 80)
    print("UNIFIED MOVIE+CINEMA MODEL METRICS")
    print("=" * 80)
    if created_dt:
        print(f"Created at      : {created_dt}")
    elif created_at:
        print(f"Created at      : {created_at}")

    print(f"Training rows   : {stats.get('n_rows')}")
    print(f"Movies          : {stats.get('n_movies')}")
    print(f"Cinemas         : {stats.get('n_cinemas')}")
    print(f"BMS target movies: {stats.get('n_bms_target_movies')}")
    print(f"Min rows/movie  : {stats.get('min_movie_rows_for_unified')}")

    metrics = stats.get("metrics", {})
    print("\n--- Test metrics (regression) ---")
    print(f"Train rows      : {metrics.get('train_rows')}")
    print(f"Test rows       : {metrics.get('test_rows')}")
    print(f"R²              : {metrics.get('r2')}")
    print(f"MAE             : {metrics.get('mae')}")
    print(f"RMSE            : {metrics.get('rmse')}")

    features = stats.get("features") or metrics.get("features") or []
    if features:
        print("\nFeatures used:")
        for f in features:
            print(f"  - {f}")

    print("\nFiles:")
    print(f"  Model file    : {stats.get('model_file')}")
    print(f"  Cinema encoder: {stats.get('cinema_encoder_file')}")
    print(f"  Movie encoder : {stats.get('movie_encoder_file')}")
    print(f"  Movie map     : {stats.get('movie_map_file')}")


if __name__ == "__main__":
    main()

