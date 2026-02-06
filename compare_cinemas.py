"""
Table of ALL cinemas x common time slots: for ONE movie, show average occupancy
(avg sold tickets) per cinema per time slot. Uses DB if USE_DB=1; else CSV.

Output: table with columns [cinema_id, slot_08:00, slot_10:00, ...] and one row
per cinema_id; each cell = average sold_tickets for that cinema in that slot.

Edit CONFIG, then run:  python compare_cinemas.py
"""

import os
import re
import pandas as pd

# Load .env if present (for DB_PASSWORD etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
#  CONFIG
# =============================================================================
MOVIE_NAME    = "RAID 2 (HINDI)"      # e.g. "WAR 2" or "War 2 Hindi" (matched case-insensitive)
USE_DB        = 0            # 1 = load from MySQL m_grouped_transactions

# Slot start hours (common show times: 10am, 2pm, 6pm, 10pm)
# Slot length = movie runtime (dynamic). Slots are built in run() from runtime.
SLOT_START_HOURS = [10, 14, 18, 22]
DEFAULT_RUNTIME_MIN = 120   # fallback if movie not in features (2 hr)
CAPACITY_PER_SCREEN = 300   # seats per screen for occupancy %

# =============================================================================
#  Load data from DB or CSV
# =============================================================================

def _normalize_movie(s: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", str(s)).strip().lower()

def _hour_label(h: int) -> str:
    if h <= 0 or h >= 24:
        return "12am" if (h == 0 or h == 24) else "%dpm" % (h % 12) if (h % 12) else "12pm"
    if h < 12:
        return "%dam" % h
    return "12pm" if h == 12 else "%dpm" % (h - 12)

def get_movie_runtime_minutes(movie_name: str) -> int:
    """Get runtime in minutes from movie_features_safe.csv; fallback DEFAULT_RUNTIME_MIN."""
    path = "movie_features_safe.csv"
    if not os.path.isfile(path):
        return DEFAULT_RUNTIME_MIN
    df = pd.read_csv(path, usecols=["original_name", "runtime"], nrows=0)
    if "runtime" not in df.columns:
        return DEFAULT_RUNTIME_MIN
    df = pd.read_csv(path, usecols=["original_name", "runtime"])
    df["clean"] = df["original_name"].astype(str).apply(_normalize_movie)
    key = _normalize_movie(movie_name)
    match = df[df["clean"].str.contains(re.escape(key), case=False, na=False)]
    if match.empty:
        return DEFAULT_RUNTIME_MIN
    runtime = match["runtime"].iloc[0]
    if pd.isna(runtime) or runtime <= 0:
        return DEFAULT_RUNTIME_MIN
    return int(runtime)

def build_slots_from_runtime(runtime_minutes: int):
    """Build (label, start_hour, end_hour) list; end_hour exclusive. One slot = one show length."""
    import math
    runtime_hours = runtime_minutes / 60.0
    slots = []
    for start in SLOT_START_HOURS:
        end_float = start + runtime_hours
        end_hour = int(math.ceil(end_float))  # exclusive
        if end_hour > 24:
            end_hour = 24
        if start >= end_hour:
            continue
        label = "%s-%s" % (_hour_label(start), _hour_label(end_hour))
        slots.append((label, start, end_hour))
    return slots

def load_from_db():
    try:
        from sqlalchemy import create_engine
        from urllib.parse import quote_plus
    except ImportError:
        print("[X] sqlalchemy required for DB. Install: pip install sqlalchemy mysql-connector-python")
        return None
    pw = os.environ.get("DB_PASSWORD", "")
    user = os.environ.get("DB_USER", "root")
    host = os.environ.get("DB_HOST", "localhost")
    db = os.environ.get("DB_NAME", "movie")
    if not pw:
        print("[X] Set DB_PASSWORD (and optionally DB_USER, DB_HOST, DB_NAME) for DB mode.")
        return None
    enc = quote_plus(pw)
    engine = create_engine(f"mysql+mysqlconnector://{user}:{enc}@{host}/{db}")
    q = """
    SELECT movie_name, cinema_id, show_time, total_count AS sold_tickets
    FROM m_grouped_transactions
    WHERE total_sum > 0
    """
    df = pd.read_sql(q, engine)
    df["show_time"] = pd.to_datetime(df["show_time"])
    df["original_name"] = df["movie_name"]
    return df

def load_from_csv():
    path = "final_training_data_v4.csv"
    if not os.path.isfile(path):
        print(f"[X] {path} not found.")
        return None
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    need = ["movie_name", "cinema_id", "show_time", "sold_tickets"]
    usecols = [c for c in need if c in all_cols]
    if "original_name" in all_cols:
        usecols.append("original_name")
    df = pd.read_csv(path, usecols=usecols)
    if "original_name" not in df.columns:
        df["original_name"] = df["movie_name"]
    df["show_time"] = pd.to_datetime(df["show_time"])
    return df

def load_data():
    if USE_DB:
        print("[*] Loading from DB (m_grouped_transactions)...")
        return load_from_db()
    print("[*] Loading from final_training_data_v4.csv...")
    return load_from_csv()

# =============================================================================
#  Build table: all_cinema_id x common time slots -> avg occupancy per slot
# =============================================================================

def run():
    df = load_data()
    if df is None or df.empty:
        return

    df["cinema_id"] = df["cinema_id"].astype(str)
    df["date"] = df["show_time"].dt.date
    df["hour"] = df["show_time"].dt.hour
    df["clean_name"] = df["original_name"].astype(str).apply(_normalize_movie)

    movie_key = _normalize_movie(MOVIE_NAME)
    movie_mask = df["clean_name"].str.contains(re.escape(movie_key), case=False, na=False)
    sub = df[movie_mask].copy()
    if sub.empty:
        print(f"[X] No rows found for movie matching '{MOVIE_NAME}'.")
        return

    # Dynamic slots from movie runtime (e.g. WAR 2 = 173 min -> ~3hr slots)
    runtime_min = get_movie_runtime_minutes(MOVIE_NAME)
    common_slots = build_slots_from_runtime(runtime_min)
    if not common_slots:
        print("[X] No slots generated from runtime.")
        return

    # Aggregate by (cinema_id, date, hour): sum sold_tickets in same slot (multiple shows)
    agg = sub.groupby(["cinema_id", "date", "hour"], as_index=False)["sold_tickets"].sum()

    all_cinema_ids = sorted(agg["cinema_id"].unique().tolist())
    slot_cols = [label for label, _s, _e in common_slots]

    # For each (cinema_id, slot_label): average sold_tickets and occupancy %
    cap = CAPACITY_PER_SCREEN
    rows = []
    for cid in all_cinema_ids:
        row = {"cinema_id": cid}
        c_agg = agg[agg["cinema_id"] == cid]
        for label, start_h, end_h in common_slots:
            in_range = c_agg[(c_agg["hour"] >= start_h) & (c_agg["hour"] < end_h)]
            if in_range.empty:
                row[label] = "-"
            else:
                avg_sold = in_range["sold_tickets"].mean()
                pct = (avg_sold / cap) * 100
                # Format: "12 (5.4%)"
                row[label] = "%d (%.1f%%)" % (int(round(avg_sold)), pct)
        rows.append(row)
    pivot = pd.DataFrame(rows)
    # Column order: cinema_id, then for each slot
    slot_cols_flat = [label for label, _s, _e in common_slots]
    pivot = pivot[["cinema_id"] + slot_cols_flat]

    runtime_hr = runtime_min / 60.0
    print()
    print("  Movie: %s | Runtime: %d min (~%.1f hr) | Cinemas: %d | Slots: %s" % (
        MOVIE_NAME, runtime_min, runtime_hr, len(all_cinema_ids), ", ".join(slot_cols)))
    print("  Capacity: %d seats per screen for occupancy %%" % cap)
    print("  [OK] Table saved to CSV and HTML (open HTML in browser for proper view).")
    print()

    # Save CSV
    csv_path = "cinema_sales_comparison.csv"
    pivot.to_csv(csv_path, index=False)
    print("  CSV:  %s" % csv_path)

    # Save HTML table (proper formatting, open in browser)
    html_path = "cinema_sales_comparison.html"
    movie_matched = sub["original_name"].iloc[0]
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cinema occupancy - %s</title>
  <style>
    body { font-family: Segoe UI, system-ui, sans-serif; margin: 24px; background: #f5f5f5; }
    h1 { font-size: 1.25rem; color: #333; }
    .meta { color: #666; margin-bottom: 16px; font-size: 0.9rem; }
    table { border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
    th { background: #2563eb; color: #fff; font-weight: 600; text-align: center; }
    td:first-child { text-align: left; font-weight: 500; }
    tr:nth-child(even) { background: #f9fafb; }
    tr:hover { background: #eff6ff; }
    .na { color: #999; }
  </style>
</head>
<body>
  <h1>Cinema average occupancy by time slot</h1>
  <p class="meta">Movie: <strong>%s</strong> &nbsp;|&nbsp; Runtime: %d min (~%.1f hr) &nbsp;|&nbsp; Cinemas: %d &nbsp;|&nbsp; Slots: %s</p>
  <p class="meta">Each slot = one show length. Cell format: <strong>Avg Sold (Occupancy%%)</strong>. Capacity: %d seats.</p>
  <table>
    <thead><tr><th>cinema_id</th>%s</tr></thead>
    <tbody>
%s
    </tbody>
  </table>
</body>
</html>
"""
    slots_str = ", ".join(slot_cols)
    th_cells = "".join("<th>%s</th>" % c for c in slot_cols_flat)
    rows_html = []
    for _, row in pivot.iterrows():
        cells = ["<td>%s</td>" % row["cinema_id"]]
        for c in slot_cols_flat:
            v = row.get(c)
            # v is now a string like "12.5 (5.4%)" or "-"
            if v == "-":
                cells.append('<td class="na">-</td>')
            else:
                cells.append("<td>%s</td>" % v)
        rows_html.append("      <tr>%s</tr>" % "".join(cells))
    tbody = "\n".join(rows_html)
    html = html % (movie_matched, movie_matched, runtime_min, runtime_hr, len(all_cinema_ids), slots_str, cap, th_cells, tbody)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("  HTML: %s" % html_path)
    try:
        import webbrowser
        webbrowser.open(os.path.abspath(html_path))
    except Exception:
        pass

if __name__ == "__main__":
    run()
