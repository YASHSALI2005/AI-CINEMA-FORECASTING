"""
Holiday weight for forecasting: uses Holiday API (API key) when available,
falls back to a manual India list for other years or if API fails.

- Set HOLIDAY_API_KEY in environment or .env to use the API.
- Set HOLIDAY_API_YEARS to comma-separated years the API has (e.g. "2025").
  Free Developer plan often has only 2025; 2024/2023 then use the built-in list.
"""

import os
from datetime import date, datetime
from typing import Union

# In-memory cache: year -> { "YYYY-MM-DD": weight, ... }
_holiday_cache: dict[int, dict[str, float]] = {}

# Years the API key has data for (e.g. Free plan = 2025 only). Other years use fallback.
def _api_years() -> set[int]:
    raw = os.environ.get("HOLIDAY_API_YEARS", "2025").strip()
    if not raw:
        return set()
    return {int(y.strip()) for y in raw.split(",") if y.strip().isdigit()}

# Manual fallback (India) when API key is missing or API fails
FALLBACK_MEGA = [
    "2024-01-01", "2024-01-26", "2023-08-15", "2024-08-15",
    "2023-10-02", "2024-10-02", "2023-12-25", "2024-12-25",
]
FALLBACK_MAJOR = [
    "2023-11-12", "2024-03-25", "2024-04-11", "2024-06-17",
    "2023-10-24", "2024-10-12",
]


def _fetch_holidays_api(country: str, year: int) -> dict[str, float]:
    """Fetch holidays for a year from Holiday API. Returns dict of date_str -> weight."""
    try:
        import requests
    except ImportError:
        return {}
    key = os.environ.get("HOLIDAY_API_KEY", "").strip()
    if not key:
        return {}
    url = "https://holidayapi.com/v1/holidays"
    params = {"key": key, "country": country, "year": year}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}
    if data.get("status") != 200 or "holidays" not in data:
        return {}
    out = {}
    for h in data["holidays"]:
        d = h.get("date") or h.get("observed")
        if not d:
            continue
        # public -> 10.0, observance -> 5.0
        w = 10.0 if h.get("public") else 5.0
        out[d] = max(out.get(d, 0), w)
    return out


def _get_cached_holidays(year: int, country: str = "IN") -> dict[str, float]:
    if year not in _holiday_cache:
        # Only call API for years the plan supports (e.g. 2025); 2024/2023 use fallback
        if year in _api_years():
            _holiday_cache[year] = _fetch_holidays_api(country, year)
        if not _holiday_cache.get(year):
            _holiday_cache[year] = _fallback_static()
    return _holiday_cache[year]


def _fallback_static() -> dict[str, float]:
    """Static fallback map (India) when API is not used."""
    result = {d: 10.0 for d in FALLBACK_MEGA}
    result.update({d: 5.0 for d in FALLBACK_MAJOR})
    return result


def _date_to_str(d: Union[date, datetime, str]) -> str:
    if isinstance(d, str):
        return d[:10] if len(d) >= 10 else d
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%Y-%m-%d")


def get_holiday_weight(d: Union[date, datetime, str], country: str = "IN") -> float:
    """
    Return holiday weight for a date: 10.0 = public holiday, 5.0 = observance, 1.0 = normal.
    Uses Holiday API if HOLIDAY_API_KEY is set; otherwise uses manual India list.
    """
    date_str = _date_to_str(d)
    try:
        year = int(date_str[:4])
    except (ValueError, TypeError):
        return 1.0
    cache = _get_cached_holidays(year, country=country)
    return cache.get(date_str, 1.0)


def clear_holiday_cache() -> None:
    """Clear in-memory holiday cache (e.g. after changing API key)."""
    _holiday_cache.clear()
