"""Download raw Banner section data for many terms.

Saves one gzipped JSON file per term to ``data/raw/banner/``. Terms that are
already downloaded are skipped, so the script can be stopped and resumed.

Usage:
    python scripts/fetch_history.py                     # Fall 2016 -> forecast term
    python scripts/fetch_history.py --start-year 2010
    python scripts/fetch_history.py --refresh 202640    # re-pull a live term
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time

from common import FIRST_YEAR, FORECAST_TERM, RAW_DIR
from scrape_schedule import fetch_sections

# Constants
# Winter, Spring, and Fall; summer quarters are skipped.
ACADEMIC_TERMS = ["10", "20", "40"]


def academic_terms(start_year: int, end_term: str) -> list[str]:
    terms = [
        f"{year}{suffix}"
        for year in range(start_year, int(end_term[:4]) + 1)
        for suffix in ACADEMIC_TERMS
    ]
    # Start with the fall term so each academic year is complete.
    return [term for term in terms if f"{start_year}40" <= term <= end_term]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--end-term", default=FORECAST_TERM)
    parser.add_argument("--refresh", nargs="*", default=[], help="Terms to re-download")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    terms = academic_terms(args.start_year, args.end_term)
    for index, term in enumerate(terms, start=1):
        path = RAW_DIR / f"{term}.json.gz"
        if path.exists() and term not in args.refresh:
            print(f"[{index}/{len(terms)}] {term}: cached", file=sys.stderr)
            continue

        started = time.perf_counter()
        print(f"[{index}/{len(terms)}] {term}: downloading", file=sys.stderr)
        sections = fetch_sections(term)
        with gzip.open(path, "wt", encoding="utf-8") as file:
            json.dump(sections, file)
        print(
            f"[{index}/{len(terms)}] {term}: {len(sections):,} sections "
            f"in {time.perf_counter() - started:.0f}s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
