"""Scrape WWU's class schedule and geocode each meeting to a campus building.

Writes one term's meetings to ``data/final_data.csv`` from two public sources:

* Banner 9 "Browse Classes" (registration.banner.wwu.edu) for sections,
  meeting times, rooms, and enrollment.
* WWU Spatial Institute's ``Building_Info__5k`` ArcGIS point layer (the layer
  behind map.wwu.edu) for building names and coordinates.

``build_map_data.py`` reuses the same flattening and geocoding for every term.

Usage:
    python scripts/scrape_schedule.py                    # Spring 2026 -> data/final_data.csv
    python scripts/scrape_schedule.py --term 202640      # Fall 2026
    python scripts/scrape_schedule.py --list-terms
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import pandas as pd
import requests

from common import ALL_DAYS, ROOT, WEEKDAYS

# Constants
BANNER_URL = "https://registration.banner.wwu.edu/StudentRegistrationSsb/ssb"
BUILDINGS_URL = (
    "https://services.arcgis.com/qboYD3ru0louQq4F/arcgis/rest/services/"
    "Building_Info__5k/FeatureServer/0/query"
)
DEFAULT_TERM = "202620"
DEFAULT_OUTPUT = ROOT / "data" / "final_data.csv"
PAGE_SIZE = 500
USER_AGENT = "wwu-daily-traffic class schedule scraper"

# Main campus is everything within this radius of Red Square. Excludes Lakewood
# (LW) and Shannon Point (MC), which are in the building layer but off campus.
CAMPUS_CENTER = (48.7367, -122.4858)
CAMPUS_RADIUS_M = 1500


# Banner building codes that don't line up cleanly with the ArcGIS layer.
BUILDING_OVERRIDES = {
    "FA": {
        "abv": "FC",
        "note": (
            "Banner uses FA for Fairhaven College; WWU ArcGIS building layer "
            "labels the same point FC. Review Fairhaven FA/FC code normalization "
            "before joining other campus datasets."
        ),
    },
    "FC": {
        "abv": "FC",
        "note": (
            "Banner says Fairhaven Housing/Dining; courses are in room 101F, "
            "matched to Fairhaven College Building. Review Fairhaven FA/FC code "
            "normalization before joining other campus datasets."
        ),
    },
}

COLUMNS = [
    "term",
    "term_desc",
    "class",
    "crn",
    "title",
    "building_code",
    "building",
    "official_map_name",
    "room",
    "schedule_days",
    "start_time",
    "end_time",
    "enrolled_students",
    "latitude",
    "longitude",
    "location_scope",
    "coordinate_validation_status",
    "coordinate_notes",
]

# Banner
def banner_session(term: str) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    # Banner ties search results to a term stored in the server-side session.
    response = session.post(
        f"{BANNER_URL}/term/search",
        params={"mode": "search"},
        data={"term": term},
        timeout=30,
    )
    response.raise_for_status()
    return session


def list_terms(max_terms: int = 20) -> list[dict]:
    response = requests.get(
        f"{BANNER_URL}/classSearch/getTerms",
        params={"searchTerm": "", "offset": 1, "max": max_terms},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_sections(term: str) -> list[dict]:
    session = banner_session(term)
    sections: list[dict] = []
    offset = 0
    while True:
        response = session.get(
            f"{BANNER_URL}/searchResults/searchResults",
            params={
                "txt_term": term,
                "pageOffset": offset,
                "pageMaxSize": PAGE_SIZE,
                "sortColumn": "subjectDescription",
                "sortDirection": "asc",
            },
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()
        if not page.get("success", True) or page.get("data") is None:
            raise RuntimeError(f"Banner returned no data for term {term}: {page}")

        sections.extend(page["data"])
        total = page["totalCount"]
        print(f"  fetched {len(sections):,} / {total:,} sections", file=sys.stderr)
        offset += PAGE_SIZE
        if offset >= total:
            return sections

# ArcGIS
def fetch_buildings() -> dict[str, dict]:
    response = requests.get(
        BUILDINGS_URL,
        params={
            "where": "1=1",
            "outFields": "name,abv",
            "outSR": 4326,
            "returnGeometry": "true",
            "f": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    return {
        feature["attributes"]["abv"]: {
            "name": feature["attributes"]["name"],
            "latitude": feature["geometry"]["y"],
            "longitude": feature["geometry"]["x"],
        }
        for feature in response.json()["features"]
    }


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(a))

# Transform
def format_time(hhmm: str) -> str:
    return f"{hhmm[:2]}:{hhmm[2:]}"


def flatten_meetings(sections: list[dict]) -> list[dict]:
    rows = []
    for section in sections:
        # Section-level campus; Banner marks the meeting itself "M" even for
        # off-campus cohorts that borrow main campus rooms.
        if section["campusDescription"] != "Main Campus":
            continue

        for meeting in section["meetingsFaculty"]:
            time = meeting["meetingTime"]
            if not time["building"] or not time["beginTime"]:
                continue
            if not any(time[day] for day in WEEKDAYS):
                continue

            rows.append(
                {
                    "term": section["term"],
                    "term_desc": section["termDesc"],
                    "class": " ".join(
                        [section["subject"], section["courseNumber"], section["sequenceNumber"] or ""]
                    ),
                    "crn": section["courseReferenceNumber"],
                    "title": html.unescape(section["courseTitle"]),
                    "building_code": time["building"],
                    "building": time["buildingDescription"],
                    "room": time["room"],
                    "schedule_days": "".join(
                        day[:3].title() for day in ALL_DAYS if time[day]
                    ),
                    "start_time": format_time(time["beginTime"]),
                    "end_time": format_time(time["endTime"]),
                    "enrolled_students": section["enrollment"],
                }
            )
    return rows


def geocode(rows: list[dict], buildings: dict[str, dict]) -> list[dict]:
    geocoded = []
    for row in rows:
        override = BUILDING_OVERRIDES.get(row["building_code"], {})
        building = buildings.get(override.get("abv", row["building_code"]))
        if building is None:
            continue

        distance = distance_m(*CAMPUS_CENTER, building["latitude"], building["longitude"])
        if distance > CAMPUS_RADIUS_M:
            continue

        geocoded.append(
            {
                **row,
                "official_map_name": building["name"],
                "latitude": building["latitude"],
                "longitude": building["longitude"],
                "location_scope": "main_campus",
                "coordinate_validation_status": "needs_review" if override else "matched",
                "coordinate_notes": override.get("note"),
            }
        )
    return geocoded


def build_dataset(sections: list[dict], buildings: dict[str, dict]) -> pd.DataFrame:
    rows = geocode(flatten_meetings(sections), buildings)
    return (
        pd.DataFrame(rows, columns=COLUMNS)
        .sort_values(["building_code", "schedule_days", "start_time", "class"], kind="stable")
        .reset_index(drop=True)
    )

# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--term", default=DEFAULT_TERM, help="Banner term code, e.g. 202620")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw", type=Path, help="Also save the raw Banner JSON here")
    parser.add_argument("--list-terms", action="store_true", help="Print available terms and exit")
    args = parser.parse_args()

    if args.list_terms:
        for term in list_terms():
            print(f"{term['code']}  {term['description']}")
        return

    print(f"Fetching Banner sections for term {args.term}...", file=sys.stderr)
    sections = fetch_sections(args.term)
    if args.raw:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        args.raw.write_text(json.dumps(sections))

    print("Fetching WWU building layer...", file=sys.stderr)
    buildings = fetch_buildings()

    data = build_dataset(sections, buildings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(
        f"Wrote {len(data):,} meetings ({data['crn'].nunique():,} sections, "
        f"{data['building_code'].nunique()} buildings) to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
