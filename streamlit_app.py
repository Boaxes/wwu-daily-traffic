from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

# Constants
DATA_PATH = Path(__file__).parent / "data" / "final_data.csv"
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")
DAYS = {
    "Monday": "Mon",
    "Tuesday": "Tue",
    "Wednesday": "Wed",
    "Thursday": "Thu",
    "Friday": "Fri",
}
DAY_NAMES = list(DAYS.keys())

# Time formatting
def format_time(minutes: int) -> str:
    hour = minutes // 60
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minutes % 60:02d} {suffix}"

# Live slider component (shared by time and day sliders)
LIVE_SLIDER = st.components.v2.component(
    "live_slider",
    html="""
        <output for="slider"></output>
        <input id="slider" type="range" />
    """,
    css="""
        :host {
            display: block;
            font-family: var(--st-font);
        }
        output {
            display: block;
            color: var(--st-text-color);
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 0.125rem;
        }
        input {
            accent-color: var(--st-primary-color);
            cursor: pointer;
            width: 100%;
        }
    """,
    js="""
        export default function(component) {
            const { data, parentElement, setStateValue } = component;
            const output = parentElement.querySelector("output");
            const input = parentElement.querySelector("input");
            let animationFrame = null;

            function formatValue(value) {
                if (data.format === "time") {
                    const hour = Math.floor(value / 60);
                    const hour12 = hour % 12 || 12;
                    const suffix = hour < 12 ? "AM" : "PM";
                    return `${hour12}:${String(value % 60).padStart(2, "0")} ${suffix}`;
                }
                return data.labels[value];
            }

            input.min = data.min;
            input.max = data.max;
            input.step = data.step;

            if (Number(input.value) !== data.value) {
                input.value = data.value;
            }
            output.textContent = formatValue(Number(input.value));

            input.oninput = (event) => {
                const value = Number(event.target.value);
                output.textContent = formatValue(value);

                if (animationFrame !== null) {
                    cancelAnimationFrame(animationFrame);
                }
                animationFrame = requestAnimationFrame(() => {
                    setStateValue("value", value);
                    animationFrame = null;
                });
            };

            return () => {
                if (animationFrame !== null) cancelAnimationFrame(animationFrame);
                input.oninput = null;
            };
        }
    """,
)

# Data loading
@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, dtype={"crn": str})
    start = data["start_time"].str.split(":", expand=True).astype(int)
    end = data["end_time"].str.split(":", expand=True).astype(int)
    data["start_minutes"] = start[0] * 60 + start[1]
    data["end_minutes"] = end[0] * 60 + end[1]
    data["enrolled_students"] = data["enrolled_students"].fillna(0).astype(int)
    return data

# Data aggregation helpers
def join_names(names: pd.Series) -> str:
    return " / ".join(sorted(set(names)))


def summarize_buildings(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["latitude", "longitude"], as_index=False)
        .agg(building=("official_map_name", join_names))
    )


def filter_active_classes(data: pd.DataFrame, day: str, minute: int) -> pd.DataFrame:
    return data[
        data["schedule_days"].str.contains(day, na=False)
        & (data["start_minutes"] <= minute)
        & (minute < data["end_minutes"])
    ].copy()


def summarize_active_buildings(active: pd.DataFrame) -> pd.DataFrame:
    buildings = (
        active.groupby(["latitude", "longitude"], as_index=False)
        .agg(
            building=("official_map_name", join_names),
            active_students=("enrolled_students", "sum"),
            active_classes=("crn", "nunique"),
        )
    )

    if buildings.empty:
        return buildings

    busiest_count = buildings["active_students"].max()
    buildings["radius"] = buildings["active_students"].map(
        lambda students: 16 + (students / busiest_count) ** 0.5 * 42
    )
    buildings["color"] = buildings["active_students"].map(occupancy_color)
    return buildings


def occupancy_color(students: int) -> list[int]:
    if students < 150:
        return [44, 123, 182, 210]
    if students < 300:
        return [54, 162, 166, 215]
    if students < 500:
        return [115, 180, 111, 220]
    if students < 700:
        return [242, 179, 76, 225]
    return [218, 91, 79, 230]

# Map builder
def build_map(campus_buildings: pd.DataFrame, active_buildings: pd.DataFrame) -> pdk.Deck:
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            id="campus-buildings",
            data=campus_buildings,
            get_position="[longitude, latitude]",
            get_radius=10,
            get_fill_color=[108, 122, 128, 100],
            stroked=False,
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            id="active-buildings",
            data=active_buildings,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 230],
            line_width_min_pixels=1,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        ),
    ]

    return pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-v9",
        map_provider="mapbox",
        api_keys={"mapbox": MAPBOX_TOKEN},
        initial_view_state=pdk.ViewState(
            latitude=48.7367,
            longitude=-122.4858,
            zoom=15.3,
            pitch=0,
        ),
        layers=layers,
        tooltip={
            "text": (
                "{building}\n"
                "{active_students} students in class\n"
                "{active_classes} active classes"
            )
        },
    )

# Slider helpers
def live_time_slider(*, default: int, key: str) -> int:
    component_state = st.session_state.get(key, {})
    value = int(component_state.get("value", default))
    result = LIVE_SLIDER(
        data={
            "min": 6 * 60,
            "max": 23 * 60,
            "step": 15,
            "value": value,
            "format": "time",
        },
        default={"value": value},
        key=key,
        on_value_change=lambda: None,
    )
    return int(result.value)


def live_day_slider(*, default_index: int, key: str) -> str:
    component_state = st.session_state.get(key, {})
    value = int(component_state.get("value", default_index))
    result = LIVE_SLIDER(
        data={
            "min": 0,
            "max": len(DAY_NAMES) - 1,
            "step": 1,
            "value": value,
            "format": "day",
            "labels": DAY_NAMES,
        },
        default={"value": value},
        key=key,
        on_value_change=lambda: None,
    )
    return DAY_NAMES[int(result.value)]

# Table builder
def build_table(active: pd.DataFrame) -> pd.DataFrame:
    table = active.copy()
    table["schedule"] = (
        table["schedule_days"]
        + " "
        + table["start_time"]
        + "-"
        + table["end_time"]
    )
    return (
        table[
            [
                "class",
                "title",
                "official_map_name",
                "room",
                "schedule",
                "enrolled_students",
            ]
        ]
        .rename(
            columns={
                "class": "Class",
                "title": "Title",
                "official_map_name": "Building",
                "room": "Room",
                "schedule": "Schedule",
                "enrolled_students": "Students",
            }
        )
        .sort_values(["Building", "Class"])
    )

# Page config
st.set_page_config(
    page_title="WWU Traffic Report -- Western Washington University Campus Map",
    page_icon=":material/location_on:",
    layout="wide",
)

if not MAPBOX_TOKEN:
    st.error(
        "Missing `MAPBOX_TOKEN`. Add it to `.streamlit/secrets.toml` "
        "or set it as an environment variable."
    )
    st.stop()

# Load data
data = load_data()
campus_buildings = summarize_buildings(data)

# Header
logo_col, title_col = st.columns([1, 10])
with logo_col:
    st.image("media/image.png", width=80)
with title_col:
    st.title("WWU Traffic Report")

_ = st.text("Use the sliders to explore where students are during a typical Spring 2026 school day.")

# Controls
day_column, time_column = st.columns(2)
with day_column:
    selected_day = live_day_slider(default_index=1, key="traffic-day-slider")
with time_column:
    selected_minute = live_time_slider(
        default=10 * 60,
        key="traffic-time-slider",
    )

# Filter active data
selected_time = format_time(selected_minute)
active = filter_active_classes(data, DAYS[selected_day], selected_minute)
active_buildings = summarize_active_buildings(active)

# Metrics
students = int(active["enrolled_students"].sum())
classes = int(active["crn"].nunique())
buildings = len(active_buildings)
busiest = (
    active_buildings.sort_values("active_students", ascending=False).iloc[0]
    if not active_buildings.empty
    else None
)

student_metric, class_metric, building_metric, busiest_metric = st.columns(4)
student_metric.metric("Students in class", f"{students:,}")
class_metric.metric("Active classes", f"{classes:,}")
building_metric.metric("Active buildings", f"{buildings:,}")
busiest_metric.metric(
    "Busiest building",
    busiest["building"] if busiest is not None else "None",
)

# Map
st.pydeck_chart(
    build_map(campus_buildings, active_buildings),
    width="stretch",
    height=600,
)

# Active classes table
st.subheader("Active classes")
st.dataframe(build_table(active), width="stretch", hide_index=True)
