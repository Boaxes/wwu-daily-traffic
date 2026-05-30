from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pydeck as pdk
import streamlit as st

# Constants
DATA_PATH = Path(__file__).parent / "data" / "map_terms.parquet"
# Forecasting write-up in the README.
WRITEUP_URL = "https://github.com/Boaxes/wwu-daily-traffic#enrollment-forecasting-xgboost"
DAYS = {
    "Monday": "Mon",
    "Tuesday": "Tue",
    "Wednesday": "Wed",
    "Thursday": "Thu",
    "Friday": "Fri",
}
DAY_NAMES = list(DAYS.keys())
FORECAST_RING_COLOR = [255, 255, 255, 240]
# Esri World Imagery: satellite tiles that need no API key. Streamlit takes the
# basemap style as a URL, so the style is inlined as a data URL.
_SATELLITE_STYLE = {
    "version": 8,
    "sources": {
        "imagery": {
            "type": "raster",
            "tiles": [
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            "tileSize": 256,
            "maxzoom": 19,
            "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        }
    },
    "layers": [{"id": "imagery", "type": "raster", "source": "imagery"}],
}
SATELLITE_STYLE = "data:application/json," + quote(json.dumps(_SATELLITE_STYLE))
TERM_NOTES = {
    "202020": (
        "Spring 2020 moved online after its schedule was published. Banner still lists "
        "the rooms assigned before the move, so this map shows the campus as it was planned."
    ),
}

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
    data = pd.read_parquet(DATA_PATH)
    start = data["start_time"].str.split(":", expand=True).astype(int)
    end = data["end_time"].str.split(":", expand=True).astype(int)
    data["start_minutes"] = start[0] * 60 + start[1]
    data["end_minutes"] = end[0] * 60 + end[1]
    data["enrolled_students"] = data["enrolled_students"].fillna(0).astype(int)
    return data


@st.cache_data
def load_terms() -> pd.DataFrame:
    return (
        load_data()
        .groupby(["term", "term_desc"], as_index=False)
        .agg(
            forecast_only=("forecast_only", "first"),
            has_forecast=("predicted_students", lambda predicted: predicted.notna().any()),
        )
        .sort_values("term", ascending=False)
        .set_index("term")
    )

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


def summarize_active_buildings(
    active: pd.DataFrame,
    primary: str,
    show_forecast: bool,
) -> pd.DataFrame:
    # A section meeting in two rooms of one building at once counts once.
    buildings = (
        active.drop_duplicates(["crn", "latitude", "longitude"])
        .assign(predicted_students=lambda frame: frame["predicted_students"].fillna(0))
        .groupby(["latitude", "longitude"], as_index=False)
        .agg(
            building=("official_map_name", join_names),
            actual_students=("enrolled_students", "sum"),
            predicted_students=("predicted_students", "sum"),
            active_classes=("crn", "nunique"),
        )
    )

    if buildings.empty:
        return buildings

    buildings["predicted_students"] = buildings["predicted_students"].astype(int)
    # Actual circles and forecast rings share one scale so their sizes compare.
    scaled = [primary, "predicted_students"] if show_forecast else [primary]
    busiest_count = max(buildings[scaled].to_numpy().max(), 1)

    def radius(students: int) -> float:
        return 16 + (students / busiest_count) ** 0.5 * 42

    buildings["radius"] = buildings[primary].map(radius)
    buildings["forecast_radius"] = buildings["predicted_students"].map(radius)
    buildings["color"] = buildings[primary].map(occupancy_color)
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
def build_map(
    campus_buildings: pd.DataFrame,
    active_buildings: pd.DataFrame,
    tooltip: str,
    forecast_rings: bool,
) -> pdk.Deck:
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
    if forecast_rings:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="forecast-rings",
                data=active_buildings,
                get_position="[longitude, latitude]",
                get_radius="forecast_radius",
                get_line_color=FORECAST_RING_COLOR,
                line_width_min_pixels=2.5,
                filled=False,
                stroked=True,
                pickable=False,
            )
        )

    return pdk.Deck(
        map_style=SATELLITE_STYLE,
        map_provider="carto",
        initial_view_state=pdk.ViewState(
            latitude=48.7367,
            longitude=-122.4858,
            zoom=15.3,
            pitch=0,
        ),
        layers=layers,
        tooltip={"text": tooltip},
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
def build_table(active: pd.DataFrame, show_actual: bool, show_forecast: bool) -> pd.DataFrame:
    table = active.copy()
    table["schedule"] = (
        table["schedule_days"]
        + " "
        + table["start_time"]
        + "-"
        + table["end_time"]
    )
    columns = {
        "class": "Class",
        "title": "Title",
        "official_map_name": "Building",
        "room": "Room",
        "schedule": "Schedule",
    }
    if show_actual:
        columns["enrolled_students"] = "Students"
    if show_forecast:
        table["predicted_students"] = table["predicted_students"].astype("Int64")
        columns["predicted_students"] = "Forecast"
        if table["predicted_low"].notna().any():
            table["range"] = (
                table["predicted_low"].astype("Int64").astype(str)
                + "–"
                + table["predicted_high"].astype("Int64").astype(str)
            ).where(table["predicted_low"].notna())
            columns["range"] = "80% range"
        table["capacity"] = table["capacity"].astype("Int64")
        columns["capacity"] = "Capacity"
    return (
        table[list(columns)]
        .rename(columns=columns)
        .sort_values(["Building", "Class"])
    )

# Page config
st.set_page_config(
    page_title="WWU Traffic Report -- Western Washington University Campus Map",
    page_icon=":material/location_on:",
    layout="wide",
)

# Load data
data = load_data()
terms = load_terms()

# Header
logo_col, title_col = st.columns([1, 10])
with logo_col:
    st.image("media/image.png", width=80)
with title_col:
    st.title("WWU Traffic Report")

# Term and forecast layer
term_column, layer_column = st.columns([3, 2], vertical_alignment="bottom")
with term_column:
    term_options = list(terms.index)
    default_term = terms.index[~terms["forecast_only"]][0]
    selected_term = st.selectbox(
        "Term",
        term_options,
        index=term_options.index(default_term),
        format_func=lambda term: terms.at[term, "term_desc"]
        + (" (forecast)" if terms.at[term, "forecast_only"] else ""),
        key="term",
    )
term_label = terms.at[selected_term, "term_desc"]
show_actual = not terms.at[selected_term, "forecast_only"]
has_forecast = bool(terms.at[selected_term, "has_forecast"])
with layer_column:
    if show_actual:
        layer_on = st.toggle(
            "Forecast layer",
            key="forecast-layer",
            disabled=not has_forecast,
            help=(
                "White rings show the model's forecast, made by a model trained only on "
                "earlier terms. Forecasts begin in 2019 because the model needs earlier "
                "terms to learn from."
            ),
        )
        show_forecast = layer_on and has_forecast
    else:
        st.caption("Forecast only. Registration for this term has not opened.")
        show_forecast = True

if show_actual:
    intro = f"Use the sliders to explore where students are during a typical {term_label} school day."
else:
    intro = (
        f"Forecast of where students will be during a typical {term_label} school day, "
        "made before registration opens."
    )
st.text(intro)
if selected_term in TERM_NOTES:
    st.info(TERM_NOTES[selected_term], icon=":material/info:")
if WRITEUP_URL:
    st.markdown(f"[How the forecast works]({WRITEUP_URL})")

term_data = data[data["term"] == selected_term]
campus_buildings = summarize_buildings(term_data)

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
active = filter_active_classes(term_data, DAYS[selected_day], selected_minute)
primary = "actual_students" if show_actual else "predicted_students"
active_buildings = summarize_active_buildings(active, primary, show_forecast)

# Metrics
sections = active.drop_duplicates("crn")
actual_total = int(sections["enrolled_students"].sum())
predicted_total = int(sections["predicted_students"].fillna(0).sum())
# Compare like with like: only sections the model forecasts.
forecast_gap = int(
    (sections["enrolled_students"] - sections["predicted_students"]).dropna().sum()
)
classes = int(active["crn"].nunique())
buildings = len(active_buildings)
busiest = (
    active_buildings.sort_values(primary, ascending=False).iloc[0]
    if not active_buildings.empty
    else None
)

student_metric, class_metric, building_metric, busiest_metric = st.columns(4)
if show_actual:
    student_metric.metric(
        "Students in class",
        f"{actual_total:,}",
        delta=f"{forecast_gap:+,} vs forecast" if show_forecast else None,
        delta_color="off",
    )
else:
    student_metric.metric("Predicted students", f"{predicted_total:,}")
class_metric.metric("Active classes", f"{classes:,}")
building_metric.metric("Active buildings", f"{buildings:,}")
busiest_metric.metric(
    "Busiest building",
    busiest["building"] if busiest is not None else "None",
)

# Map
tooltip_lines = ["{building}"]
if show_actual:
    tooltip_lines.append("{actual_students} students in class")
if show_forecast:
    tooltip_lines.append("{predicted_students} " + ("forecast" if show_actual else "predicted students"))
tooltip_lines.append("{active_classes} active classes")
tooltip = "\n".join(tooltip_lines)

st.pydeck_chart(
    build_map(campus_buildings, active_buildings, tooltip, show_actual and show_forecast),
    width="stretch",
    height=600,
)
if show_actual and show_forecast:
    st.caption(
        "Filled circles: students enrolled. White rings: the model's forecast for the same "
        "building, from a model trained only on earlier terms. Both use the same size scale. "
        "Independent studies and other individually arranged sections have no forecast."
    )

# Active classes table
st.subheader("Active classes")
st.dataframe(
    build_table(active, show_actual, show_forecast),
    width="stretch",
    hide_index=True,
)
