import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from fl3xx_api import Fl3xxApiConfig, fetch_flights

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Short Turns Highlighter", layout="wide")
st.title("✈️ Short Turns — Lightweight Viewer")


def _purge_autorefresh_session_state() -> None:
    """Remove stale session-state keys left behind by the old autorefresh widget."""

    stale_keys = [key for key in st.session_state if "autorefresh" in key.lower()]
    for key in stale_keys:
        st.session_state.pop(key, None)


_purge_autorefresh_session_state()

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "America/Edmonton"))
DEFAULT_TURN_THRESHOLD_MIN = int(os.getenv("TURN_THRESHOLD_MIN", "45"))

# ----------------------------
# Helper: Normalize / Parse datetimes
# ----------------------------
def parse_dt(x):
    if pd.isna(x) or x == "":
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    # Try multiple formats
    for fmt in (None, "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return pd.to_datetime(x, format=fmt, utc=True).tz_convert(LOCAL_TZ) if fmt else pd.to_datetime(x, utc=True).tz_convert(LOCAL_TZ)
        except Exception:
            continue
    # Last resort
    try:
        return pd.to_datetime(x, utc=True).tz_convert(LOCAL_TZ)
    except Exception:
        return pd.NaT

# ----------------------------
# Data Model we need
# ----------------------------
# Minimal normalized columns required for the turn calculation:
#   tail: str (aircraft reg)
#   station: str (ICAO where the turn happens; typically ARR airport of leg N and DEP airport of leg N+1)
#   arr_onblock: datetime (actual or scheduled on-block for arriving leg)
#   dep_offblock_next: datetime (actual or scheduled off-block for next departing leg from same station)
#   arr_leg_id / dep_leg_id: identifiers (optional but helpful)
#
# The app provides two input paths:
#   1) Direct FL3XX API fetch (configure below in fetch_fl3xx_legs)
#   2) Upload CSV/JSON already exported by your FF Dashboard or FL3XX

# ----------------------------
# FL3XX Fetch (skeleton — adapt endpoint & mapping to your account)
# ----------------------------
def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_nested(mapping, path):
    if isinstance(path, str):
        parts = path.split(".")
    else:
        parts = list(path)
    current = mapping
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _first_stripped(*values):
    for value in values:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
        elif value is not None:
            candidate = str(value).strip()
            if candidate:
                return candidate
    return None


def _coerce_datetime(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return parse_dt(value)
    if isinstance(value, str):
        return parse_dt(value)
    if isinstance(value, dict):
        for key in (
            "actual",
            "actualTime",
            "actualUTC",
            "actualUtc",
            "actualDateTime",
            "scheduled",
            "scheduledTime",
            "scheduledUTC",
            "scheduledUtc",
            "scheduledDateTime",
            "offBlock",
            "offBlockActual",
            "offBlockScheduled",
            "out",
            "outActual",
            "outScheduled",
            "in",
            "inActual",
            "inScheduled",
        ):
            if key in value:
                dt = _coerce_datetime(value[key])
                if dt is not pd.NaT:
                    return dt
        for sub_value in value.values():
            dt = _coerce_datetime(sub_value)
            if dt is not pd.NaT:
                return dt
        return pd.NaT
    if isinstance(value, (list, tuple, set)):
        for item in value:
            dt = _coerce_datetime(item)
            if dt is not pd.NaT:
                return dt
    return pd.NaT


def _extract_field(payload: dict, options):
    for option in options:
        value = _get_nested(payload, option)
        if value is None:
            continue
        result = _first_stripped(value)
        if result:
            return result
    return None


def _is_placeholder_tail(tail: str) -> bool:
    """Return ``True`` when the provided tail number is a placeholder."""

    if not tail:
        return False
    first_word = tail.split()[0]
    return first_word in {"ADD", "REMOVE"}


def _normalise_flights(flights):
    """Return a dataframe of legs and diagnostics about skipped flights."""

    stats = {
        "raw_count": 0,
        "normalised": 0,
        "skipped_non_mapping": 0,
        "skipped_missing_tail": 0,
        "skipped_placeholder_tail": 0,
        "skipped_missing_airports": 0,
        "skipped_missing_dep_airport": 0,
        "skipped_missing_arr_airport": 0,
        "skipped_missing_times": 0,
    }

    dep_time_keys = [
        "scheduledOut",
        "actualOut",
        "outActual",
        "outScheduled",
        "offBlock",
        "offBlockActual",
        "offBlockScheduled",
        "blockOffEstUTC",
        "blockOffEstLocal",
        "blocksoffestimated",
        "departureScheduledTime",
        "departureActualTime",
        "departureTimeScheduled",
        "departureTimeActual",
        "departureScheduledUtc",
        "departureActualUtc",
        "departure.scheduled",
        "departure.actual",
        "departure.scheduledTime",
        "departure.actualTime",
        "departure.scheduledUtc",
        "departure.actualUtc",
        "times.departure.scheduled",
        "times.departure.actual",
        "times.offBlock.scheduled",
        "times.offBlock.actual",
    ]

    arr_time_keys = [
        "scheduledIn",
        "actualIn",
        "inActual",
        "inScheduled",
        "onBlock",
        "onBlockActual",
        "onBlockScheduled",
        "blockOnEstUTC",
        "blockOnEstLocal",
        "blocksonestimated",
        "arrivalScheduledTime",
        "arrivalActualTime",
        "arrivalTimeScheduled",
        "arrivalTimeActual",
        "arrivalScheduledUtc",
        "arrivalActualUtc",
        "arrival.scheduled",
        "arrival.actual",
        "arrival.scheduledTime",
        "arrival.actualTime",
        "arrival.scheduledUtc",
        "arrival.actualUtc",
        "times.arrival.scheduled",
        "times.arrival.actual",
        "times.onBlock.scheduled",
        "times.onBlock.actual",
    ]

    tail_keys = [
        "aircraftRegistration",
        "aircraft.registration",
        "aircraft.reg",
        "aircraft.registrationNumber",
        "registrationNumber",
        "aircraft.tailNumber",
        "aircraft.name",
        "tailNumber",
        "tail",
        "registration",
    ]

    dep_airport_keys = [
        "departureAirportIcao",
        "departureAirport.icao",
        "departure.airportIcao",
        "departure.airport.icao",
        "departureAirport",
        "departure.icao",
        "departure.airport",
        "departureStation",
        "airportFrom",
        "realAirportFrom",
    ]

    arr_airport_keys = [
        "arrivalAirportIcao",
        "arrivalAirport.icao",
        "arrival.airportIcao",
        "arrival.airport.icao",
        "arrivalAirport",
        "arrival.icao",
        "arrival.airport",
        "arrivalStation",
        "airportTo",
        "realAirportTo",
    ]

    leg_id_keys = [
        "bookingIdentifier",
        "booking.identifier",
        "flightId",
        "id",
        "uuid",
        "legId",
        "scheduleId",
    ]

    rows = []
    for flight in flights:
        stats["raw_count"] += 1
        if not isinstance(flight, dict):
            stats["skipped_non_mapping"] += 1
            continue
        tail = _extract_field(flight, tail_keys)
        dep_ap = _extract_field(flight, dep_airport_keys)
        arr_ap = _extract_field(flight, arr_airport_keys)
        dep_time = _extract_datetime(flight, dep_time_keys)
        arr_time = _extract_datetime(flight, arr_time_keys)
        leg_id = _extract_field(flight, leg_id_keys)

        if tail:
            tail = tail.upper()
            if _is_placeholder_tail(tail):
                stats["skipped_placeholder_tail"] += 1
                continue
        if dep_ap:
            dep_ap = dep_ap.upper()
        if arr_ap:
            arr_ap = arr_ap.upper()

        if not tail:
            stats["skipped_missing_tail"] += 1
            continue

        missing_airport = False
        if not dep_ap:
            stats["skipped_missing_dep_airport"] += 1
            missing_airport = True
        if not arr_ap:
            stats["skipped_missing_arr_airport"] += 1
            missing_airport = True

        if missing_airport:
            stats["skipped_missing_airports"] += 1
            continue

        if dep_time is pd.NaT:
            dep_time = None
        if arr_time is pd.NaT:
            arr_time = None

        if dep_time is None and arr_time is None:
            stats["skipped_missing_times"] += 1
            continue

        rows.append(
            {
                "tail": tail,
                "dep_airport": dep_ap,
                "arr_airport": arr_ap,
                "dep_offblock": dep_time,
                "arr_onblock": arr_time,
                "leg_id": leg_id,
            }
        )

        stats["normalised"] += 1

    return pd.DataFrame(rows), stats


def _extract_datetime(payload: dict, options):
    for option in options:
        value = _get_nested(payload, option)
        if value is None:
            continue
        dt = _coerce_datetime(value)
        if dt is not pd.NaT:
            return dt
    return None


def _build_fl3xx_config(token: str) -> Fl3xxApiConfig:
    secrets_section = st.secrets.get("fl3xx_api", {})

    base_url = secrets_section.get("base_url") or os.getenv("FL3XX_BASE_URL") or Fl3xxApiConfig().base_url

    auth_header_name = secrets_section.get("auth_header_name") or os.getenv("FL3XX_AUTH_HEADER", "Authorization")

    auth_header = secrets_section.get("auth_header") or os.getenv("FL3XX_AUTH_HEADER_VALUE")

    api_token_scheme = secrets_section.get("api_token_scheme") or os.getenv("FL3XX_TOKEN_SCHEME")

    extra_headers = {}
    if "extra_headers" in secrets_section and isinstance(secrets_section["extra_headers"], dict):
        extra_headers = dict(secrets_section["extra_headers"])

    extra_params = {}
    if "extra_params" in secrets_section and isinstance(secrets_section["extra_params"], dict):
        extra_params = dict(secrets_section["extra_params"])

    verify_ssl = secrets_section.get("verify_ssl")
    if verify_ssl is None:
        verify_ssl = os.getenv("FL3XX_VERIFY_SSL")
    verify_ssl = True if verify_ssl is None else _coerce_bool(verify_ssl)

    timeout = secrets_section.get("timeout") or os.getenv("FL3XX_TIMEOUT")
    if timeout is not None:
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = None

    config_kwargs = {
        "base_url": base_url,
        "api_token": token or secrets_section.get("api_token"),
        "auth_header": auth_header,
        "auth_header_name": auth_header_name,
        "api_token_scheme": api_token_scheme,
        "extra_headers": extra_headers,
        "extra_params": extra_params,
        "verify_ssl": verify_ssl,
    }

    if timeout is not None:
        config_kwargs["timeout"] = timeout

    return Fl3xxApiConfig(**config_kwargs)


@st.cache_data(show_spinner=True, ttl=180)
def fetch_fl3xx_legs(token: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Fetch FL3XX flights and normalise them into the dataframe the app expects."""

    config = _build_fl3xx_config(token)

    if not (config.api_token or config.auth_header):
        st.error("No FL3XX API token found. Provide a token or configure it in Streamlit secrets.")
        return pd.DataFrame()

    from_date = start_utc.date()
    to_date = end_utc.date()

    try:
        flights, metadata = fetch_flights(config, from_date=from_date, to_date=to_date)
    except Exception as exc:
        st.error(f"FL3XX fetch failed: {exc}")
        return pd.DataFrame()

    legs_df, normalise_stats = _normalise_flights(flights)

    st.session_state["fl3xx_last_metadata"] = {
        "count": len(flights),
        **metadata,
        "normalisation": normalise_stats,
    }

    return legs_df

# ----------------------------
# Upload parser (CSV/JSON) — expects the columns listed above, but will try to infer
# ----------------------------
def load_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".json"):
        raw = pd.read_json(file)
    else:
        raw = pd.read_csv(file)

    # Try to normalize column names
    cols = {c.lower(): c for c in raw.columns}
    def pick(*options):
        for opt in options:
            if opt in cols:
                return cols[opt]
        return None

    tail_col = pick("tail", "aircraftregistration", "aircraft", "reg")
    dep_ap_col = pick("dep_airport", "departureairporticao", "depicao", "departure")
    arr_ap_col = pick("arr_airport", "arrivalairporticao", "arricao", "arrival")
    dep_off_col = pick("dep_offblock", "scheduledout", "outtime", "offblock")
    arr_on_col = pick("arr_onblock", "scheduledin", "intime", "onblock")
    leg_id_col = pick("leg_id", "bookingidentifier", "id", "legid", "uuid")

    df = pd.DataFrame({
        "tail": raw[tail_col] if tail_col else None,
        "dep_airport": raw[dep_ap_col] if dep_ap_col else None,
        "arr_airport": raw[arr_ap_col] if arr_ap_col else None,
        "dep_offblock": raw[dep_off_col].apply(parse_dt) if dep_off_col else pd.NaT,
        "arr_onblock": raw[arr_on_col].apply(parse_dt) if arr_on_col else pd.NaT,
        "leg_id": raw[leg_id_col] if leg_id_col else None,
    })
    return df.dropna(subset=["tail"]) if "tail" in df else df

# ----------------------------
# Core: Compute Turns
# ----------------------------
def compute_short_turns(legs: pd.DataFrame, threshold_min: int) -> pd.DataFrame:
    if legs.empty:
        return pd.DataFrame(columns=[
            "tail", "station", "arr_leg_id", "arr_onblock", "dep_leg_id", "dep_offblock", "turn_min"
        ])

    # Ensure dtypes
    legs = legs.copy()
    legs["dep_offblock"] = legs["dep_offblock"].apply(parse_dt)
    legs["arr_onblock"] = legs["arr_onblock"].apply(parse_dt)

    # We'll compute turns per tail per station: find next departure from the ARR station after ARR onblock
    # Prepare two views: arrivals and departures
    arrs = legs.dropna(subset=["arr_airport", "arr_onblock"]).copy()
    arrs.rename(columns={"arr_airport": "station", "arr_onblock": "arr_onblock"}, inplace=True)

    deps = legs.dropna(subset=["dep_airport", "dep_offblock"]).copy()
    deps.rename(columns={"dep_airport": "station", "dep_offblock": "dep_offblock"}, inplace=True)

    arrs = arrs[["tail", "station", "arr_onblock", "leg_id"]].rename(columns={"leg_id": "arr_leg_id"})
    deps = deps[["tail", "station", "dep_offblock", "leg_id"]].rename(columns={"leg_id": "dep_leg_id"})

    # Sort for asof merge (next departure after arrival)
    arrs = arrs.sort_values(["tail", "station", "arr_onblock"]).reset_index(drop=True)
    deps = deps.sort_values(["tail", "station", "dep_offblock"]).reset_index(drop=True)

    # Merge by tail & station; for each arrival, find the FIRST departure strictly after arrival
    short_turn_rows = []
    # We'll iterate per (tail, station) to keep memory small
    for (tail, station), arr_grp in arrs.groupby(["tail", "station"], sort=False):
        dep_grp = deps[(deps["tail"] == tail) & (deps["station"] == station)]
        if dep_grp.empty:
            continue
        dep_times = dep_grp["dep_offblock"].tolist()
        dep_ids = dep_grp["dep_leg_id"].tolist()
        for _, r in arr_grp.iterrows():
            arr_t = r["arr_onblock"]
            # find first dep time > arr_t
            idx = next((i for i, t in enumerate(dep_times) if pd.notna(arr_t) and pd.notna(t) and t > arr_t), None)
            if idx is None:
                continue
            dep_t = dep_times[idx]
            dep_id = dep_ids[idx]
            turn_min = (dep_t - arr_t).total_seconds() / 60.0
            if turn_min < threshold_min:
                short_turn_rows.append({
                    "tail": tail,
                    "station": station,
                    "arr_leg_id": r.get("arr_leg_id"),
                    "arr_onblock": arr_t,
                    "dep_leg_id": dep_id,
                    "dep_offblock": dep_t,
                    "turn_min": round(turn_min, 1),
                })

    out = pd.DataFrame(short_turn_rows)
    if not out.empty:
        out = out.sort_values(["turn_min", "tail", "station"]).reset_index(drop=True)
    return out

# ----------------------------
# UI — Sidebar Controls
# ----------------------------
st.sidebar.header("Data Source")
source = st.sidebar.radio("Choose source", ["FL3XX API", "Upload CSV/JSON"], index=0)
threshold = st.sidebar.number_input("Short-turn threshold (minutes)", min_value=5, max_value=240, value=DEFAULT_TURN_THRESHOLD_MIN, step=5)

# Date selector defaults: night shift usually looks at "tomorrow" for the next few days
local_today = datetime.now(LOCAL_TZ).date()
default_start = local_today + timedelta(days=1)
default_end = default_start + timedelta(days=4)
selected_dates = st.sidebar.date_input(
    "Date range (local)",
    value=(default_start, default_end),
)

if isinstance(selected_dates, list):
    selected_dates = tuple(selected_dates)

if not selected_dates:
    start_date, end_date = default_start, default_end
elif isinstance(selected_dates, tuple):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date = end_date = default_start
else:
    start_date = end_date = selected_dates

if start_date > end_date:
    start_date, end_date = end_date, start_date

start_local = datetime.combine(start_date, datetime.min.time(), tzinfo=LOCAL_TZ)
end_local = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=LOCAL_TZ)
start_utc = start_local.astimezone(ZoneInfo("UTC"))
end_utc = end_local.astimezone(ZoneInfo("UTC"))
window_label = f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"

# ----------------------------
# Load Data
# ----------------------------
legs_df = pd.DataFrame()

if source != "FL3XX API":
    st.session_state.pop("fl3xx_last_metadata", None)

if source == "FL3XX API":
    default_token = ""
    if "fl3xx_api" in st.secrets:
        default_token = st.secrets["fl3xx_api"].get("api_token", "")
    if not default_token:
        default_token = st.secrets.get("FL3XX_TOKEN", "")

    token = default_token
    if not token:
        st.sidebar.warning(
            "Configure an FL3XX API token in Streamlit secrets to enable fetching."
        )

    fetch_btn = st.sidebar.button(
        "Fetch from FL3XX", type="primary", disabled=not bool(token)
    )
    if fetch_btn:
        legs_df = fetch_fl3xx_legs(token, start_utc, end_utc)
        if legs_df.empty:
            message = (
                "No legs returned. Check your endpoint/mapping and date range "
                f"({window_label})."
            )
            metadata = st.session_state.get("fl3xx_last_metadata", {})
            stats = metadata.get("normalisation", {})
            raw_count = stats.get("raw_count")
            normalised = stats.get("normalised")

            if raw_count:
                parts = [
                    f"The API returned {raw_count} flight{'s' if raw_count != 1 else ''}"
                ]
                if normalised is not None:
                    parts.append(
                        f"but {normalised} could be converted into legs"
                    )

                reasons = []

                def _format_reason(key, description):
                    count = stats.get(key, 0)
                    if not count:
                        return None
                    suffix = "s" if count != 1 else ""
                    return f"{count} flight{suffix} {description}"

                for key, desc in (
                    ("skipped_missing_tail", "were missing tail numbers"),
                    ("skipped_missing_dep_airport", "were missing departure airports"),
                    ("skipped_missing_arr_airport", "were missing arrival airports"),
                    (
                        "skipped_missing_times",
                        "were missing both departure and arrival times",
                    ),
                    ("skipped_non_mapping", "had an unexpected format"),
                ):
                    reason = _format_reason(key, desc)
                    if reason:
                        reasons.append(reason)

                details = ". ".join(parts)
                if reasons:
                    details += ". Flights were skipped because " + ", ".join(reasons)

                message = f"{message} {details}."

            st.warning(message)
else:
    up = st.sidebar.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if up is not None:
        legs_df = load_uploaded(up)

# ----------------------------
# Compute & Display Short Turns
# ----------------------------
if not legs_df.empty:
    with st.expander("Raw legs (normalized)", expanded=False):
        st.dataframe(legs_df, use_container_width=True, hide_index=True)

    short_df = compute_short_turns(legs_df, threshold)

    st.subheader(
        f"Short turns under {threshold} min for {window_label} ({LOCAL_TZ.key})"
    )

    if short_df.empty:
        st.success(f"No short turns found in the selected window ({window_label}). 🎉")
    else:
        # Nice column formatting
        col_config = {
            "arr_leg_id": st.column_config.TextColumn(
                "Arrival Booking",
                help="Booking identifier for the arrival leg",
            ),
            "dep_leg_id": st.column_config.TextColumn(
                "Departure Booking",
                help="Booking identifier for the departure leg",
            ),
            "arr_onblock": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "dep_offblock": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "turn_min": st.column_config.NumberColumn(
                "Turn (min)",
                help="Minutes between ARR on-block and next DEP off-block at the same station",
                step=0.1,
            ),
        }
        st.dataframe(
            short_df,
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
        )

        # Download
        csv = short_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"short_turns_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        # Quick summary chips
        st.markdown("### Summary")
        by_tail = (
            short_df.groupby("tail")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        st.dataframe(by_tail, use_container_width=True, hide_index=True)
else:
    st.info("Select a data source and load legs to see short turns.")

if source == "FL3XX API" and "fl3xx_last_metadata" in st.session_state:
    with st.expander("FL3XX fetch metadata", expanded=False):
        st.json(st.session_state["fl3xx_last_metadata"])
