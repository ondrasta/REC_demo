import hashlib
import html
import io
import json
import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode


# ----------------------------
# Constants / assumptions (immutable defaults)
# ----------------------------
CO2_FACTOR = 0.2263     # kg CO2 per kWh imported from grid (226.3 gCO2/kWh)
PV_COST_PER_KWP = 1000
BATT_COST_PER_KWH = 300
DISCOUNT_RATE = 0.05
LIFETIME_YEARS = 20
ELECTRICITY_INFLATION_RATE = 0.033  # 3.3% per year (decimal form for formulas)

DEFAULT_OPEX_PCT = 1.0  # % of scenario CAPEX — sidebar default

DEFAULT_EXPORT_RATE = 0.1886  # €/kWh (flat export for all tariffs)

# Default annual standing charges (€/year) — sidebar starting values
DEFAULT_STANDING_CHARGE_STANDARD_EUR = 286.60  # Standard (smart meter)
DEFAULT_STANDING_CHARGE_WEEKEND_EUR = 338.85  # Weekend Saver
DEFAULT_STANDING_CHARGE_FLAT_EUR = 286.60  # Flat rate (aligned with Standard)

DEFAULT_PSO_LEVY_EUR_PER_YEAR = 19.10  # annual PSO levy (€), same for all tariffs; escalates with electricity inflation in long-run metrics

DEFAULT_BATTERY_REPLACEMENT_YEAR = 10  # calendar year in horizon; 0 in UI = none
DEFAULT_INVERTER_REPLACEMENT_YEAR = 10
DEFAULT_BATTERY_REPLACEMENT_COST_PCT = 60.0  # % of battery CAPEX
DEFAULT_INVERTER_REPLACEMENT_COST_PCT = 15.0  # % of PV CAPEX

DEFAULT_TARIFFS = {
    "standard": {"day": 0.32067, "peak": 0.36526, "night": 0.2076},
    "weekend": {
        "weekday": {"day": 0.3268, "peak": 0.38095, "night": 0.2476},
        "weekend": {"day": 0.22205, "peak": 0.23265, "night": 0.19025},
    },
    "flat": 0.29966,
}


def get_active_tariff_config(
    override: bool,
    std_day: float,
    std_peak: float,
    std_night: float,
    wk_day: float,
    wk_peak: float,
    wk_night: float,
    we_day: float,
    we_peak: float,
    we_night: float,
    flat_rate: float,
    export_rate: float,
) -> Tuple[Dict, float]:
    """Build active tariff config from sidebar inputs. Never mutates globals."""
    if not override:
        return DEFAULT_TARIFFS, DEFAULT_EXPORT_RATE
    config = {
        "standard": {"day": std_day, "peak": std_peak, "night": std_night},
        "weekend": {
            "weekday": {"day": wk_day, "peak": wk_peak, "night": wk_night},
            "weekend": {"day": we_day, "peak": we_peak, "night": we_night},
        },
        "flat": flat_rate,
    }
    return config, export_rate


def _tariff_cache_key(config: Dict, export_rate: float) -> str:
    """Stable cache key from tariff config."""
    return hashlib.sha256(
        (json.dumps(config, sort_keys=True) + str(export_rate)).encode("utf-8")
    ).hexdigest()[:16]


def build_run_assumptions_overview_rows(
    *,
    pv_capex: float,
    batt_capex: float,
    opex_pct: float,
    discount_rate_pct: float,
    electricity_inflation_pct: float,
    battery_replacement_year: Optional[int],
    battery_replacement_cost_pct: float,
    inverter_replacement_year: Optional[int],
    inverter_replacement_cost_pct: float,
    pv_min: int,
    pv_max: int,
    batt_min: int,
    batt_max: int,
    opt_pv_step: int,
    opt_batt_step: int,
    speed_preset: str,
    standing_charges: Dict[str, float],
    pso_levy_annual: float,
    rt_eff_pct: float,
    dod_pct: float,
    init_soc_pct: float,
    c_rate: float,
    charge_from_pv: bool,
    charge_from_grid_at_night: bool,
    discharge_schedule: str,
    override_tariffs: bool,
) -> List[Dict]:
    """Snapshot of sidebar run inputs for display while optimization runs."""
    sc = standing_charges
    sc_line = (
        f"Standard €{sc['Standard']:,.0f}/y, "
        f"Weekend Saver €{sc['Weekend Saver']:,.0f}/y, "
        f"Flat €{sc['Flat']:,.0f}/y"
    )
    return [
        {"Setting": "PV CAPEX (€/kWp)", "Value": float(pv_capex)},
        {"Setting": "Battery CAPEX (€/kWh)", "Value": float(batt_capex)},
        {"Setting": "OPEX (% of CAPEX)", "Value": float(opex_pct)},
        {"Setting": "PSO levy (annual, €)", "Value": float(pso_levy_annual)},
        {"Setting": "Standing charges (annual)", "Value": sc_line},
        {"Setting": "Discount rate (%)", "Value": round(float(discount_rate_pct), 4)},
        {"Setting": "Electricity inflation (%/y)", "Value": round(float(electricity_inflation_pct), 4)},
        {"Setting": "Battery replacement year", "Value": battery_replacement_year if battery_replacement_year is not None else "—"},
        {"Setting": "Battery replacement (% batt CAPEX)", "Value": float(battery_replacement_cost_pct)},
        {"Setting": "Inverter replacement year", "Value": inverter_replacement_year if inverter_replacement_year is not None else "—"},
        {"Setting": "Inverter replacement (% PV CAPEX)", "Value": float(inverter_replacement_cost_pct)},
        {"Setting": "PV range (kWp)", "Value": f"{pv_min} – {pv_max}"},
        {"Setting": "Battery range (kWh)", "Value": f"{batt_min} – {batt_max}"},
        {"Setting": "Optimizer steps (PV / battery kWh)", "Value": f"{opt_pv_step} / {opt_batt_step}"},
        {"Setting": "Speed preset", "Value": str(speed_preset)},
        {"Setting": "Round-trip efficiency (%)", "Value": float(rt_eff_pct)},
        {"Setting": "DoD (%)", "Value": float(dod_pct)},
        {"Setting": "Initial SOC (%)", "Value": float(init_soc_pct)},
        {"Setting": "C-rate", "Value": float(c_rate)},
        {"Setting": "Battery dispatch", "Value": str(discharge_schedule)},
        {"Setting": "Charge from PV", "Value": bool(charge_from_pv)},
        {"Setting": "Charge from grid at night", "Value": bool(charge_from_grid_at_night)},
        {"Setting": "Tariff override enabled", "Value": bool(override_tariffs)},
    ]


def tariffs_in_use_info_text(cfg: Dict, export_rate: float) -> str:
    """Multi-line message for st.info: active import bands and export (€/kWh)."""
    return (
        "Tariffs in use (€/kWh):\n"
        f"- Standard: day={cfg['standard']['day']:.5f}, peak={cfg['standard']['peak']:.5f}, night={cfg['standard']['night']:.5f}\n"
        f"- Weekend Saver weekday: day={cfg['weekend']['weekday']['day']:.5f}, peak={cfg['weekend']['weekday']['peak']:.5f}, night={cfg['weekend']['weekday']['night']:.5f}\n"
        f"- Weekend Saver weekend: day={cfg['weekend']['weekend']['day']:.5f}, peak={cfg['weekend']['weekend']['peak']:.5f}, night={cfg['weekend']['weekend']['night']:.5f}\n"
        f"- Flat: {float(cfg['flat']):.5f}\n"
        f"- Export: {float(export_rate):.4f}"
    )


def render_last_run_tariffs_and_assumptions_section() -> None:
    """Tariff list + assumption table from last successful run (Settings & KPI guide tab)."""
    st.markdown("### Tariffs and assumptions (last completed run)")
    if st.session_state.get("opt_dfs") is None:
        st.info("Complete **Run analysis** once to see the tariff and assumption snapshot for your last run here.")
        return

    cfg = st.session_state.last_tariff_config
    er = float(st.session_state.last_export_rate)
    st.info(tariffs_in_use_info_text(cfg, er))

    loc = st.session_state.last_opt_cfg
    bs = st.session_state.last_battery_settings
    rows = build_run_assumptions_overview_rows(
        pv_capex=float(st.session_state.last_pv_capex),
        batt_capex=float(st.session_state.last_batt_capex),
        opex_pct=float(st.session_state.last_opex_pct),
        discount_rate_pct=float(st.session_state.last_discount_rate) * 100.0,
        electricity_inflation_pct=float(st.session_state.last_electricity_inflation_rate) * 100.0,
        battery_replacement_year=st.session_state.last_battery_replacement_year,
        battery_replacement_cost_pct=float(st.session_state.last_battery_replacement_cost_pct),
        inverter_replacement_year=st.session_state.last_inverter_replacement_year,
        inverter_replacement_cost_pct=float(st.session_state.last_inverter_replacement_cost_pct),
        pv_min=int(loc["pv_min"]),
        pv_max=int(loc["pv_max"]),
        batt_min=int(loc["batt_min"]),
        batt_max=int(loc["batt_max"]),
        opt_pv_step=int(loc["pv_step"]),
        opt_batt_step=int(loc["batt_step"]),
        speed_preset=str(loc.get("speed_preset", "")),
        standing_charges=dict(st.session_state.last_standing_charges),
        pso_levy_annual=float(st.session_state.last_pso_levy),
        rt_eff_pct=float(bs.eff_round_trip) * 100.0,
        dod_pct=float(bs.dod) * 100.0,
        init_soc_pct=float(bs.init_soc) * 100.0,
        c_rate=float(bs.c_rate),
        charge_from_pv=bool(bs.charge_from_pv),
        charge_from_grid_at_night=bool(bs.charge_from_grid_at_night),
        discharge_schedule=str(bs.discharge_schedule),
        override_tariffs=bool(st.session_state.get("last_override_tariffs", False)),
    )
    with st.expander("Run assumptions (last completed run)", expanded=True):
        st.caption(
            "Frozen from your last successful **Run analysis**. If sidebar inputs differ, rerun to refresh results and this snapshot."
        )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=360,
        )


def tariff_band(hour: int) -> str:
    # Peak: 17:00–19:00
    if 17 <= hour < 19:
        return "peak"
    # Night: 23:00–08:00
    if hour >= 23 or hour < 8:
        return "night"
    return "day"


def get_tariff_value_from_config(ts: pd.Timestamp, tariff_col: str, config: Dict) -> float:
    """
    tariff_col is one of: 'tariff_standard', 'tariff_weekend', 'tariff_flat'
    config: from get_active_tariff_config
    """
    hour = int(ts.hour)
    band = tariff_band(hour)
    if tariff_col == "tariff_flat":
        return float(config["flat"])
    is_weekend = ts.dayofweek >= 5  # Sat=5, Sun=6
    if tariff_col == "tariff_weekend":
        key = "weekend" if is_weekend else "weekday"
        return float(config["weekend"][key][band])
    return float(config["standard"][band])


# ----------------------------
# Data parsing
# ----------------------------
def _shift_year_boundary_midnight_back(ts: pd.Series) -> pd.Series:
    """
    Normalize end-of-year hourly convention:
    01/01/(Y+1) 00:00 is treated as the last hour of year Y (23:00-24:00),
    so shift that specific timestamp back by one hour.
    """
    if ts.empty:
        return ts
    out = ts.copy()
    year_vals = out.dt.year
    if not (year_vals == 2020).any():
        return out
    boundary_mask = (
        (out.dt.year == 2021)
        & (out.dt.month == 1)
        & (out.dt.day == 1)
        & (out.dt.hour == 0)
        & (out.dt.minute == 0)
    )
    if boundary_mask.any():
        out.loc[boundary_mask] = out.loc[boundary_mask] - pd.Timedelta(hours=1)
    return out


def _parse_consumption_csv(cons_bytes: bytes) -> pd.DataFrame:
    """
    Expected columns:
      - date: 'DD/MM/YYYY HH:00'
      - consumption: usually 'Final_Community_Sum'
    """
    from io import BytesIO

    df = pd.read_csv(BytesIO(cons_bytes))
    if "date" not in df.columns:
        raise ValueError("Consumption CSV must contain a 'date' column.")

    # Parse timestamps and round down to the hour
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M", errors="coerce").dt.floor("h")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = _shift_year_boundary_midnight_back(df["date"])

    if "Final_Community_Sum" in df.columns:
        consumption_col = "Final_Community_Sum"
    elif len(df.columns) >= 2:
        consumption_col = df.columns[1]
    else:
        raise ValueError("Consumption CSV must contain at least two columns (date + consumption).")
    df = df[["date", consumption_col]].rename(columns={consumption_col: "consumption"})

    # If year 2020 exists, keep only 2020 to match the notebook/methodology usage
    if (df["date"].dt.year == 2020).any():
        df = df[df["date"].dt.year == 2020].copy()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _find_pvgis_header_row(text: str) -> int:
    # PVGIS exports vary; find the line containing both time and P/G(i)
    lines = text.splitlines()
    for i, line in enumerate(lines[:300]):  # header should be early
        lower = line.lower()
        if "time" in lower and ("p" in lower or "g(i)" in lower):
            return i
    return 10


def _parse_pv_timeseries_csv(pv_bytes: bytes) -> pd.DataFrame:
    """
    PVGIS timeseries file:
      - skip header rows
      - 'time' values like '20200101:1011'
      - 'P' column (Wh for 1 kWp)
    """
    from io import BytesIO

    raw_text = pv_bytes.decode("utf-8", errors="ignore")
    header_row = _find_pvgis_header_row(raw_text)

    df = pd.read_csv(BytesIO(pv_bytes), skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    if "time" not in df.columns:
        time_col = df.columns[0]
    else:
        time_col = "time"

    def parse_pv_time(s):
        s = str(s).strip()
        # Expect YYYYMMDD:HHmm => validate strictly
        if len(s) < 12:
            return pd.NaT
        if not s[:8].isdigit() or s[8] != ":":
            return pd.NaT
        if not s[9:11].isdigit():
            return pd.NaT
        date_part = s[:8]
        hour_part = s[9:11]  # treat HHmm -> HH (drop minutes)
        return pd.Timestamp(f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {int(hour_part):02d}:00:00")

    df["date"] = df[time_col].apply(parse_pv_time)
    df = df.dropna(subset=["date"]).copy()
    df["date"] = _shift_year_boundary_midnight_back(df["date"])

    # PV energy column can be 'P' (Wh) in your file
    if "P" in df.columns:
        p_col = "P"
    elif len(df.columns) >= 2:
        p_col = df.columns[1]
    else:
        raise ValueError("PV CSV must contain at least two columns (time + production).")
    df["pv_per_kwp"] = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0) / 1000.0

    # Optional: keep 2020 if present
    if (df["date"].dt.year == 2020).any():
        df = df[df["date"].dt.year == 2020].copy()

    df = df[["date", "pv_per_kwp"]].sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_and_prepare_data(cons_bytes: bytes, pv_bytes: bytes, tariff_config: Dict, tariff_cache_key: str) -> pd.DataFrame:
    """Prepare df with tariff columns from config. Cache key must reflect config."""
    df_cons = _parse_consumption_csv(cons_bytes)
    df_pv = _parse_pv_timeseries_csv(pv_bytes)

    df = df_cons.merge(df_pv, on="date", how="left")
    df["pv_per_kwp"] = df["pv_per_kwp"].fillna(0.0)

    for col in ["tariff_standard", "tariff_weekend", "tariff_flat"]:
        df[col] = df["date"].apply(lambda t, c=col: get_tariff_value_from_config(t, c, tariff_config))

    df = df.reset_index(drop=True)
    return df


# ----------------------------
# Battery model settings
# ----------------------------
DEFAULT_BATT_EFF_ROUND_TRIP = 0.9  # round-trip efficiency
DEFAULT_BATT_DOD = 0.9            # usable depth of discharge
DEFAULT_BATT_INIT_SOC = 0.0      # initial state of charge (fraction)
DEFAULT_BATT_C_RATE = 0.5         # max charge/discharge power as fraction of capacity per hour


@dataclass(frozen=True)
class BatterySettings:
    eff_round_trip: float = DEFAULT_BATT_EFF_ROUND_TRIP
    dod: float = DEFAULT_BATT_DOD
    init_soc: float = DEFAULT_BATT_INIT_SOC
    c_rate: float = DEFAULT_BATT_C_RATE

    # Dispatch/rules
    charge_from_pv: bool = True
    charge_from_grid_at_night: bool = True
    discharge_schedule: str = "Peak only"  # or "Day+Peak"


def run_scenario_grid_only(df: pd.DataFrame, tariff_col: str) -> pd.DataFrame:
    d = df.copy()
    d["grid_import"] = d["consumption"]
    d["feed_in"] = 0.0
    d["pv_generation"] = 0.0
    d["self_consumed_pv"] = 0.0
    d["local_renewable_to_load"] = 0.0
    return d


def run_scenario_pv_grid(df: pd.DataFrame, pv_kwp: int, tariff_col: str) -> pd.DataFrame:
    d = df.copy()
    pv_prod = d["pv_per_kwp"].to_numpy(dtype=float) * float(pv_kwp)
    cons = d["consumption"].to_numpy(dtype=float)

    grid_import = np.maximum(cons - pv_prod, 0.0)
    feed_in = np.maximum(pv_prod - cons, 0.0)

    d["grid_import"] = grid_import
    d["feed_in"] = feed_in
    d["pv_generation"] = pv_prod
    d["self_consumed_pv"] = pv_prod - feed_in
    d["local_renewable_to_load"] = pv_prod - feed_in  # all self-consumed is local renewable
    return d


def run_scenario_battery_grid(df: pd.DataFrame, batt_kwh: int, tariff_col: str, battery_settings: BatterySettings) -> pd.DataFrame:
    """Battery + Grid: charge from grid at night, discharge during peak. Symmetric round-trip efficiency."""
    d = df.copy()
    cons = d["consumption"].to_numpy(dtype=float)
    charge_eff = np.sqrt(battery_settings.eff_round_trip)
    discharge_eff = np.sqrt(battery_settings.eff_round_trip)
    max_power = batt_kwh * float(battery_settings.c_rate)
    usable = batt_kwh * float(battery_settings.dod)
    soc = batt_kwh * float(battery_settings.init_soc)

    grid_import = np.zeros_like(cons, dtype=float)
    feed_in = np.zeros_like(cons, dtype=float)

    for i in range(len(cons)):
        hour = int(d.loc[i, "date"].hour)
        is_night = hour >= 23 or hour < 8

        ch_from_grid = 0.0
        dch = 0.0

        if is_night:
            space = usable - soc
            if space > 0 and max_power > 0 and battery_settings.charge_from_grid_at_night:
                # Charge: draw from grid, SOC += drawn * charge_eff
                ch_from_grid = min(space / charge_eff, max_power)
                soc += ch_from_grid * charge_eff
        else:
            hour_is_peak = 17 <= hour < 19
            discharge_ok = battery_settings.discharge_schedule == "Day+Peak" or (
                battery_settings.discharge_schedule == "Peak only" and hour_is_peak
            )
            if discharge_ok and soc > 0 and max_power > 0:
                # Discharge: deliver to load; max deliverable = soc * discharge_eff; SOC -= delivered / discharge_eff
                max_deliverable = soc * discharge_eff
                dch = min(max_deliverable, cons[i], max_power)
                soc -= dch / discharge_eff

        grid_import[i] = max(0.0, cons[i] - dch) + ch_from_grid

    d["grid_import"] = grid_import
    d["feed_in"] = feed_in
    d["pv_generation"] = 0.0
    d["self_consumed_pv"] = 0.0
    d["local_renewable_to_load"] = 0.0  # battery-only: grid-charged, so 0% self-sufficiency
    return d


def run_scenario_pv_battery_grid(
    df: pd.DataFrame,
    pv_kwp: int,
    batt_kwh: int,
    tariff_col: str,
    battery_settings: BatterySettings,
) -> pd.DataFrame:
    """PV + Battery + Grid with symmetric round-trip efficiency. Tracks PV vs grid origin for self-sufficiency."""
    d = df.copy()
    cons = d["consumption"].to_numpy(dtype=float)
    pv_prod = d["pv_per_kwp"].to_numpy(dtype=float) * float(pv_kwp)

    charge_eff = np.sqrt(battery_settings.eff_round_trip)
    discharge_eff = np.sqrt(battery_settings.eff_round_trip)
    max_power = batt_kwh * float(battery_settings.c_rate)
    usable = batt_kwh * float(battery_settings.dod)
    soc = batt_kwh * float(battery_settings.init_soc)
    soc_pv = 0.0  # PV-origin energy in battery
    soc_grid = 0.0  # grid-origin energy in battery

    grid_import = np.zeros_like(cons, dtype=float)
    feed_in = np.zeros_like(cons, dtype=float)
    pv_to_load_direct = np.zeros_like(cons, dtype=float)
    battery_to_load_pv_origin = np.zeros_like(cons, dtype=float)

    for i in range(len(cons)):
        hour = int(d.loc[i, "date"].hour)
        is_night = hour >= 23 or hour < 8
        is_peak_or_day = not is_night

        pv = pv_prod[i]
        cons_load = cons[i]

        # 1) PV self-consumption first
        pv_to_load = min(pv, cons_load)
        pv_surplus = pv - pv_to_load
        cons_remaining = cons_load - pv_to_load
        pv_to_load_direct[i] = pv_to_load

        # 2) Charge battery from PV surplus
        ch_from_pv = 0.0
        if (
            batt_kwh > 0
            and pv_surplus > 0
            and soc < usable
            and max_power > 0
            and battery_settings.charge_from_pv
        ):
            ch_from_pv = min(pv_surplus, (usable - soc) / charge_eff, max_power)
            soc += ch_from_pv * charge_eff
            soc_pv += ch_from_pv * charge_eff
            pv_surplus -= ch_from_pv

        # 3) Charge battery from grid at night
        ch_from_grid = 0.0
        if (
            batt_kwh > 0
            and is_night
            and soc < usable
            and max_power > 0
            and battery_settings.charge_from_grid_at_night
        ):
            ch_from_grid = min((usable - soc) / charge_eff, max_power)
            soc += ch_from_grid * charge_eff
            soc_grid += ch_from_grid * charge_eff

        # 4) Discharge based on schedule
        dch = 0.0
        dch_pv_origin = 0.0
        hour_is_peak = 17 <= hour < 19
        discharge_ok = battery_settings.discharge_schedule == "Day+Peak" or (
            battery_settings.discharge_schedule == "Peak only" and hour_is_peak
        )
        if batt_kwh > 0 and cons_remaining > 0 and soc > 0 and discharge_ok and max_power > 0:
            max_deliverable = soc * discharge_eff
            dch = min(max_deliverable, cons_remaining, max_power)
            if soc > 1e-9:
                frac_pv = soc_pv / soc
                dch_pv_origin = dch * frac_pv
            soc -= dch / discharge_eff
            soc_pv = max(0.0, soc_pv - dch_pv_origin / discharge_eff)
            soc_grid = max(0.0, soc_grid - (dch - dch_pv_origin) / discharge_eff)
            cons_remaining -= dch

        battery_to_load_pv_origin[i] = dch_pv_origin
        grid_import[i] = cons_remaining + ch_from_grid
        feed_in[i] = max(0.0, pv_surplus)

    d["grid_import"] = grid_import
    d["feed_in"] = feed_in
    d["pv_generation"] = pv_prod
    # Self-consumed PV = direct PV to load + PV-origin battery discharge to load
    # (excludes PV lost to round-trip loss and PV remaining in battery at end)
    pv_used_locally = pv_to_load_direct + battery_to_load_pv_origin
    d["self_consumed_pv"] = pv_used_locally
    d["pv_to_load_direct"] = pv_to_load_direct
    d["battery_to_load_pv_origin"] = battery_to_load_pv_origin
    d["local_renewable_to_load"] = pv_used_locally
    return d


# ----------------------------
# Financial / KPI helpers
# ----------------------------
def compute_kpis_for_scenario(d: pd.DataFrame, tariff_col: str, export_rate: float) -> Dict[str, float]:
    grid_import_kwh = float(np.sum(d["grid_import"].to_numpy(dtype=float)))
    export_kwh = float(np.sum(d["feed_in"].to_numpy(dtype=float)))
    pv_gen_kwh = float(np.sum(d["pv_generation"].to_numpy(dtype=float)))
    self_cons_kwh = float(np.sum(d["self_consumed_pv"].to_numpy(dtype=float))) if "self_consumed_pv" in d.columns else max(pv_gen_kwh - export_kwh, 0.0)

    # Cost of grid imports only
    cost_grid_import = float(np.sum(d["grid_import"].to_numpy(dtype=float) * d[tariff_col].to_numpy(dtype=float)))
    # Export income
    export_income = export_kwh * export_rate
    annual_cost = cost_grid_import - export_income

    # CO2 only from imports
    co2_kg = grid_import_kwh * CO2_FACTOR

    total_consumption = float(d["consumption"].to_numpy(dtype=float).sum())
    local_renewable = float(d["local_renewable_to_load"].to_numpy(dtype=float).sum()) if "local_renewable_to_load" in d.columns else self_cons_kwh
    self_sufficiency_ratio = 100.0 * (local_renewable / total_consumption) if total_consumption > 0 else 0.0

    # Baseline is computed outside (grid-only), so do not set CO2 savings here
    return {
        "Total annual community consumption (kWh)": total_consumption,
        "Total annual PV generation (kWh)": pv_gen_kwh,
        "Self-consumed PV (kWh)": self_cons_kwh,
        "Export to grid (kWh)": export_kwh,
        "Export income (€)": export_income,
        "Self-consumption ratio (%)": (100.0 * (self_cons_kwh / pv_gen_kwh)) if pv_gen_kwh > 0 else 0.0,
        "Self-sufficiency ratio (%)": self_sufficiency_ratio,
        "Grid import (kWh)": grid_import_kwh,
        "Cost of grid import (€)": cost_grid_import,
        "Annual cost (€)": annual_cost,
        "CO2 (kg)": co2_kg,
    }


def _gross_savings_20y(annual_savings_year1: float, electricity_inflation_rate: float) -> float:
    """Sum of inflated annual savings over LIFETIME_YEARS. When infl=0, returns annual_savings * 20."""
    if electricity_inflation_rate <= 0:
        return annual_savings_year1 * LIFETIME_YEARS
    infl = float(electricity_inflation_rate)
    return annual_savings_year1 * float(((1 + infl) ** LIFETIME_YEARS - 1) / infl)


def compute_payback_and_npv(
    capex_eur: float,
    annual_savings_eur: float,
    discount_rate: float | None = None,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_cost_eur: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_cost_eur: float = 0.0,
) -> Tuple[float, float]:
    r = discount_rate if discount_rate is not None else DISCOUNT_RATE
    npv = -capex_eur
    payback = float("inf") if annual_savings_eur <= 0 else (capex_eur / annual_savings_eur)
    # Always include the 20-year discounted savings stream in NPV,
    # including zero/negative annual savings.
    if electricity_inflation_rate <= 0:
        discount_factors = (1 + r) ** (-np.arange(1, LIFETIME_YEARS + 1))
        npv += annual_savings_eur * float(discount_factors.sum())
    else:
        infl = float(electricity_inflation_rate)
        for t in range(1, LIFETIME_YEARS + 1):
            savings_t = annual_savings_eur * ((1 + infl) ** (t - 1))
            npv += savings_t / ((1 + r) ** t)
    if (
        battery_replacement_year is not None
        and 1 <= int(battery_replacement_year) <= LIFETIME_YEARS
        and float(battery_replacement_cost_eur) > 0
    ):
        npv -= float(battery_replacement_cost_eur) / ((1 + r) ** int(battery_replacement_year))
    if (
        inverter_replacement_year is not None
        and 1 <= int(inverter_replacement_year) <= LIFETIME_YEARS
        and float(inverter_replacement_cost_eur) > 0
    ):
        npv -= float(inverter_replacement_cost_eur) / ((1 + r) ** int(inverter_replacement_year))
    return payback, npv


def compute_irr(
    capex_eur: float,
    annual_savings_eur: float,
    n_years: int = LIFETIME_YEARS,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_cost_eur: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_cost_eur: float = 0.0,
) -> float:
    """IRR: discount rate r where NPV = 0. Returns as decimal (e.g. 0.08 for 8%)."""
    if capex_eur <= 0 or annual_savings_eur <= 0:
        return float("nan")
    infl = float(electricity_inflation_rate)

    def npv_at_r(rate: float) -> float:
        if rate <= -0.999:
            return float("inf")
        total = -capex_eur
        for t in range(1, n_years + 1):
            savings_t = annual_savings_eur * ((1 + infl) ** (t - 1))
            total += savings_t / ((1 + rate) ** t)
        if (
            battery_replacement_year is not None
            and 1 <= int(battery_replacement_year) <= n_years
            and float(battery_replacement_cost_eur) > 0
        ):
            total -= float(battery_replacement_cost_eur) / ((1 + rate) ** int(battery_replacement_year))
        if (
            inverter_replacement_year is not None
            and 1 <= int(inverter_replacement_year) <= n_years
            and float(inverter_replacement_cost_eur) > 0
        ):
            total -= float(inverter_replacement_cost_eur) / ((1 + rate) ** int(inverter_replacement_year))
        return total

    # Bisection: find r in [lo, hi] where npv_at_r(r) = 0
    lo, hi = 1e-6, 2.0  # 0.0001% to 200%
    if npv_at_r(lo) < 0:
        return float("nan")  # No positive IRR
    if npv_at_r(hi) > 0:
        return hi  # IRR > 200%, cap at 200%
    for _ in range(80):  # ~2^-80 precision
        mid = (lo + hi) / 2
        val = npv_at_r(mid)
        if abs(val) < 1e-6:
            return mid
        if val > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_financial_metrics(
    energy_cost: float,
    baseline_energy_cost: float,
    capex: float,
    standing_charge: float,
    opex_pct: float,
    discount_rate: float | None = None,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_cost_eur: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_cost_eur: float = 0.0,
    pso_levy_annual: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Shared finance logic: annual_cost (year 1), annual_savings (year 1), payback, npv."""
    opex = capex * (opex_pct / 100.0)
    fixed_annual = float(standing_charge) + float(pso_levy_annual)
    annual_cost = energy_cost + fixed_annual + opex
    baseline_cost = baseline_energy_cost + fixed_annual
    annual_savings = baseline_cost - annual_cost
    payback, npv = compute_payback_and_npv(
        capex,
        annual_savings,
        discount_rate,
        electricity_inflation_rate,
        battery_replacement_year,
        battery_replacement_cost_eur,
        inverter_replacement_year,
        inverter_replacement_cost_eur,
    )
    return annual_cost, annual_savings, payback, npv


def build_pv_grid_sweep_table(
    df: pd.DataFrame,
    tariff_col: str,
    pv_capex_per_kwp: float,
    export_rate: float,
    standing_charge: float = 0.0,
    opex_pct: float = 0.0,
    discount_rate: float | None = None,
    electricity_inflation_rate: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_pct_of_pv_capex: float = 0.0,
    pso_levy_annual: float = 0.0,
    pv_min: int = 0,
    pv_max: int = 100,
) -> pd.DataFrame:
    """
    PV + Grid only: one row per PV size (pv_min–pv_max kWp, step 1 kWp).
    Uses same KPI definitions as the main results view.
    """
    d_base = run_scenario_grid_only(df, tariff_col)
    baseline_cost = float(np.sum(d_base["grid_import"].to_numpy(dtype=float) * df[tariff_col].to_numpy(dtype=float)))

    rows = []
    if int(pv_max) < int(pv_min):
        return pd.DataFrame(rows)
    for pv in range(int(pv_min), int(pv_max) + 1, 1):
        d = run_scenario_pv_grid(df, pv, tariff_col)
        k = compute_kpis_for_scenario(d, tariff_col, export_rate)
        capex = pv * pv_capex_per_kwp
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k["Annual cost (€)"] - opex
        inverter_replacement_cost = capex * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        payback, npv = compute_payback_and_npv(
            capex,
            annual_savings,
            discount_rate,
            electricity_inflation_rate,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inverter_replacement_cost,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            electricity_inflation_rate=electricity_inflation_rate,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inverter_replacement_cost,
        )
        rows.append(
            {
                "PV (kWp)": pv,
                "Grid import (kWh)": k["Grid import (kWh)"],
                "Annual PV generation (kWh)": k["Total annual PV generation (kWh)"],
                "Self-Consumption (kWh)": k["Self-consumed PV (kWh)"],
                "Export (kWh)": k["Export to grid (kWh)"],
                "Annual cost of grid import (€)": k["Cost of grid import (€)"],
                "Annual electricity cost (€)": k["Annual cost (€)"] + standing_charge + pso_levy_annual + opex,
                "Annual export earnings (€)": k["Export income (€)"],
                "Annual savings vs grid only (€)": annual_savings,
                "CAPEX (€)": capex,
                "Payback period (years)": payback,
                "NPV (20y, €)": npv,
                "IRR (20y, %)": 100.0 * irr if np.isfinite(irr) else float("nan"),
                "Self-sufficiency ratio (%)": k["Self-sufficiency ratio (%)"],
                "Self-consumption ratio (%)": k["Self-consumption ratio (%)"],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# Optimizer
# ----------------------------
@dataclass(frozen=True)
class OptimizerConfig:
    pv_min: int = 0
    pv_max: int = 150
    pv_step: int = 5      # quick default
    batt_min: int = 0
    batt_max: int = 300
    batt_step: int = 10   # quick default


def count_optimizer_evaluations(config: OptimizerConfig) -> int:
    """Count scenario evaluations for one tariff run."""
    pv_count = len(range(config.pv_min, config.pv_max + 1, config.pv_step))
    batt_sizes = [b for b in range(config.batt_min, config.batt_max + 1, config.batt_step) if b != 0]
    batt_only_count = len(batt_sizes)
    batt_start = config.batt_min if config.batt_min > 0 else config.batt_step
    pv_batt_count = pv_count * len(range(batt_start, config.batt_max + 1, config.batt_step))
    return pv_count + batt_only_count + pv_batt_count


def optimize(
    df: pd.DataFrame,
    tariff_col: str,
    config: OptimizerConfig,
    battery_settings: BatterySettings,
    export_rate: float,
    standing_charge: float = 0.0,
    opex_pct: float = 0.0,
    discount_rate: float | None = None,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_pct_of_batt_capex: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_pct_of_pv_capex: float = 0.0,
    pso_levy_annual: float = 0.0,
    progress_callback: Callable[[], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    d_base = run_scenario_grid_only(df, tariff_col)
    baseline_energy_cost = float(np.sum(d_base["grid_import"].to_numpy(dtype=float) * df[tariff_col].to_numpy(dtype=float)))
    baseline_co2 = float(d_base["grid_import"].to_numpy(dtype=float).sum() * CO2_FACTOR)

    total_consumption = float(df["consumption"].to_numpy(dtype=float).sum())

    rows = []
    eval_counter = 0

    # PV only
    for pv in range(config.pv_min, config.pv_max + 1, config.pv_step):
        if stop_requested is not None and stop_requested():
            return pd.DataFrame(rows)
        d = run_scenario_pv_grid(df, pv, tariff_col)
        cost = float((d["grid_import"].to_numpy() * d[tariff_col].to_numpy()).sum() - (d["feed_in"].to_numpy().sum() * export_rate))

        pv_gen_kwh = float(d["pv_generation"].to_numpy(dtype=float).sum())
        self_cons_kwh = float(d["self_consumed_pv"].to_numpy(dtype=float).sum())
        self_consumption_ratio_pct = (100.0 * self_cons_kwh / pv_gen_kwh) if pv_gen_kwh > 0 else 0.0

        inv = pv * PV_COST_PER_KWP
        batt_repl_cost = 0.0
        inv_repl_cost = inv * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        annual_cost, savings, payback, npv = compute_financial_metrics(
            cost,
            baseline_energy_cost,
            inv,
            standing_charge,
            opex_pct,
            discount_rate,
            electricity_inflation_rate,
            battery_replacement_year,
            batt_repl_cost,
            inverter_replacement_year,
            inv_repl_cost,
            pso_levy_annual=pso_levy_annual,
        )
        irr = compute_irr(
            inv,
            savings,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl_cost,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl_cost,
        )
        k = compute_kpis_for_scenario(d, tariff_col, export_rate)
        ss = k["Self-sufficiency ratio (%)"]
        co2_kg = float(d["grid_import"].to_numpy(dtype=float).sum() * CO2_FACTOR)
        co2_save = max(0.0, baseline_co2 - co2_kg)
        rows.append(
            {
                "config": "PV only",
                "pv_kwp": pv,
                "batt_kwh": 0,
                "cost": annual_cost,
                "savings": savings,
                "co2_save_kg": co2_save,
                "self_suff_pct": ss,
                "self_consumption_ratio_pct": self_consumption_ratio_pct,
                "pv_gen_kwh": pv_gen_kwh,
                "payback": payback,
                "npv": npv,
                "irr": irr,
            }
        )
        eval_counter += 1
        if progress_callback is not None:
            progress_callback()

    # Battery only
    for batt in range(config.batt_min, config.batt_max + 1, config.batt_step):
        if stop_requested is not None and stop_requested():
            return pd.DataFrame(rows)
        if batt == 0:
            continue
        d = run_scenario_battery_grid(df, batt, tariff_col, battery_settings)
        energy_cost = float(np.sum(d["grid_import"].to_numpy(dtype=float) * d[tariff_col].to_numpy(dtype=float)))

        inv = batt * BATT_COST_PER_KWH
        batt_repl_cost = inv * (float(battery_replacement_pct_of_batt_capex) / 100.0)
        inv_repl_cost = 0.0
        annual_cost, savings, payback, npv = compute_financial_metrics(
            energy_cost,
            baseline_energy_cost,
            inv,
            standing_charge,
            opex_pct,
            discount_rate,
            electricity_inflation_rate,
            battery_replacement_year,
            batt_repl_cost,
            inverter_replacement_year,
            inv_repl_cost,
            pso_levy_annual=pso_levy_annual,
        )
        irr = compute_irr(
            inv,
            savings,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl_cost,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl_cost,
        )
        k = compute_kpis_for_scenario(d, tariff_col, export_rate)
        ss = k["Self-sufficiency ratio (%)"]
        co2_kg = float(d["grid_import"].to_numpy(dtype=float).sum() * CO2_FACTOR)
        co2_save = max(0.0, baseline_co2 - co2_kg)
        rows.append(
            {
                "config": "Battery only",
                "pv_kwp": 0,
                "batt_kwh": batt,
                "cost": annual_cost,
                "savings": savings,
                "co2_save_kg": co2_save,
                "self_suff_pct": ss,
                "self_consumption_ratio_pct": 0.0,
                "pv_gen_kwh": 0.0,
                "payback": payback,
                "npv": npv,
                "irr": irr,
            }
        )
        eval_counter += 1
        if progress_callback is not None:
            progress_callback()

    # PV + Battery: respect batt_min, batt_max, batt_step
    batt_start = config.batt_min if config.batt_min > 0 else config.batt_step
    for pv in range(config.pv_min, config.pv_max + 1, config.pv_step):
        if stop_requested is not None and stop_requested():
            return pd.DataFrame(rows)
        for batt in range(batt_start, config.batt_max + 1, config.batt_step):
            if stop_requested is not None and stop_requested():
                return pd.DataFrame(rows)
            d = run_scenario_pv_battery_grid(df, pv, batt, tariff_col, battery_settings)
            energy_cost = float((d["grid_import"].to_numpy() * d[tariff_col].to_numpy()).sum() - (d["feed_in"].to_numpy().sum() * export_rate))

            pv_gen_kwh = float(d["pv_generation"].to_numpy(dtype=float).sum())
            self_cons_kwh = float(d["self_consumed_pv"].to_numpy(dtype=float).sum())
            self_consumption_ratio_pct = (100.0 * self_cons_kwh / pv_gen_kwh) if pv_gen_kwh > 0 else 0.0

            inv = pv * PV_COST_PER_KWP + batt * BATT_COST_PER_KWH
            batt_repl_cost = (batt * BATT_COST_PER_KWH) * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            inv_repl_cost = (pv * PV_COST_PER_KWP) * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            annual_cost, savings, payback, npv = compute_financial_metrics(
                energy_cost,
                baseline_energy_cost,
                inv,
                standing_charge,
                opex_pct,
                discount_rate,
                electricity_inflation_rate,
                battery_replacement_year,
                batt_repl_cost,
                inverter_replacement_year,
                inv_repl_cost,
                pso_levy_annual=pso_levy_annual,
            )
            irr = compute_irr(
                inv,
                savings,
                electricity_inflation_rate=electricity_inflation_rate,
                battery_replacement_year=battery_replacement_year,
                battery_replacement_cost_eur=batt_repl_cost,
                inverter_replacement_year=inverter_replacement_year,
                inverter_replacement_cost_eur=inv_repl_cost,
            )
            k = compute_kpis_for_scenario(d, tariff_col, export_rate)
            ss = k["Self-sufficiency ratio (%)"]
            co2_kg = float(d["grid_import"].to_numpy(dtype=float).sum() * CO2_FACTOR)
            co2_save = max(0.0, baseline_co2 - co2_kg)
            rows.append(
                {
                    "config": "PV + Battery",
                    "pv_kwp": pv,
                    "batt_kwh": batt,
                    "cost": annual_cost,
                    "savings": savings,
                    "co2_save_kg": co2_save,
                    "self_suff_pct": ss,
                    "self_consumption_ratio_pct": self_consumption_ratio_pct,
                    "pv_gen_kwh": pv_gen_kwh,
                    "payback": payback,
                    "npv": npv,
                    "irr": irr,
                }
            )
            eval_counter += 1
            if progress_callback is not None:
                progress_callback()
    return pd.DataFrame(rows)


def pick_best(opt_df: pd.DataFrame, config_name: str, goal: str) -> pd.Series:
    sub = opt_df[opt_df["config"] == config_name].copy()
    if len(sub) == 0:
        raise ValueError(f"No optimizer rows found for config '{config_name}'.")

    # Map your goals to optimizer columns
    # Goal names used in the UI are chosen below.
    if goal == "Lowest annual electricity cost":
        return sub.loc[sub["cost"].idxmin()]
    if goal == "Highest annual savings":
        return sub.loc[sub["savings"].idxmax()]
    if goal == "Best payback":
        # shortest payback; ignore inf
        finite = sub[np.isfinite(sub["payback"])]
        if len(finite) == 0:
            return sub.loc[sub["payback"].idxmin()]
        return finite.loc[finite["payback"].idxmin()]
    if goal == "Best self-sufficiency / lowest grid import":
        return sub.loc[sub["self_suff_pct"].idxmax()]
    if goal == "Most CO2 savings":
        return sub.loc[sub["co2_save_kg"].idxmax()]
    if goal == "Best NPV":
        return sub.loc[sub["npv"].idxmax()]
    if goal == "Best IRR":
        # Exclude NaN/negative IRR; maximize among valid
        valid = sub[sub["irr"].notna() & np.isfinite(sub["irr"]) & (sub["irr"] > 0)]
        if len(valid) == 0:
            return sub.iloc[0]
        return valid.loc[valid["irr"].idxmax()]

    raise ValueError(f"Unknown goal: {goal}")


# ----------------------------
# Decision-support helpers
# ----------------------------
_CONFIG_TO_SCENARIO = {
    "PV only": "PV + Grid",
    "PV + Battery": "PV + Battery + Grid",
    "Battery only": "Battery + Grid",
}

_SCENARIO_TYPES_ALL = ["Grid only", "PV + Grid", "Battery + Grid", "PV + Battery + Grid"]


def build_full_scenario_results_df(
    opt_dfs: Dict[str, pd.DataFrame],
    prepared_df: pd.DataFrame,
    standing_charges_by_name: Dict[str, float],
    pv_cost_per_kwp: float,
    batt_cost_per_kwh: float,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_pct_of_batt_capex: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_pct_of_pv_capex: float = 0.0,
    pso_levy_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Build a consolidated scenario table across ALL optimizer configurations.

    Source:
      - opt_dfs[tcol] rows for PV + Grid, PV + Battery + Grid, Battery + Grid
      - a synthetic Grid-only baseline row for each tariff
    """
    if opt_dfs is None or prepared_df is None:
        return pd.DataFrame()

    tcol_to_name = {
        "tariff_standard": "Standard",
        "tariff_weekend": "Weekend Saver",
        "tariff_flat": "Flat",
    }

    cons = prepared_df["consumption"].to_numpy(dtype=float)

    rows: list[pd.DataFrame] = []
    for tcol, tname in tcol_to_name.items():
        opt_df = opt_dfs.get(tcol)
        if opt_df is None or len(opt_df) == 0:
            continue

        df = opt_df.copy()
        df["Tariff"] = tname
        df["Scenario"] = df["config"].map(_CONFIG_TO_SCENARIO).fillna(df["config"])

        pv_kwp = df["pv_kwp"].astype(int)
        batt_kwh = df["batt_kwh"].astype(int)

        capex_eur = pv_kwp * float(pv_cost_per_kwp) + batt_kwh * float(batt_cost_per_kwh)
        df["CAPEX (€)"] = capex_eur

        df["PV (kWp)"] = pv_kwp
        df["Battery (kWh)"] = batt_kwh
        df["Total annual PV generation (kWh)"] = df["pv_gen_kwh"]
        df["Annual cost (€)"] = df["cost"]
        df["Annual savings (€)"] = df["savings"]
        df["Payback (yrs)"] = df["payback"]
        df["NPV (€)"] = df["npv"]
        df["IRR (%)"] = 100.0 * df["irr"]
        df["Self-sufficiency (%)"] = df["self_suff_pct"]
        df["Self-consumption ratio (%)"] = df["self_consumption_ratio_pct"]
        df["CO2 savings (kg)"] = df["co2_save_kg"]

        gross_20y = df["Annual savings (€)"].apply(
            lambda s: _gross_savings_20y(float(s), electricity_inflation_rate)
        )
        batt_capex_eur = batt_kwh * float(batt_cost_per_kwh)
        pv_capex_eur = pv_kwp * float(pv_cost_per_kwp)
        batt_replacement_nominal = (
            batt_capex_eur * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= LIFETIME_YEARS)
            else 0.0
        )
        inverter_replacement_nominal = (
            pv_capex_eur * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= LIFETIME_YEARS)
            else 0.0
        )
        df["Gross savings over 20 years (€)"] = gross_20y
        df["Net benefit over 20 years (€)"] = gross_20y - df["CAPEX (€)"] - batt_replacement_nominal - inverter_replacement_nominal

        keep_cols = [
            "Tariff",
            "Scenario",
            "PV (kWp)",
            "Battery (kWh)",
            "Total annual PV generation (kWh)",
            "Annual cost (€)",
            "Annual savings (€)",
            "CAPEX (€)",
            "Payback (yrs)",
            "NPV (€)",
            "IRR (%)",
            "Self-sufficiency (%)",
            "Self-consumption ratio (%)",
            "CO2 savings (kg)",
            "Gross savings over 20 years (€)",
            "Net benefit over 20 years (€)",
        ]
        rows.append(df[keep_cols])

        # Synthetic Grid-only baseline per tariff
        tariff_series = prepared_df[tcol].to_numpy(dtype=float)
        baseline_energy_cost = float((cons * tariff_series).sum())
        standing_charge = float(standing_charges_by_name.get(tname, 0.0))
        baseline_cost = baseline_energy_cost + standing_charge + float(pso_levy_annual)

        baseline_row = {
            "Tariff": tname,
            "Scenario": "Grid only",
            "PV (kWp)": 0,
            "Battery (kWh)": 0,
            "Total annual PV generation (kWh)": 0.0,
            "Annual cost (€)": baseline_cost,
            "Annual savings (€)": 0.0,
            "CAPEX (€)": 0.0,
            "Payback (yrs)": float("inf"),
            "NPV (€)": 0.0,
            "IRR (%)": float("nan"),
            "Self-sufficiency (%)": 0.0,
            "Self-consumption ratio (%)": 0.0,
            "CO2 savings (kg)": 0.0,
            "Gross savings over 20 years (€)": 0.0,
            "Net benefit over 20 years (€)": 0.0,
        }
        rows.append(pd.DataFrame([baseline_row]))

    if len(rows) == 0:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _prep_df_for_aggrid(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN so Ag Grid number filters behave sensibly."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def _format_numeric_columns_for_aggrid(df: pd.DataFrame) -> pd.DataFrame:
    """One decimal for numeric KPIs; whole numbers for PV / battery size columns."""
    out = df.copy()
    int_cols = {"PV (kWp)", "Battery (kWh)"}
    for c in out.columns:
        if c in int_cols:
            num = pd.to_numeric(out[c], errors="coerce")
            out[c] = num.round(0).astype("Int64")
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(1)
    return out


# KPI line charts vs PV size on the PV + Grid sweep tab (order: financial / energy / ratios).
PV_GRID_SWEEP_CHART_COLUMNS: list[str] = [
    "Annual electricity cost (€)",
    "Annual savings vs grid only (€)",
    "Annual cost of grid import (€)",
    "Annual export earnings (€)",
    "CAPEX (€)",
    "Payback period (years)",
    "NPV (20y, €)",
    "IRR (20y, %)",
    "Grid import (kWh)",
    "Annual PV generation (kWh)",
    "Self-Consumption (kWh)",
    "Export (kWh)",
    "Self-sufficiency ratio (%)",
    "Self-consumption ratio (%)",
]


def goal_to_pv_sweep_chart_column(goal: str) -> str:
    """Map 'Rank results by' goal to a column in build_pv_grid_sweep_table output."""
    m = {
        "Lowest annual electricity cost": "Annual electricity cost (€)",
        "Highest annual savings": "Annual savings vs grid only (€)",
        "Best payback": "Payback period (years)",
        "Best self-sufficiency / lowest grid import": "Self-sufficiency ratio (%)",
        "Most CO2 savings": "Annual savings vs grid only (€)",
        "Best NPV": "NPV (20y, €)",
        "Best IRR": "IRR (20y, %)",
        "Largest PV meeting self-consumption ratio >= X%": "Self-consumption ratio (%)",
        "PV size closest to annual community demand": "PV (kWp)",
    }
    return m.get(goal, "Annual electricity cost (€)")


def goal_to_tariff_compare_chart_column(goal: str) -> str:
    """Map 'Rank results by' to a column in evaluate_for_tariff() output (all-tariff comparison)."""
    m = {
        "Lowest annual electricity cost": "Annual cost (€)",
        "Highest annual savings": "Annual savings (€)",
        "Best payback": "Payback period (years)",
        "Best self-sufficiency / lowest grid import": "Self-sufficiency ratio (%)",
        "Most CO2 savings": "CO2 savings (kg)",
        "Best NPV": "NPV (20y, €)",
        "Best IRR": "IRR (20y, %)",
        "Largest PV meeting self-consumption ratio >= X%": "Self-consumption ratio (%)",
        "PV size closest to annual community demand": "PV (kWp)",
    }
    return m.get(goal, "Annual cost (€)")


def render_aggrid_results_table(
    df: pd.DataFrame,
    *,
    grid_key: str,
    height: int = 380,
    caption: str | None = None,
    default_rank_goal: str | None = None,
    rank_goal_table: str | None = None,
    annual_community_demand_kwh: float | None = None,
    goal5_threshold_pct: float | None = None,
    grid_key_extra: str | None = None,
) -> pd.DataFrame:
    """Excel-style per-column filters (client-side). Returns filtered + sorted rows from the grid.

    When ``default_rank_goal`` and ``rank_goal_table`` are set, rows are pre-ordered like **Rank results by**
    (``full`` = consolidated All scenario table, ``sweep`` = PV + Grid sweep). Widget ``grid_key`` is suffixed
    with a goal hash so changing the goal remounts the grid with the new default order.
    """
    if df is None or len(df) == 0:
        st.info("No rows to display.")
        return pd.DataFrame()

    work_df = df
    effective_key = grid_key
    g5_thr = float(goal5_threshold_pct) if goal5_threshold_pct is not None else 90.0
    if default_rank_goal and rank_goal_table == "full":
        demand = float(annual_community_demand_kwh) if annual_community_demand_kwh is not None else 0.0
        work_df = _sort_consolidated_scenarios_for_goal(
            df,
            default_rank_goal,
            annual_community_demand_kwh=demand,
            goal5_threshold_pct=g5_thr,
        )
        effective_key = f"{grid_key}__g{_aggrid_goal_key_fragment(default_rank_goal)}"
        if default_rank_goal == "Largest PV meeting self-consumption ratio >= X%":
            effective_key = f"{effective_key}__t{int(round(g5_thr))}"
    elif default_rank_goal and rank_goal_table == "sweep":
        demand = float(annual_community_demand_kwh) if annual_community_demand_kwh is not None else 0.0
        work_df = _sort_pv_sweep_for_goal(
            df, default_rank_goal, annual_community_demand_kwh=demand, goal5_threshold_pct=g5_thr
        )
        effective_key = f"{grid_key}__g{_aggrid_goal_key_fragment(default_rank_goal)}"
        if default_rank_goal == "Largest PV meeting self-consumption ratio >= X%":
            effective_key = f"{effective_key}__t{int(round(g5_thr))}"
    if grid_key_extra:
        effective_key = f"{effective_key}__x{hashlib.md5(grid_key_extra.encode('utf-8')).hexdigest()[:10]}"

    display_df = _format_numeric_columns_for_aggrid(_prep_df_for_aggrid(work_df))
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(
        filter=True,
        floatingFilter=True,
        sortable=True,
        resizable=True,
        editable=False,
    )
    gb.configure_selection(selection_mode="disabled", suppressRowClickSelection=True)
    gb.configure_grid_options(rowHeight=28, headerHeight=34, suppressCellFocus=True)
    go = gb.build()

    response = AgGrid(
        display_df,
        gridOptions=go,
        update_mode=GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.SORTING_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        height=height,
        width="100%",
        key=effective_key,
        reload_data=False,
    )
    if caption:
        st.caption(caption)

    raw = response.get("data")
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        return raw if len(raw) else pd.DataFrame()
    try:
        out = pd.DataFrame(raw)
        return out if len(out) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_kpi_guide_table() -> pd.DataFrame:
    """Single source of truth for KPI definitions (landing page + Settings tab)."""
    return pd.DataFrame(
        [
            {"KPI": "Annual cost of grid import (€)", "Meaning / formula": "Grid import energy multiplied by the applicable tariff over the year."},
            {"KPI": "Annual export earnings (€)", "Meaning / formula": "Exported energy multiplied by export rate over the year."},
            {"KPI": "Annual electricity cost (€)", "Meaning / formula": "Grid import cost minus export earnings, plus standing charge, PSO levy, and OPEX."},
            {"KPI": "Annual savings vs grid only (€)", "Meaning / formula": "Grid-only annual cost minus scenario annual cost (year 1 basis)."},
            {"KPI": "Self-Consumption (kWh)", "Meaning / formula": "PV energy used locally: direct PV to load plus PV-origin battery discharge to load."},
            {"KPI": "Self-consumption ratio (%)", "Meaning / formula": "Self-Consumption (kWh) divided by total PV generation, times 100."},
            {"KPI": "Self-sufficiency ratio (%)", "Meaning / formula": "Local renewable energy supplied to load divided by total consumption, times 100."},
            {"KPI": "Payback period (years)", "Meaning / formula": "CAPEX divided by year-1 annual savings (simple payback)."},
            {"KPI": "NPV (20y, €)", "Meaning / formula": "Present value of 20-year savings stream minus CAPEX and discounted replacement costs."},
            {"KPI": "IRR (20y, %)", "Meaning / formula": "Discount rate at which NPV equals zero for the same cashflow assumptions."},
            {"KPI": "Gross savings over 20 years (€)", "Meaning / formula": "Sum of yearly savings over 20 years (inflated if electricity inflation > 0)."},
            {"KPI": "Net benefit over 20 years (€)", "Meaning / formula": "Gross 20-year savings minus CAPEX and nominal replacement costs."},
            {"KPI": "Annual CO2 savings (kg)", "Meaning / formula": "Grid-only CO2 minus scenario CO2 (not below zero)."},
            {"KPI": "CO2 reduction (%)", "Meaning / formula": "Annual CO2 savings divided by grid-only CO2, times 100."},
        ]
    )


def render_landing_quick_start_kpi_guide() -> None:
    """Shown on the main page until the first successful Run analysis completes."""
    st.markdown("## Quick start & KPI guide")
    st.caption(
        "After your first successful run, this block is hidden here; the same reference material stays in the **Settings & KPI guide** tab."
    )

    st.markdown("### Files you need (sidebar uploads)")
    st.markdown(
        """
- **Consumption CSV:** hourly kWh with a **`date`** column (e.g. `DD/MM/YYYY HH:00`) and consumption (often **`Final_Community_Sum`**). One row per hour.
- **PV CSV (PVGIS-style):** columns **`time`** (e.g. `YYYYMMDD:HH11`) and **`P`** = production in **Wh for 1 kWp**. The app converts Wh to kWh per kWp and scales by PV size.

Both files must cover the same period so hourly rows align.
"""
    )

    st.markdown("### Main KPIs")
    st.markdown(
        """
- **Annual electricity cost:** what you pay for imports (minus export income), plus standing charge and OPEX.
- **Annual savings vs grid only:** how much cheaper the scenario is than grid-only in year 1.
- **Payback / NPV / IRR:** how fast the investment pays back and how attractive it is over 20 years (with your discount rate, inflation, and optional replacements).
- **Self-sufficiency:** share of consumption met by **local renewable** supply (PV to load and PV-origin battery discharge).
- **Self-consumption ratio:** share of **PV generation** used on site (directly or via battery from PV).
- **CO2 savings / reduction:** grid-only emissions minus scenario emissions (savings never shown below zero).
"""
    )

    st.markdown("### Key assumptions (sidebar, before Run)")
    st.markdown(
        f"""
#### Costs and finance
- **PV CAPEX (€/kWp):** Upfront PV cost per kWp; sets part of each scenario’s CAPEX. Default: **€{PV_COST_PER_KWP:,.0f}**/kWp.
- **Battery CAPEX (€/kWh):** Upfront battery cost per kWh; sets part of each scenario’s CAPEX. Default: **€{BATT_COST_PER_KWH:,.0f}**/kWh.
- **OPEX %:** Annual operating cost as a percentage of that scenario’s total CAPEX. Default: **{DEFAULT_OPEX_PCT:g}%**.
- **Standing charges:** Fixed €/year per tariff family, added to both grid-only and scenario bills. Defaults: Standard **€{DEFAULT_STANDING_CHARGE_STANDARD_EUR:,.2f}**/y, Weekend Saver **€{DEFAULT_STANDING_CHARGE_WEEKEND_EUR:,.2f}**/y, Flat **€{DEFAULT_STANDING_CHARGE_FLAT_EUR:,.2f}**/y.
- **PSO levy (annual, €):** Fixed annual Public Service Obligation charge, same amount for every tariff and scenario (added to grid-only and scenario bills). Default: **€{DEFAULT_PSO_LEVY_EUR_PER_YEAR:,.2f}**/y. Escalates with electricity inflation in long-run metrics.
- **Discount rate:** Used to build NPV and IRR from future savings. Default: **{DISCOUNT_RATE * 100:g}%** per year.
- **Electricity inflation:** Grows import costs, export income, standing charges, PSO levy, and OPEX year by year in long-run metrics (CAPEX is not inflated). Default: **{ELECTRICITY_INFLATION_RATE * 100:g}%** per year.
- **Battery replacement:** One replacement in a chosen year, cost as % of battery CAPEX. Defaults: year **{DEFAULT_BATTERY_REPLACEMENT_YEAR}**, **{DEFAULT_BATTERY_REPLACEMENT_COST_PCT:g}%** of battery CAPEX. Set year **0** to turn off (no battery replacement cash flow).
- **Inverter replacement:** One replacement in a chosen year, cost as % of PV CAPEX. Defaults: year **{DEFAULT_INVERTER_REPLACEMENT_YEAR}**, **{DEFAULT_INVERTER_REPLACEMENT_COST_PCT:g}%** of PV CAPEX. Set year **0** to turn off.

#### Optimizer
- **PV size bounds (kWp):** Slider min/max; only PV sizes in this range are tried.
- **Battery size bounds (kWh):** Slider min/max; only battery sizes in this range are tried.
- **Speed preset (Quick / Fast / Full):** Sets step sizes on those grids—larger steps → fewer combinations and a faster run, but a coarser search.

#### Battery model
- **Round-trip efficiency:** How much energy is lost over a full charge–discharge cycle.
- **Depth of discharge (DoD):** How much of the battery’s capacity you allow to be used.
- **Initial SOC:** Battery state of charge at the start of the simulated year.
- **C-rate:** Caps how fast the battery can charge or discharge relative to its capacity.
- **Charge from surplus PV:** Whether the battery may store excess PV instead of exporting it.
- **Charge from grid at night:** Whether the battery may import cheap power overnight.
- **Discharge schedule:** When stored energy is sent to the load (e.g. peak only vs day + peak).

#### Tariff overrides (optional)
- **Overrides off:** The app uses built-in default €/kWh for Standard (day / peak / night), Weekend Saver (weekday vs weekend bands), Flat import, and a default export price.
- **Overrides on:** You enter your own €/kWh for those import bands and for export; the run uses your values instead of the defaults.
- **Typical use:** Your supplier’s prices differ from the defaults, or you want to test price sensitivity.

Changing any of these means you should **Run analysis** again so results match your inputs.
"""
    )

    st.markdown("### Full KPI reference table")
    st.dataframe(build_kpi_guide_table(), use_container_width=True, hide_index=True)


def render_settings_kpi_guide_tab() -> None:
    """Full guide content for the Settings & KPI guide tab (after a run)."""
    st.subheader("Settings and KPI guide")
    st.caption("Plain-language explanation of model inputs, outputs, and formulas used in the app.")

    render_last_run_tariffs_and_assumptions_section()

    st.markdown("### How to read results in 5 steps")
    st.markdown(
        """
1. **Run analysis** with your chosen assumptions in the sidebar.
2. **Choose Scenario type** and **Rank results by** in Results Controls, then **Show ranked result** (Best / 2nd / 3rd) for the KPI block.
3. **Check core economics** first: annual electricity cost, annual savings, payback, NPV, IRR.
4. **Check energy and carbon metrics**: self-sufficiency, self-consumption, CO2 savings/reduction.
5. **Filter the full scenario table** (Ag Grid, Excel-style column filters) and **export CSV** when needed.
"""
    )

    st.markdown("### Run settings (sidebar)")
    st.markdown(
        f"""
- **PV CAPEX / Battery CAPEX:** upfront investment cost per unit size. Defaults: **€{PV_COST_PER_KWP:,.0f}**/kWp and **€{BATT_COST_PER_KWH:,.0f}**/kWh. CAPEX affects payback, NPV, IRR, and net benefit.
- **Standing charge:** fixed annual tariff charge per tariff family. Defaults: Standard **€{DEFAULT_STANDING_CHARGE_STANDARD_EUR:,.2f}**/y, Weekend Saver **€{DEFAULT_STANDING_CHARGE_WEEKEND_EUR:,.2f}**/y, Flat **€{DEFAULT_STANDING_CHARGE_FLAT_EUR:,.2f}**/y.
- **PSO levy (annual, €):** fixed annual charge added to every tariff/scenario bill. Default: **€{DEFAULT_PSO_LEVY_EUR_PER_YEAR:,.2f}**/y; escalates with electricity inflation like standing charge.
- **OPEX (% of CAPEX):** annual operating cost estimated from CAPEX. Default: **{DEFAULT_OPEX_PCT:g}%**.
- **Discount rate:** used to discount future yearly savings in NPV. Default: **{DISCOUNT_RATE * 100:g}%** per year.
- **Electricity inflation:** yearly escalation applied to recurring costs and savings streams, including PSO levy (not to CAPEX). Default: **{ELECTRICITY_INFLATION_RATE * 100:g}%** per year.
- **Battery replacement:** one cash outflow in a chosen year, % of battery CAPEX. Defaults: year **{DEFAULT_BATTERY_REPLACEMENT_YEAR}**, **{DEFAULT_BATTERY_REPLACEMENT_COST_PCT:g}%**; set year **0** to omit. Applies when the scenario includes a battery.
- **Inverter replacement:** one cash outflow in a chosen year, % of PV CAPEX. Defaults: year **{DEFAULT_INVERTER_REPLACEMENT_YEAR}**, **{DEFAULT_INVERTER_REPLACEMENT_COST_PCT:g}%**; set year **0** to omit. Applies when the scenario includes PV.
- **Battery model settings:** efficiency, DoD, initial SOC, C-rate, whether PV/grid can charge, and dispatch rules for discharging to the load.
- **Optimizer bounds and speed:** define min/max PV and battery sizes and the step sizes (via speed preset) for the search grid.
- **Tariff overrides:** when off, built-in default day/peak/night and flat import rates plus default export apply. When on, your entered €/kWh values for each band and export replace those defaults for the run.
"""
    )

    st.markdown("### Results controls (post-run)")
    st.markdown(
        """
- **Rank results by:** chooses which KPI orders scenarios after **Scenario type** filtering.
- **Scenario type:** limits which scenario family rows are eligible (All, Grid only, PV + Grid, PV + Battery + Grid, Battery + Grid); does not rerun optimization.
- **Show ranked result:** picks 1st / 2nd / 3rd from that filtered list, using the ranking goal above.
- **All scenario results:** interactive table (Ag Grid) with per-column filters like Excel; CSV export of full or filtered view.
- **Season filter (Consumption tab):** limits charted periods only; does not change optimization outputs. Charts follow **Show ranked result** from the Results tab.
"""
    )

    st.markdown("### KPI formulas")
    st.dataframe(build_kpi_guide_table(), use_container_width=True, hide_index=True)


_KPI_EUR_WHOLE_KEYS = frozenset(
    {
        "Annual cost (€)",
        "Export income (€)",
        "Annual savings (€)",
        "NPV (20y, €)",
        "Gross savings over 20 years (€)",
        "Net benefit over 20 years (€)",
    }
)
_KPI_KWH_WHOLE_KEYS = frozenset(
    {
        "Grid import (kWh)",
        "Total annual PV generation (kWh)",
        "Self-consumed PV (kWh)",
        "Export to grid (kWh)",
    }
)
_KPI_PCT_ONE_DECIMAL_KEYS = frozenset(
    {
        "CO2 reduction (%)",
        "Self-sufficiency ratio (%)",
        "Self-consumption ratio (%)",
    }
)


def _format_kpi_tile_value(col_key: str, val) -> str:
    """Whole euros for money KPIs; one decimal for payback / IRR / %; whole units for kWh and kg CO₂."""
    if val is None:
        return "—"
    if isinstance(val, (int, float, np.floating)):
        if not np.isfinite(val):
            return "inf" if val > 0 else "—"
        x = float(val)
        if col_key in _KPI_EUR_WHOLE_KEYS:
            return f"€{x:,.0f}"
        if col_key == "Payback period (years)":
            return f"{x:,.1f}"
        if col_key == "IRR (20y, %)":
            return f"{x:,.1f}%"
        if col_key in _KPI_PCT_ONE_DECIMAL_KEYS:
            return f"{x:,.1f}%"
        if col_key in _KPI_KWH_WHOLE_KEYS or col_key == "CO2 savings (kg)":
            return f"{x:,.0f}"
        return f"{x:,.1f}"
    return str(val)


def render_compact_kpi_tile_grid(row: pd.Series, kpi_labels: List[Tuple[str, str]]) -> None:
    """Dense 4-column card-style KPI tiles for the Results dashboard (no extra vertical chrome)."""
    tile_css = (
        "background:#f8fafc;border:1px solid #e8edf3;border-radius:7px;padding:7px 10px 8px;"
        "box-shadow:0 1px 2px rgba(15,23,42,0.04);"
    )
    lbl_css = "font-size:11px;font-weight:500;color:#64748b;line-height:1.2;margin:0 0 3px 0;"
    val_css = (
        "font-size:22px;font-weight:700;color:#0f172a;line-height:1.18;"
        "font-variant-numeric:tabular-nums;letter-spacing:-0.02em;"
    )
    parts: List[str] = [
        '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px 8px;margin:2px 0 10px 0;">'
    ]
    for label, col_key in kpi_labels:
        val = row.get(col_key, "—")
        val_disp = "—" if val == "—" else _format_kpi_tile_value(col_key, val)
        esc_l = html.escape(label)
        esc_v = html.escape(val_disp)
        parts.append(
            f'<div style="{tile_css}"><div style="{lbl_css}">{esc_l}</div>'
            f'<div style="{val_css}">{esc_v}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _optimizer_progress_overlay_html(tariff_name: str, completed: int, total: int, pct: int) -> str:
    """Small fixed-position panel (Windows copy–style) shown during optimize()."""
    safe = html.escape(str(tariff_name))
    return (
        f'<div style="position:fixed;bottom:24px;right:24px;z-index:999999;'
        f"width:min(400px,calc(100vw - 32px));background:#fffef5;border:1px solid #d9d0b8;"
        f"border-radius:10px;box-shadow:0 8px 28px rgba(0,0,0,0.22);padding:14px 16px;"
        f'font-family:Segoe UI,system-ui,sans-serif;">'
        f'<div style="font-size:11px;letter-spacing:0.06em;color:#666;text-transform:uppercase;">Optimizer</div>'
        f'<div style="font-size:16px;font-weight:600;margin-top:4px;color:#111;">{safe}</div>'
        f'<div style="font-size:13px;margin-top:10px;color:#333;">{completed:,} / {total:,} evaluations · <b>{pct}%</b></div>'
        f"</div>"
    )


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="REC Feasibility Analyzer", layout="wide")
st.title("REC Feasibility Analyzer (PV + Battery + Tariffs)")

with st.sidebar:
    st.header("Inputs")
    cons_file = st.file_uploader(
        "Upload consumption CSV (hourly kWh)",
        type=["csv"],
        key="cons",
        help="Expected: column 'date' in DD/MM/YYYY HH:00 format and hourly consumption (typically 'Final_Community_Sum').",
    )
    pv_file = st.file_uploader(
        "Upload PV CSV (PVGIS: 'time' + 'P' in Wh for 1 kWp)",
        type=["csv"],
        key="pv",
        help="Expected PVGIS timeseries with 'time' like YYYYMMDD:HH11 and 'P' in Wh for 1 kWp. The app converts it to kWh/kWp.",
    )

    st.divider()
    st.header("Run setup")
    st.caption("Changes in this sidebar require clicking **Run analysis**.")

    # CAPEX inputs (used for payback/NPV and for cost-based KPI reporting)
    st.subheader("Costs")
    pv_capex = st.number_input(
        "PV CAPEX (€/kWp)",
        min_value=0.0,
        value=float(PV_COST_PER_KWP),
        step=50.0,
    )
    batt_capex = st.number_input(
        "Battery CAPEX (€/kWh)",
        min_value=0.0,
        value=float(BATT_COST_PER_KWH),
        step=50.0,
    )
    with st.expander("Standing charge (annual, €)", expanded=False):
        st.caption("Fixed annual charge per tariff family. Defaults are starting values—set to 0 if not applicable.")
        standing_charge_standard = st.number_input(
            "Standard (smart meter)",
            min_value=0.0,
            value=float(DEFAULT_STANDING_CHARGE_STANDARD_EUR),
            step=0.05,
            key="sc_standard",
        )
        standing_charge_weekend = st.number_input(
            "Weekend Saver",
            min_value=0.0,
            value=float(DEFAULT_STANDING_CHARGE_WEEKEND_EUR),
            step=0.05,
            key="sc_weekend",
        )
        standing_charge_flat = st.number_input(
            "Flat rate",
            min_value=0.0,
            value=float(DEFAULT_STANDING_CHARGE_FLAT_EUR),
            step=0.05,
            key="sc_flat",
        )
    standing_charges = {
        "Standard": float(standing_charge_standard),
        "Weekend Saver": float(standing_charge_weekend),
        "Flat": float(standing_charge_flat),
    }
    pso_levy = st.number_input(
        "PSO levy (annual, €)",
        min_value=0.0,
        value=float(DEFAULT_PSO_LEVY_EUR_PER_YEAR),
        step=0.05,
        help="Fixed annual PSO (Public Service Obligation) levy added to grid-only and all scenario bills. "
        "Escalates with electricity inflation in NPV, IRR, and gross 20-year savings (same pattern as standing charge).",
    )
    opex_pct = st.number_input(
        "OPEX (annual, % of CAPEX)",
        min_value=0.0,
        value=float(DEFAULT_OPEX_PCT),
        step=0.5,
        help="Annual operating cost as % of scenario total CAPEX. Counts as cost.",
    )
    discount_rate_pct = st.number_input(
        "Discount rate for NPV (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(DISCOUNT_RATE * 100),
        step=0.5,
        help="Annual discount rate used in NPV formula: NPV = −CAPEX + Σ(annual_savings / (1+r)^t). Default 5%.",
    )
    discount_rate = discount_rate_pct / 100.0
    electricity_inflation_pct = st.number_input(
        "Electricity inflation rate (% per year)",
        min_value=0.0,
        max_value=15.0,
        value=float(ELECTRICITY_INFLATION_RATE * 100),
        step=0.1,
        help="Escalation of electricity costs, export income, standing charge, PSO levy, and OPEX each year. CAPEX is not inflated. 0% = flat costs.",
    )
    electricity_inflation_rate = electricity_inflation_pct / 100.0
    with st.expander("Replacement assumptions", expanded=False):
        battery_replacement_year_input = st.number_input(
            "Battery replacement year (0 = none)",
            min_value=0,
            max_value=LIFETIME_YEARS,
            value=int(DEFAULT_BATTERY_REPLACEMENT_YEAR),
            step=1,
        )
        battery_replacement_cost_pct = st.number_input(
            "Battery replacement cost (% of battery CAPEX)",
            min_value=0.0,
            max_value=300.0,
            value=float(DEFAULT_BATTERY_REPLACEMENT_COST_PCT),
            step=5.0,
        )
        inverter_replacement_year_input = st.number_input(
            "Inverter replacement year (0 = none)",
            min_value=0,
            max_value=LIFETIME_YEARS,
            value=int(DEFAULT_INVERTER_REPLACEMENT_YEAR),
            step=1,
        )
        inverter_replacement_cost_pct = st.number_input(
            "Inverter replacement cost (% of PV CAPEX)",
            min_value=0.0,
            max_value=300.0,
            value=float(DEFAULT_INVERTER_REPLACEMENT_COST_PCT),
            step=5.0,
        )
    battery_replacement_year = int(battery_replacement_year_input) if int(battery_replacement_year_input) > 0 else None
    inverter_replacement_year = int(inverter_replacement_year_input) if int(inverter_replacement_year_input) > 0 else None

    # Optimizer speed control (biggest performance lever)
    st.subheader("Optimizer speed")
    speed_preset = st.selectbox(
        "Speed preset",
        [
            "Quick (PV step 5, battery step 10)",
            "Fast (PV step 10, battery step 20)",
            "Full (PV step 1, battery step 5)",
        ],
        index=0,
    )
    if speed_preset == "Quick (PV step 5, battery step 10)":
        opt_pv_step, opt_batt_step = 5, 10
    elif speed_preset == "Fast (PV step 10, battery step 20)":
        opt_pv_step, opt_batt_step = 10, 20
    else:
        opt_pv_step, opt_batt_step = 1, 5

    # Optimizer search bounds (true run inputs)
    pv_range = st.slider(
        "PV size range (kWp) — optimizer bounds",
        min_value=0,
        max_value=150,
        value=(0, 150),
        step=5,
        key="pv_range",
    )
    pv_min, pv_max = pv_range[0], pv_range[1]
    batt_range = st.slider(
        "Battery size range (kWh) — optimizer bounds",
        min_value=0,
        max_value=300,
        value=(0, 300),
        step=10,
        key="batt_range",
    )
    batt_min, batt_max = batt_range[0], batt_range[1]

    with st.expander("Battery model settings (optional)", expanded=False):
        rt_eff_pct = st.slider(
            "Round-trip efficiency (%)",
            min_value=50,
            max_value=100,
            value=90,
            step=1,
        )
        dod_pct = st.slider(
            "Usable depth of discharge (DoD) (%)",
            min_value=50,
            max_value=100,
            value=90,
            step=1,
        )
        init_soc_pct = st.slider(
            "Initial state of charge (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
        c_rate = st.slider(
            "Max charge/discharge power as C-rate",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        charge_from_pv = st.checkbox("Allow battery charging from PV surplus", value=True)
        charge_from_grid_at_night = st.checkbox("Allow battery charging from grid during night", value=True)
        discharge_schedule = st.selectbox(
            "Battery discharging schedule",
            ["Peak only", "Day+Peak"],
            index=0,
        )

    with st.expander("Tariff rates (optional overrides)", expanded=False):
        override_tariffs = st.checkbox("Override default tariff rates", value=False)
        # Standard
        std_day = st.number_input("Standard: Day (€/kWh)", value=float(DEFAULT_TARIFFS["standard"]["day"]), step=0.001, disabled=not override_tariffs)
        std_peak = st.number_input("Standard: Peak (€/kWh)", value=float(DEFAULT_TARIFFS["standard"]["peak"]), step=0.001, disabled=not override_tariffs)
        std_night = st.number_input("Standard: Night (€/kWh)", value=float(DEFAULT_TARIFFS["standard"]["night"]), step=0.001, disabled=not override_tariffs)

        # Weekend saver
        wk_day = st.number_input("Weekend Saver (weekday): Day (€/kWh)", value=float(DEFAULT_TARIFFS["weekend"]["weekday"]["day"]), step=0.001, disabled=not override_tariffs)
        wk_peak = st.number_input("Weekend Saver (weekday): Peak (€/kWh)", value=float(DEFAULT_TARIFFS["weekend"]["weekday"]["peak"]), step=0.001, disabled=not override_tariffs)
        wk_night = st.number_input("Weekend Saver (weekday): Night (€/kWh)", value=float(DEFAULT_TARIFFS["weekend"]["weekday"]["night"]), step=0.001, disabled=not override_tariffs)

        we_day = st.number_input("Weekend Saver (weekend): Day (€/kWh)", value=float(DEFAULT_TARIFFS["weekend"]["weekend"]["day"]), step=0.001, disabled=not override_tariffs)
        we_peak = st.number_input("Weekend Saver (weekend): Peak (€/kWh)", value=float(DEFAULT_TARIFFS["weekend"]["weekend"]["peak"]), step=0.001, disabled=not override_tariffs)
        we_night = st.number_input("Weekend Saver (weekend): Night (€/kWh)", value=float(DEFAULT_TARIFFS["weekend"]["weekend"]["night"]), step=0.001, disabled=not override_tariffs)

        # Flat + export
        flat_rate = st.number_input("Flat tariff: Rate (€/kWh)", value=float(DEFAULT_TARIFFS["flat"]), step=0.001, disabled=not override_tariffs)
        export_rate_input = st.number_input("Export rate (€/kWh)", value=float(DEFAULT_EXPORT_RATE), step=0.0001, disabled=not override_tariffs)

    if "stop_run_requested" not in st.session_state:
        st.session_state.stop_run_requested = False
    run_button = st.button("Run analysis", type="primary")
    if st.button("Stop current run", help="Requests optimization stop at the next progress checkpoint."):
        st.session_state.stop_run_requested = True

_has_completed_run = (
    st.session_state.get("prepared_df") is not None and st.session_state.get("opt_dfs") is not None
)
if not _has_completed_run and not run_button:
    st.caption("Read **Quick start & KPI guide** below, then use the **sidebar** to upload files and click **Run analysis**.")
    render_landing_quick_start_kpi_guide()

if "prepared_df" not in st.session_state:
    st.session_state.prepared_df = None
if "opt_dfs" not in st.session_state:
    st.session_state.opt_dfs = None
if "prepared_meta" not in st.session_state:
    st.session_state.prepared_meta = None
if "active_tariff_config" not in st.session_state:
    st.session_state.active_tariff_config = DEFAULT_TARIFFS
if "active_export_rate" not in st.session_state:
    st.session_state.active_export_rate = DEFAULT_EXPORT_RATE
if "active_discount_rate" not in st.session_state:
    st.session_state.active_discount_rate = DISCOUNT_RATE
if "view_goal" not in st.session_state:
    st.session_state.view_goal = "Lowest annual electricity cost"
if "view_scenario_type" not in st.session_state:
    st.session_state.view_scenario_type = "All scenarios"
if "view_goal5_threshold_pct" not in st.session_state:
    st.session_state.view_goal5_threshold_pct = 90
if "view_filter_tariff" not in st.session_state:
    st.session_state.view_filter_tariff = "Standard"
if "last_opex_pct" not in st.session_state:
    st.session_state.last_opex_pct = 0.0
if "last_discount_rate" not in st.session_state:
    st.session_state.last_discount_rate = DISCOUNT_RATE
if "last_pv_capex" not in st.session_state:
    st.session_state.last_pv_capex = float(PV_COST_PER_KWP)
if "last_batt_capex" not in st.session_state:
    st.session_state.last_batt_capex = float(BATT_COST_PER_KWH)
if "last_electricity_inflation_rate" not in st.session_state:
    st.session_state.last_electricity_inflation_rate = ELECTRICITY_INFLATION_RATE
if "last_battery_replacement_year" not in st.session_state:
    st.session_state.last_battery_replacement_year = None
if "last_battery_replacement_cost_pct" not in st.session_state:
    st.session_state.last_battery_replacement_cost_pct = 0.0
if "last_inverter_replacement_year" not in st.session_state:
    st.session_state.last_inverter_replacement_year = None
if "last_inverter_replacement_cost_pct" not in st.session_state:
    st.session_state.last_inverter_replacement_cost_pct = 0.0
if "last_battery_settings" not in st.session_state:
    st.session_state.last_battery_settings = BatterySettings()
if "last_tariff_config" not in st.session_state:
    st.session_state.last_tariff_config = DEFAULT_TARIFFS
if "last_export_rate" not in st.session_state:
    st.session_state.last_export_rate = DEFAULT_EXPORT_RATE
if "last_override_tariffs" not in st.session_state:
    st.session_state.last_override_tariffs = False
if "last_input_hashes" not in st.session_state:
    st.session_state.last_input_hashes = {"cons_sha": None, "pv_sha": None}
if "last_opt_cfg" not in st.session_state:
    st.session_state.last_opt_cfg = {
        "pv_min": 0,
        "pv_max": 150,
        "batt_min": 0,
        "batt_max": 300,
        "pv_step": 5,
        "batt_step": 10,
        "speed_preset": "Quick (PV step 5, battery step 10)",
    }
if "last_standing_charges" not in st.session_state:
    st.session_state.last_standing_charges = {"Standard": 0.0, "Weekend Saver": 0.0, "Flat": 0.0}
if "last_pso_levy" not in st.session_state:
    st.session_state.last_pso_levy = float(DEFAULT_PSO_LEVY_EUR_PER_YEAR)

tariff_map = {
    "Standard": "tariff_standard",
    "Weekend Saver": "tariff_weekend",
    "Flat": "tariff_flat",
}

# Season definitions (Northern hemisphere)
SEASON_MONTHS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Autumn": (9, 10, 11),
    "All Year": tuple(range(1, 13)),
}


def _month_to_nh_season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _consumption_time_band(hour: int) -> str:
    if 17 <= hour < 19:
        return "Peak"
    if hour >= 23 or hour < 8:
        return "Night"
    return "Day"


def _community_consumption_features(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly rows with demand-side columns only (no scenario / tariff)."""
    out = prepared_df[["date", "consumption"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["consumption"] = pd.to_numeric(out["consumption"], errors="coerce").fillna(0.0)
    out["month"] = out["date"].dt.month
    out["hour"] = out["date"].dt.hour
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["dow"] >= 5
    out["day"] = out["date"].dt.normalize()
    out["season"] = out["month"].map(_month_to_nh_season)
    out["time_band"] = out["hour"].map(_consumption_time_band)
    return out


def render_community_consumption_patterns(prepared_df: pd.DataFrame) -> None:
    """Scenario-independent demand charts: how the community consumes electricity."""
    if prepared_df is None or len(prepared_df) == 0:
        st.info("Load consumption data to see community demand patterns.")
        return
    h = _community_consumption_features(prepared_df)
    _band_colors = {"Night": "#bbf7d0", "Day": "#bfdbfe", "Peak": "#fed7aa"}
    annual_kwh = float(h["consumption"].sum())
    n_hours = len(h)
    daily_totals = h.groupby("day")["consumption"].sum()
    n_days = max(len(daily_totals), 1)
    avg_daily_kwh = float(daily_totals.mean()) if len(daily_totals) else 0.0
    k0a, k0b, k0c = st.columns(3)
    with k0a:
        st.metric(
            "Annual electricity consumption (kWh)",
            f"{annual_kwh:,.0f}",
            help=f"Sum of hourly kWh over {n_hours:,} hours in the uploaded series.",
        )
    with k0b:
        st.metric(
            "Average daily demand (kWh/day)",
            f"{avg_daily_kwh:,.0f}",
            help=f"Mean total kWh per calendar day over {n_days:,} days in the series.",
        )
    with k0c:
        st.metric(
            "Average hourly demand (kWh/h)",
            f"{annual_kwh / max(n_hours, 1):,.2f}",
            help="Mean consumption per hour over the series (not a kW power reading).",
        )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = h.groupby("month", sort=True)["consumption"].sum()
    seasonal_order = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = h.groupby("season", sort=False)["consumption"].sum().reindex(seasonal_order).fillna(0)

    band_labels_share = ["Night", "Day", "Peak"]

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_m = go.Figure(
            data=[go.Bar(x=[month_names[m - 1] for m in monthly.index], y=monthly.values, marker_color="#3b82f6")]
        )
        fig_m.update_layout(
            title="Monthly consumption (kWh)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=220,
            showlegend=False,
            xaxis_title="",
            yaxis_title="kWh",
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})
    with r1c2:
        fig_s = go.Figure(data=[go.Bar(x=seasonal.index.tolist(), y=seasonal.values, marker_color="#6366f1")])
        fig_s.update_layout(
            title="Seasonal consumption (kWh)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=220,
            showlegend=False,
            xaxis_title="",
            yaxis_title="kWh",
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    r4c1, r4c2 = st.columns(2)
    daily = daily_totals.sort_index()
    with r4c1:
        fig_d = go.Figure(data=[go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(color="#2563eb", width=1))])
        fig_d.update_layout(
            title="Daily community consumption (kWh)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=240,
            xaxis_title="Date",
            yaxis_title="kWh",
            xaxis=dict(tickformat="%b %Y", nticks=10),
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})
    with r4c2:
        hr_all = h.groupby("hour")["consumption"].mean().reindex(range(24)).fillna(0.0)
        fig_av = go.Figure()
        for band in band_labels_share:
            xs = [hr for hr in range(24) if _consumption_time_band(hr) == band]
            ys = [float(hr_all.iloc[hr]) for hr in xs]
            fig_av.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=band,
                    marker_color=_band_colors[band],
                    marker_line_width=0,
                    hovertemplate="Hour %{x}<br>Mean kWh/h %{y:.2f}<extra></extra>",
                )
            )
        fig_av.update_layout(
            title="Average daily load (kWh/h)",
            margin=dict(l=8, r=120, t=36, b=40),
            height=240,
            xaxis_title="Hour",
            yaxis_title="kWh",
            barmode="overlay",
            bargap=0.15,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                x=1.02,
                xanchor="left",
                font=dict(size=10),
            ),
            xaxis=dict(
                range=[-0.5, 23.5],
                tickmode="linear",
                tick0=0,
                dtick=1,
                tickfont=dict(size=9),
            ),
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig_av, use_container_width=True, config={"displayModeBar": False})

    def _time_band_share_fig(title: str, sub: pd.DataFrame) -> go.Figure:
        if len(sub) == 0:
            vals = [0.0, 0.0, 0.0]
        else:
            sums = sub.groupby("time_band")["consumption"].sum()
            vals = [float(sums.get(b, 0.0)) for b in band_labels_share]
        if sum(vals) <= 0:
            vals = [1.0, 1.0, 1.0]
        colors = [_band_colors[b] for b in band_labels_share]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=band_labels_share,
                    values=vals,
                    hole=0.52,
                    sort=False,
                    direction="clockwise",
                    marker=dict(colors=colors, line=dict(color="#ffffff", width=1)),
                    textinfo="label+percent",
                    textposition="inside",
                    textfont=dict(size=10),
                    insidetextorientation="horizontal",
                    hovertemplate="<b>%{label}</b><br>%{value:,.0f} kWh<br>(%{percent})<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=title,
            margin=dict(l=2, r=2, t=36, b=2),
            height=220,
            showlegend=False,
        )
        return fig

    # Section label: same visual weight as Plotly figure titles (~14px semibold), not st.caption.
    st.markdown("##### Share of consumption by time band")
    share_cols = st.columns(5)
    share_specs = [
        ("Total", h),
        ("Winter", h[h["season"] == "Winter"]),
        ("Spring", h[h["season"] == "Spring"]),
        ("Summer", h[h["season"] == "Summer"]),
        ("Autumn", h[h["season"] == "Autumn"]),
    ]
    for col, (ttl, sub) in zip(share_cols, share_specs):
        with col:
            st.plotly_chart(
                _time_band_share_fig(ttl, sub),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.markdown("")
    hr_wd = h[~h["is_weekend"]].groupby("hour")["consumption"].mean().reindex(range(24)).fillna(0.0)
    hr_we = h[h["is_weekend"]].groupby("hour")["consumption"].mean().reindex(range(24)).fillna(0.0)
    fig_wd = go.Figure()
    fig_wd.add_trace(
        go.Scatter(
            x=list(range(24)),
            y=hr_wd.values,
            mode="lines",
            name="Weekday",
            line=dict(color="#0ea5e9", width=2),
        )
    )
    fig_wd.add_trace(
        go.Scatter(
            x=list(range(24)),
            y=hr_we.values,
            mode="lines",
            name="Weekend",
            line=dict(color="#f97316", width=2),
        )
    )
    fig_wd.update_layout(
        title=dict(text="Average hourly load — weekday vs weekend (kWh/h)", y=0.97, x=0, xanchor="left"),
        margin=dict(l=8, r=120, t=48, b=40),
        height=260,
        xaxis_title="Hour",
        yaxis_title="kWh",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            x=1.02,
            xanchor="left",
            font=dict(size=10),
        ),
        xaxis=dict(
            range=[-0.5, 23.5],
            tickmode="linear",
            tick0=0,
            dtick=1,
            tickfont=dict(size=9),
        ),
        yaxis=dict(rangemode="tozero"),
    )

    season_line_colors = {"Winter": "#38bdf8", "Spring": "#4ade80", "Summer": "#fbbf24", "Autumn": "#fb923c"}
    fig_season_h = go.Figure()
    for s in seasonal_order:
        sub = h[h["season"] == s]
        if len(sub) == 0:
            continue
        hr = sub.groupby("hour")["consumption"].mean().reindex(range(24)).fillna(0.0)
        fig_season_h.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=hr.values,
                mode="lines",
                name=s,
                line=dict(color=season_line_colors.get(s, "#64748b"), width=2),
            )
        )
    fig_season_h.update_layout(
        title=dict(text="Average hourly load by season (kWh/h)", y=0.97, x=0, xanchor="left"),
        margin=dict(l=8, r=120, t=48, b=40),
        height=260,
        xaxis_title="Hour",
        yaxis_title="kWh",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            x=1.02,
            xanchor="left",
            font=dict(size=9),
        ),
        xaxis=dict(
            range=[-0.5, 23.5],
            tickmode="linear",
            tick0=0,
            dtick=1,
            tickfont=dict(size=9),
        ),
        yaxis=dict(rangemode="tozero"),
    )

    line_wd_col, line_season_col = st.columns(2)
    with line_wd_col:
        st.plotly_chart(fig_wd, use_container_width=True, config={"displayModeBar": False})
    with line_season_col:
        st.plotly_chart(fig_season_h, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "**Average daily load** (right-hand chart in the row above the time-band donuts): bars by time band — "
        "**Peak** 17:00–19:00 (orange), **Night** 23:00–08:00 (green), **Day** otherwise (blue). "
        "Seasons: Northern hemisphere."
    )
    hm_left, hm_right = st.columns(2)
    pivot_hm = h.pivot_table(values="consumption", index="hour", columns="month", aggfunc="mean")
    pivot_hm = pivot_hm.reindex(index=range(24), columns=range(1, 13)).fillna(0)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot_dow = h.pivot_table(values="consumption", index="hour", columns="dow", aggfunc="mean")
    pivot_dow = pivot_dow.reindex(index=range(24), columns=range(7)).fillna(0)
    with hm_left:
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot_hm.values,
                x=[month_names[m - 1] for m in pivot_hm.columns],
                y=pivot_hm.index,
                colorscale="Blues",
                colorbar=dict(title="kWh", len=0.45),
            )
        )
        fig_hm.update_layout(
            title="Mean load — hour × month (kWh/h)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=260,
            xaxis_title="Month",
            yaxis_title="Hour",
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
    with hm_right:
        fig_heat_dow = go.Figure(
            data=go.Heatmap(
                z=pivot_dow.values,
                x=dow_names,
                y=pivot_dow.index,
                colorscale="YlOrRd",
                colorbar=dict(title="kWh", len=0.45),
            )
        )
        fig_heat_dow.update_layout(
            title="Mean load — hour × day of week (kWh/h)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=260,
            xaxis_title="",
            yaxis_title="Hour",
        )
        st.plotly_chart(fig_heat_dow, use_container_width=True, config={"displayModeBar": False})


SCENARIO_TYPE_UI_OPTIONS = [
    "All scenarios",
    "Grid only",
    "PV + Grid",
    "PV + Battery + Grid",
    "Battery + Grid",
]


def _include_flags_for_scenario_type(scenario_type_ui: str) -> tuple[bool, bool]:
    """Map Results Controls scenario type to optimizer include_pv, include_battery."""
    if scenario_type_ui == "All scenarios":
        return True, True
    if scenario_type_ui == "Grid only":
        return False, False
    if scenario_type_ui == "PV + Grid":
        return True, False
    if scenario_type_ui == "PV + Battery + Grid":
        return True, True
    if scenario_type_ui == "Battery + Grid":
        return False, True
    return True, True


def _filter_by_scenario_type(res_df: pd.DataFrame, scenario_type_ui: str) -> pd.DataFrame:
    """Keep rows allowed by Scenario type; drop degenerate non–Grid-only 0/0 clones."""
    if res_df is None or len(res_df) == 0:
        return pd.DataFrame()
    out = res_df.copy()
    if scenario_type_ui == "All scenarios":
        allowed = set(_SCENARIO_TYPES_ALL)
    else:
        allowed = {scenario_type_ui}
    out = out[out["Scenario"].isin(allowed)].copy()
    deg_mask = (
        (out["Scenario"] != "Grid only")
        & (pd.to_numeric(out["PV (kWp)"], errors="coerce").fillna(0) <= 0)
        & (pd.to_numeric(out["Battery (kWh)"], errors="coerce").fillna(0) <= 0)
    )
    out = out[~deg_mask].copy()
    return out


def _apply_hard_filters_to_results_df(
    df: pd.DataFrame,
    *,
    capex_max_eur: float | None = None,
    payback_max_years: float | None = None,
    npv_min_eur: float | None = None,
    irr_min_pct: float | None = None,
    self_sufficiency_min_pct: float | None = None,
    annual_co2_savings_min_kg: float | None = None,
    annual_electricity_cost_max_eur: float | None = None,
    self_consumption_ratio_min_pct: float | None = None,
) -> pd.DataFrame:
    """Post-run hard constraints filter for the consolidated scenario results table.

    Filters are applied with numeric comparisons and are optional (None = no filtering).
    NaN values never pass constraints (e.g. NaN IRR with IRR min set is excluded).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()

    def _col_numeric(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce")

    def _col_numeric_finite(col: str) -> pd.Series:
        x = _col_numeric(col)
        # Constraints should not accept inf / -inf values.
        return x.where(np.isfinite(x), np.nan)

    if capex_max_eur is not None:
        if "CAPEX (€)" in out.columns:
            capex = _col_numeric_finite("CAPEX (€)")
            out = out[capex <= float(capex_max_eur)].copy()

    if payback_max_years is not None:
        # Consolidated table KPI key is "Payback (yrs)".
        if "Payback (yrs)" in out.columns:
            pb = _col_numeric_finite("Payback (yrs)")
            out = out[pb <= float(payback_max_years)].copy()

    if npv_min_eur is not None:
        # Consolidated table KPI key is "NPV (€)".
        if "NPV (€)" in out.columns:
            npv = _col_numeric_finite("NPV (€)")
            out = out[npv >= float(npv_min_eur)].copy()

    if irr_min_pct is not None:
        # Consolidated table KPI key is "IRR (%)".
        if "IRR (%)" in out.columns:
            irr = _col_numeric_finite("IRR (%)")
            out = out[irr >= float(irr_min_pct)].copy()

    if self_sufficiency_min_pct is not None:
        if "Self-sufficiency (%)" in out.columns:
            ss = _col_numeric_finite("Self-sufficiency (%)")
            out = out[ss >= float(self_sufficiency_min_pct)].copy()

    if annual_co2_savings_min_kg is not None:
        if "CO2 savings (kg)" in out.columns:
            co2 = _col_numeric_finite("CO2 savings (kg)")
            out = out[co2 >= float(annual_co2_savings_min_kg)].copy()

    if annual_electricity_cost_max_eur is not None:
        if "Annual cost (€)" in out.columns:
            ac = _col_numeric_finite("Annual cost (€)")
            out = out[ac <= float(annual_electricity_cost_max_eur)].copy()

    if self_consumption_ratio_min_pct is not None:
        if "Self-consumption ratio (%)" in out.columns:
            scr = _col_numeric_finite("Self-consumption ratio (%)")
            out = out[scr >= float(self_consumption_ratio_min_pct)].copy()

    return out


def _rank_scenarios_by_goal(res_df: pd.DataFrame, goal: str) -> list:
    """Rank scenarios in res_df by the selected goal. Returns list of (scenario_name, row) best first."""
    if res_df is None or len(res_df) == 0:
        return []
    cand = res_df.copy()
    if goal == "Lowest annual electricity cost":
        cand = cand.sort_values("Annual cost (€)", ascending=True)
    elif goal == "Highest annual savings":
        cand = cand.sort_values("Annual savings (€)", ascending=False)
    elif goal == "Best payback":
        cand["_pb"] = cand["Payback period (years)"].replace([float("inf")], np.nan)
        cand = cand.sort_values("_pb", ascending=True, na_position="last")
    elif goal == "Best self-sufficiency / lowest grid import":
        cand = cand.sort_values("Self-sufficiency ratio (%)", ascending=False)
    elif goal == "Most CO2 savings":
        cand = cand.sort_values("CO2 savings (kg)", ascending=False)
    elif goal == "Best NPV":
        cand = cand.sort_values("NPV (20y, €)", ascending=False)
    elif goal == "Best IRR":
        cand["_irr"] = cand["IRR (20y, %)"].fillna(-1)
        cand = cand.sort_values("_irr", ascending=False)
    elif goal == "Largest PV meeting self-consumption ratio >= X%":
        cand = cand.sort_values(["PV (kWp)", "Self-consumption ratio (%)"], ascending=[False, False])
    elif goal == "PV size closest to annual community demand":
        demand = float(cand["Total annual community consumption (kWh)"].iloc[0])
        cand["_diff"] = (cand["Total annual PV generation (kWh)"] - demand).abs()
        cand = cand.sort_values("_diff", ascending=True)
    else:
        cand = cand.sort_values("Annual savings (€)", ascending=False)
    return [(row["Scenario"], row) for _, row in cand.iterrows()]


def _order_dataframe_goal5_largest_pv_selfconsumption(cand: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """
    Order rows for goal "Largest PV meeting self-consumption ratio >= X%".
    Matches optimizer ``pick_largest_pv_selfconsumption``: among PV>0 rows, prefer those with
    self-consumption ratio >= threshold; rank by largest PV then highest ratio. If none qualify,
    fall back to highest ratio then largest PV (among PV>0), then non-PV rows last.
    """
    if cand is None or len(cand) == 0:
        return cand
    out = cand.copy()
    ratio = pd.to_numeric(out["Self-consumption ratio (%)"], errors="coerce")
    pv = pd.to_numeric(out["PV (kWp)"], errors="coerce").fillna(0)
    thr = float(threshold_pct)
    with_pv = pv > 0
    meets = with_pv & ratio.notna() & (ratio >= thr)
    parts: list[pd.DataFrame] = []
    if meets.any():
        a = out.loc[meets].sort_values(
            ["PV (kWp)", "Self-consumption ratio (%)"],
            ascending=[False, False],
            kind="mergesort",
        )
        parts.append(a)
        rest = out.loc[~meets]
    else:
        rest = out
    rest_pv = rest.loc[pd.to_numeric(rest["PV (kWp)"], errors="coerce").fillna(0) > 0]
    rest_nopv = rest.loc[pd.to_numeric(rest["PV (kWp)"], errors="coerce").fillna(0) <= 0]
    if len(rest_pv) > 0:
        b = rest_pv.sort_values(
            ["Self-consumption ratio (%)", "PV (kWp)"],
            ascending=[False, False],
            kind="mergesort",
        )
        parts.append(b)
    if len(rest_nopv) > 0:
        c = rest_nopv.sort_values("Annual cost (€)", ascending=True, kind="mergesort")
        parts.append(c)
    return pd.concat(parts, ignore_index=True) if parts else out


def _rank_scenarios_from_consolidated_table(
    df: pd.DataFrame,
    goal: str,
    annual_community_demand_kwh: float,
    *,
    goal5_threshold_pct: float = 90.0,
) -> list[tuple[str, pd.Series]]:
    """Rank rows from ``build_full_scenario_results_df`` (same universe as the All scenario grid)."""
    if df is None or len(df) == 0:
        return []
    cand = df.copy()
    if goal == "Lowest annual electricity cost":
        cand = cand.sort_values("Annual cost (€)", ascending=True, kind="mergesort")
    elif goal == "Highest annual savings":
        cand = cand.sort_values("Annual savings (€)", ascending=False, kind="mergesort")
    elif goal == "Best payback":
        pb = pd.to_numeric(cand["Payback (yrs)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cand = cand.assign(_pb=pb).sort_values("_pb", ascending=True, na_position="last", kind="mergesort").drop(
            columns=["_pb"]
        )
    elif goal == "Best self-sufficiency / lowest grid import":
        cand = cand.sort_values("Self-sufficiency (%)", ascending=False, kind="mergesort")
    elif goal == "Most CO2 savings":
        cand = cand.sort_values("CO2 savings (kg)", ascending=False, kind="mergesort")
    elif goal == "Best NPV":
        cand = cand.sort_values("NPV (€)", ascending=False, kind="mergesort")
    elif goal == "Best IRR":
        irr = pd.to_numeric(cand["IRR (%)"], errors="coerce").fillna(-1.0)
        cand = cand.assign(_irr=irr).sort_values("_irr", ascending=False, kind="mergesort").drop(columns=["_irr"])
    elif goal == "Largest PV meeting self-consumption ratio >= X%":
        cand = _order_dataframe_goal5_largest_pv_selfconsumption(cand, goal5_threshold_pct)
    elif goal == "PV size closest to annual community demand":
        if "Total annual PV generation (kWh)" in cand.columns:
            gen = pd.to_numeric(cand["Total annual PV generation (kWh)"], errors="coerce")
            cand = cand.assign(_diff=(gen - float(annual_community_demand_kwh)).abs()).sort_values(
                "_diff", ascending=True, kind="mergesort"
            ).drop(columns=["_diff"])
        else:
            cand = cand.sort_values("PV (kWp)", ascending=True, kind="mergesort")
    else:
        cand = cand.sort_values("Annual savings (€)", ascending=False, kind="mergesort")
    return [(str(r["Scenario"]), r) for _, r in cand.iterrows()]


def metrics_and_hourly_for_scenario_at_sizes(
    df: pd.DataFrame,
    tcol: str,
    scenario_name: str,
    pv_kwp: int,
    batt_kwh: int,
    export_rate: float,
    standing_charge: float,
    pso_levy_annual: float,
    opex_pct: float,
    discount_rate: float | None,
    pv_cost_per_kwp: float,
    batt_cost_per_kwh: float,
    electricity_inflation_rate: float,
    battery_settings: BatterySettings,
    battery_replacement_year: int | None,
    battery_replacement_pct_of_batt_capex: float,
    inverter_replacement_year: int | None,
    inverter_replacement_pct_of_pv_capex: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Dispatch + KPI row (evaluate_for_tariff-shaped) and hourly dataframe for one scenario size."""
    d_base = run_scenario_grid_only(df, tcol)
    baseline_co2 = float(d_base["grid_import"].to_numpy(dtype=float).sum()) * CO2_FACTOR
    baseline_cost = float(np.sum(d_base["grid_import"].to_numpy(dtype=float) * df[tcol].to_numpy(dtype=float)))
    pv_cost_eff = float(pv_cost_per_kwp)
    batt_cost_eff = float(batt_cost_per_kwh)

    if scenario_name == "Grid only":
        d = d_base
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        row = {
            "Scenario": "Grid only",
            "PV (kWp)": 0,
            "Battery (kWh)": 0,
            **k,
            "Annual cost (€)": k["Annual cost (€)"] + standing_charge + pso_levy_annual,
            "CAPEX (€)": 0.0,
            "Annual savings (€)": 0.0,
            "Gross savings over 20 years (€)": 0.0,
            "Net benefit over 20 years (€)": 0.0,
            "Payback period (years)": float("inf"),
            "NPV (20y, €)": 0.0,
            "IRR (20y, %)": float("nan"),
            "CO2 savings (kg)": 0.0,
            "CO2 reduction (%)": 0.0,
        }
        return pd.Series(row), d

    if scenario_name == "PV + Grid":
        d = run_scenario_pv_grid(df, int(pv_kwp), tcol)
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        capex = int(pv_kwp) * pv_cost_eff
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k["Annual cost (€)"] - opex
        batt_repl = 0.0
        inv_repl = (
            capex * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= LIFETIME_YEARS)
            else 0.0
        )
        payback, npv = compute_payback_and_npv(
            capex,
            annual_savings,
            discount_rate,
            electricity_inflation_rate,
            battery_replacement_year,
            batt_repl,
            inverter_replacement_year,
            inv_repl,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl,
        )
        gross = _gross_savings_20y(annual_savings, electricity_inflation_rate)
        row = {
            "Scenario": "PV + Grid",
            "PV (kWp)": int(pv_kwp),
            "Battery (kWh)": 0,
            **k,
            "Annual cost (€)": k["Annual cost (€)"] + standing_charge + pso_levy_annual + opex,
            "CAPEX (€)": capex,
            "Annual savings (€)": annual_savings,
            "Gross savings over 20 years (€)": gross,
            "Net benefit over 20 years (€)": gross - capex - inv_repl,
            "Payback period (years)": payback,
            "NPV (20y, €)": npv,
            "IRR (20y, %)": 100.0 * irr if np.isfinite(irr) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
        return pd.Series(row), d

    if scenario_name == "PV + Battery + Grid":
        d = run_scenario_pv_battery_grid(df, int(pv_kwp), int(batt_kwh), tcol, battery_settings)
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        capex = int(pv_kwp) * pv_cost_eff + int(batt_kwh) * batt_cost_eff
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k["Annual cost (€)"] - opex
        batt_repl = (
            (int(batt_kwh) * batt_cost_eff) * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= LIFETIME_YEARS)
            else 0.0
        )
        inv_repl = (
            (int(pv_kwp) * pv_cost_eff) * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= LIFETIME_YEARS)
            else 0.0
        )
        payback, npv = compute_payback_and_npv(
            capex,
            annual_savings,
            discount_rate,
            electricity_inflation_rate,
            battery_replacement_year,
            batt_repl,
            inverter_replacement_year,
            inv_repl,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl,
        )
        gross = _gross_savings_20y(annual_savings, electricity_inflation_rate)
        row = {
            "Scenario": "PV + Battery + Grid",
            "PV (kWp)": int(pv_kwp),
            "Battery (kWh)": int(batt_kwh),
            **k,
            "Annual cost (€)": k["Annual cost (€)"] + standing_charge + pso_levy_annual + opex,
            "CAPEX (€)": capex,
            "Annual savings (€)": annual_savings,
            "Gross savings over 20 years (€)": gross,
            "Net benefit over 20 years (€)": gross - capex - batt_repl - inv_repl,
            "Payback period (years)": payback,
            "NPV (20y, €)": npv,
            "IRR (20y, %)": 100.0 * irr if np.isfinite(irr) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
        return pd.Series(row), d

    if scenario_name == "Battery + Grid":
        d = run_scenario_battery_grid(df, int(batt_kwh), tcol, battery_settings)
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        capex = int(batt_kwh) * batt_cost_eff
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k["Annual cost (€)"] - opex
        batt_repl = (
            capex * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= LIFETIME_YEARS)
            else 0.0
        )
        inv_repl = 0.0
        payback, npv = compute_payback_and_npv(
            capex,
            annual_savings,
            discount_rate,
            electricity_inflation_rate,
            battery_replacement_year,
            batt_repl,
            inverter_replacement_year,
            inv_repl,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl,
        )
        gross = _gross_savings_20y(annual_savings, electricity_inflation_rate)
        row = {
            "Scenario": "Battery + Grid",
            "PV (kWp)": 0,
            "Battery (kWh)": int(batt_kwh),
            **k,
            "Annual cost (€)": k["Annual cost (€)"] + standing_charge + pso_levy_annual + opex,
            "CAPEX (€)": capex,
            "Annual savings (€)": annual_savings,
            "Gross savings over 20 years (€)": gross,
            "Net benefit over 20 years (€)": gross - capex - batt_repl,
            "Payback period (years)": payback,
            "NPV (20y, €)": npv,
            "IRR (20y, %)": 100.0 * irr if np.isfinite(irr) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
        return pd.Series(row), d

    raise ValueError(f"Unknown scenario_name for metrics/hourly: {scenario_name!r}")


def _aggrid_goal_key_fragment(goal: str) -> str:
    """Stable short id for Streamlit widget keys when goal-based grid order changes."""
    return hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]


def _sort_consolidated_scenarios_for_goal(
    df: pd.DataFrame,
    goal: str,
    *,
    annual_community_demand_kwh: float,
    goal5_threshold_pct: float = 90.0,
) -> pd.DataFrame:
    """Row order for All scenario results — aligned with Rank results by (_rank_scenarios_by_goal)."""
    if df is None or len(df) == 0:
        return df
    cand = df.copy()
    if goal == "Lowest annual electricity cost":
        return cand.sort_values("Annual cost (€)", ascending=True, kind="mergesort")
    if goal == "Highest annual savings":
        return cand.sort_values("Annual savings (€)", ascending=False, kind="mergesort")
    if goal == "Best payback":
        pb = pd.to_numeric(cand["Payback (yrs)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return cand.assign(_pb=pb).sort_values("_pb", ascending=True, na_position="last", kind="mergesort").drop(
            columns=["_pb"]
        )
    if goal == "Best self-sufficiency / lowest grid import":
        return cand.sort_values("Self-sufficiency (%)", ascending=False, kind="mergesort")
    if goal == "Most CO2 savings":
        return cand.sort_values("CO2 savings (kg)", ascending=False, kind="mergesort")
    if goal == "Best NPV":
        return cand.sort_values("NPV (€)", ascending=False, kind="mergesort")
    if goal == "Best IRR":
        irr = pd.to_numeric(cand["IRR (%)"], errors="coerce").fillna(-1.0)
        return cand.assign(_irr=irr).sort_values("_irr", ascending=False, kind="mergesort").drop(columns=["_irr"])
    if goal == "Largest PV meeting self-consumption ratio >= X%":
        return _order_dataframe_goal5_largest_pv_selfconsumption(cand, goal5_threshold_pct)
    if goal == "PV size closest to annual community demand":
        col = "Total annual PV generation (kWh)"
        if col not in cand.columns:
            return cand.sort_values("PV (kWp)", ascending=True, kind="mergesort")
        gen = pd.to_numeric(cand[col], errors="coerce")
        diff = (gen - float(annual_community_demand_kwh)).abs()
        return cand.assign(_diff=diff).sort_values("_diff", ascending=True, kind="mergesort").drop(columns=["_diff"])
    return cand.sort_values("Annual savings (€)", ascending=False, kind="mergesort")


def _sort_pv_sweep_for_goal(
    df: pd.DataFrame,
    goal: str,
    *,
    annual_community_demand_kwh: float,
    goal5_threshold_pct: float = 90.0,
) -> pd.DataFrame:
    """Row order for PV + Grid sweep table — aligned with goal_to_pv_sweep_chart_column / pick_best."""
    if df is None or len(df) == 0:
        return df
    cand = df.copy()
    if goal == "Largest PV meeting self-consumption ratio >= X%":
        return _order_dataframe_goal5_largest_pv_selfconsumption(cand, goal5_threshold_pct)
    if goal == "PV size closest to annual community demand":
        col = "Annual PV generation (kWh)"
        if col not in cand.columns:
            return cand.sort_values("PV (kWp)", ascending=True, kind="mergesort")
        gen = pd.to_numeric(cand[col], errors="coerce")
        diff = (gen - float(annual_community_demand_kwh)).abs()
        return cand.assign(_diff=diff).sort_values("_diff", ascending=True, kind="mergesort").drop(columns=["_diff"])
    metric = goal_to_pv_sweep_chart_column(goal)
    if metric not in cand.columns:
        return cand
    series = pd.to_numeric(cand[metric], errors="coerce")
    if goal == "Lowest annual electricity cost":
        return cand.assign(_v=series).sort_values("_v", ascending=True, na_position="last", kind="mergesort").drop(
            columns=["_v"]
        )
    if goal == "Best payback":
        return cand.assign(_v=series.replace([np.inf, -np.inf], np.nan)).sort_values(
            "_v", ascending=True, na_position="last", kind="mergesort"
        ).drop(columns=["_v"])
    if goal == "Best IRR":
        return cand.assign(_v=series.fillna(-1.0)).sort_values("_v", ascending=False, kind="mergesort").drop(
            columns=["_v"]
        )
    # Highest annual savings, self-sufficiency, CO2 proxy (savings col), NPV, default
    return cand.assign(_v=series).sort_values("_v", ascending=False, na_position="last", kind="mergesort").drop(
        columns=["_v"]
    )


def evaluate_for_tariff(
    df: pd.DataFrame,
    opt_dfs: Dict[str, pd.DataFrame],
    tcol: str,
    tname: str,
    goal: str,
    include_pv: bool,
    include_battery: bool,
    goal5_threshold_pct: float,
    battery_settings: BatterySettings,
    export_rate: float,
    standing_charge: float = 0.0,
    pso_levy_annual: float = 0.0,
    opex_pct: float = 0.0,
    discount_rate: float | None = None,
    pv_cost_per_kwp: float | None = None,
    batt_cost_per_kwh: float | None = None,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_pct_of_batt_capex: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_pct_of_pv_capex: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    opt_df = opt_dfs[tcol]
    annual_demand_kwh = float(df["consumption"].to_numpy(dtype=float).sum())

    pv_cost_eff = float(PV_COST_PER_KWP if pv_cost_per_kwp is None else pv_cost_per_kwp)
    batt_cost_eff = float(BATT_COST_PER_KWH if batt_cost_per_kwh is None else batt_cost_per_kwh)

    def has_config(config_name: str) -> bool:
        return len(opt_df[opt_df["config"] == config_name]) > 0

    def pick_lowest_cost(config_name: str) -> pd.Series | None:
        sub = opt_df[opt_df["config"] == config_name]
        if len(sub) == 0:
            return None
        return sub.loc[sub["cost"].idxmin()]

    def pick_largest_pv_selfconsumption(config_name: str, threshold_pct: float) -> pd.Series | None:
        sub = opt_df[opt_df["config"] == config_name].copy()
        if len(sub) == 0:
            return None
        sub = sub[sub["self_consumption_ratio_pct"].notna()]
        if len(sub) == 0:
            return None
        filtered = sub[sub["self_consumption_ratio_pct"] >= threshold_pct]
        if len(filtered) > 0:
            pv_max = float(filtered["pv_kwp"].max())
            cand = filtered[filtered["pv_kwp"] == pv_max]
            return cand.loc[cand["self_consumption_ratio_pct"].idxmax()]

        # Fallback: maximize self-consumption ratio, then PV size
        max_ratio = float(sub["self_consumption_ratio_pct"].max()) if len(sub) else 0.0
        cand = sub[sub["self_consumption_ratio_pct"] == max_ratio] if len(sub) else sub
        if len(cand) == 0:
            return None
        return cand.loc[cand["pv_kwp"].idxmax()]

    def pick_closest_pv_to_demand(config_name: str, demand_kwh: float) -> pd.Series | None:
        sub = opt_df[opt_df["config"] == config_name].copy()
        if len(sub) == 0:
            return None
        sub["diff"] = (sub["pv_gen_kwh"] - demand_kwh).abs()
        min_diff = float(sub["diff"].min())
        cand = sub[sub["diff"] == min_diff]
        # Tie-breaker: lowest annual cost
        return cand.loc[cand["cost"].idxmin()]

    pv_only_kwp = 0
    pv_batt_kwp = 0
    batt_kwp = 0
    batt_only_kwh = 0

    # Goal 5: Largest PV that satisfies self-consumption ratio >= X%
    if goal == "Largest PV meeting self-consumption ratio >= X%":
        if include_pv and include_battery:
            pv_only_row = pick_largest_pv_selfconsumption("PV only", goal5_threshold_pct)
            if pv_only_row is not None:
                pv_only_kwp = int(pv_only_row["pv_kwp"])

            pv_batt_row = pick_largest_pv_selfconsumption("PV + Battery", goal5_threshold_pct)
            if pv_batt_row is not None:
                pv_batt_kwp = int(pv_batt_row["pv_kwp"])
                batt_kwp = int(pv_batt_row["batt_kwh"])

            batt_only_row = pick_lowest_cost("Battery only")
            if batt_only_row is not None:
                batt_only_kwh = int(batt_only_row["batt_kwh"])
        elif include_pv and not include_battery:
            pv_only_row = pick_largest_pv_selfconsumption("PV only", goal5_threshold_pct)
            if pv_only_row is not None:
                pv_only_kwp = int(pv_only_row["pv_kwp"])
                pv_batt_kwp = pv_only_kwp
            batt_kwp = 0
            batt_only_kwh = 0
        elif not include_pv and include_battery:
            batt_only_row = pick_lowest_cost("Battery only")
            if batt_only_row is not None:
                batt_only_kwh = int(batt_only_row["batt_kwh"])
            pv_only_kwp = 0
            pv_batt_kwp = 0
            batt_kwp = batt_only_kwh
        else:
            # grid-only
            pv_only_kwp = 0
            pv_batt_kwp = 0
            batt_kwp = 0
            batt_only_kwh = 0

    # Goal 6: PV size closest to annual community demand (kWh)
    elif goal == "PV size closest to annual community demand":
        if include_pv and include_battery:
            pv_only_row = pick_closest_pv_to_demand("PV only", annual_demand_kwh)
            if pv_only_row is not None:
                pv_only_kwp = int(pv_only_row["pv_kwp"])

            pv_batt_row = pick_closest_pv_to_demand("PV + Battery", annual_demand_kwh)
            if pv_batt_row is not None:
                pv_batt_kwp = int(pv_batt_row["pv_kwp"])
                batt_kwp = int(pv_batt_row["batt_kwh"])

            batt_only_row = pick_lowest_cost("Battery only")
            if batt_only_row is not None:
                batt_only_kwh = int(batt_only_row["batt_kwh"])
        elif include_pv and not include_battery:
            pv_only_row = pick_closest_pv_to_demand("PV only", annual_demand_kwh)
            if pv_only_row is not None:
                pv_only_kwp = int(pv_only_row["pv_kwp"])
                pv_batt_kwp = pv_only_kwp
            batt_kwp = 0
            batt_only_kwh = 0
        elif not include_pv and include_battery:
            batt_only_row = pick_lowest_cost("Battery only")
            if batt_only_row is not None:
                batt_only_kwh = int(batt_only_row["batt_kwh"])
            pv_only_kwp = 0
            pv_batt_kwp = 0
            batt_kwp = batt_only_kwh
        else:
            pv_only_kwp = 0
            pv_batt_kwp = 0
            batt_kwp = 0
            batt_only_kwh = 0

    # All other goals: use normal optimizer objective, but still respect PV/Battery inclusion
    else:
        if include_pv and has_config("PV only"):
            pv_only_row = pick_best(opt_df, "PV only", goal)
            pv_only_kwp = int(pv_only_row["pv_kwp"])
        if include_battery and has_config("Battery only"):
            batt_only_row = pick_best(opt_df, "Battery only", goal)
            batt_only_kwh = int(batt_only_row["batt_kwh"])

        if include_pv and include_battery and has_config("PV + Battery"):
            pv_batt_row = pick_best(opt_df, "PV + Battery", goal)
            pv_batt_kwp = int(pv_batt_row["pv_kwp"])
            batt_kwp = int(pv_batt_row["batt_kwh"])
        elif include_pv and not include_battery:
            pv_batt_kwp = pv_only_kwp
            batt_kwp = 0
        elif not include_pv and include_battery:
            pv_batt_kwp = 0
            batt_kwp = batt_only_kwh
        else:
            pv_batt_kwp = 0
            batt_kwp = 0

    # Baseline for savings + CO2 savings
    d_base = run_scenario_grid_only(df, tcol)
    baseline_grid_import = float(d_base["grid_import"].to_numpy(dtype=float).sum())
    baseline_co2 = baseline_grid_import * CO2_FACTOR
    baseline_cost = float(np.sum(d_base["grid_import"].to_numpy(dtype=float) * df[tcol].to_numpy(dtype=float)))

    scenarios = []

    # Grid only
    d1 = run_scenario_grid_only(df, tcol)
    k1 = compute_kpis_for_scenario(d1, tcol, export_rate)
    scenarios.append(
        {
            "Scenario": "Grid only",
            "PV (kWp)": 0,
            "Battery (kWh)": 0,
            **k1,
            "Annual cost (€)": k1["Annual cost (€)"] + standing_charge + pso_levy_annual,
            "CAPEX (€)": 0.0,
            "Annual savings (€)": 0.0,
            "Gross savings over 20 years (€)": 0.0,
            "Net benefit over 20 years (€)": 0.0,
            "Payback period (years)": float("inf"),
            "NPV (20y, €)": 0.0,
            "IRR (20y, %)": float("nan"),
            "CO2 savings (kg)": 0.0,
            "CO2 reduction (%)": 0.0,
        }
    )

    # PV + Grid
    d2 = run_scenario_pv_grid(df, pv_only_kwp, tcol)
    k2 = compute_kpis_for_scenario(d2, tcol, export_rate)
    capex2 = pv_only_kwp * pv_cost_eff
    opex2 = capex2 * (opex_pct / 100.0)
    annual_savings2 = baseline_cost - k2["Annual cost (€)"] - opex2
    batt_repl2 = 0.0
    inv_repl2 = (
        capex2 * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= LIFETIME_YEARS)
        else 0.0
    )
    payback2, npv2 = compute_payback_and_npv(
        capex2,
        annual_savings2,
        discount_rate,
        electricity_inflation_rate,
        battery_replacement_year,
        batt_repl2,
        inverter_replacement_year,
        inv_repl2,
    )
    irr2 = compute_irr(
        capex2,
        annual_savings2,
        electricity_inflation_rate=electricity_inflation_rate,
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_eur=batt_repl2,
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_eur=inv_repl2,
    )
    gross2 = _gross_savings_20y(annual_savings2, electricity_inflation_rate)
    scenarios.append(
        {
            "Scenario": "PV + Grid",
            "PV (kWp)": pv_only_kwp,
            "Battery (kWh)": 0,
            **k2,
            "Annual cost (€)": k2["Annual cost (€)"] + standing_charge + pso_levy_annual + opex2,
            "CAPEX (€)": capex2,
            "Annual savings (€)": annual_savings2,
            "Gross savings over 20 years (€)": gross2,
            "Net benefit over 20 years (€)": gross2 - capex2 - inv_repl2,
            "Payback period (years)": payback2,
            "NPV (20y, €)": npv2,
            "IRR (20y, %)": 100.0 * irr2 if np.isfinite(irr2) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k2["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k2["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
    )

    # PV + Battery + Grid
    d3 = run_scenario_pv_battery_grid(df, pv_batt_kwp, batt_kwp, tcol, battery_settings)
    k3 = compute_kpis_for_scenario(d3, tcol, export_rate)
    capex3 = pv_batt_kwp * pv_cost_eff + batt_kwp * batt_cost_eff
    opex3 = capex3 * (opex_pct / 100.0)
    annual_savings3 = baseline_cost - k3["Annual cost (€)"] - opex3
    batt_repl3 = (
        (batt_kwp * batt_cost_eff) * (float(battery_replacement_pct_of_batt_capex) / 100.0)
        if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= LIFETIME_YEARS)
        else 0.0
    )
    inv_repl3 = (
        (pv_batt_kwp * pv_cost_eff) * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= LIFETIME_YEARS)
        else 0.0
    )
    payback3, npv3 = compute_payback_and_npv(
        capex3,
        annual_savings3,
        discount_rate,
        electricity_inflation_rate,
        battery_replacement_year,
        batt_repl3,
        inverter_replacement_year,
        inv_repl3,
    )
    irr3 = compute_irr(
        capex3,
        annual_savings3,
        electricity_inflation_rate=electricity_inflation_rate,
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_eur=batt_repl3,
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_eur=inv_repl3,
    )
    gross3 = _gross_savings_20y(annual_savings3, electricity_inflation_rate)
    scenarios.append(
        {
            "Scenario": "PV + Battery + Grid",
            "PV (kWp)": pv_batt_kwp,
            "Battery (kWh)": batt_kwp,
            **k3,
            "Annual cost (€)": k3["Annual cost (€)"] + standing_charge + pso_levy_annual + opex3,
            "CAPEX (€)": capex3,
            "Annual savings (€)": annual_savings3,
            "Gross savings over 20 years (€)": gross3,
            "Net benefit over 20 years (€)": gross3 - capex3 - batt_repl3 - inv_repl3,
            "Payback period (years)": payback3,
            "NPV (20y, €)": npv3,
            "IRR (20y, %)": 100.0 * irr3 if np.isfinite(irr3) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k3["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k3["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
    )

    # Battery + Grid (no PV)
    d4 = run_scenario_battery_grid(df, batt_only_kwh, tcol, battery_settings)
    k4 = compute_kpis_for_scenario(d4, tcol, export_rate)
    capex4 = batt_only_kwh * batt_cost_eff
    opex4 = capex4 * (opex_pct / 100.0)
    annual_savings4 = baseline_cost - k4["Annual cost (€)"] - opex4
    batt_repl4 = (
        capex4 * (float(battery_replacement_pct_of_batt_capex) / 100.0)
        if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= LIFETIME_YEARS)
        else 0.0
    )
    inv_repl4 = 0.0
    payback4, npv4 = compute_payback_and_npv(
        capex4,
        annual_savings4,
        discount_rate,
        electricity_inflation_rate,
        battery_replacement_year,
        batt_repl4,
        inverter_replacement_year,
        inv_repl4,
    )
    irr4 = compute_irr(
        capex4,
        annual_savings4,
        electricity_inflation_rate=electricity_inflation_rate,
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_eur=batt_repl4,
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_eur=inv_repl4,
    )
    gross4 = _gross_savings_20y(annual_savings4, electricity_inflation_rate)
    scenarios.append(
        {
            "Scenario": "Battery + Grid",
            "PV (kWp)": 0,
            "Battery (kWh)": batt_only_kwh,
            **k4,
            "Annual cost (€)": k4["Annual cost (€)"] + standing_charge + pso_levy_annual + opex4,
            "CAPEX (€)": capex4,
            "Annual savings (€)": annual_savings4,
            "Gross savings over 20 years (€)": gross4,
            "Net benefit over 20 years (€)": gross4 - capex4 - batt_repl4,
            "Payback period (years)": payback4,
            "NPV (20y, €)": npv4,
            "IRR (20y, %)": 100.0 * irr4 if np.isfinite(irr4) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k4["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k4["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
    )

    res = pd.DataFrame(scenarios)
    res.insert(0, "Tariff", tname)
    hourly_by_scenario = {
        "Grid only": d1,
        "PV + Grid": d2,
        "PV + Battery + Grid": d3,
        "Battery + Grid": d4,
    }
    return res, hourly_by_scenario


if run_button:
    st.session_state.stop_run_requested = False
    if cons_file is None or pv_file is None:
        st.warning("Please upload both the consumption CSV and the PV timeseries CSV.")
    else:
        # Only compute expensive stuff once per upload
        cons_bytes = cons_file.getvalue()
        pv_bytes = pv_file.getvalue()
        st.session_state.prepared_meta = {
            "cons_sha": hashlib.sha256(cons_bytes).hexdigest(),
            "pv_sha": hashlib.sha256(pv_bytes).hexdigest(),
        }

        with st.spinner("Loading and aligning data..."):
            cfg, exp_rate = get_active_tariff_config(
                override_tariffs,
                float(std_day), float(std_peak), float(std_night),
                float(wk_day), float(wk_peak), float(wk_night),
                float(we_day), float(we_peak), float(we_night),
                float(flat_rate), float(export_rate_input),
            )
            st.session_state.active_tariff_config = cfg
            st.session_state.active_export_rate = exp_rate
            tck = "defaults" if not override_tariffs else _tariff_cache_key(cfg, exp_rate)
            try:
                st.session_state.prepared_df = load_and_prepare_data(cons_bytes, pv_bytes, cfg, tck)
            except Exception as e:
                st.error(f"Could not parse input files. {e}")
                st.stop()

        # Update global CAPEX assumptions (used inside optimizer and KPI calculations)
        PV_COST_PER_KWP = float(pv_capex)
        BATT_COST_PER_KWH = float(batt_capex)

        # Battery dispatch settings
        st.session_state.battery_settings = BatterySettings(
            eff_round_trip=float(rt_eff_pct) / 100.0,
            dod=float(dod_pct) / 100.0,
            init_soc=float(init_soc_pct) / 100.0,
            c_rate=float(c_rate),
            charge_from_pv=bool(charge_from_pv),
            charge_from_grid_at_night=bool(charge_from_grid_at_night),
            discharge_schedule=str(discharge_schedule),
        )

        st.session_state.active_discount_rate = discount_rate
        opt_cfg = OptimizerConfig(
            pv_min=pv_min,
            pv_max=pv_max,
            batt_min=batt_min,
            batt_max=batt_max,
            pv_step=opt_pv_step,
            batt_step=opt_batt_step,
        )
        tariffs_to_run = ["tariff_standard", "tariff_weekend", "tariff_flat"]
        opt_dfs = {}
        tcol_to_name = {"tariff_standard": "Standard", "tariff_weekend": "Weekend Saver", "tariff_flat": "Flat"}
        total_evals_per_tariff = count_optimizer_evaluations(opt_cfg)
        total_work = max(1, total_evals_per_tariff * len(tariffs_to_run))
        completed_work_ref = {"value": 0}
        current_tariff_name = "Standard"
        progress_overlay = st.empty()
        with st.status("Optimizer running…", expanded=True) as run_status:
            run_status.update(label="Starting…", state="running")
            progress_bar = st.progress(0, text="Starting optimizer... 0%")
            st.caption("Tip: Use 'Stop current run' in the sidebar to stop at the next checkpoint.")

            def _stop_requested() -> bool:
                return bool(st.session_state.get("stop_run_requested", False))

            def _progress_tick() -> None:
                completed_work_ref["value"] += 1
                completed_work = int(completed_work_ref["value"])
                if completed_work == 1 or completed_work % 25 == 0 or completed_work >= total_work:
                    pct = int(100 * completed_work / total_work)
                    txt = f"Running optimizer for {current_tariff_name} — {completed_work:,}/{total_work:,} ({pct}%)"
                    progress_bar.progress(
                        min(1.0, completed_work / total_work),
                        text=txt,
                    )
                    run_status.update(label=txt, state="running")
                    progress_overlay.markdown(
                        _optimizer_progress_overlay_html(current_tariff_name, completed_work, total_work, pct),
                        unsafe_allow_html=True,
                    )

            aborted = False
            for tcol in tariffs_to_run:
                current_tariff_name = tcol_to_name[tcol]
                opt_dfs[tcol] = optimize(
                    st.session_state.prepared_df,
                    tcol,
                    opt_cfg,
                    st.session_state.battery_settings,
                    st.session_state.active_export_rate,
                    standing_charge=standing_charges[tcol_to_name[tcol]],
                    opex_pct=float(opex_pct),
                    discount_rate=discount_rate,
                    electricity_inflation_rate=electricity_inflation_rate,
                    battery_replacement_year=battery_replacement_year,
                    battery_replacement_pct_of_batt_capex=float(battery_replacement_cost_pct),
                    inverter_replacement_year=inverter_replacement_year,
                    inverter_replacement_pct_of_pv_capex=float(inverter_replacement_cost_pct),
                    pso_levy_annual=float(pso_levy),
                    progress_callback=_progress_tick,
                    stop_requested=_stop_requested,
                )
                if _stop_requested():
                    aborted = True
                    break

            if aborted:
                completed_work = int(completed_work_ref["value"])
                pct_ab = int(100 * completed_work / total_work)
                progress_bar.progress(min(1.0, completed_work / total_work), text=f"Run stopped at {pct_ab}%")
                run_status.update(label=f"Stopped at {pct_ab}%", state="error")
                progress_overlay.empty()
                st.warning("Optimization stopped by user request. Previous completed results are kept.")
                st.session_state.stop_run_requested = False
                st.stop()

            progress_bar.progress(1.0, text="Optimization complete. 100%")
            run_status.update(label="Optimization complete — 100%", state="complete")

        progress_overlay.empty()
        st.session_state.opt_dfs = opt_dfs
        st.session_state.full_results_df = None
        st.session_state.last_standing_charges = standing_charges.copy()
        st.session_state.last_pso_levy = float(pso_levy)
        st.session_state.last_opex_pct = float(opex_pct)
        # Freeze discount rate used for NPV/IRR calculations
        st.session_state.last_discount_rate = discount_rate
        # Freeze electricity inflation rate used for NPV/IRR and 20y metrics
        st.session_state.last_electricity_inflation_rate = electricity_inflation_rate
        st.session_state.last_battery_replacement_year = battery_replacement_year
        st.session_state.last_battery_replacement_cost_pct = float(battery_replacement_cost_pct)
        st.session_state.last_inverter_replacement_year = inverter_replacement_year
        st.session_state.last_inverter_replacement_cost_pct = float(inverter_replacement_cost_pct)
        st.session_state.last_battery_settings = st.session_state.battery_settings
        _cfg_f = st.session_state.active_tariff_config
        _er_f = float(st.session_state.active_export_rate)
        st.session_state.last_tariff_config = _cfg_f
        st.session_state.last_export_rate = _er_f
        st.session_state.last_override_tariffs = bool(override_tariffs)
        st.session_state.last_input_hashes = dict(st.session_state.prepared_meta)
        st.session_state.last_opt_cfg = {
            "pv_min": int(pv_min),
            "pv_max": int(pv_max),
            "batt_min": int(batt_min),
            "batt_max": int(batt_max),
            "pv_step": int(opt_pv_step),
            "batt_step": int(opt_batt_step),
            "speed_preset": str(speed_preset),
        }
        # Freeze CAPEX assumptions used during post-run evaluation
        st.session_state.last_pv_capex = float(PV_COST_PER_KWP)
        st.session_state.last_batt_capex = float(BATT_COST_PER_KWH)


# If data is prepared, evaluate instantly when dropdowns change
if st.session_state.prepared_df is not None and st.session_state.opt_dfs is not None:
    def fmt_num_1(x):
        try:
            if x is None:
                return "—"
            if isinstance(x, (int, float, np.floating)):
                if not np.isfinite(x):
                    return "inf" if x > 0 else "—"
                return f"{x:,.1f}"
        except Exception:
            pass
        return str(x)

    def fmt_eur_1(x):
        s = fmt_num_1(x)
        if s in ("inf", "—"):
            return s
        return f"€{s}"

    current_battery_settings = BatterySettings(
        eff_round_trip=float(rt_eff_pct) / 100.0,
        dod=float(dod_pct) / 100.0,
        init_soc=float(init_soc_pct) / 100.0,
        c_rate=float(c_rate),
        charge_from_pv=bool(charge_from_pv),
        charge_from_grid_at_night=bool(charge_from_grid_at_night),
        discharge_schedule=str(discharge_schedule),
    )
    current_tariff_config, current_export_rate = get_active_tariff_config(
        override_tariffs,
        float(std_day), float(std_peak), float(std_night),
        float(wk_day), float(wk_peak), float(wk_night),
        float(we_day), float(we_peak), float(we_night),
        float(flat_rate), float(export_rate_input),
    )
    current_cons_sha = hashlib.sha256(cons_file.getvalue()).hexdigest() if cons_file is not None else None
    current_pv_sha = hashlib.sha256(pv_file.getvalue()).hexdigest() if pv_file is not None else None

    # Show subtle warning if current sidebar assumptions differ from last completed run
    assumptions_changed = (
        (st.session_state.last_input_hashes.get("cons_sha") != current_cons_sha)
        or (st.session_state.last_input_hashes.get("pv_sha") != current_pv_sha)
        or (abs(float(st.session_state.last_opex_pct) - float(opex_pct)) > 1e-12)
        or (abs(float(st.session_state.last_discount_rate) - float(discount_rate)) > 1e-12)
        or (abs(float(st.session_state.last_electricity_inflation_rate) - float(electricity_inflation_rate)) > 1e-12)
        or (st.session_state.last_battery_replacement_year != battery_replacement_year)
        or (abs(float(st.session_state.last_battery_replacement_cost_pct) - float(battery_replacement_cost_pct)) > 1e-12)
        or (st.session_state.last_inverter_replacement_year != inverter_replacement_year)
        or (abs(float(st.session_state.last_inverter_replacement_cost_pct) - float(inverter_replacement_cost_pct)) > 1e-12)
        or (st.session_state.last_battery_settings != current_battery_settings)
        or (st.session_state.last_tariff_config != current_tariff_config)
        or (abs(float(st.session_state.last_export_rate) - float(current_export_rate)) > 1e-12)
        or (bool(st.session_state.last_override_tariffs) != bool(override_tariffs))
        or (int(st.session_state.last_opt_cfg.get("pv_min", -1)) != int(pv_min))
        or (int(st.session_state.last_opt_cfg.get("pv_max", -1)) != int(pv_max))
        or (int(st.session_state.last_opt_cfg.get("batt_min", -1)) != int(batt_min))
        or (int(st.session_state.last_opt_cfg.get("batt_max", -1)) != int(batt_max))
        or (int(st.session_state.last_opt_cfg.get("pv_step", -1)) != int(opt_pv_step))
        or (int(st.session_state.last_opt_cfg.get("batt_step", -1)) != int(opt_batt_step))
        or (str(st.session_state.last_opt_cfg.get("speed_preset", "")) != str(speed_preset))
        or (abs(float(st.session_state.last_pv_capex) - float(pv_capex)) > 1e-12)
        or (abs(float(st.session_state.last_batt_capex) - float(batt_capex)) > 1e-12)
        or any(
            abs(float(st.session_state.last_standing_charges.get(k, 0.0)) - float(standing_charges.get(k, 0.0))) > 1e-12
            for k in ["Standard", "Weekend Saver", "Flat"]
        )
        or (abs(float(st.session_state.last_pso_levy) - float(pso_levy)) > 1e-12)
    )
    if assumptions_changed:
        st.info("Sidebar assumptions changed after the last run. Displayed results still reflect the last completed Run analysis.")

    _tariff_opts = ["Standard", "Weekend Saver", "Flat"]
    if st.session_state.view_filter_tariff not in _tariff_opts:
        st.session_state.view_filter_tariff = _tariff_opts[0]
    _last_loc = st.session_state.get("last_opt_cfg", {}) or {}
    _pv_min_last = int(_last_loc.get("pv_min", pv_min))
    _pv_max_last = int(_last_loc.get("pv_max", pv_max))
    tab_results, tab_consumption, tab_pv_sweep, tab_explainer = st.tabs(
        [
            "Results (KPIs)",
            "Consumption patterns",
            f"PV + Grid sweep ({_pv_min_last}–{_pv_max_last} kWp)",
            "Settings & KPI guide",
        ]
    )

    with tab_results:
        st.markdown(
            '<p style="font-size:0.95rem;font-weight:600;margin:0 0 0.35rem 0;color:#1a1a1a;">Results Controls</p>',
            unsafe_allow_html=True,
        )
        ctl1, ctl2 = st.columns(2)
        goal_options = [
            "Lowest annual electricity cost",
            "Highest annual savings",
            "Best payback",
            "Best self-sufficiency / lowest grid import",
            "Most CO2 savings",
            "Best NPV",
            "Best IRR",
            "Largest PV meeting self-consumption ratio >= X%",
            "PV size closest to annual community demand",
        ]
        if st.session_state.view_goal not in goal_options:
            st.session_state.view_goal = goal_options[0]

        with ctl1:
            st.markdown(
                '<div style="font-size:0.78rem;font-weight:500;color:#444;margin:0 0 0.15rem 0;">Rank results by</div>',
                unsafe_allow_html=True,
            )
            st.selectbox(
                " ",
                goal_options,
                index=goal_options.index(st.session_state.view_goal),
                key="view_goal",
                label_visibility="collapsed",
            )
        with ctl2:
            st.markdown(
                '<div style="font-size:0.78rem;font-weight:500;color:#444;margin:0 0 0.15rem 0;">Show tariff</div>',
                unsafe_allow_html=True,
            )
            st.selectbox(
                " ",
                _tariff_opts,
                index=_tariff_opts.index(st.session_state.view_filter_tariff),
                key="view_filter_tariff",
                label_visibility="collapsed",
            )
        if st.session_state.view_scenario_type not in SCENARIO_TYPE_UI_OPTIONS:
            st.session_state.view_scenario_type = "All scenarios"
        st.markdown(
            '<div style="font-size:0.78rem;font-weight:500;color:#444;margin:0.35rem 0 0.15rem 0;">Scenario type</div>',
            unsafe_allow_html=True,
        )
        st.selectbox(
            " ",
            SCENARIO_TYPE_UI_OPTIONS,
            index=SCENARIO_TYPE_UI_OPTIONS.index(st.session_state.view_scenario_type),
            key="view_scenario_type",
            label_visibility="collapsed",
            help="Post-run filter: which scenario rows are ranked (same filter as the All scenario results table for this tariff).",
        )

        # ----------------------------
        # Decision constraints (hard filters)
        # ----------------------------
        st.markdown(
            '<div style="font-size:0.78rem;font-weight:600;color:#444;margin:0.35rem 0 0.15rem 0;">Decision constraints (hard filters)</div>',
            unsafe_allow_html=True,
        )
        c_cap1, c_cap2, c_strat = st.columns(3)

        with c_cap1:
            capex_max_en = st.checkbox("CAPEX max (€)", value=False, key="hard_capex_max_en")
            capex_max_eur = st.number_input(
                "CAPEX max (€)",
                value=0.0,
                min_value=0.0,
                step=1000.0,
                disabled=not capex_max_en,
                label_visibility="collapsed",
                key="hard_capex_max_eur",
            )
            payback_max_en = st.checkbox("Payback max (years)", value=False, key="hard_payback_max_en")
            payback_max_years = st.number_input(
                "Payback max (years)",
                value=0.0,
                min_value=0.0,
                step=0.5,
                disabled=not payback_max_en,
                label_visibility="collapsed",
                key="hard_payback_max_years",
            )

        with c_cap2:
            npv_min_en = st.checkbox("NPV min (€)", value=False, key="hard_npv_min_en")
            npv_min_eur = st.number_input(
                "NPV min (€)",
                value=0.0,
                min_value=-1_000_000_000.0,
                max_value=1_000_000_000.0,
                step=1000.0,
                disabled=not npv_min_en,
                label_visibility="collapsed",
                key="hard_npv_min_eur",
            )
            irr_min_en = st.checkbox("IRR min (%)", value=False, key="hard_irr_min_en")
            irr_min_pct = st.number_input(
                "IRR min (%)",
                value=0.0,
                min_value=-100.0,
                max_value=100.0,
                step=0.5,
                disabled=not irr_min_en,
                label_visibility="collapsed",
                key="hard_irr_min_pct",
            )

        with c_strat:
            ss_min_en = st.checkbox("Self-sufficiency min (%)", value=False, key="hard_ss_min_en")
            self_sufficiency_min_pct = st.number_input(
                "Self-sufficiency min (%)",
                value=0.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                disabled=not ss_min_en,
                label_visibility="collapsed",
                key="hard_ss_min_pct",
            )
            co2_min_en = st.checkbox("Annual CO2 savings min (kg)", value=False, key="hard_co2_min_en")
            annual_co2_savings_min_kg = st.number_input(
                "Annual CO2 savings min (kg)",
                value=0.0,
                min_value=0.0,
                step=100.0,
                disabled=not co2_min_en,
                label_visibility="collapsed",
                key="hard_co2_min_kg",
            )

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            ann_cost_max_en = st.checkbox("Annual electricity cost max (€)", value=False, key="hard_ann_cost_max_en")
            annual_electricity_cost_max_eur = st.number_input(
                "Annual electricity cost max (€)",
                value=0.0,
                min_value=0.0,
                step=1000.0,
                disabled=not ann_cost_max_en,
                label_visibility="collapsed",
                key="hard_ann_cost_max_eur",
            )
        with c_opt2:
            self_cons_min_en = st.checkbox("Self-consumption ratio min (%)", value=False, key="hard_self_cons_min_en")
            self_consumption_ratio_min_pct = st.number_input(
                "Self-consumption ratio min (%)",
                value=0.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                disabled=not self_cons_min_en,
                label_visibility="collapsed",
                key="hard_self_cons_min_pct",
            )

        # Build constraints dict; None means "no filtering".
        capex_max = float(capex_max_eur) if capex_max_en else None
        payback_max = float(payback_max_years) if payback_max_en else None
        npv_min = float(npv_min_eur) if npv_min_en else None
        irr_min = float(irr_min_pct) if irr_min_en else None
        ss_min = float(self_sufficiency_min_pct) if ss_min_en else None
        co2_min = float(annual_co2_savings_min_kg) if co2_min_en else None
        ann_cost_max = float(annual_electricity_cost_max_eur) if ann_cost_max_en else None
        scr_min = float(self_consumption_ratio_min_pct) if self_cons_min_en else None

        if st.session_state.view_goal == "Largest PV meeting self-consumption ratio >= X%":
            st.markdown(
                '<div style="font-size:0.78rem;font-weight:500;color:#444;margin:0.35rem 0 0.15rem 0;">'
                "Self-consumption ratio threshold (%)</div>",
                unsafe_allow_html=True,
            )
            st.slider(
                " ",
                min_value=0,
                max_value=100,
                step=1,
                key="view_goal5_threshold_pct",
                label_visibility="collapsed",
            )
            st.caption(
                "Ranking and table sort use **only** rows with self-consumption ≥ this value (among PV > 0), "
                "largest PV first. If none qualify, the app falls back to highest self-consumption, then largest PV."
            )

        goal = st.session_state.view_goal
        filter_tariff = st.session_state.view_filter_tariff
        scenario_type_ui = st.session_state.view_scenario_type
        goal5_threshold_pct = float(st.session_state.view_goal5_threshold_pct)

        chosen_tcol = tariff_map[filter_tariff]
        if st.session_state.full_results_df is None:
            st.session_state.full_results_df = build_full_scenario_results_df(
                st.session_state.opt_dfs,
                st.session_state.prepared_df,
                st.session_state.last_standing_charges,
                pv_cost_per_kwp=st.session_state.last_pv_capex,
                batt_cost_per_kwh=st.session_state.last_batt_capex,
                electricity_inflation_rate=st.session_state.last_electricity_inflation_rate,
                battery_replacement_year=st.session_state.last_battery_replacement_year,
                battery_replacement_pct_of_batt_capex=st.session_state.last_battery_replacement_cost_pct,
                inverter_replacement_year=st.session_state.last_inverter_replacement_year,
                inverter_replacement_pct_of_pv_capex=st.session_state.last_inverter_replacement_cost_pct,
                pso_levy_annual=float(st.session_state.last_pso_levy),
            )
        full_table_rank = st.session_state.full_results_df
        slice_rank = full_table_rank[full_table_rank["Tariff"] == filter_tariff].copy()
        slice_rank = _filter_by_scenario_type(slice_rank, scenario_type_ui)

        # Apply decision constraints before ranking.
        hard_filtered_rank_df = _apply_hard_filters_to_results_df(
            slice_rank,
            capex_max_eur=capex_max,
            payback_max_years=payback_max,
            npv_min_eur=npv_min,
            irr_min_pct=irr_min,
            self_sufficiency_min_pct=ss_min,
            annual_co2_savings_min_kg=co2_min,
            annual_electricity_cost_max_eur=ann_cost_max,
            self_consumption_ratio_min_pct=scr_min,
        )

        st.caption(f"Rows after filters: **{len(hard_filtered_rank_df):,}**")
        if len(hard_filtered_rank_df) == 0:
            st.warning("No scenarios meet the current constraints. Relax one or more Decision constraints filters.")
        elif payback_max is not None and "Payback (yrs)" in hard_filtered_rank_df.columns:
            # Small sanity check: ensure the filtered ranking universe respects the payback limit.
            pb_vals = pd.to_numeric(hard_filtered_rank_df["Payback (yrs)"], errors="coerce")
            pb_vals = pb_vals[pd.Series(np.isfinite(pb_vals), index=pb_vals.index)]
            if len(pb_vals) > 0:
                st.caption(
                    f"Max payback in filtered set: **{float(pb_vals.max()):.2f}** years (limit **{float(payback_max):.2f}**) "
                )

        demand_kwh_rank = float(st.session_state.prepared_df["consumption"].sum())
        ranked = _rank_scenarios_from_consolidated_table(
            hard_filtered_rank_df, goal, demand_kwh_rank, goal5_threshold_pct=goal5_threshold_pct
        )

        st.caption(
            "Best / 2nd / 3rd are selected from the hard-filtered rows (tariff + scenario type + Decision constraints), "
            "ordered by **Rank results by**."
        )
        if len(ranked) > 0:
            st.markdown(
                '<div style="font-size:0.78rem;font-weight:500;color:#444;margin:0 0 0.15rem 0;">Show ranked result</div>',
                unsafe_allow_html=True,
            )
            rank_pick_options = ["Best", "2nd best", "3rd best"][: min(3, len(ranked))]
            if st.session_state.get("view_rank_pick") not in rank_pick_options:
                st.session_state.view_rank_pick = rank_pick_options[0]
            st.selectbox(
                " ",
                rank_pick_options,
                key="view_rank_pick",
                label_visibility="collapsed",
                help="Best / 2nd / 3rd over filtered rows, ordered by Rank results by (same default sort as the table).",
            )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#666;margin-top:0.5rem;">No ranked rows for this scenario type.</div>',
                unsafe_allow_html=True,
            )

        esc_tariff = html.escape(str(filter_tariff))
        esc_goal = html.escape(str(goal))
        st.markdown(
            f'<h4 style="margin:0.75rem 0 0.25rem 0;font-size:1.05rem;font-weight:600;color:#111;line-height:1.3;">'
            f"Results for {esc_tariff} "
            f'<span style="font-weight:500;color:#555;font-size:0.95rem;">(Goal: {esc_goal})</span></h4>',
            unsafe_allow_html=True,
        )

        if len(ranked) == 0:
            st.info(
                "No scenarios match the current Scenario type (or the filtered table is empty). "
                "Try All scenarios or another type that includes computed rows."
            )
        else:
            rank_order = ["Best", "2nd best", "3rd best"]
            pick = st.session_state.get("view_rank_pick", "Best")
            if pick not in rank_order:
                pick = "Best"
            rank_idx = rank_order.index(pick)
            if rank_idx >= len(ranked):
                rank_idx = len(ranked) - 1
            scenario_name, cons_row = ranked[rank_idx]
            pv_kwp = int(cons_row["PV (kWp)"]) if "PV (kWp)" in cons_row.index else 0
            batt_kwh = int(cons_row["Battery (kWh)"]) if "Battery (kWh)" in cons_row.index else 0
            row, _ = metrics_and_hourly_for_scenario_at_sizes(
                st.session_state.prepared_df,
                chosen_tcol,
                scenario_name,
                pv_kwp,
                batt_kwh,
                st.session_state.active_export_rate,
                st.session_state.last_standing_charges[filter_tariff],
                float(st.session_state.last_pso_levy),
                float(st.session_state.last_opex_pct),
                st.session_state.active_discount_rate,
                st.session_state.last_pv_capex,
                st.session_state.last_batt_capex,
                st.session_state.last_electricity_inflation_rate,
                st.session_state.battery_settings,
                st.session_state.last_battery_replacement_year,
                float(st.session_state.last_battery_replacement_cost_pct),
                st.session_state.last_inverter_replacement_year,
                float(st.session_state.last_inverter_replacement_cost_pct),
            )

            # Keep finance KPI tile values aligned with the consolidated table values that hard filters use.
            # (Dispatch/hourly data is still recomputed; only the finance KPIs shown in tiles are overwritten.)
            if "Payback (yrs)" in cons_row.index:
                row["Payback period (years)"] = float(cons_row["Payback (yrs)"])
            if "NPV (€)" in cons_row.index:
                row["NPV (20y, €)"] = float(cons_row["NPV (€)"])
            if "IRR (%)" in cons_row.index:
                irrv = cons_row["IRR (%)"]
                row["IRR (20y, %)"] = float(irrv) if pd.notna(irrv) else row.get("IRR (20y, %)", float("nan"))

            win_line = html.escape(f"{scenario_name} · PV {pv_kwp} kWp · Battery {batt_kwh} kWh")
            st.markdown(
                f'<div style="background:linear-gradient(180deg,#e8f5e9 0%,#c8e6c9 100%);'
                f"border:2px solid #2e7d32;border-radius:10px;padding:12px 16px;margin:0.35rem 0 0.85rem 0;"
                f'box-shadow:0 1px 3px rgba(46,125,50,0.15);">'
                f'<div style="font-size:1.4rem;font-weight:700;color:#145a32;line-height:1.4;">{win_line}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            kpi_labels = [
                ("Annual grid import (kWh)", "Grid import (kWh)"),
                ("Annual PV generation (kWh)", "Total annual PV generation (kWh)"),
                ("Self-Consumption (kWh)", "Self-consumed PV (kWh)"),
                ("Export (kWh)", "Export to grid (kWh)"),
                ("Annual electricity cost (€)", "Annual cost (€)"),
                ("Annual export earnings (€)", "Export income (€)"),
                ("Annual savings vs grid only (€)", "Annual savings (€)"),
                ("Payback period (years)", "Payback period (years)"),
                ("NPV (20y, €)", "NPV (20y, €)"),
                ("IRR (20y, %)", "IRR (20y, %)"),
                ("Annual CO2 savings (kg)", "CO2 savings (kg)"),
                ("CO2 reduction (%)", "CO2 reduction (%)"),
                ("Gross savings over 20 years (€)", "Gross savings over 20 years (€)"),
                ("Net benefit over 20 years (€)", "Net benefit over 20 years (€)"),
                ("Self-sufficiency ratio (%)", "Self-sufficiency ratio (%)"),
                ("Self-consumption ratio (%)", "Self-consumption ratio (%)"),
            ]
            render_compact_kpi_tile_grid(row, kpi_labels)

            # Cumulative charts for the selected scenario row (same assumptions as the KPI block)
            # - Cumulative savings (€): inflation-applied annual savings accumulation (no CAPEX / replacements)
            # - Cumulative CO2 savings (kg): assume annual CO2 savings constant each year (no inflation)
            # - Cumulative discounted net cash flow (€): CAPEX at year 0 + discounted savings - discounted replacements
            years = np.arange(0, LIFETIME_YEARS + 1, dtype=int)  # 0..20
            t = years[1:]  # 1..20
            infl = float(st.session_state.last_electricity_inflation_rate)
            r = float(st.session_state.active_discount_rate)

            annual_savings_y1 = float(row.get("Annual savings (€)", 0.0))
            annual_co2_savings_y1 = float(row.get("CO2 savings (kg)", 0.0))
            capex = float(row.get("CAPEX (€)", 0.0))

            # Replacement outflows nominal (undiscounted); apply only if replacement year is within horizon.
            batt_repl_nominal = 0.0
            if st.session_state.last_battery_replacement_year is not None and 1 <= int(st.session_state.last_battery_replacement_year) <= LIFETIME_YEARS:
                batt_repl_nominal = (batt_kwh * float(st.session_state.last_batt_capex)) * (
                    float(st.session_state.last_battery_replacement_cost_pct) / 100.0
                )
            inv_repl_nominal = 0.0
            if st.session_state.last_inverter_replacement_year is not None and 1 <= int(st.session_state.last_inverter_replacement_year) <= LIFETIME_YEARS:
                inv_repl_nominal = (pv_kwp * float(st.session_state.last_pv_capex)) * (
                    float(st.session_state.last_inverter_replacement_cost_pct) / 100.0
                )

            # Cumulative operating savings (no discount, no replacements)
            if infl <= 0:
                savings_stream = annual_savings_y1 * np.ones_like(t, dtype=float)
            else:
                savings_stream = annual_savings_y1 * (1.0 + infl) ** (t - 1)
            cumulative_savings = np.concatenate([[0.0], np.cumsum(savings_stream)])

            # Cumulative CO2 savings (no inflation assumption)
            cumulative_co2 = annual_co2_savings_y1 * years.astype(float)

            # Discounted net cash flow accumulation
            cashflow_disc = np.zeros_like(years, dtype=float)
            cashflow_disc[0] = -capex
            disc_den = (1.0 + r) ** t if r != -1 else np.inf
            savings_disc = savings_stream / disc_den

            replacement_disc = np.zeros_like(t, dtype=float)
            if st.session_state.last_battery_replacement_year is not None:
                by = int(st.session_state.last_battery_replacement_year)
                if 1 <= by <= LIFETIME_YEARS:
                    replacement_disc += batt_repl_nominal * (by == t).astype(float) / disc_den
            if st.session_state.last_inverter_replacement_year is not None:
                iy = int(st.session_state.last_inverter_replacement_year)
                if 1 <= iy <= LIFETIME_YEARS:
                    replacement_disc += inv_repl_nominal * (iy == t).astype(float) / disc_den

            cashflow_disc[1:] = savings_disc - replacement_disc
            cumulative_disc_net_cashflow = np.cumsum(cashflow_disc)

            c_sav, c_co2, c_cash = st.columns(3)
            with c_sav:
                fig_sav = go.Figure()
                fig_sav.add_trace(
                    go.Scatter(x=years, y=cumulative_savings, mode="lines+markers", name="Cumulative savings")
                )
                fig_sav.update_layout(
                    title="Cumulative savings over time (€)",
                    margin=dict(l=8, r=8, t=36, b=8),
                    height=240,
                    xaxis_title="Year",
                    yaxis_title="€",
                    xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, LIFETIME_YEARS]),
                    yaxis=dict(rangemode="tozero"),
                    showlegend=False,
                )
                st.plotly_chart(fig_sav, use_container_width=True, config={"displayModeBar": False})

            with c_co2:
                fig_co2 = go.Figure()
                fig_co2.add_trace(
                    go.Scatter(x=years, y=cumulative_co2, mode="lines+markers", name="Cumulative CO2 savings")
                )
                fig_co2.update_layout(
                    title="Cumulative CO₂ savings over time (kg)",
                    margin=dict(l=8, r=8, t=36, b=8),
                    height=240,
                    xaxis_title="Year",
                    yaxis_title="kg",
                    xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, LIFETIME_YEARS]),
                    yaxis=dict(rangemode="tozero"),
                    showlegend=False,
                )
                st.plotly_chart(fig_co2, use_container_width=True, config={"displayModeBar": False})

            with c_cash:
                fig_cf = go.Figure()
                fig_cf.add_trace(
                    go.Scatter(
                        x=years,
                        y=cumulative_disc_net_cashflow,
                        mode="lines+markers",
                        name="Cumulative discounted net cash flow",
                    )
                )
                fig_cf.update_layout(
                    title="Cumulative discounted net cash flow (€)",
                    margin=dict(l=8, r=8, t=36, b=8),
                    height=240,
                    xaxis_title="Year",
                    yaxis_title="€",
                    xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, LIFETIME_YEARS]),
                    yaxis=dict(rangemode="tozero"),
                    showlegend=False,
                )
                st.plotly_chart(fig_cf, use_container_width=True, config={"displayModeBar": False})

        st.divider()
        st.subheader("All scenario results")
        st.caption(
            "Decision constraints + your selected tariff and Scenario type. Rows start sorted by **Rank results by**; "
            "use column headers to re-sort. Filter under headers—like Excel. Client-side only; does not rerun the model."
        )
        if st.session_state.full_results_df is None:
            st.session_state.full_results_df = build_full_scenario_results_df(
                st.session_state.opt_dfs,
                st.session_state.prepared_df,
                st.session_state.last_standing_charges,
                pv_cost_per_kwp=st.session_state.last_pv_capex,
                batt_cost_per_kwh=st.session_state.last_batt_capex,
                electricity_inflation_rate=st.session_state.last_electricity_inflation_rate,
                battery_replacement_year=st.session_state.last_battery_replacement_year,
                battery_replacement_pct_of_batt_capex=st.session_state.last_battery_replacement_cost_pct,
                inverter_replacement_year=st.session_state.last_inverter_replacement_year,
                inverter_replacement_pct_of_pv_capex=st.session_state.last_inverter_replacement_cost_pct,
                pso_levy_annual=float(st.session_state.last_pso_levy),
            )
        full_table_df = st.session_state.full_results_df
        if full_table_df is None or len(full_table_df) == 0:
            st.warning("No consolidated scenario table available.")
        else:
            # Keep the table aligned with the ranking universe.
            # hard_filtered_rank_df is defined in the Results Controls block.
            try:
                base_table_df = hard_filtered_rank_df
            except NameError:
                base_table_df = full_table_df

            n_all = len(base_table_df)
            demand_kwh = float(st.session_state.prepared_df["consumption"].sum())
            filtered_view = render_aggrid_results_table(
                base_table_df,
                grid_key="all_scenarios_aggrid",
                height=420,
                default_rank_goal=goal,
                rank_goal_table="full",
                annual_community_demand_kwh=demand_kwh,
                goal5_threshold_pct=goal5_threshold_pct,
            )
            n_f = len(filtered_view)
            st.caption(f"**{n_f:,}** / **{n_all:,}** rows visible after grid filters and column sort.")
            dlc1, dlc2 = st.columns(2)
            with dlc1:
                st.download_button(
                    "Download grid view (CSV)",
                    data=filtered_view.to_csv(index=False).encode("utf-8-sig") if n_f else b"\xef\xbb\xbf",
                    file_name="scenario_results_grid_view.csv",
                    mime="text/csv",
                    key="dl_scenarios_grid_csv",
                    disabled=n_f == 0,
                )
            with dlc2:
                st.download_button(
                    "Download full table (CSV)",
                    data=base_table_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="scenario_results_all.csv",
                    mime="text/csv",
                    key="dl_scenarios_all_csv",
                )

        include_pv, include_battery = _include_flags_for_scenario_type(scenario_type_ui)

        st.divider()
        st.subheader("All tariffs — comparison")
        compare_metric = goal_to_tariff_compare_chart_column(goal)
        st.caption(f"Grouped bars use **{compare_metric}**, matching **Rank results by**.")
        cmp_rows: List[pd.DataFrame] = []
        for tname, tcol in tariff_map.items():
            res_cmp, _ = evaluate_for_tariff(
                st.session_state.prepared_df,
                st.session_state.opt_dfs,
                tcol,
                tname,
                goal,
                include_pv=include_pv,
                include_battery=include_battery,
                goal5_threshold_pct=float(goal5_threshold_pct),
                battery_settings=st.session_state.battery_settings,
                export_rate=st.session_state.active_export_rate,
                standing_charge=st.session_state.last_standing_charges[tname],
                pso_levy_annual=float(st.session_state.last_pso_levy),
                opex_pct=float(st.session_state.last_opex_pct),
                discount_rate=st.session_state.active_discount_rate,
                pv_cost_per_kwp=st.session_state.last_pv_capex,
                batt_cost_per_kwh=st.session_state.last_batt_capex,
                electricity_inflation_rate=st.session_state.last_electricity_inflation_rate,
                battery_replacement_year=st.session_state.last_battery_replacement_year,
                battery_replacement_pct_of_batt_capex=st.session_state.last_battery_replacement_cost_pct,
                inverter_replacement_year=st.session_state.last_inverter_replacement_year,
                inverter_replacement_pct_of_pv_capex=st.session_state.last_inverter_replacement_cost_pct,
            )
            cmp_rows.append(_filter_by_scenario_type(res_cmp, scenario_type_ui))
        all_cmp_df = pd.concat(cmp_rows, ignore_index=True) if cmp_rows else pd.DataFrame()
        if len(all_cmp_df) == 0 or compare_metric not in all_cmp_df.columns:
            st.info("Nothing to chart for the current tariff / scenario-type settings.")
        else:
            fig_cmp = go.Figure()
            for tname, _ in tariff_map.items():
                sub = all_cmp_df[all_cmp_df["Tariff"] == tname]
                if len(sub) == 0:
                    continue
                yv = pd.to_numeric(sub[compare_metric], errors="coerce").to_numpy(dtype=float)
                yv = np.where(np.isfinite(yv), yv, np.nan)
                fig_cmp.add_trace(go.Bar(name=tname, x=sub["Scenario"], y=yv))
            fig_cmp.update_layout(
                barmode="group",
                title=compare_metric,
                xaxis_title="Scenario",
                yaxis_title=compare_metric,
                height=320,
                margin=dict(l=20, r=20, t=44, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

    with tab_consumption:
        st.subheader("Consumption patterns")
        st.caption(
            "Community **demand** only — how electricity is consumed over time. "
            "Uses the uploaded hourly consumption series; independent of PV, battery, or tariff scenarios."
        )
        render_community_consumption_patterns(st.session_state.prepared_df)

    with tab_pv_sweep:
        st.subheader("PV + Grid — all PV sizes (no battery)")
        pv_min_last = _pv_min_last
        pv_max_last = _pv_max_last
        st.caption(
            f"Table shows KPIs for the **PV + Grid** scenario only, for PV from **{pv_min_last} to {pv_max_last} kWp** in **1 kWp** steps. "
            "CAPEX uses your sidebar **PV CAPEX (€/kWp)** from the last Run."
        )
        sweep_default_idx = _tariff_opts.index(st.session_state.view_filter_tariff) if st.session_state.view_filter_tariff in _tariff_opts else 0
        sweep_tariff = st.selectbox(
            "Tariff",
            _tariff_opts,
            index=sweep_default_idx,
            key="pv_grid_sweep_tariff",
        )
        sweep_tcol = tariff_map[sweep_tariff]
        pv_cap = float(st.session_state.last_pv_capex)
        sweep_df = build_pv_grid_sweep_table(
            st.session_state.prepared_df,
            sweep_tcol,
            pv_cap,
            st.session_state.active_export_rate,
            standing_charge=st.session_state.last_standing_charges[sweep_tariff],
            opex_pct=float(st.session_state.last_opex_pct),
            discount_rate=st.session_state.active_discount_rate,
            electricity_inflation_rate=st.session_state.last_electricity_inflation_rate,
            inverter_replacement_year=st.session_state.last_inverter_replacement_year,
            inverter_replacement_pct_of_pv_capex=st.session_state.last_inverter_replacement_cost_pct,
            pso_levy_annual=float(st.session_state.last_pso_levy),
            pv_min=pv_min_last,
            pv_max=pv_max_last,
        )
        sweep_base = sweep_df.copy()

        st.caption(
            f"**{len(sweep_base):,}** rows ({pv_min_last}–{pv_max_last} kWp, 1 kWp steps). "
            "Use column header filters (Ag Grid), same idea as Excel."
        )
        sweep_goal = st.session_state.view_goal

        demand_kwh_sweep = float(st.session_state.prepared_df["consumption"].sum())
        grid_sweep = render_aggrid_results_table(
            sweep_base,
            grid_key="pv_grid_sweep_aggrid",
            height=380,
            default_rank_goal=sweep_goal,
            rank_goal_table="sweep",
            annual_community_demand_kwh=demand_kwh_sweep,
            goal5_threshold_pct=float(st.session_state.view_goal5_threshold_pct),
        )
        n_sweep_vis = len(grid_sweep)
        st.caption(f"**{n_sweep_vis:,}** / **{len(sweep_base):,}** rows visible after grid filters.")
        sw_dl1, sw_dl2 = st.columns(2)
        with sw_dl1:
            st.download_button(
                "Download grid view (CSV)",
                data=grid_sweep.to_csv(index=False).encode("utf-8-sig") if n_sweep_vis else b"\xef\xbb\xbf",
                file_name="pv_grid_sweep_grid_view.csv",
                mime="text/csv",
                key="pv_sweep_dl_grid",
                disabled=n_sweep_vis == 0,
            )
        with sw_dl2:
            st.download_button(
                "Download sampled table (CSV)",
                data=sweep_base.to_csv(index=False).encode("utf-8-sig"),
                file_name="pv_grid_sweep_sampled.csv",
                mime="text/csv",
                key="pv_sweep_dl_sampled",
            )

        st.subheader("KPI vs PV size")
        st.caption(
            f"One line chart per KPI vs **PV (kWp)** — full sweep ({pv_min_last}–{pv_max_last} kWp), not affected by table filters. "
            "Y-axis always includes **zero** where Plotly allows; dashed line = grid-only baseline where applicable."
        )
        d_base = run_scenario_grid_only(st.session_state.prepared_df, sweep_tcol)
        k_base = compute_kpis_for_scenario(d_base, sweep_tcol, st.session_state.active_export_rate)
        total_cons = float(st.session_state.prepared_df["consumption"].sum())
        baseline_cost = float(np.sum(d_base["grid_import"].to_numpy() * st.session_state.prepared_df[sweep_tcol].to_numpy()))
        baseline_map = {
            "Grid import (kWh)": total_cons,
            "Annual PV generation (kWh)": 0.0,
            "Self-Consumption (kWh)": 0.0,
            "Export (kWh)": 0.0,
            "Annual cost of grid import (€)": k_base["Cost of grid import (€)"],
            "Annual electricity cost (€)": baseline_cost
            + st.session_state.last_standing_charges[sweep_tariff]
            + float(st.session_state.last_pso_levy),
            "Annual export earnings (€)": 0.0,
            "Annual savings vs grid only (€)": 0.0,
            "CAPEX (€)": 0.0,
            "Payback period (years)": np.nan,
            "NPV (20y, €)": 0.0,
            "IRR (20y, %)": np.nan,
            "Self-sufficiency ratio (%)": 0.0,
            "Self-consumption ratio (%)": 0.0,
        }
        plot_sweep = sweep_base
        if len(plot_sweep) == 0:
            st.info("No rows to plot.")
        else:
            cols_to_plot = [c for c in PV_GRID_SWEEP_CHART_COLUMNS if c in plot_sweep.columns]
            for i in range(0, len(cols_to_plot), 2):
                c_left, c_right = st.columns(2)
                for col_container, chart_choice in zip(
                    (c_left, c_right),
                    cols_to_plot[i : i + 2],
                ):
                    with col_container:
                        baseline_val = baseline_map.get(chart_choice, np.nan)
                        y_vals = pd.to_numeric(plot_sweep[chart_choice], errors="coerce").to_numpy(dtype=float)
                        y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)
                        x_vals = pd.to_numeric(plot_sweep["PV (kWp)"], errors="coerce").to_numpy(dtype=float)
                        fig_sweep = go.Figure()
                        fig_sweep.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="PV + Grid"))
                        if np.isfinite(baseline_val):
                            fig_sweep.add_hline(
                                y=baseline_val,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Grid-only baseline",
                            )
                        fig_sweep.update_layout(
                            title=f"{chart_choice} vs PV size",
                            xaxis_title="PV (kWp)",
                            yaxis_title=chart_choice,
                            margin=dict(l=8, r=8, t=36, b=8),
                            height=240,
                            yaxis=dict(rangemode="tozero"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_sweep, use_container_width=True, config={"displayModeBar": False})

    with tab_explainer:
        render_settings_kpi_guide_tab()

