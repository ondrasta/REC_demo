from __future__ import annotations

import base64
import colorsys
import hashlib
import html
import io
import json
import os
import re
import time
import warnings
import datetime
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from saved_run_bundle import (
    BUNDLE_SCHEMA_VERSION,
    battery_settings_to_json_dict,
    build_saved_run_zip_bytes,
    load_bundle_from_zip,
    read_manifest_from_zip,
)
import plotly.graph_objects as go
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder

from bundled_research import (
    RESEARCH_WINNER_RULES,
    build_all_winners_summary_df,
    build_research_display_dataframe,
    format_research_display_dataframe,
    load_bundled_research_xlsx,
    research_metric_grouped_bars,
)


# ----------------------------
# Constants / assumptions (immutable defaults)
# ----------------------------
DEFAULT_CO2_FACTOR = 0.2462  # kg CO₂ per kWh grid import — default in Model setup


def _grid_co2_factor() -> float:
    """Emission factor for grid electricity imports (kg CO₂/kWh); follows **last Run analysis** or default."""
    try:
        return float(st.session_state.last_co2_factor)
    except Exception:
        return float(DEFAULT_CO2_FACTOR)


CO2_FACTOR = DEFAULT_CO2_FACTOR  # alias for tests and backward compatibility
PV_COST_PER_KWP = 1000
BATT_COST_PER_KWH = 300
DISCOUNT_RATE = 0.035
DEFAULT_LIFETIME_YEARS = 20


def col_npv(lifetime_years: int) -> str:
    return f"NPV ({lifetime_years}y, €)"


def col_irr(lifetime_years: int) -> str:
    return f"IRR ({lifetime_years}y, %)"


def col_gross_savings(lifetime_years: int) -> str:
    return f"Gross savings over {lifetime_years} years (€)"


def col_net_benefit(lifetime_years: int) -> str:
    return f"Net benefit over {lifetime_years} years (€)"


def per_capex_ratio_column_names(lifetime_years: int) -> tuple[str, ...]:
    """Consolidated-table column names for NPV / CO₂ / savings per € CAPEX (same ``ly`` as NPV / gross savings)."""
    ly = int(lifetime_years)
    return (
        f"NPV per € CAPEX ({ly}y, €/€)",
        "Annual CO2 reduction per € CAPEX (kg/€)",
        f"Lifetime CO2 avoided per € CAPEX ({ly}y, kg/€)",
        "Annual savings per € CAPEX (€/€)",
        f"Gross savings per € CAPEX ({ly}y, €/€)",
    )


ELECTRICITY_INFLATION_RATE = 0.0  # default 0% per year (decimal form for formulas)

DEFAULT_OPEX_PCT = 1.0  # % of scenario CAPEX — UI default

DEFAULT_EXPORT_RATE = 0.1886  # €/kWh (flat export for all tariffs)

# Default annual standing charges (€/year) — UI starting values
DEFAULT_STANDING_CHARGE_STANDARD_EUR = 286.60  # Standard (smart meter)
DEFAULT_STANDING_CHARGE_WEEKEND_EUR = 338.85  # Weekend Saver
DEFAULT_STANDING_CHARGE_FLAT_EUR = 286.60  # Flat rate (aligned with Standard)

DEFAULT_PSO_LEVY_EUR_PER_YEAR = 19.10  # annual PSO levy (€), same for all tariffs; escalates with electricity inflation in long-run metrics

DEFAULT_BATTERY_REPLACEMENT_YEAR = 10  # calendar year in horizon; 0 in UI = none
DEFAULT_INVERTER_REPLACEMENT_YEAR = 0
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

# User-facing consolidated results / rankings: full year-1 bill (import − export + standing + PSO + OPEX).
# All-scenario Ag Grid column name; detail tiles still use ``COL_ANNUAL_ELECTRICITY_COST_EUR``.
COL_ANNUAL_ELECTRICITY_BILL_EUR = "Annual electricity bill (€)"

# Recommended setups Ag Grid: show lifetime NPV and year-1 savings as whole euros (no decimals).
RECOMMENDED_SETUPS_AGGRID_INTEGER_NUMERIC_COLS = frozenset({"NPV (€)", "Annual savings (€)"})
COL_ANNUAL_ELECTRICITY_COST_EUR = "Annual electricity cost (€)"
# Per-scenario energy balance from hourly sim only (no standing charge, PSO, or OPEX).
COL_NET_IMPORT_EXPORT_COST_EUR = "Net import/export cost (€)"
# All-scenario results grid: energy / carbon from the same hourly dispatch as the optimizer row.
COL_GRID_IMPORT_KWH = "Grid import (kWh)"
COL_EXPORT_TO_GRID_KWH = "Export to grid (kWh)"
COL_SELF_CONSUMED_PV_KWH = "Self-consumed PV (kWh)"
COL_BATTERY_CHARGE_KWH = "Battery charge (kWh)"
COL_BATTERY_DISCHARGE_KWH = "Battery discharge (kWh)"
COL_ANNUAL_GRID_IMPORT_COST_EUR = "Annual grid import cost (€)"
# Scenario grid-import CO₂ for the simulated year (not savings vs baseline).
COL_ANNUAL_GRID_CO2_EMISSIONS_KG = "Annual grid CO₂ emissions (kg)"
COL_LIFETIME_CO2_KG = "Lifetime CO2 (kg)"
COL_ANNUAL_CO2_REDUCTION_KG = "Annual CO2 reduction (kg)"
# Year-1 savings as % of tariff-specific grid-only annual bill (consolidated grid).
COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT = "Annual electricity bill reduction (%)"
# Sidebar: warn if CAPEX max is enabled and threshold looks like “thousands” confusion (e.g. 21 vs 21000).
DECISION_CONSTRAINT_CAPEX_WARN_BELOW_EUR = 500.0


def _perf_profiling_enabled() -> bool:
    """Set ``REC_FEASIBILITY_PROFILE=1`` (or ``true`` / ``yes``) to record section timings in session state."""
    v = os.environ.get("REC_FEASIBILITY_PROFILE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _perf_record(name: str, seconds: float) -> None:
    if not _perf_profiling_enabled():
        return
    st.session_state["_perf_last_section"] = name
    st.session_state["_perf_last_section_s"] = float(seconds)
    log = st.session_state.setdefault("_perf_log", [])
    log.append((name, float(seconds)))
    if len(log) > 80:
        del log[: len(log) - 80]


def _scenario_explorer_core_display_columns(lifetime_years: int = DEFAULT_LIFETIME_YEARS) -> list[str]:
    """Subset of consolidated columns for a readable default grid (sort still uses full data first)."""
    ly = int(lifetime_years)
    return [
        SCENARIO_ROW_KEY_FIELD,
        "Tariff",
        "Scenario",
        "PV (kWp)",
        "Battery (kWh)",
        COL_ANNUAL_ELECTRICITY_BILL_EUR,
        "Annual savings (€)",
        COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT,
        "Payback (yrs)",
        "NPV (€)",
        "IRR (%)",
        "CAPEX (€)",
        COL_ANNUAL_CO2_REDUCTION_KG,
        "CO2 reduction (%)",
        "Self-sufficiency (%)",
        "Self-consumption ratio (%)",
        "Export ratio (% of PV gen)",
        "Total annual PV generation (kWh)",
        COL_GRID_IMPORT_KWH,
        COL_BATTERY_CHARGE_KWH,
        COL_BATTERY_DISCHARGE_KWH,
        COL_ANNUAL_GRID_IMPORT_COST_EUR,
        *list(per_capex_ratio_column_names(ly)),
    ]


def _subset_dataframe_display_columns(df: pd.DataFrame, allowlist: list[str]) -> pd.DataFrame:
    cols = [c for c in allowlist if c in df.columns]
    if not cols:
        return df
    return df.loc[:, cols].copy()


def _apply_full_results_explorer_table_filters(
    df: pd.DataFrame,
    *,
    tariff_pick: list[str],
    tariff_universe: list[str],
    scenario_pick: list[str],
    scenario_universe: list[str],
    search_text: str,
    pv_min: float,
    pv_max: float,
    batt_min: float,
    batt_max: float,
    npv_min: float,
    npv_max: float,
    payback_min: float,
    payback_max: float,
) -> pd.DataFrame:
    """Sidebar-hard-filtered consolidated table → further client filters for **Full results** (no AgGrid required)."""
    if df is None or len(df) == 0:
        return df
    out = df
    tp = tariff_pick if tariff_pick else tariff_universe
    sp = scenario_pick if scenario_pick else scenario_universe
    if "Tariff" in out.columns and tp:
        out = out[out["Tariff"].astype(str).isin(tp)]
    if "Scenario" in out.columns and sp:
        out = out[out["Scenario"].astype(str).isin(sp)]
    if "PV (kWp)" in out.columns and np.isfinite(pv_min) and np.isfinite(pv_max) and pv_max >= pv_min:
        _pv = pd.to_numeric(out["PV (kWp)"], errors="coerce")
        out = out[(_pv >= float(pv_min)) & (_pv <= float(pv_max))]
    if "Battery (kWh)" in out.columns and np.isfinite(batt_min) and np.isfinite(batt_max) and batt_max >= batt_min:
        _bh = pd.to_numeric(out["Battery (kWh)"], errors="coerce")
        out = out[(_bh >= float(batt_min)) & (_bh <= float(batt_max))]
    if "NPV (€)" in out.columns and np.isfinite(npv_min) and np.isfinite(npv_max) and npv_max >= npv_min:
        _np = pd.to_numeric(out["NPV (€)"], errors="coerce")
        out = out[(_np >= float(npv_min)) & (_np <= float(npv_max))]
    if "Payback (yrs)" in out.columns and np.isfinite(payback_min) and np.isfinite(payback_max) and payback_max >= payback_min:
        _pb = pd.to_numeric(out["Payback (yrs)"], errors="coerce")
        out = out[(_pb >= float(payback_min)) & (_pb <= float(payback_max))]
    q = (search_text or "").strip()
    if q and len(out) > 0:
        ql = q.lower()

        def _row_hits(r: pd.Series) -> bool:
            for v in r.values:
                try:
                    if ql in str(v).lower():
                        return True
                except Exception:
                    continue
            return False

        out = out[out.apply(_row_hits, axis=1)]
    return out


def _numeric_column_bounds(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Finite min/max for slider defaults on a numeric column."""
    if df is None or len(df) == 0 or col not in df.columns:
        return (0.0, 0.0)
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return (0.0, 0.0)
    return (float(s.min()), float(s.max()))


def _assumption_value_column_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Arrow-serialize Setting/Value tables: mixed numeric + text in ``Value`` must not infer float dtype."""
    if df is None or len(df) == 0 or "Value" not in df.columns:
        return df
    out = df.copy()
    out["Value"] = out["Value"].map(
        lambda v: ""
        if v is None or (isinstance(v, float) and pd.isna(v))
        else str(v)
    )
    return out


def render_sidebar_performance_panel() -> None:
    """Show last measured timings when ``REC_FEASIBILITY_PROFILE`` is set."""
    if not _perf_profiling_enabled():
        return
    st.sidebar.markdown("##### Performance (debug)")
    st.sidebar.caption("Env: **REC_FEASIBILITY_PROFILE=1**. Times are seconds (last full script run).")
    rows: list[dict[str, object]] = []
    for key, label in [
        ("_perf_load_data_s", "Load & prepare data"),
        ("_perf_optimizer_total_s", "Optimizer (all tariffs)"),
        ("_perf_build_full_s", "build_full_scenario_results_df"),
        ("_perf_all_tariffs_eval_s", "All-tariffs evaluate_for_tariff loop"),
    ]:
        if key in st.session_state and st.session_state[key] is not None:
            rows.append({"Section": label, "Seconds": float(st.session_state[key])})
    by_t = st.session_state.get("_perf_optimizer_by_tariff")
    if isinstance(by_t, dict) and by_t:
        slow = sorted(by_t.items(), key=lambda x: -float(x[1]))[:8]
        for tcol, sec in slow:
            rows.append({"Section": f"  · optimize ({tcol})", "Seconds": float(sec)})
    if rows:
        st.sidebar.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.sidebar.caption("No timings yet — run **Run analysis** or open results after setting the env var.")
    log = st.session_state.get("_perf_log")
    if isinstance(log, list) and log:
        with st.sidebar.expander("Raw perf log (latest first)", expanded=False):
            for name, sec in reversed(log[-25:]):
                st.text(f"{sec:,.3f}s  {name}")


def _df_bill_column(df: pd.DataFrame | pd.Series | None) -> str:
    """Consolidated All-scenarios table uses ``COL_ANNUAL_ELECTRICITY_BILL_EUR``; detail tiles use cost."""
    if df is None:
        return COL_ANNUAL_ELECTRICITY_COST_EUR
    keys = df.columns if isinstance(df, pd.DataFrame) else df.index
    if COL_ANNUAL_ELECTRICITY_BILL_EUR in keys:
        return COL_ANNUAL_ELECTRICITY_BILL_EUR
    return COL_ANNUAL_ELECTRICITY_COST_EUR


def _df_co2_avoided_column(df: pd.DataFrame | pd.Series | None) -> str:
    """Consolidated table exposes ``COL_ANNUAL_CO2_REDUCTION_KG``; other frames keep ``CO2 savings (kg)``."""
    if df is None:
        return "CO2 savings (kg)"
    keys = df.columns if isinstance(df, pd.DataFrame) else df.index
    if COL_ANNUAL_CO2_REDUCTION_KG in keys:
        return COL_ANNUAL_CO2_REDUCTION_KG
    return "CO2 savings (kg)"


def _annual_co2_savings_kg_from_consolidated_row(row: pd.Series) -> float:
    """Year-1 CO₂ avoided (kg) — same field as KPI tiles (consolidated column name varies)."""
    col = _df_co2_avoided_column(row)
    v = row.get(col)
    if v is None or (isinstance(v, (float, np.floating)) and pd.isna(v)):
        v = row.get("CO2 savings (kg)", 0.0)
    try:
        out = float(pd.to_numeric(v, errors="coerce"))
        return float(out) if np.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


# Plotly export / toolbar config for all charts
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "rec_chart",
        "scale": 2,
    },
}

# Night / Day / Peak — hourly band bars and band-share pies (consumption, production, scenario detail).
# Saturated mid-tones so Night (green) and Day (blue) stay distinguishable; very dark navy + forest read as one.
TIME_BAND_CHART_COLORS: Dict[str, str] = {
    "Night": "#16a34a",
    "Day": "#2563eb",
    "Peak": "#c2410c",
}

# Recommended setups — monthly “notebook” style charts (load / PV / dispatch colours; not time-band semantics).
_MONTH_ABB: List[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_NOTEBOOK_CHART_LOAD_BLUE = "#2563eb"
_NOTEBOOK_CHART_PV_ORANGE = "#ea580c"
_NOTEBOOK_CHART_STACK_EXPORT = "#16a34a"
_NOTEBOOK_CHART_STACK_SELF = "#2563eb"
_NOTEBOOK_CHART_STACK_GRID = "#ea580c"


def _format_bar_value_label_two_decimals(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{float(v):,.2f}"


def _apply_bar_chart_value_labels(fig: go.Figure) -> None:
    """Attach visible **y** labels on ``Bar`` traces only (two decimal places); other trace types unchanged."""
    if fig is None or not getattr(fig, "data", None):
        return
    for tr in fig.data:
        if type(tr).__name__ != "Bar":
            continue
        y_raw = getattr(tr, "y", None)
        if y_raw is None:
            continue
        texts: list[str] = []
        for v in list(y_raw):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                texts.append("")
                continue
            texts.append(_format_bar_value_label_two_decimals(fv))
        # Plotly uses outsidetextfont / insidetextfont for bar labels, not only textfont; omitting them
        # lets the library auto-scale, so labels look different sizes across bars and traces.
        _bar_lbl_font = dict(size=10, color="#334155")
        tr.update(
            text=texts,
            textposition="outside",
            textfont=_bar_lbl_font,
            outsidetextfont=_bar_lbl_font,
            insidetextfont=_bar_lbl_font,
            cliponaxis=False,
        )


def render_plotly_figure(
    fig: go.Figure,
    *,
    use_container_width: bool = True,
    key: str | None = None,
    apply_bar_value_labels: bool = False,
) -> None:
    if apply_bar_value_labels:
        _apply_bar_chart_value_labels(fig)
    kw: dict[str, object] = {"config": PLOTLY_CONFIG}
    if use_container_width:
        kw["width"] = "stretch"
    if key is not None:
        kw["key"] = key
    st.plotly_chart(fig, **kw)


# Set True to show battery setup inputs, scenario types, and battery rows/columns in the Streamlit UI.
ENABLE_BATTERY_UI = True


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
    """Build active tariff config from model setup tariff inputs. Never mutates globals."""
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


def _tariff_profiles_cache_key(tariff_profiles: List[Dict]) -> str:
    """Stable cache key from a list of tariff profiles."""
    payload = json.dumps(tariff_profiles, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _apply_yaxis_range_from_values(
    fig: go.Figure,
    y_values: object,
    *,
    pad_frac: float = 0.08,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """
    Ensure the y-axis range and tick marks include the plotted data.
    Only touches the y-axis range/ticks (no styling changes).
    """
    try:
        arr = np.asarray(y_values, dtype=float)
    except Exception:
        return
    if arr.size == 0:
        return
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return

    # Add headroom, then round to "nice" bounds so tick labels don't stop below the max bar/line.
    pad = float(pad_frac)
    raw_top = y_max + (abs(y_max) * pad)
    raw_bottom = y_min - (abs(y_min) * pad)

    if y_min >= 0:
        raw_bottom = 0.0
    if y_max <= 0:
        raw_top = 0.0

    span = float(raw_top - raw_bottom)
    if not np.isfinite(span) or span <= 0:
        return

    # Choose a "nice" dtick aiming for ~5 ticks.
    rough = span / 5.0
    mag = 10 ** np.floor(np.log10(max(rough, 1e-12)))
    for m in (1.0, 2.0, 5.0, 10.0):
        dtick = float(m * mag)
        if dtick >= rough:
            break

    if y_min >= 0:
        bottom = 0.0
    else:
        bottom = float(np.floor(raw_bottom / dtick) * dtick)
    top = float(np.ceil(raw_top / dtick) * dtick)

    yaxis_kw: dict[str, object] = {"range": [bottom, top], "tick0": bottom, "dtick": dtick}
    if row is not None and col is not None:
        yaxis_kw["row"] = row
        yaxis_kw["col"] = col
    fig.update_yaxes(**yaxis_kw)

    # Plotly often omits the gridline at the axis boundary; draw it explicitly at the top tick.
    try:
        grid_color = fig.layout.yaxis.gridcolor or "#e5e7eb"
    except Exception:
        grid_color = "#e5e7eb"
    hline_kw: dict[str, object] = dict(y=top, line_width=1, line_color=grid_color, layer="below")
    if row is not None and col is not None:
        hline_kw["row"] = row
        hline_kw["col"] = col
    fig.add_hline(**hline_kw)


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
    grid_co2_factor_kg_per_kwh: float,
    rt_eff_pct: float,
    dod_pct: float,
    init_soc_pct: float,
    min_soc_pct: float,
    max_soc_pct: float,
    c_rate: float,
    charge_from_pv: bool,
    charge_from_grid_at_night: bool,
    discharge_schedule: str,
    tariff_variant_count: int,
    tariff_matrix_source_last_run: str,
) -> List[Dict]:
    """Snapshot of model setup inputs for display (e.g. last run assumptions table)."""
    sc = standing_charges or {}
    if len(sc) == 0:
        sc_line = "—"
    else:
        parts = [f"{k} €{float(v):,.0f}/y" for k, v in sc.items()]
        sc_line = ", ".join(parts)
    return [
        {"Setting": "PV CAPEX (€/kWp)", "Value": float(pv_capex)},
        {"Setting": "Battery CAPEX (€/kWh)", "Value": float(batt_capex)},
        {"Setting": "OPEX (% of CAPEX)", "Value": float(opex_pct)},
        {"Setting": "PSO levy (annual, €)", "Value": float(pso_levy_annual)},
        {"Setting": "Standing charges (annual)", "Value": sc_line},
        {"Setting": "Discount rate (%)", "Value": round(float(discount_rate_pct), 4)},
        {"Setting": "Electricity inflation (%/y)", "Value": round(float(electricity_inflation_pct), 4)},
        {"Setting": "Grid CO₂ factor (kg/kWh)", "Value": round(float(grid_co2_factor_kg_per_kwh), 4)},
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
        {"Setting": "Min SOC (%)", "Value": float(min_soc_pct)},
        {"Setting": "Max SOC (%)", "Value": float(max_soc_pct)},
        {"Setting": "C-rate", "Value": float(c_rate)},
        {"Setting": "Battery dispatch", "Value": str(discharge_schedule)},
        {"Setting": "Charge from PV", "Value": bool(charge_from_pv)},
        {"Setting": "Charge from grid at night", "Value": bool(charge_from_grid_at_night)},
        {
            "Setting": "Tariff matrix (last run)",
            "Value": (
                f"{int(tariff_variant_count)} variant(s); source: {tariff_matrix_source_last_run}"
                if str(tariff_matrix_source_last_run).strip()
                else f"{int(tariff_variant_count)} variant(s)"
            ),
        },
    ]


_ASSUMPTION_OVERVIEW_SETTINGS_HIDDEN_WITHOUT_BATTERY_UI = frozenset(
    {
        "Battery CAPEX (€/kWh)",
        "Battery replacement year",
        "Battery replacement (% batt CAPEX)",
        "Battery range (kWh)",
        "Round-trip efficiency (%)",
        "DoD (%)",
        "Initial SOC (%)",
        "Min SOC (%)",
        "Max SOC (%)",
        "C-rate",
        "Battery dispatch",
        "Charge from PV",
        "Charge from grid at night",
    }
)


def visible_assumption_overview_rows(rows: List[Dict], *, opt_pv_step: int) -> List[Dict]:
    """Drop battery-only assumption lines from the last-run table when ENABLE_BATTERY_UI is False."""
    if ENABLE_BATTERY_UI:
        return rows
    out: List[Dict] = []
    for r in rows:
        s = r.get("Setting", "")
        if s in _ASSUMPTION_OVERVIEW_SETTINGS_HIDDEN_WITHOUT_BATTERY_UI:
            continue
        if s == "Optimizer steps (PV / battery kWh)":
            out.append({"Setting": "Optimizer step (PV kWp)", "Value": int(opt_pv_step)})
            continue
        out.append(r)
    return out


def last_run_assumptions_snapshot_df() -> Optional[pd.DataFrame]:
    """Two-column snapshot (Setting / Value) for the last completed **Run analysis**, or ``None`` if none."""
    if st.session_state.get("opt_dfs") is None:
        return None
    profiles = list(st.session_state.get("last_tariff_profiles") or _default_tariff_profiles())
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
        standing_charges={str(p.get("name")): float(p.get("standing_charge", 0.0) or 0.0) for p in profiles},
        pso_levy_annual=float(st.session_state.last_pso_levy),
        grid_co2_factor_kg_per_kwh=float(st.session_state.last_co2_factor),
        rt_eff_pct=float(bs.eff_round_trip) * 100.0,
        dod_pct=float(bs.dod) * 100.0,
        init_soc_pct=float(bs.init_soc) * 100.0,
        min_soc_pct=float(getattr(bs, "min_soc", 0.0)) * 100.0,
        max_soc_pct=float(getattr(bs, "max_soc", 1.0)) * 100.0,
        c_rate=float(bs.c_rate),
        charge_from_pv=bool(bs.charge_from_pv),
        charge_from_grid_at_night=bool(bs.charge_from_grid_at_night),
        discharge_schedule=str(bs.discharge_schedule),
        tariff_variant_count=len(profiles),
        tariff_matrix_source_last_run=str(st.session_state.last_tariff_matrix_source_label or "").strip(),
    )
    rows = visible_assumption_overview_rows(rows, opt_pv_step=int(loc["pv_step"]))
    return _assumption_value_column_to_string(pd.DataFrame(rows))


def encode_csv_assumptions_block_then_results_df(
    assumptions_df: Optional[pd.DataFrame],
    results_df: pd.DataFrame,
) -> bytes:
    """
    UTF-8-SIG text file: assumptions as a **Setting**,**Value** CSV table, one blank line, then the results CSV.
    If ``assumptions_df`` is missing or empty, export results only (same as a plain results CSV).
    """
    if assumptions_df is None or len(assumptions_df) == 0:
        return results_df.to_csv(index=False).encode("utf-8-sig")
    buf_a = io.StringIO()
    assumptions_df.to_csv(buf_a, index=False, lineterminator="\n")
    a = buf_a.getvalue().rstrip("\n")
    buf_b = io.StringIO()
    results_df.to_csv(buf_b, index=False, lineterminator="\n")
    b = buf_b.getvalue()
    return (a + "\n\n" + b).encode("utf-8-sig")


def _export_results_df_for_csv(results_df: pd.DataFrame) -> pd.DataFrame:
    """Remove internal-only columns before writing scenario results to CSV."""
    if results_df is None or len(results_df) == 0:
        return results_df
    out = results_df
    if SCENARIO_ROW_KEY_FIELD in out.columns:
        out = out.drop(columns=[SCENARIO_ROW_KEY_FIELD])
    return out


# Recommended setups — **Recommendation preset** ids/labels (shared feasible set; ranking differs by preset).
RECOMMENDED_WINNER_PRESET_DEFAULT = "balanced"
RECOMMENDED_WINNER_PRESETS: tuple[tuple[str, str], ...] = (
    ("balanced", "Balanced recommendation"),
    ("financial", "Best financial value"),
    ("lowest_bill", "Lowest annual bill"),
    ("fast_payback", "Fastest payback"),
    ("highest_co2", "Highest CO₂ saving"),
    ("highest_scr", "Highest self-consumption"),
)
RECOMMENDED_WINNER_PRESET_IDS: tuple[str, ...] = tuple(p[0] for p in RECOMMENDED_WINNER_PRESETS)
RECOMMENDED_WINNER_PRESET_LABEL_BY_ID: dict[str, str] = {p[0]: p[1] for p in RECOMMENDED_WINNER_PRESETS}
RECOMMENDED_WINNER_PRESET_ID_BY_LABEL: dict[str, str] = {p[1]: p[0] for p in RECOMMENDED_WINNER_PRESETS}

# Migrate saved sessions that used pre–v1 **Rank results by** strings.
_LEGACY_RANK_GOAL_TO_PRESET_LABEL: dict[str, str] = {
    "Lowest annual electricity cost": "Lowest annual bill",
    "Highest annual savings": "Best financial value",
    "Best payback": "Fastest payback",
    "Best self-sufficiency / lowest grid import": "Highest CO₂ saving",
    "Highest annual CO2 savings": "Highest CO₂ saving",
    "Best cost–CO2 trade-off": "Balanced recommendation",
    "Best NPV": "Balanced recommendation",
    "Best IRR": "Best financial value",
    "Largest PV meeting self-consumption ratio >= X%": "Balanced recommendation",
    "PV size closest to annual community demand": "Balanced recommendation",
}

RECOMMENDED_WINNER_PRESET_HELP = """
Pick how the app chooses **one winner** per tariff × scenario among rows that already pass **Decision constraints** (same feasible set for every preset).

**Balanced recommendation** — After NPV (highest first), tie-break **CO₂ savings (kg)**. **Best financial value** — After NPV, tie-break **annual savings (€)** first instead. Then payback, bill, and CAPEX as below.

**Balanced recommendation:** 1. NPV (€) ↑ 2. CO₂ savings (kg) ↑ 3. SCR (%) ↑ 4. Annual savings (€) ↑ 5. CAPEX (€) ↓

**Best financial value:** 1. NPV (€) ↑ 2. Annual savings (€) ↑ 3. Payback (years) ↓ 4. Annual electricity bill (€) ↓ 5. CAPEX (€) ↓

**Lowest annual bill:** 1. Bill (€) ↓ 2. Annual savings (€) ↑ 3. NPV (€) ↑ 4. CO₂ savings (kg) ↑ 5. CAPEX (€) ↓

**Fastest payback:** 1. Payback (years) ↓ 2. NPV (€) ↑ 3. Annual savings (€) ↑ 4. Bill (€) ↓ 5. CAPEX (€) ↓

**Highest CO₂ saving:** 1. CO₂ savings (kg) ↑ 2. SSR (%) ↑ 3. NPV (€) ↑ 4. SCR (%) ↑ 5. CAPEX (€) ↓

**Highest self-consumption:** 1. SCR (%) ↑ 2. Export ratio (% of PV gen) ↓ 3. NPV (€) ↑ 4. Annual savings (€) ↑ 5. CAPEX (€) ↓
""".strip()


def _recommended_winner_selection_export_text(preset_id: str) -> str:
    """One-line description for CSV assumptions (matches :data:`RECOMMENDED_WINNER_PRESET_HELP`)."""
    pid = preset_id if preset_id in RECOMMENDED_WINNER_PRESET_LABEL_BY_ID else RECOMMENDED_WINNER_PRESET_DEFAULT
    lines = {
        "balanced": "Among feasible: max NPV → max CO₂ (kg) → max SCR → max annual savings → min CAPEX",
        "financial": "Among feasible: max NPV → max annual savings → min payback → min bill → min CAPEX",
        "lowest_bill": "Among feasible: min bill → max annual savings → max NPV → max CO₂ (kg) → min CAPEX",
        "fast_payback": "Among feasible: min payback → max NPV → max annual savings → min bill → min CAPEX",
        "highest_co2": "Among feasible: max CO₂ (kg) → max SSR → max NPV → max SCR → min CAPEX",
        "highest_scr": "Among feasible: max SCR → min export ratio → max NPV → max annual savings → min CAPEX",
    }
    return lines.get(pid, lines["balanced"])


def recommended_setups_constraint_assumptions_df(
    *,
    max_payback_years: float,
    min_self_consumption_pct: float,
    max_export_ratio_pct: float,
    require_positive_npv: bool,
    require_positive_co2_savings: bool,
    npv_min_eur: float | None = None,
    min_co2_reduction_pct: float | None = None,
    winner_preset_id: str = RECOMMENDED_WINNER_PRESET_DEFAULT,
) -> pd.DataFrame:
    """Extra assumption rows for the Recommended setups tab (constraint controls for this export)."""
    prefix = "Recommended setups (this tab)"
    _wl = RECOMMENDED_WINNER_PRESET_LABEL_BY_ID.get(
        winner_preset_id if winner_preset_id in RECOMMENDED_WINNER_PRESET_LABEL_BY_ID else RECOMMENDED_WINNER_PRESET_DEFAULT,
        "Balanced recommendation",
    )
    winner_line = f"{_wl}: {_recommended_winner_selection_export_text(winner_preset_id)}"
    rows: list[dict[str, object]] = [
        {"Setting": f"{prefix}: max payback (years)", "Value": float(max_payback_years)},
        {
            "Setting": f"{prefix}: min self-consumption ratio (%)",
            "Value": float(min_self_consumption_pct),
        },
        {
            "Setting": f"{prefix}: max export ratio (% of PV gen)",
            "Value": float(max_export_ratio_pct),
        },
        {"Setting": f"{prefix}: require NPV > 0", "Value": bool(require_positive_npv)},
        {
            "Setting": f"{prefix}: require CO₂ savings > 0",
            "Value": bool(require_positive_co2_savings),
        },
    ]
    if npv_min_eur is not None and (not require_positive_npv):
        rows.append({"Setting": f"{prefix}: NPV min (€)", "Value": float(npv_min_eur)})
    if min_co2_reduction_pct is not None and float(min_co2_reduction_pct) > 0.0:
        rows.append(
            {"Setting": f"{prefix}: min CO₂ reduction vs grid-only (%)", "Value": float(min_co2_reduction_pct)}
        )
    rows.append({"Setting": f"{prefix}: recommendation preset (ranking among feasible rows)", "Value": winner_line})
    return _assumption_value_column_to_string(pd.DataFrame(rows))


def tariffs_in_use_info_text(tariff_profiles: List[Dict]) -> str:
    """Multi-line message for st.info: active tariffs (import + per-variant export)."""
    lines = ["Tariffs in use (€/kWh):"]
    for p in (tariff_profiles or []):
        name = str(p.get("name", "") or p.get("col", "Tariff"))
        kind = str(p.get("kind", "standard"))
        rates = p.get("rates", {})
        er = float(p.get("export_rate", DEFAULT_EXPORT_RATE))
        if kind == "flat":
            flat_entry = rates.get("flat", DEFAULT_TARIFFS["flat"])
            r = (
                float(flat_entry.get("flat", DEFAULT_TARIFFS["flat"]))
                if isinstance(flat_entry, dict)
                else float(flat_entry)
            )
            lines.append(f"- {name}: flat={r:.5f}, export={er:.4f}")
        elif kind == "weekend":
            wk = rates.get("weekend", DEFAULT_TARIFFS["weekend"])
            lines.append(
                f"- {name} weekday: day={wk['weekday']['day']:.5f}, peak={wk['weekday']['peak']:.5f}, night={wk['weekday']['night']:.5f}"
            )
            lines.append(
                f"- {name} weekend: day={wk['weekend']['day']:.5f}, peak={wk['weekend']['peak']:.5f}, night={wk['weekend']['night']:.5f}, export={er:.4f}"
            )
        else:
            std = rates.get("standard", DEFAULT_TARIFFS["standard"])
            lines.append(f"- {name}: day={std['day']:.5f}, peak={std['peak']:.5f}, night={std['night']:.5f}, export={er:.4f}")
    return "\n".join(lines)


def render_last_run_tariffs_and_assumptions_section() -> None:
    """Tariff list + assumption table from last successful run (Settings & App guide tab)."""
    st.markdown("### Tariffs and assumptions (last completed run)")
    if st.session_state.get("opt_dfs") is None:
        st.info("Complete **Run analysis** once to see the tariff and assumption snapshot for your last run here.")
        return

    profiles = st.session_state.get("last_tariff_profiles") or _default_tariff_profiles()
    st.info(tariffs_in_use_info_text(list(profiles)))

    rows_df = last_run_assumptions_snapshot_df()
    assert rows_df is not None
    with st.expander("Run assumptions (last completed run)", expanded=True):
        st.caption(
            "Frozen from your last successful **Run analysis**. If **Edit assumptions and rerun** differs from this snapshot, click **Run analysis** again."
        )
        st.dataframe(
            rows_df,
            width="stretch",
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
    Backwards-compatible helper for the built-in tariff columns:
      - 'tariff_standard', 'tariff_weekend', 'tariff_flat'
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


def get_tariff_value_from_profile(ts: pd.Timestamp, profile: Dict) -> float:
    """€/kWh import rate for a given timestamp under a tariff profile."""
    kind = str(profile.get("kind", "standard"))
    rates = profile.get("rates", {})
    hour = int(ts.hour)
    band = tariff_band(hour)
    if kind == "flat":
        flat_entry = rates.get("flat", DEFAULT_TARIFFS["flat"])
        # Support both scalar and nested dict shapes for backward compatibility.
        if isinstance(flat_entry, dict):
            return float(flat_entry.get("flat", DEFAULT_TARIFFS["flat"]))
        return float(flat_entry)
    if kind == "weekend":
        is_weekend = ts.dayofweek >= 5
        key = "weekend" if is_weekend else "weekday"
        src = rates.get("weekend", DEFAULT_TARIFFS["weekend"])
        return float(src[key][band])
    src = rates.get("standard", DEFAULT_TARIFFS["standard"])
    return float(src[band])


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
        s = " ".join(str(s).strip().split())
        # Also accept consumption-style timestamps (DD/MM/YYYY HH:MM[:SS])
        if "/" in s:
            dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
            if pd.isna(dt):
                return pd.NaT
            return pd.Timestamp(dt).floor("h")
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
def load_and_prepare_data(
    cons_bytes: bytes,
    pv_bytes: bytes,
    tariff_profiles: object,
    tariff_cache_key: str,
) -> pd.DataFrame:
    """
    Prepare df with tariff columns. Cache key must reflect the tariffs used.

    Backwards compatible:
      - If tariff_profiles is a dict shaped like DEFAULT_TARIFFS, creates the 3 built-in columns.
      - If tariff_profiles is a list of profile dicts, creates a column per profile.
    """
    df_cons = _parse_consumption_csv(cons_bytes)
    df_pv = _parse_pv_timeseries_csv(pv_bytes)

    df = df_cons.merge(df_pv, on="date", how="left")
    df["pv_per_kwp"] = df["pv_per_kwp"].fillna(0.0)

    if isinstance(tariff_profiles, dict) and "standard" in tariff_profiles:
        # Legacy call style: build 3 default columns.
        tariff_config = tariff_profiles
        for col in ["tariff_standard", "tariff_weekend", "tariff_flat"]:
            df[col] = df["date"].apply(lambda t, c=col: get_tariff_value_from_config(t, c, tariff_config))
    else:
        profiles = list(tariff_profiles or [])
        for p in profiles:
            col = str(p.get("col", "") or "")
            if not col:
                continue
            df[col] = df["date"].apply(lambda t, pr=p: get_tariff_value_from_profile(t, pr))

    df = df.reset_index(drop=True)
    return df


# Repo-rooted sample data (same parsing path as uploads via bytes)
_APP_ROOT = Path(__file__).resolve().parent
BUILTIN_DEFAULT_CONSUMPTION_CSV = _APP_ROOT / "data" / "default_consumption.csv"
BUILTIN_DEFAULT_PV_CSV = _APP_ROOT / "data" / "default_pv.csv"
LOCAL_OVERRIDE_CONSUMPTION_CSV = _APP_ROOT / "data" / "local_consumption.csv"
LOCAL_OVERRIDE_PV_CSV = _APP_ROOT / "data" / "local_pv.csv"
BUILTIN_DEFAULT_TARIFFS_CSV = _APP_ROOT / "data" / "default_tariffs.csv"
LOCAL_OVERRIDE_TARIFFS_CSV = _APP_ROOT / "data" / "local_tariffs.csv"
HEADER_BANNER_IMAGE = _APP_ROOT / "assets" / "banners" / "banner_rec_residential_02.png"
EMBEDDED_SAVED_RUNS_DIR = _APP_ROOT / "assets" / "saved_runs"
BUNDLED_RESEARCH_XLSX = _APP_ROOT / "assets" / "research" / "res.xlsx"
RESEARCH_OVERALL_COMPARISON_IMAGE = _APP_ROOT / "assets" / "research" / "overall_comparison.png"
EMBEDDED_RUN_NIGHT_CHARGING_OFF_ZIP = EMBEDDED_SAVED_RUNS_DIR / "rec_saved_run_batt_night_off.zip"
EMBEDDED_RUN_NIGHT_CHARGING_ON_ZIP = EMBEDDED_SAVED_RUNS_DIR / "rec_saved_run_batt_night_on.zip"
# Keep bundled demo runs available in code, but disabled by default so users start
# with "Run analysis" / uploaded saved-run flow unless explicitly re-enabled.
ENABLE_EMBEDDED_SAVED_RUNS = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# Packaged demo: hide **Run your own analysis**, enable embedded ZIPs + sidebar picker. Set ``REC_FEASIBILITY_DEMO=1``.
DEMO_MODE = True


def embedded_saved_runs_active() -> bool:
    """Embedded demo ZIPs (sidebar + first-open autoload): on when ``REC_FEASIBILITY_DEMO`` or ``ENABLE_EMBEDDED_SAVED_RUNS``."""
    return bool(DEMO_MODE) or ENABLE_EMBEDDED_SAVED_RUNS


def _render_header_banner_strip(path: Path, *, max_height_px: int = 112) -> None:
    """Full-width banner with short fixed height (~1/5 of a typical uncropped hero); crops with ``object-fit: cover``."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    h = int(max_height_px)
    st.markdown(
        f'<div style="width:100%;height:{h}px;overflow:hidden;border-radius:6px;margin-bottom:0.5rem;">'
        f'<img src="data:{mime};base64,{b64}" alt="" '
        'style="width:100%;height:100%;object-fit:cover;object-position:center 40%;display:block;" />'
        f"</div>",
        unsafe_allow_html=True,
    )


def _consumption_default_path_and_label() -> Tuple[Path, str]:
    """
    When upload is empty: env REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV, then
    data/local_consumption.csv, then data/default_consumption.csv.
    """
    env = os.environ.get("REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p, f"default ({p.name})"
    if LOCAL_OVERRIDE_CONSUMPTION_CSV.is_file():
        return LOCAL_OVERRIDE_CONSUMPTION_CSV, "default (local_consumption.csv)"
    if BUILTIN_DEFAULT_CONSUMPTION_CSV.is_file():
        return BUILTIN_DEFAULT_CONSUMPTION_CSV, "default (bundled sample)"
    raise FileNotFoundError(
        "No consumption CSV: set REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV, add data/local_consumption.csv, "
        f"or ship {BUILTIN_DEFAULT_CONSUMPTION_CSV.name} under data/."
    )


def _pv_default_path_and_label() -> Tuple[Path, str]:
    """
    When upload is empty: env REC_FEASIBILITY_DEFAULT_PV_CSV, then data/local_pv.csv,
    then data/default_pv.csv.
    """
    env = os.environ.get("REC_FEASIBILITY_DEFAULT_PV_CSV", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p, f"default ({p.name})"
    if LOCAL_OVERRIDE_PV_CSV.is_file():
        return LOCAL_OVERRIDE_PV_CSV, "default (local_pv.csv)"
    if BUILTIN_DEFAULT_PV_CSV.is_file():
        return BUILTIN_DEFAULT_PV_CSV, "default (bundled sample)"
    raise FileNotFoundError(
        "No PV timeseries CSV: set REC_FEASIBILITY_DEFAULT_PV_CSV, add data/local_pv.csv, "
        f"or ship {BUILTIN_DEFAULT_PV_CSV.name} under data/."
    )


def resolve_consumption_csv_bytes(cons_file: Optional[object]) -> Tuple[bytes, str]:
    """Bytes for `load_and_prepare_data` and SHA tracking. Upload wins; else default path chain."""
    if cons_file is not None:
        return cons_file.getvalue(), "uploaded"
    path, label = _consumption_default_path_and_label()
    return path.read_bytes(), label


def resolve_pv_csv_bytes(pv_file: Optional[object]) -> Tuple[bytes, str]:
    """Bytes for `load_and_prepare_data` and SHA tracking. Upload wins; else default path chain."""
    if pv_file is not None:
        return pv_file.getvalue(), "uploaded"
    path, label = _pv_default_path_and_label()
    return path.read_bytes(), label


def _resolve_csv_bytes_for_comparison(cons_file: Optional[object], pv_file: Optional[object]) -> Tuple[bytes, bytes]:
    """Same bytes as a run would use; falls back to upload-only if built-ins are missing (rare)."""
    try:
        cb, _ = resolve_consumption_csv_bytes(cons_file)
    except FileNotFoundError:
        cb = cons_file.getvalue() if cons_file is not None else b""
    try:
        pb, _ = resolve_pv_csv_bytes(pv_file)
    except FileNotFoundError:
        pb = pv_file.getvalue() if pv_file is not None else b""
    return cb, pb


# ----------------------------
# Battery model settings
# ----------------------------
DEFAULT_BATT_EFF_ROUND_TRIP = 0.95  # round-trip efficiency (aligned with UI default)
DEFAULT_BATT_DOD = 0.9            # usable depth of discharge
DEFAULT_BATT_INIT_SOC = 0.0      # initial state of charge (fraction)
DEFAULT_BATT_C_RATE = 0.5         # max charge/discharge power as fraction of capacity per hour


@dataclass(frozen=True)
class BatterySettings:
    eff_round_trip: float = DEFAULT_BATT_EFF_ROUND_TRIP
    dod: float = DEFAULT_BATT_DOD
    init_soc: float = DEFAULT_BATT_INIT_SOC
    min_soc: float = 0.0
    max_soc: float = 1.0
    c_rate: float = DEFAULT_BATT_C_RATE

    # Dispatch/rules
    charge_from_pv: bool = True
    charge_from_grid_at_night: bool = False
    discharge_schedule: str = "Peak only"  # or "Day+Peak"


def _battery_discharge_ok_hour(hour: int, discharge_schedule: str) -> bool:
    """
    Hours when the battery may discharge to serve load (same bands for PV+battery and battery-only).

    - **Peak only:** 17:00-19:00 (hours 17 and 18 at hourly resolution).
    - **Day+Peak:** that peak window, then 19:00-23:00 (hours 19-22) if energy remains — no discharge
      at night (23:00-08:00) or in the earlier daytime band (08:00-17:00).
    """
    h = int(hour)
    in_peak = 17 <= h < 19
    evening_after_peak = 19 <= h < 23
    if discharge_schedule == "Peak only":
        return in_peak
    if discharge_schedule == "Day+Peak":
        return in_peak or evening_after_peak
    return False


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
    soc_min_dod = batt_kwh * max(0.0, 1.0 - float(battery_settings.dod))
    soc_min_user = batt_kwh * max(0.0, min(1.0, float(battery_settings.min_soc)))
    soc_max_user = batt_kwh * max(0.0, min(1.0, float(battery_settings.max_soc)))
    soc_min = max(0.0, soc_min_dod, min(soc_min_user, soc_max_user))
    soc_max = min(float(batt_kwh), max(soc_min_user, soc_max_user))
    soc = batt_kwh * float(battery_settings.init_soc)
    soc = min(max(soc, soc_min), soc_max)

    grid_import = np.zeros_like(cons, dtype=float)
    feed_in = np.zeros_like(cons, dtype=float)
    n = len(cons)
    soc_end = np.zeros(n, dtype=float)
    battery_discharge_kwh = np.zeros(n, dtype=float)
    battery_charge_from_grid_kwh = np.zeros(n, dtype=float)

    for i in range(n):
        hour = int(d.loc[i, "date"].hour)
        is_night = hour >= 23 or hour < 8

        ch_from_grid = 0.0
        dch = 0.0

        if is_night:
            space = soc_max - soc
            if space > 0 and max_power > 0 and battery_settings.charge_from_grid_at_night:
                # Charge: draw from grid, SOC += drawn * charge_eff
                ch_from_grid = min(space / charge_eff, max_power)
                soc += ch_from_grid * charge_eff
        else:
            discharge_ok = _battery_discharge_ok_hour(hour, battery_settings.discharge_schedule)
            if discharge_ok and soc > soc_min and max_power > 0:
                max_deliverable = max(0.0, (soc - soc_min) * discharge_eff)
                dch = min(max_deliverable, cons[i], max_power)
                soc -= dch / discharge_eff

        grid_import[i] = max(0.0, cons[i] - dch) + ch_from_grid
        soc_end[i] = float(soc)
        battery_discharge_kwh[i] = float(dch)
        battery_charge_from_grid_kwh[i] = float(ch_from_grid)

    d["grid_import"] = grid_import
    d["feed_in"] = feed_in
    d["pv_generation"] = 0.0
    d["self_consumed_pv"] = 0.0
    d["local_renewable_to_load"] = 0.0  # battery-only: grid-charged, so 0% self-sufficiency
    d["battery_soc_kwh"] = soc_end
    d["battery_discharge_to_load_kwh"] = battery_discharge_kwh
    d["battery_charge_from_pv_kwh"] = np.zeros(n, dtype=float)
    d["battery_charge_from_grid_kwh"] = battery_charge_from_grid_kwh
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
    soc_min_dod = batt_kwh * max(0.0, 1.0 - float(battery_settings.dod))
    soc_min_user = batt_kwh * max(0.0, min(1.0, float(battery_settings.min_soc)))
    soc_max_user = batt_kwh * max(0.0, min(1.0, float(battery_settings.max_soc)))
    soc_min = max(0.0, soc_min_dod, min(soc_min_user, soc_max_user))
    soc_max = min(float(batt_kwh), max(soc_min_user, soc_max_user))
    soc = batt_kwh * float(battery_settings.init_soc)
    soc = min(max(soc, soc_min), soc_max)
    soc_pv = 0.0  # PV-origin energy in battery
    soc_grid = 0.0  # grid-origin energy in battery

    grid_import = np.zeros_like(cons, dtype=float)
    feed_in = np.zeros_like(cons, dtype=float)
    pv_to_load_direct = np.zeros_like(cons, dtype=float)
    battery_to_load_pv_origin = np.zeros_like(cons, dtype=float)
    n = len(cons)
    soc_end = np.zeros(n, dtype=float)
    battery_discharge_kwh = np.zeros(n, dtype=float)
    battery_charge_from_pv_kwh = np.zeros(n, dtype=float)
    battery_charge_from_grid_kwh = np.zeros(n, dtype=float)

    for i in range(n):
        hour = int(d.loc[i, "date"].hour)
        is_night = hour >= 23 or hour < 8

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
            and soc < soc_max
            and max_power > 0
            and battery_settings.charge_from_pv
        ):
            ch_from_pv = min(pv_surplus, (soc_max - soc) / charge_eff, max_power)
            soc += ch_from_pv * charge_eff
            soc_pv += ch_from_pv * charge_eff
            pv_surplus -= ch_from_pv

        # 3) Charge battery from grid at night
        ch_from_grid = 0.0
        if (
            batt_kwh > 0
            and is_night
            and soc < soc_max
            and max_power > 0
            and battery_settings.charge_from_grid_at_night
        ):
            ch_from_grid = min((soc_max - soc) / charge_eff, max_power)
            soc += ch_from_grid * charge_eff
            soc_grid += ch_from_grid * charge_eff

        # 4) Discharge based on schedule
        dch = 0.0
        dch_pv_origin = 0.0
        discharge_ok = _battery_discharge_ok_hour(hour, battery_settings.discharge_schedule)
        if batt_kwh > 0 and cons_remaining > 0 and soc > soc_min and discharge_ok and max_power > 0:
            max_deliverable = max(0.0, (soc - soc_min) * discharge_eff)
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
        soc_end[i] = float(soc)
        battery_discharge_kwh[i] = float(dch)
        battery_charge_from_pv_kwh[i] = float(ch_from_pv)
        battery_charge_from_grid_kwh[i] = float(ch_from_grid)

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
    d["battery_soc_kwh"] = soc_end
    d["battery_discharge_to_load_kwh"] = battery_discharge_kwh
    d["battery_charge_from_pv_kwh"] = battery_charge_from_pv_kwh
    d["battery_charge_from_grid_kwh"] = battery_charge_from_grid_kwh
    return d


# ----------------------------
# Financial / KPI helpers
# ----------------------------
def compute_kpis_for_scenario(d: pd.DataFrame, tariff_col: str, export_rate: float) -> Dict[str, float]:
    grid_import_kwh = float(np.sum(d["grid_import"].to_numpy(dtype=float)))
    export_kwh = float(np.sum(d["feed_in"].to_numpy(dtype=float)))
    pv_gen_kwh = float(np.sum(d["pv_generation"].to_numpy(dtype=float)))
    self_cons_kwh = float(np.sum(d["self_consumed_pv"].to_numpy(dtype=float))) if "self_consumed_pv" in d.columns else max(pv_gen_kwh - export_kwh, 0.0)
    batt_discharge_kwh = (
        float(np.sum(pd.to_numeric(d["battery_discharge_to_load_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)))
        if "battery_discharge_to_load_kwh" in d.columns
        else 0.0
    )
    batt_charge_kwh = 0.0
    if "battery_charge_from_pv_kwh" in d.columns:
        batt_charge_kwh += float(np.sum(pd.to_numeric(d["battery_charge_from_pv_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)))
    if "battery_charge_from_grid_kwh" in d.columns:
        batt_charge_kwh += float(np.sum(pd.to_numeric(d["battery_charge_from_grid_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)))

    # Cost of grid imports only
    cost_grid_import = float(np.sum(d["grid_import"].to_numpy(dtype=float) * d[tariff_col].to_numpy(dtype=float)))
    # Export income
    export_income = export_kwh * export_rate
    annual_cost = cost_grid_import - export_income

    # CO2 only from imports
    co2_kg = grid_import_kwh * _grid_co2_factor()

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
        COL_BATTERY_CHARGE_KWH: batt_charge_kwh,
        COL_BATTERY_DISCHARGE_KWH: batt_discharge_kwh,
        "Cost of grid import (€)": cost_grid_import,
        COL_NET_IMPORT_EXPORT_COST_EUR: annual_cost,
        "CO2 (kg)": co2_kg,
    }


def _gross_savings_lifetime(
    annual_savings_year1: float,
    electricity_inflation_rate: float,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> float:
    """Sum of inflated annual savings over ``lifetime_years``."""
    ly = int(lifetime_years)
    if electricity_inflation_rate <= 0:
        return annual_savings_year1 * ly
    infl = float(electricity_inflation_rate)
    return annual_savings_year1 * float(((1 + infl) ** ly - 1) / infl)


def compute_payback_and_npv(
    capex_eur: float,
    annual_savings_eur: float,
    discount_rate: float | None = None,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_cost_eur: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_cost_eur: float = 0.0,
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> Tuple[float, float]:
    ly = int(lifetime_years)
    r = discount_rate if discount_rate is not None else DISCOUNT_RATE
    npv = -capex_eur
    payback = float("inf") if annual_savings_eur <= 0 else (capex_eur / annual_savings_eur)
    # Always include the full-horizon discounted savings stream in NPV,
    # including zero/negative annual savings.
    if electricity_inflation_rate <= 0:
        discount_factors = (1 + r) ** (-np.arange(1, ly + 1))
        npv += annual_savings_eur * float(discount_factors.sum())
    else:
        infl = float(electricity_inflation_rate)
        for t in range(1, ly + 1):
            savings_t = annual_savings_eur * ((1 + infl) ** (t - 1))
            npv += savings_t / ((1 + r) ** t)
    if (
        battery_replacement_year is not None
        and 1 <= int(battery_replacement_year) <= ly
        and float(battery_replacement_cost_eur) > 0
    ):
        npv -= float(battery_replacement_cost_eur) / ((1 + r) ** int(battery_replacement_year))
    if (
        inverter_replacement_year is not None
        and 1 <= int(inverter_replacement_year) <= ly
        and float(inverter_replacement_cost_eur) > 0
    ):
        npv -= float(inverter_replacement_cost_eur) / ((1 + r) ** int(inverter_replacement_year))
    return payback, npv


def compute_irr(
    capex_eur: float,
    annual_savings_eur: float,
    n_years: int = DEFAULT_LIFETIME_YEARS,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_cost_eur: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_cost_eur: float = 0.0,
) -> float:
    """IRR: discount rate r where NPV = 0. Returns as decimal (e.g. 0.08 for 8%)."""
    if capex_eur <= 0:
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

    # Bisection: find r in [lo, hi] where npv_at_r(r) = 0.
    # Include negative rates so economically weak cases can report negative IRR.
    lo, hi = -0.99, 2.0  # -99% to 200%
    v_lo = npv_at_r(lo)
    v_hi = npv_at_r(hi)
    if not np.isfinite(v_lo) or not np.isfinite(v_hi):
        return float("nan")
    if v_lo == 0:
        return lo
    if v_hi == 0:
        return hi
    if v_lo * v_hi > 0:
        return float("nan")  # No IRR root in bracket.
    for _ in range(80):  # ~2^-80 precision
        mid = (lo + hi) / 2
        val = npv_at_r(mid)
        if abs(val) < 1e-6:
            return mid
        if v_lo * val > 0:
            lo = mid
            v_lo = val
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
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
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
        lifetime_years=int(lifetime_years),
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
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> pd.DataFrame:
    """
    PV + Grid only: one row per PV size (pv_min–pv_max kWp, step 1 kWp).
    Uses same KPI definitions as the main results view. (Used by tests / tooling; not shown in the UI.)
    """
    ly = int(lifetime_years)
    _cn = col_npv(ly)
    _ci = col_irr(ly)
    d_base = run_scenario_grid_only(df, tariff_col)
    baseline_cost = float(np.sum(d_base["grid_import"].to_numpy(dtype=float) * df[tariff_col].to_numpy(dtype=float)))
    baseline_co2_kg = float(d_base["grid_import"].to_numpy(dtype=float).sum() * _grid_co2_factor())

    rows = []
    if int(pv_max) < int(pv_min):
        return pd.DataFrame(rows)
    for pv in range(int(pv_min), int(pv_max) + 1, 1):
        d = run_scenario_pv_grid(df, pv, tariff_col)
        k = compute_kpis_for_scenario(d, tariff_col, export_rate)
        co2_kg = float(d["grid_import"].to_numpy(dtype=float).sum() * _grid_co2_factor())
        co2_save = max(0.0, baseline_co2_kg - co2_kg)
        capex = pv * pv_capex_per_kwp
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k[COL_NET_IMPORT_EXPORT_COST_EUR] - opex
        inverter_replacement_cost = capex * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        payback, npv = compute_payback_and_npv(
            capex,
            annual_savings,
            discount_rate,
            electricity_inflation_rate,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inverter_replacement_cost,
            lifetime_years=ly,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            n_years=ly,
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
                COL_ANNUAL_ELECTRICITY_COST_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
                "Annual export earnings (€)": k["Export income (€)"],
                "Annual savings vs grid only (€)": annual_savings,
                "CAPEX (€)": capex,
                "Payback period (years)": payback,
                _cn: npv,
                _ci: 100.0 * irr if np.isfinite(irr) else float("nan"),
                "Self-sufficiency ratio (%)": k["Self-sufficiency ratio (%)"],
                "Self-consumption ratio (%)": k["Self-consumption ratio (%)"],
                "CO2 savings (kg)": co2_save,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# Optimizer
# ----------------------------
@dataclass(frozen=True)
class OptimizerConfig:
    pv_min: int = 5
    pv_max: int = 60
    pv_step: int = 5      # dataclass fallback; UI default preset is Full (step 1)
    batt_min: int = 0
    batt_max: int = 40
    batt_step: int = 5   # dataclass fallback; UI default preset is Full (step 1)


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
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
    progress_callback: Callable[[], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    ly = int(lifetime_years)
    d_base = run_scenario_grid_only(df, tariff_col)
    baseline_energy_cost = float(np.sum(d_base["grid_import"].to_numpy(dtype=float) * df[tariff_col].to_numpy(dtype=float)))
    baseline_co2 = float(d_base["grid_import"].to_numpy(dtype=float).sum() * _grid_co2_factor())

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
            lifetime_years=ly,
        )
        irr = compute_irr(
            inv,
            savings,
            n_years=ly,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl_cost,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl_cost,
        )
        k = compute_kpis_for_scenario(d, tariff_col, export_rate)
        ss = k["Self-sufficiency ratio (%)"]
        co2_kg = float(k["CO2 (kg)"])
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
                "grid_import_kwh": float(k["Grid import (kWh)"]),
                "export_kwh": float(k["Export to grid (kWh)"]),
                "self_consumed_pv_kwh": float(k["Self-consumed PV (kWh)"]),
                "battery_charge_kwh": float(k[COL_BATTERY_CHARGE_KWH]),
                "battery_discharge_kwh": float(k[COL_BATTERY_DISCHARGE_KWH]),
                "grid_import_cost_eur": float(k["Cost of grid import (€)"]),
                "annual_co2_kg": co2_kg,
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
            lifetime_years=ly,
        )
        irr = compute_irr(
            inv,
            savings,
            n_years=ly,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl_cost,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl_cost,
        )
        k = compute_kpis_for_scenario(d, tariff_col, export_rate)
        ss = k["Self-sufficiency ratio (%)"]
        co2_kg = float(k["CO2 (kg)"])
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
                "grid_import_kwh": float(k["Grid import (kWh)"]),
                "export_kwh": float(k["Export to grid (kWh)"]),
                "self_consumed_pv_kwh": float(k["Self-consumed PV (kWh)"]),
                "battery_charge_kwh": float(k[COL_BATTERY_CHARGE_KWH]),
                "battery_discharge_kwh": float(k[COL_BATTERY_DISCHARGE_KWH]),
                "grid_import_cost_eur": float(k["Cost of grid import (€)"]),
                "annual_co2_kg": co2_kg,
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
                lifetime_years=ly,
            )
            irr = compute_irr(
                inv,
                savings,
                n_years=ly,
                electricity_inflation_rate=electricity_inflation_rate,
                battery_replacement_year=battery_replacement_year,
                battery_replacement_cost_eur=batt_repl_cost,
                inverter_replacement_year=inverter_replacement_year,
                inverter_replacement_cost_eur=inv_repl_cost,
            )
            k = compute_kpis_for_scenario(d, tariff_col, export_rate)
            ss = k["Self-sufficiency ratio (%)"]
            co2_kg = float(k["CO2 (kg)"])
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
                    "grid_import_kwh": float(k["Grid import (kWh)"]),
                    "export_kwh": float(k["Export to grid (kWh)"]),
                    "self_consumed_pv_kwh": float(k["Self-consumed PV (kWh)"]),
                    "battery_charge_kwh": float(k[COL_BATTERY_CHARGE_KWH]),
                    "battery_discharge_kwh": float(k[COL_BATTERY_DISCHARGE_KWH]),
                    "grid_import_cost_eur": float(k["Cost of grid import (€)"]),
                    "annual_co2_kg": co2_kg,
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
    if goal in RECOMMENDED_WINNER_PRESET_ID_BY_LABEL:
        work = sub.copy()
        if "_export_ratio_pct" not in work.columns:
            pv_gen = pd.to_numeric(work["pv_gen_kwh"], errors="coerce")
            if "export_kwh" in work.columns:
                export_kwh = pd.to_numeric(work["export_kwh"], errors="coerce").fillna(0.0)
            else:
                export_kwh = pd.Series(np.zeros(len(work), dtype=float), index=work.index)
            _pv_a = pv_gen.to_numpy(dtype=float)
            _ex_a = export_kwh.to_numpy(dtype=float)
            _ratio = np.zeros_like(_pv_a, dtype=float)
            np.divide(100.0 * np.maximum(0.0, _ex_a), _pv_a, out=_ratio, where=_pv_a > 1e-9)
            work["_export_ratio_pct"] = _ratio
        return _sort_feasible_for_recommended_winner_preset(
            work, RECOMMENDED_WINNER_PRESET_ID_BY_LABEL[goal]
        ).iloc[0]

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
    if goal == "Highest annual CO2 savings":
        return sub.loc[sub["co2_save_kg"].idxmax()]
    if goal == "Best cost–CO2 trade-off":
        c = pd.to_numeric(sub["cost"], errors="coerce").astype(float)
        g = pd.to_numeric(sub["co2_save_kg"], errors="coerce").fillna(0.0).astype(float)
        cmin, cmax = float(c.min()), float(c.max())
        gmin, gmax = float(g.min()), float(g.max())
        cr = (c - cmin) / (cmax - cmin + 1e-9)
        gr = (g - gmin) / (gmax - gmin + 1e-9)
        bal = 0.5 * cr + 0.5 * (1.0 - gr)
        return sub.loc[bal.idxmin()]
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

# Recommended setups tab — defaults for constraint panel (shared feasible set; winner from **Recommendation preset**).
RECOMMENDED_SETUP_MAX_PAYBACK_YEARS = 10.0
RECOMMENDED_SETUP_DEFAULT_MIN_SELF_CONSUMPTION_PCT = 80.0
RECOMMENDED_SETUP_DEFAULT_MAX_EXPORT_RATIO_PCT = 20.0


def _sort_feasible_for_recommended_winner_preset(feas: pd.DataFrame, preset_id: str) -> pd.DataFrame:
    """Lexicographic sort on feasible optimizer rows; last key is always CAPEX (€), lowest wins."""
    if len(feas) == 0:
        return feas
    pid = preset_id if preset_id in RECOMMENDED_WINNER_PRESET_LABEL_BY_ID else RECOMMENDED_WINNER_PRESET_DEFAULT
    work = feas.copy()
    _pv = pd.to_numeric(work["pv_kwp"], errors="coerce").fillna(0.0)
    _bt = pd.to_numeric(work["batt_kwh"], errors="coerce").fillna(0.0)
    work["_rec_capex"] = _pv * float(PV_COST_PER_KWP) + _bt * float(BATT_COST_PER_KWH)
    if "cost" not in work.columns:
        work["cost"] = np.nan
    else:
        work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
    if "self_suff_pct" not in work.columns:
        work["self_suff_pct"] = np.nan
    else:
        work["self_suff_pct"] = pd.to_numeric(work["self_suff_pct"], errors="coerce")
    work["npv"] = pd.to_numeric(work["npv"], errors="coerce")
    work["co2_save_kg"] = pd.to_numeric(work["co2_save_kg"], errors="coerce")
    work["self_consumption_ratio_pct"] = pd.to_numeric(work["self_consumption_ratio_pct"], errors="coerce").fillna(0.0)
    work["savings"] = pd.to_numeric(work["savings"], errors="coerce")
    work["payback"] = pd.to_numeric(work["payback"], errors="coerce")
    work["_export_ratio_pct"] = pd.to_numeric(work["_export_ratio_pct"], errors="coerce").fillna(0.0)

    if pid == "balanced":
        by = ["npv", "co2_save_kg", "self_consumption_ratio_pct", "savings", "_rec_capex"]
        asc = [False, False, False, False, True]
    elif pid == "financial":
        by = ["npv", "savings", "payback", "cost", "_rec_capex"]
        asc = [False, False, True, True, True]
    elif pid == "lowest_bill":
        by = ["cost", "savings", "npv", "co2_save_kg", "_rec_capex"]
        asc = [True, False, False, False, True]
    elif pid == "fast_payback":
        by = ["payback", "npv", "savings", "cost", "_rec_capex"]
        asc = [True, False, False, True, True]
    elif pid == "highest_co2":
        by = ["co2_save_kg", "self_suff_pct", "npv", "self_consumption_ratio_pct", "_rec_capex"]
        asc = [False, False, False, False, True]
    elif pid == "highest_scr":
        by = ["self_consumption_ratio_pct", "_export_ratio_pct", "npv", "savings", "_rec_capex"]
        asc = [False, True, False, False, True]
    else:
        by = ["npv", "co2_save_kg", "self_consumption_ratio_pct", "savings", "_rec_capex"]
        asc = [False, False, False, False, True]

    return work.sort_values(by=by, ascending=asc, na_position="last", kind="mergesort")


def _recommended_build_params_from_sidebar_hard_filters() -> Dict[str, object]:
    """Map sidebar **Decision constraints** widgets → :func:`build_recommended_setups_summary_df` kwargs.

    No new controls: only interprets existing ``hard_*`` session keys.
    """
    pay_en = bool(st.session_state.get("hard_payback_max_en"))
    pb_years = float(st.session_state.get("hard_payback_max_years") or 0.0)
    max_payback_years = float(RECOMMENDED_SETUP_MAX_PAYBACK_YEARS) if pay_en else float("inf")
    if pay_en:
        max_payback_years = float(pb_years) if pb_years > 0 else float(RECOMMENDED_SETUP_MAX_PAYBACK_YEARS)

    scr_en = bool(st.session_state.get("hard_self_cons_min_en"))
    min_self_consumption_pct = (
        float(st.session_state.get("hard_self_cons_min_pct") or 0.0) if scr_en else 0.0
    )

    exp_en = bool(st.session_state.get("hard_export_max_en"))
    max_export_ratio_pct = float(st.session_state.get("hard_export_max_pct") or 100.0) if exp_en else 100.0

    npv_en = bool(st.session_state.get("hard_npv_min_en"))
    npv_min_eur = float(st.session_state.get("hard_npv_min_eur") or 0.0) if npv_en else None
    # Keep semantics aligned with Full results hard filters:
    # enabling NPV min applies a numeric floor (>= value), including 0.0.
    # Do not silently upgrade 0.0 to a strict > 0 gate here.
    require_positive_npv = False

    co2_en = bool(st.session_state.get("hard_co2_min_en"))
    co2_pct = float(st.session_state.get("hard_co2_min_pct") or 0.0) if co2_en else 0.0
    min_co2_reduction_pct: float | None = None
    # Keep semantics aligned with Full results hard filters:
    # enabling CO2 reduction min applies >= threshold; 0.0 should not become strict > 0.
    require_positive_co2_savings = False
    if co2_en and co2_pct > 0.0:
        min_co2_reduction_pct = float(co2_pct)

    return {
        "max_payback_years": float(max_payback_years),
        "min_self_consumption_pct": float(min_self_consumption_pct),
        "max_export_ratio_pct": float(max_export_ratio_pct),
        "require_positive_npv": bool(require_positive_npv),
        "npv_min_eur": npv_min_eur,
        "require_positive_co2_savings": bool(require_positive_co2_savings),
        "min_co2_reduction_pct": min_co2_reduction_pct,
    }


def build_recommended_setups_summary_df(
    opt_dfs: Optional[Dict[str, pd.DataFrame]],
    tariff_profiles: List[Dict],
    *,
    enable_battery_ui: bool,
    scenario_type_ui: str = "All scenarios",
    tariff_family_ui: str = "All tariff types",
    max_payback_years: float = RECOMMENDED_SETUP_MAX_PAYBACK_YEARS,
    min_self_consumption_pct: float = 0.0,
    max_export_ratio_pct: float = 100.0,
    require_positive_npv: bool = True,
    require_positive_co2_savings: bool = True,
    npv_min_eur: float | None = None,
    min_co2_reduction_pct: float | None = None,
    grid_baseline_annual_co2_kg: float | None = None,
    charge_from_grid_at_night_last_run: Optional[bool] = None,
    winner_preset: str = RECOMMENDED_WINNER_PRESET_DEFAULT,
    prepared_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Post-processes **only** the optimizer grid from the last **Run analysis** (`opt_dfs`) — no second pass.

    **Feasible set:** rows for each tariff × scenario family that satisfy **all** enabled constraints
    (max payback, optional NPV floor / strict NPV > 0, optional CO₂ reduction (%) vs grid and/or CO₂ savings > 0,
    min self-consumption, max export ratio).

    **Selected row:** among feasible rows, one winner per tariff × scenario family chosen by ``winner_preset``
    (lexicographic ranking; final tie-break **lowest CAPEX (€)**). See :data:`RECOMMENDED_WINNER_PRESETS`.

    When ``charge_from_grid_at_night_last_run`` is not ``None`` (completed run with battery UI), the table
    includes **Battery charging (last run)** as **PV only charging** vs **PV + night-grid charging**.

    Sizing is **PV (kWp) + battery (kWh)** only (same as the main optimizer — no inverter size).

    ``scenario_type_ui`` and ``tariff_family_ui`` follow the sidebar **Scenario type** and **Tariff family**
    results filters (same allowed scenario names / tariff kinds as the consolidated table).
    """
    fams_all: List[Tuple[str, str]] = [("PV only", "PV + Grid")]
    if enable_battery_ui:
        fams_all.extend([("PV + Battery", "PV + Battery + Grid"), ("Battery only", "Battery + Grid")])
    allowed_scenarios = _scenario_allowed_for_filter(str(scenario_type_ui))
    fams = [(ck, sl) for ck, sl in fams_all if sl in allowed_scenarios]

    rows: List[Dict[str, object]] = []
    _rec_batt_charging_extra: Dict[str, object] = {}
    if enable_battery_ui and charge_from_grid_at_night_last_run is not None:
        _rec_batt_charging_extra["Battery charging (last run)"] = (
            "PV + night-grid charging" if charge_from_grid_at_night_last_run else "PV only charging"
        )
    od = opt_dfs or {}
    profiles = _filter_tariff_profiles_by_family_ui(list(tariff_profiles or []), str(tariff_family_ui))

    _co2_pct_active = min_co2_reduction_pct is not None and float(min_co2_reduction_pct) > 0.0
    _baseline_co2_kg: float | None = None
    if _co2_pct_active:
        _baseline_co2_kg = float(grid_baseline_annual_co2_kg) if grid_baseline_annual_co2_kg is not None else None
        if _baseline_co2_kg is None or not np.isfinite(_baseline_co2_kg) or _baseline_co2_kg <= 1e-9:
            # Without a baseline, % reduction is undefined — skip this filter (keeps tests + edge cases safe).
            _co2_pct_active = False

    _display_co2_baseline_kg: float | None = None
    if grid_baseline_annual_co2_kg is not None:
        try:
            _dbc = float(grid_baseline_annual_co2_kg)
            if np.isfinite(_dbc) and _dbc > 1e-9:
                _display_co2_baseline_kg = _dbc
        except (TypeError, ValueError):
            _display_co2_baseline_kg = None

    # Prefer deriving a run-consistent emission factor from the same baseline used for filters.
    _effective_grid_co2_factor = float(_grid_co2_factor())
    if prepared_df is not None and "consumption" in prepared_df.columns and _display_co2_baseline_kg is not None:
        _cons_sum = float(pd.to_numeric(prepared_df["consumption"], errors="coerce").fillna(0.0).sum())
        if np.isfinite(_cons_sum) and _cons_sum > 1e-9:
            _effective_grid_co2_factor = float(_display_co2_baseline_kg) / _cons_sum

    def _row_co2_reduction_pct(co2_save_kg: float) -> float:
        if _display_co2_baseline_kg is None:
            return float("nan")
        return 100.0 * max(0.0, float(co2_save_kg)) / float(_display_co2_baseline_kg)

    for p in profiles:
        tcol = str(p.get("col", "") or "")
        tname = str(p.get("name", "") or tcol)
        if not tcol:
            continue
        odf = od.get(tcol)
        if odf is None or len(odf) == 0:
            for _ck, slabel in fams:
                rows.append(
                    {
                        "Tariff": tname,
                        "Scenario family": slabel,
                        "PV (kWp)": np.nan,
                        "Battery (kWh)": np.nan,
                        "NPV (€)": np.nan,
                        "Payback (yrs)": np.nan,
                        "SCR": np.nan,
                        "Export ratio (% of PV gen)": np.nan,
                        COL_ANNUAL_ELECTRICITY_BILL_EUR: np.nan,
                        "CO2 reduction (%)": np.nan,
                        "SSR": np.nan,
                        "Annual savings (€)": np.nan,
                        "CO₂ savings (kg)": np.nan,
                        "Note": "No optimizer results for this tariff — run analysis first.",
                        **_rec_batt_charging_extra,
                    }
                )
            continue

        for cfg_key, scen_label in fams:
            sub = odf.loc[odf["config"] == cfg_key].copy()
            if len(sub) == 0:
                rows.append(
                    {
                        "Tariff": tname,
                        "Scenario family": scen_label,
                        "PV (kWp)": np.nan,
                        "Battery (kWh)": np.nan,
                        "NPV (€)": np.nan,
                        "Payback (yrs)": np.nan,
                        "SCR": np.nan,
                        "Export ratio (% of PV gen)": np.nan,
                        COL_ANNUAL_ELECTRICITY_BILL_EUR: np.nan,
                        "CO2 reduction (%)": np.nan,
                        "SSR": np.nan,
                        "Annual savings (€)": np.nan,
                        "CO₂ savings (kg)": np.nan,
                        "Note": "No rows for this scenario family in the optimizer grid.",
                        **_rec_batt_charging_extra,
                    }
                )
                continue

            pv_gen = pd.to_numeric(sub["pv_gen_kwh"], errors="coerce")
            if "export_kwh" in sub.columns:
                export_kwh = pd.to_numeric(sub["export_kwh"], errors="coerce").fillna(0.0)
            else:
                export_kwh = pd.Series(np.zeros(len(sub), dtype=float), index=sub.index)
            _pv_a = pv_gen.to_numpy(dtype=float)
            _ex_a = export_kwh.to_numpy(dtype=float)
            _ratio = np.zeros_like(_pv_a, dtype=float)
            np.divide(100.0 * np.maximum(0.0, _ex_a), _pv_a, out=_ratio, where=_pv_a > 1e-9)
            sub["_export_ratio_pct"] = _ratio
            pb = pd.to_numeric(sub["payback"], errors="coerce")
            npv_c = pd.to_numeric(sub["npv"], errors="coerce")
            _grid_imp = (
                pd.to_numeric(sub["grid_import_kwh"], errors="coerce")
                if "grid_import_kwh" in sub.columns
                else None
            )
            if _display_co2_baseline_kg is not None and isinstance(_grid_imp, pd.Series):
                _row_co2_kg = _grid_imp.fillna(0.0).astype(float) * float(_effective_grid_co2_factor)
                co2_c = pd.Series(
                    np.maximum(0.0, float(_display_co2_baseline_kg) - _row_co2_kg.to_numpy(dtype=float)),
                    index=sub.index,
                )
            else:
                co2_c = pd.to_numeric(sub["co2_save_kg"], errors="coerce")
            scr_c = pd.to_numeric(sub["self_consumption_ratio_pct"], errors="coerce").fillna(0.0)
            exr = pd.to_numeric(sub["_export_ratio_pct"], errors="coerce").fillna(0.0)
            # Winner-preset sorting and display should use canonical per-row CO2 savings.
            sub["co2_save_kg"] = co2_c

            m = pb.notna() & np.isfinite(pb) & (pb > 0) & (pb <= float(max_payback_years))
            if require_positive_npv:
                m &= npv_c > 0
            if npv_min_eur is not None:
                m &= npv_c.notna() & np.isfinite(npv_c) & (npv_c >= float(npv_min_eur))
            if require_positive_co2_savings:
                m &= co2_c.notna() & np.isfinite(co2_c) & (co2_c > 0)
            if _co2_pct_active and _baseline_co2_kg is not None:
                co2_red_pct = np.where(
                    float(_baseline_co2_kg) > 1e-9,
                    100.0 * co2_c.to_numpy(dtype=float) / float(_baseline_co2_kg),
                    0.0,
                )
                m &= co2_c.notna() & np.isfinite(co2_c) & (co2_red_pct >= float(min_co2_reduction_pct) - 1e-12)
            m &= scr_c >= float(min_self_consumption_pct)
            m &= exr <= float(max_export_ratio_pct) + 1e-9

            feas = sub.loc[m].copy()
            if len(feas) == 0:
                _pb_note = (
                    f"payback ≤ {float(max_payback_years):g} y"
                    if np.isfinite(float(max_payback_years))
                    else "finite payback"
                )
                rows.append(
                    {
                        "Tariff": tname,
                        "Scenario family": scen_label,
                        "PV (kWp)": np.nan,
                        "Battery (kWh)": np.nan,
                        "NPV (€)": np.nan,
                        "Payback (yrs)": np.nan,
                        "SCR": np.nan,
                        "Export ratio (% of PV gen)": np.nan,
                        COL_ANNUAL_ELECTRICITY_BILL_EUR: np.nan,
                        "CO2 reduction (%)": np.nan,
                        "SSR": np.nan,
                        "Annual savings (€)": np.nan,
                        "CO₂ savings (kg)": np.nan,
                        "Note": (
                            "No PV/battery size in the grid satisfies **all** constraints "
                            f"({_pb_note}"
                            + (", NPV > 0" if require_positive_npv else "")
                            + (
                                f", NPV ≥ €{float(npv_min_eur):g}"
                                if (not require_positive_npv and npv_min_eur is not None)
                                else ""
                            )
                            + (", CO₂ savings > 0" if require_positive_co2_savings else "")
                            + (
                                f", CO₂ reduction ≥ {float(min_co2_reduction_pct):g}%"
                                if (_co2_pct_active and min_co2_reduction_pct is not None)
                                else ""
                            )
                            + (f", self-consumption ≥ {min_self_consumption_pct:g}%" if min_self_consumption_pct > 0 else "")
                            + (f", export ratio ≤ {max_export_ratio_pct:g}%" if max_export_ratio_pct < 100 else "")
                            + ")."
                        ),
                        **_rec_batt_charging_extra,
                    }
                )
                continue

            _wp = winner_preset if winner_preset in RECOMMENDED_WINNER_PRESET_LABEL_BY_ID else RECOMMENDED_WINNER_PRESET_DEFAULT
            feas = _sort_feasible_for_recommended_winner_preset(feas, _wp)
            r = feas.iloc[0]
            exp_pct = float(r["_export_ratio_pct"])
            _co2_sv = float(r["co2_save_kg"])
            _bill_y1 = (
                float(r["cost"])
                if "cost" in r.index and r.get("cost") is not None and pd.notna(r.get("cost"))
                else float("nan")
            )
            _ssr_v = (
                float(r["self_suff_pct"])
                if "self_suff_pct" in r.index and r.get("self_suff_pct") is not None and pd.notna(r.get("self_suff_pct"))
                else float("nan")
            )
            rows.append(
                {
                    "Tariff": tname,
                    "Scenario family": scen_label,
                    "PV (kWp)": int(r["pv_kwp"]),
                    "Battery (kWh)": int(r["batt_kwh"]),
                    "NPV (€)": float(r["npv"]),
                    "Payback (yrs)": float(r["payback"]),
                    "SCR": float(r["self_consumption_ratio_pct"]),
                    "Export ratio (% of PV gen)": exp_pct,
                    COL_ANNUAL_ELECTRICITY_BILL_EUR: _bill_y1,
                    "CO2 reduction (%)": _row_co2_reduction_pct(_co2_sv),
                    "SSR": _ssr_v,
                    "Annual savings (€)": float(r["savings"]),
                    "CO₂ savings (kg)": _co2_sv,
                    "Note": "",
                    **_rec_batt_charging_extra,
                }
            )

    return pd.DataFrame(rows)


_SCENARIO_TYPES_ALL = ["Grid only", "PV + Grid", "Battery + Grid", "PV + Battery + Grid"]

_SCENARIO_LABELS_NO_BATTERY_UI = frozenset({"Grid only", "PV + Grid"})


def scenario_type_ui_options() -> List[str]:
    """Scenario-type filter choices in the sidebar; omit battery families when ENABLE_BATTERY_UI is False."""
    opts = ["All scenarios", "Grid only", "PV + Grid"]
    if ENABLE_BATTERY_UI:
        opts.extend(["PV + Battery + Grid", "Battery + Grid"])
    return opts


def _scenario_allowed_for_filter(scenario_type_ui: str) -> set[str]:
    """Scenario column values included for the given Results-filter selection."""
    if scenario_type_ui == "All scenarios":
        return set(_SCENARIO_TYPES_ALL) if ENABLE_BATTERY_UI else set(_SCENARIO_LABELS_NO_BATTERY_UI)
    return {scenario_type_ui}


# Internal-only stable row id for consolidated results (tariff column + scenario + int PV + int battery).
SCENARIO_ROW_KEY_FIELD = "_scenario_row_key"
SCENARIO_ROW_KEY_SEP = "\x1e"


def compose_scenario_row_key(tcol: str, scenario: str, pv_kwp: float | int, batt_kwh: float | int) -> str:
    """Build a stable key: internal tariff column id, scenario label, integer kWp, integer kWh.

    Grid-only rows are always encoded as PV=0 and Battery=0 regardless of source values.
    """
    tcol_s = str(tcol or "").strip()
    scen_s = str(scenario or "").strip()
    pv_i = int(round(float(pv_kwp)))
    batt_i = int(round(float(batt_kwh)))
    if scen_s == "Grid only":
        pv_i, batt_i = 0, 0
    return SCENARIO_ROW_KEY_SEP.join([tcol_s, scen_s, str(pv_i), str(batt_i)])


# Recommended setups rows without PV/battery (no feasible sizing) use this prefix — never matches consolidated keys.
RECOMMENDED_NO_SIZING_KEY_PREFIX = "rec_no_sizing\x1e"


def augment_recommended_df_with_scenario_row_keys(
    rec_df: pd.DataFrame,
    tariff_profiles: List[Dict],
) -> pd.DataFrame:
    """Add ``SCENARIO_ROW_KEY_FIELD`` so Recommended AgGrid selection maps to ``build_full_scenario_results_df`` rows."""
    if rec_df is None or len(rec_df) == 0:
        return rec_df
    out = rec_df.copy()
    by_name = {str(p.get("name", "")): str(p.get("col", "") or "") for p in (tariff_profiles or [])}
    keys: list[str] = []
    for _, row in out.iterrows():
        tname = str(row.get("Tariff", "") or "")
        tcol = by_name.get(tname, "")
        scen = str(row.get("Scenario family", "") or "")
        pv = row.get("PV (kWp)")
        bt = row.get("Battery (kWh)")
        if pd.notna(pv) and pd.notna(bt):
            keys.append(compose_scenario_row_key(tcol, scen, int(pv), int(bt)))
        else:
            keys.append(f"{RECOMMENDED_NO_SIZING_KEY_PREFIX}{tcol}{SCENARIO_ROW_KEY_SEP}{scen}")
    out[SCENARIO_ROW_KEY_FIELD] = keys
    return out


def _inject_recommended_metrics_from_consolidated(
    rec_aug: pd.DataFrame,
    full_table: pd.DataFrame | None,
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> pd.DataFrame:
    """Populate Recommended KPI columns from canonical consolidated rows by ``_scenario_row_key``.

    Recommended selection/winner logic still comes from optimizer-space feasible sets, but the numbers shown in
    the table should come from the same canonical consolidated source used across the rest of the app.
    """
    if rec_aug is None or len(rec_aug) == 0:
        return rec_aug
    if full_table is None or len(full_table) == 0:
        return rec_aug
    if SCENARIO_ROW_KEY_FIELD not in rec_aug.columns or SCENARIO_ROW_KEY_FIELD not in full_table.columns:
        return rec_aug

    out = rec_aug.copy()
    _ly_inj = int(lifetime_years)
    for _cn in per_capex_ratio_column_names(_ly_inj):
        if _cn not in out.columns:
            out[_cn] = np.nan
    full = full_table.copy()
    full[SCENARIO_ROW_KEY_FIELD] = full[SCENARIO_ROW_KEY_FIELD].astype(str)
    full = full.drop_duplicates(subset=[SCENARIO_ROW_KEY_FIELD], keep="first")

    bill_col = _df_bill_column(full)
    co2_col = _df_co2_avoided_column(full)
    metric_map = {
        "NPV (€)": "NPV (€)",
        "Payback (yrs)": "Payback (yrs)",
        "SCR": "Self-consumption ratio (%)",
        "Export ratio (% of PV gen)": "Export ratio (% of PV gen)",
        COL_ANNUAL_ELECTRICITY_BILL_EUR: bill_col,
        "CO2 reduction (%)": "CO2 reduction (%)",
        "SSR": "Self-sufficiency (%)",
        "Annual savings (€)": "Annual savings (€)",
        "CO₂ savings (kg)": co2_col,
    }

    full_by_key = full.set_index(SCENARIO_ROW_KEY_FIELD)
    for i, key in enumerate(out[SCENARIO_ROW_KEY_FIELD].astype(str).tolist()):
        if key.startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX):
            continue
        if key not in full_by_key.index:
            continue
        row = full_by_key.loc[key]
        for rec_col, full_col in metric_map.items():
            if full_col in row.index:
                out.at[out.index[i], rec_col] = row[full_col]
        for _pk in per_capex_ratio_column_names(_ly_inj):
            if _pk in row.index and pd.notna(row.get(_pk)):
                out.at[out.index[i], _pk] = row[_pk]
    return out


def _sort_recommended_setups_df_by_sidebar_rank(
    rec_aug: pd.DataFrame,
    ranked: list[tuple[str, pd.Series]],
) -> pd.DataFrame:
    """Order rows like **Rank results by** on the filtered consolidated table: best keys first, infeasible rows last."""
    if rec_aug is None or len(rec_aug) == 0:
        return rec_aug
    if SCENARIO_ROW_KEY_FIELD not in rec_aug.columns:
        return rec_aug
    order_map: dict[str, int] = {}
    for i, (_sn, srow) in enumerate(ranked):
        if SCENARIO_ROW_KEY_FIELD not in srow.index:
            continue
        k = srow.get(SCENARIO_ROW_KEY_FIELD)
        if k is None or pd.isna(k):
            continue
        order_map[str(k)] = i
    keys = rec_aug[SCENARIO_ROW_KEY_FIELD].astype(str)

    def _sort_tuple(i: int) -> tuple:
        k = str(keys.iloc[i])
        if k.startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX):
            return (2, k)
        if k in order_map:
            return (0, order_map[k])
        return (1, k)

    order = sorted(range(len(rec_aug)), key=lambda i: _sort_tuple(i))
    return rec_aug.iloc[order].reset_index(drop=True)


RECOMMENDED_SETUPS_EXPORT_NOTE_COL = "Recommended setups note"


def recommended_setups_join_consolidated_kpis_df(
    rec_aug: pd.DataFrame,
    full_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    One row per Recommended setups grid row: same KPI columns as the consolidated / Full results export
    when ``SCENARIO_ROW_KEY_FIELD`` matches ``full_table``. Infeasible / unmatched rows keep KPI cells empty
    and set ``Recommended setups note``.
    """
    if rec_aug is None or len(rec_aug) == 0:
        return pd.DataFrame()
    if full_table is None or len(full_table) == 0:
        return pd.DataFrame()
    if SCENARIO_ROW_KEY_FIELD not in full_table.columns or SCENARIO_ROW_KEY_FIELD not in rec_aug.columns:
        return pd.DataFrame()
    note_col = RECOMMENDED_SETUPS_EXPORT_NOTE_COL
    base_cols = list(full_table.columns)
    rows: list[dict[str, object]] = []
    for _, r in rec_aug.iterrows():
        k = str(r[SCENARIO_ROW_KEY_FIELD])
        if k.startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX):
            blank = {c: np.nan for c in base_cols}
            if "Tariff" in base_cols:
                blank["Tariff"] = r.get("Tariff")
            if "Scenario" in base_cols:
                blank["Scenario"] = r.get("Scenario family")
            blank[note_col] = str(r.get("Note", "") or "")
            rows.append(blank)
            continue
        hit = full_table[full_table[SCENARIO_ROW_KEY_FIELD].astype(str) == k]
        if len(hit) == 0:
            blank = {c: np.nan for c in base_cols}
            if "Tariff" in base_cols:
                blank["Tariff"] = r.get("Tariff")
            if "Scenario" in base_cols:
                blank["Scenario"] = r.get("Scenario family")
            blank[note_col] = "No matching row in consolidated results table."
            rows.append(blank)
        else:
            row_d = hit.iloc[0].to_dict()
            row_d[note_col] = ""
            rows.append(row_d)
    out = pd.DataFrame(rows)
    if note_col in out.columns:
        others = [c for c in out.columns if c != note_col]
        out = out[others + [note_col]]
    return out


def render_recommended_setups_tab_section(
    *,
    subheader: str,
    header_caption: str = "",
    grid_key: str,
    selection_session_key: str,
    download_key_plain: str,
    download_key_full: str,
    csv_filename_plain: str,
    csv_filename_full: str,
    plotly_chart_key_prefix: str,
    selection_detail_heading: str,
    tradeoff_expander_title: str,
    download_help_full: str,
    constraints_label: str,
    goal: str,
    ly: int,
    full_table_rank: pd.DataFrame,
    hard_filtered_rank_df: pd.DataFrame,
    ranked: list[tuple[str, pd.Series]],
) -> pd.DataFrame:
    """UI for **Recommended setups**: Decision constraints define the feasible set; sidebar **Rank results by** ranks winners."""
    st.subheader(subheader)
    if header_caption and str(header_caption).strip():
        st.caption(header_caption)

    _rank_goal = str(st.session_state.get("view_goal", RECOMMENDED_WINNER_PRESETS[0][1]))
    _preset_id = RECOMMENDED_WINNER_PRESET_ID_BY_LABEL.get(_rank_goal, RECOMMENDED_WINNER_PRESET_DEFAULT)
    _preset_label = RECOMMENDED_WINNER_PRESET_LABEL_BY_ID.get(_preset_id, _rank_goal)

    _payback_max_rec = float(st.session_state.hard_payback_max_years) if st.session_state.get("hard_payback_max_en") else None

    _rec_night_last_run: Optional[bool] = None
    if ENABLE_BATTERY_UI and st.session_state.get("opt_dfs") is not None:
        _rec_night_last_run = bool(st.session_state.last_battery_settings.charge_from_grid_at_night)
    _rec_kw = _recommended_build_params_from_sidebar_hard_filters()
    _rec_scenario_type_ui = str(st.session_state.get("view_scenario_type", "All scenarios"))
    _rec_tariff_family_ui = str(st.session_state.get("view_tariff_family", "All tariff types"))
    _prep = st.session_state.get("prepared_df")
    _base_co2_kg: float | None = None
    if _prep is not None and len(_prep) > 0 and "consumption" in _prep.columns:
        try:
            _base_co2_kg = float(pd.to_numeric(_prep["consumption"], errors="coerce").fillna(0.0).sum()) * float(
                _grid_co2_factor()
            )
        except Exception:
            _base_co2_kg = None
    _rec_profiles_full = list(st.session_state.get("last_tariff_profiles") or _default_tariff_profiles())
    _rec_df = build_recommended_setups_summary_df(
        st.session_state.opt_dfs,
        _rec_profiles_full,
        enable_battery_ui=ENABLE_BATTERY_UI,
        scenario_type_ui=_rec_scenario_type_ui,
        tariff_family_ui=_rec_tariff_family_ui,
        max_payback_years=float(_rec_kw["max_payback_years"]),
        min_self_consumption_pct=float(_rec_kw["min_self_consumption_pct"]),
        max_export_ratio_pct=float(_rec_kw["max_export_ratio_pct"]),
        require_positive_npv=bool(_rec_kw["require_positive_npv"]),
        require_positive_co2_savings=bool(_rec_kw["require_positive_co2_savings"]),
        npv_min_eur=_rec_kw["npv_min_eur"],  # type: ignore[arg-type]
        min_co2_reduction_pct=_rec_kw["min_co2_reduction_pct"],  # type: ignore[arg-type]
        grid_baseline_annual_co2_kg=_base_co2_kg,
        charge_from_grid_at_night_last_run=_rec_night_last_run,
        winner_preset=_preset_id,
        prepared_df=_prep if isinstance(_prep, pd.DataFrame) else None,
    )
    if len(_rec_df) == 0:
        if st.session_state.get("opt_dfs") is not None:
            st.info(
                "No **Recommended setups** rows for the current **Scenario type** / **Tariff family** filters "
                "(or **Grid only** is selected — there is no PV/battery sizing row). "
                "Try **All scenarios** and **All tariff types**, or change the sidebar filters."
            )
        else:
            st.info("Run **Run analysis** first — this table is filled from the optimizer results.")
    else:
        _rec_profiles = _filter_tariff_profiles_by_family_ui(_rec_profiles_full, _rec_tariff_family_ui)
        _rec_df_grid = augment_recommended_df_with_scenario_row_keys(_rec_df, _rec_profiles)
        _rec_df_grid = _inject_recommended_metrics_from_consolidated(
            _rec_df_grid, full_table_rank, lifetime_years=ly
        )
        _rec_df_grid = _sort_recommended_setups_df_by_sidebar_rank(_rec_df_grid, ranked)
        render_recommended_snapshot_cards_from_table(
            _rec_df_grid,
            goal=goal,
            payback_max=_payback_max_rec,
        )
        st.divider()
        _feasible_row_keys = [
            str(k)
            for k in _rec_df_grid[SCENARIO_ROW_KEY_FIELD].tolist()
            if not str(k).startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX)
        ]
        if _feasible_row_keys:
            _cur_rec_sel = st.session_state.get(selection_session_key)
            if (
                _cur_rec_sel is None
                or str(_cur_rec_sel).startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX)
                or str(_cur_rec_sel) not in _feasible_row_keys
            ):
                st.session_state[selection_session_key] = _feasible_row_keys[0]
        _hf_sig = "".join("1" if bool(st.session_state.get(k)) else "0" for k in HARD_FILTER_ENABLE_KEYS)
        _rec_grid_extra = (
            f"{_preset_id}|{float(_rec_kw['max_payback_years'])}|{float(_rec_kw['min_self_consumption_pct'])}|"
            f"{float(_rec_kw['max_export_ratio_pct'])}|{int(bool(_rec_kw['require_positive_npv']))}|"
            f"{'' if _rec_kw['npv_min_eur'] is None else float(_rec_kw['npv_min_eur'])}|"
            f"{int(bool(_rec_kw['require_positive_co2_savings']))}|"
            f"{'' if _rec_kw['min_co2_reduction_pct'] is None else float(_rec_kw['min_co2_reduction_pct'])}|"
            f"{_rec_night_last_run!s}|h{_hf_sig}|sc:{_rec_scenario_type_ui}|tf:{_rec_tariff_family_ui}|g:{goal}"
        )
        filtered_rec = render_aggrid_results_table(
            _rec_df_grid,
            grid_key=grid_key,
            height=380,
            default_rank_goal=None,
            rank_goal_table=None,
            lifetime_years=ly,
            selection_session_key=selection_session_key,
            grid_key_extra=_rec_grid_extra,
            aggrid_integer_round_cols=RECOMMENDED_SETUPS_AGGRID_INTEGER_NUMERIC_COLS,
            enable_column_filters=False,
        )
        _preset_disp = RECOMMENDED_WINNER_PRESET_LABEL_BY_ID.get(_preset_id, _preset_label)
        st.caption(
            f"**{len(filtered_rec):,}** row{'s' if len(filtered_rec) != 1 else ''} · per-cell winner: **{_preset_disp}** · "
            "table order = **Rank results by** (infeasible rows at bottom)."
        )
        _rec_ass_base_dl = last_run_assumptions_snapshot_df()
        _rec_ass_extra_dl = recommended_setups_constraint_assumptions_df(
            max_payback_years=float(_rec_kw["max_payback_years"]),
            min_self_consumption_pct=float(_rec_kw["min_self_consumption_pct"]),
            max_export_ratio_pct=float(_rec_kw["max_export_ratio_pct"]),
            require_positive_npv=bool(_rec_kw["require_positive_npv"]),
            require_positive_co2_savings=bool(_rec_kw["require_positive_co2_savings"]),
            npv_min_eur=_rec_kw["npv_min_eur"],  # type: ignore[arg-type]
            min_co2_reduction_pct=_rec_kw["min_co2_reduction_pct"],  # type: ignore[arg-type]
            winner_preset_id=_preset_id,
        )
        if _rec_ass_base_dl is None:
            _rec_ass_combined_dl = _rec_ass_extra_dl
        else:
            _rec_ass_combined_dl = pd.concat([_rec_ass_base_dl, _rec_ass_extra_dl], ignore_index=True)
        _rec_kpi_all = recommended_setups_join_consolidated_kpis_df(_rec_df_grid, full_table_rank)
        _csv_rec_kpi_plain = (
            _export_results_df_for_csv(_rec_kpi_all).to_csv(index=False).encode("utf-8-sig")
            if len(_rec_kpi_all)
            else b"\xef\xbb\xbf"
        )
        _csv_rec_kpi_full = encode_csv_assumptions_block_then_results_df(
            _rec_ass_combined_dl, _export_results_df_for_csv(_rec_kpi_all)
        )
        st.markdown("##### Download CSV")
        st.caption(
            "Same **consolidated KPI columns** as **Full results** (one row per recommended sizing; last column "
            f"**{RECOMMENDED_SETUPS_EXPORT_NOTE_COL}** is empty when the row matched the full scenario table)."
        )
        rda, rdb = st.columns(2)
        with rda:
            st.download_button(
                "All rows — full KPIs (CSV)",
                data=_csv_rec_kpi_plain,
                file_name=csv_filename_plain,
                mime="text/csv",
                key=download_key_plain,
                help="Consolidated scenario columns (like Full results), one row per recommended row.",
            )
        with rdb:
            st.download_button(
                "All rows — full KPIs + assumptions (CSV)",
                data=_csv_rec_kpi_full,
                file_name=csv_filename_full,
                mime="text/csv",
                key=download_key_full,
                help=download_help_full,
            )
        _rk_rec = st.session_state.get(selection_session_key)
        _rec_sel_row = None
        if (
            _rk_rec is not None
            and len(filtered_rec) > 0
            and SCENARIO_ROW_KEY_FIELD in filtered_rec.columns
        ):
            _hit_rec = filtered_rec[filtered_rec[SCENARIO_ROW_KEY_FIELD].astype(str) == str(_rk_rec)]
            if len(_hit_rec) > 0:
                _rec_sel_row = _hit_rec.iloc[0]
        _rec_bottom_cons_row: pd.Series | None = None
        if _rec_sel_row is not None:
            st.divider()
            st.markdown(f"##### {selection_detail_heading} — KPIs and charts")
            _rec_key_s = (
                str(_rec_sel_row[SCENARIO_ROW_KEY_FIELD])
                if SCENARIO_ROW_KEY_FIELD in _rec_sel_row.index
                else ""
            )
            if _rec_key_s.startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX):
                st.info(
                    str(
                        _rec_sel_row.get(
                            "Note",
                            f"No feasible PV/battery sizing for this tariff × scenario under the {constraints_label} constraints.",
                        )
                    )
                )
            else:
                _cons_hits = full_table_rank[
                    full_table_rank[SCENARIO_ROW_KEY_FIELD].astype(str) == _rec_key_s
                ]
                if len(_cons_hits) == 0:
                    st.warning(
                        "Could not match this row to the full scenario table. Run **Run analysis** again if you changed tariffs or bounds."
                    )
                else:
                    _rec_nb_wk = hashlib.sha256(str(_rec_key_s).encode("utf-8")).hexdigest()[:14]
                    _rec_bottom_cons_row = _cons_hits.iloc[0]
                    render_consolidated_selection_detail_block(
                        _rec_bottom_cons_row,
                        full_table_rank=full_table_rank,
                        hard_filtered_rank_df=hard_filtered_rank_df,
                        ranked=ranked,
                        goal=goal,
                        ly=ly,
                        tradeoff_expander_title=tradeoff_expander_title,
                        comparison_selection_caption="",
                        plotly_chart_key_prefix=plotly_chart_key_prefix,
                        prominent_header=True,
                        show_secondary_detail=False,
                        show_tradeoff_expanded_inline=True,
                        show_cumulative_expanded_inline=False,
                    )
                    if st.session_state.get("prepared_df") is not None and len(st.session_state.prepared_df) > 0:
                        render_recommended_monthly_notebook_style_charts(
                            _rec_bottom_cons_row,
                            st.session_state.prepared_df,
                            _rec_profiles,
                            st.session_state.last_battery_settings,
                            plotly_chart_key_prefix=plotly_chart_key_prefix,
                            widget_key_suffix=_rec_nb_wk,
                        )
        st.divider()
        if _rec_bottom_cons_row is not None:
            _pv_bb = int(_rec_bottom_cons_row["PV (kWp)"]) if "PV (kWp)" in _rec_bottom_cons_row.index else 0
            _bt_bb = int(_rec_bottom_cons_row["Battery (kWh)"]) if "Battery (kWh)" in _rec_bottom_cons_row.index else 0
            _render_cumulative_outlook_expander_for_row(
                _rec_bottom_cons_row,
                _pv_bb,
                _bt_bb,
                ly,
                plotly_chart_key_prefix=f"{plotly_chart_key_prefix}_bottom",
                expanded_inline=True,
            )
        else:
            st.markdown("##### Cumulative outlook (selected scenario)")
            st.caption(
                f"{ly}-year views for the **selected** scenario only (same cash-flow assumptions as elsewhere)."
            )
            st.info(
                "Select a row in the **Recommended setups** table above to see cumulative savings, CO₂, and discounted cash flow."
            )
        _rec_sel_tariff = (
            str(_rec_bottom_cons_row.get("Tariff", "") or "").strip() if _rec_bottom_cons_row is not None else ""
        )
        render_all_tariffs_comparison_grouped_bars(
            hard_filtered_rank_df=hard_filtered_rank_df,
            scenario_type_ui=_rec_scenario_type_ui,
            tariff_family_ui=_rec_tariff_family_ui,
            goal=goal,
            ly=ly,
            selected_kpi_tariff=_rec_sel_tariff if _rec_sel_tariff else None,
            radio_session_key="recommended_all_tariffs_bar_scope",
        )
    return _rec_df


def build_full_scenario_results_df(
    opt_dfs: Dict[str, pd.DataFrame],
    prepared_df: pd.DataFrame,
    tariff_profiles: List[Dict],
    pv_cost_per_kwp: float,
    batt_cost_per_kwh: float,
    electricity_inflation_rate: float = 0.0,
    battery_replacement_year: int | None = None,
    battery_replacement_pct_of_batt_capex: float = 0.0,
    inverter_replacement_year: int | None = None,
    inverter_replacement_pct_of_pv_capex: float = 0.0,
    pso_levy_annual: float = 0.0,
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> pd.DataFrame:
    """
    Build a consolidated scenario table across ALL optimizer configurations.

    Source:
      - opt_dfs[tcol] rows for PV + Grid, PV + Battery + Grid, Battery + Grid
      - a synthetic Grid-only baseline row for each tariff
    """
    if opt_dfs is None or prepared_df is None:
        return pd.DataFrame()

    ly = int(lifetime_years)
    _cg = col_gross_savings(ly)
    _cnb = col_net_benefit(ly)

    cons = prepared_df["consumption"].to_numpy(dtype=float)

    rows: list[pd.DataFrame] = []
    profiles = list(tariff_profiles or [])
    for p in profiles:
        tcol = str(p.get("col", "") or "")
        tname = str(p.get("name", "") or tcol)
        standing_charge = float(p.get("standing_charge", 0.0) or 0.0)
        if not tcol:
            continue
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
        df[COL_GRID_IMPORT_KWH] = pd.to_numeric(df["grid_import_kwh"], errors="coerce")
        _batt_charge = (
            pd.to_numeric(df["battery_charge_kwh"], errors="coerce")
            if "battery_charge_kwh" in df.columns
            else pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        ).fillna(0.0)
        _batt_discharge = (
            pd.to_numeric(df["battery_discharge_kwh"], errors="coerce")
            if "battery_discharge_kwh" in df.columns
            else pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        ).fillna(0.0)
        df[COL_BATTERY_CHARGE_KWH] = _batt_charge
        df[COL_BATTERY_DISCHARGE_KWH] = _batt_discharge
        df[COL_EXPORT_TO_GRID_KWH] = pd.to_numeric(df["export_kwh"], errors="coerce")
        df[COL_SELF_CONSUMED_PV_KWH] = pd.to_numeric(df["self_consumed_pv_kwh"], errors="coerce")
        df[COL_ANNUAL_GRID_IMPORT_COST_EUR] = pd.to_numeric(df["grid_import_cost_eur"], errors="coerce")
        _grid_co2 = float(_grid_co2_factor())
        _grid_import_num = pd.to_numeric(df["grid_import_kwh"], errors="coerce").fillna(0.0)
        _aco2 = _grid_import_num.astype(float) * _grid_co2
        df[COL_ANNUAL_GRID_CO2_EMISSIONS_KG] = _aco2
        df[COL_LIFETIME_CO2_KG] = _aco2 * float(ly)
        df[COL_ANNUAL_ELECTRICITY_BILL_EUR] = df["cost"]
        df["Annual savings (€)"] = df["savings"]
        df["Payback (yrs)"] = df["payback"]
        df["NPV (€)"] = df["npv"]
        df["IRR (%)"] = 100.0 * df["irr"]
        df["Self-sufficiency (%)"] = df["self_suff_pct"]
        df["Self-consumption ratio (%)"] = df["self_consumption_ratio_pct"]
        _pvgen_num = pd.to_numeric(df["pv_gen_kwh"], errors="coerce")
        _export_num = pd.to_numeric(df["export_kwh"], errors="coerce").fillna(0.0)
        _pv = _pvgen_num.to_numpy(dtype=float)
        _ex = _export_num.to_numpy(dtype=float)
        _export_ratio = np.full(len(_pv), np.nan, dtype=float)
        _ok = _pv > 1e-9
        _export_ratio[_ok] = 100.0 * np.maximum(0.0, _ex[_ok]) / _pv[_ok]
        df["Export ratio (% of PV gen)"] = _export_ratio
        baseline_co2_kg = float(cons.sum() * _grid_co2_factor())
        df["CO2 savings (kg)"] = np.maximum(
            0.0,
            baseline_co2_kg - df[COL_ANNUAL_GRID_CO2_EMISSIONS_KG].to_numpy(dtype=float),
        )
        df["CO2 reduction (%)"] = np.where(
            baseline_co2_kg > 1e-9,
            100.0 * df["CO2 savings (kg)"] / baseline_co2_kg,
            0.0,
        )
        df["Grid import reduction (kWh)"] = (
            np.where(_grid_co2 > 0, pd.to_numeric(df["CO2 savings (kg)"], errors="coerce") / _grid_co2, 0.0)
        )
        df[COL_ANNUAL_CO2_REDUCTION_KG] = df["CO2 savings (kg)"]

        df[SCENARIO_ROW_KEY_FIELD] = [
            compose_scenario_row_key(tcol, str(s), int(pv), int(bh))
            for s, pv, bh in zip(df["Scenario"].astype(str), df["PV (kWp)"], df["Battery (kWh)"])
        ]

        gross_lifetime = df["Annual savings (€)"].apply(
            lambda s: _gross_savings_lifetime(float(s), electricity_inflation_rate, ly)
        )
        batt_capex_eur = batt_kwh * float(batt_cost_per_kwh)
        pv_capex_eur = pv_kwp * float(pv_cost_per_kwp)
        batt_replacement_nominal = (
            batt_capex_eur * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= ly)
            else 0.0
        )
        inverter_replacement_nominal = (
            pv_capex_eur * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= ly)
            else 0.0
        )
        df[_cg] = gross_lifetime
        df[_cnb] = gross_lifetime - df["CAPEX (€)"] - batt_replacement_nominal - inverter_replacement_nominal

        (
            k_npv_per,
            k_co2_ann_per,
            k_co2_life_per,
            k_asav_per,
            k_gross_per,
        ) = per_capex_ratio_column_names(ly)
        _capex_a = pd.to_numeric(df["CAPEX (€)"], errors="coerce").to_numpy(dtype=float)
        _npv_a = pd.to_numeric(df["NPV (€)"], errors="coerce").to_numpy(dtype=float)
        _co2r_a = pd.to_numeric(df[COL_ANNUAL_CO2_REDUCTION_KG], errors="coerce").to_numpy(dtype=float)
        _asav_a = pd.to_numeric(df["Annual savings (€)"], errors="coerce").to_numpy(dtype=float)
        _gross_a = pd.to_numeric(df[_cg], errors="coerce").to_numpy(dtype=float)
        _ok_capex = np.isfinite(_capex_a) & (_capex_a > 1e-9)
        df[k_npv_per] = np.where(_ok_capex, _npv_a / _capex_a, np.nan)
        df[k_co2_ann_per] = np.where(_ok_capex, _co2r_a / _capex_a, np.nan)
        df[k_co2_life_per] = np.where(_ok_capex, (_co2r_a * float(ly)) / _capex_a, np.nan)
        df[k_asav_per] = np.where(_ok_capex, _asav_a / _capex_a, np.nan)
        df[k_gross_per] = np.where(_ok_capex, _gross_a / _capex_a, np.nan)

        keep_cols = [
            "Tariff",
            "Scenario",
            "PV (kWp)",
            "Battery (kWh)",
            COL_ANNUAL_ELECTRICITY_BILL_EUR,
            COL_ANNUAL_GRID_IMPORT_COST_EUR,
            COL_GRID_IMPORT_KWH,
            COL_BATTERY_CHARGE_KWH,
            COL_BATTERY_DISCHARGE_KWH,
            COL_EXPORT_TO_GRID_KWH,
            COL_SELF_CONSUMED_PV_KWH,
            "Total annual PV generation (kWh)",
            "Annual savings (€)",
            "CAPEX (€)",
            "Payback (yrs)",
            "NPV (€)",
            "IRR (%)",
            COL_ANNUAL_GRID_CO2_EMISSIONS_KG,
            COL_LIFETIME_CO2_KG,
            COL_ANNUAL_CO2_REDUCTION_KG,
            "CO2 reduction (%)",
            "Self-sufficiency (%)",
            "Self-consumption ratio (%)",
            "Export ratio (% of PV gen)",
            "Grid import reduction (kWh)",
            _cg,
            _cnb,
            k_npv_per,
            k_co2_ann_per,
            k_co2_life_per,
            k_asav_per,
            k_gross_per,
            SCENARIO_ROW_KEY_FIELD,
        ]
        rows.append(df[keep_cols])

        # Synthetic Grid-only baseline per tariff
        tariff_series = prepared_df[tcol].to_numpy(dtype=float)
        baseline_energy_cost = float((cons * tariff_series).sum())
        baseline_cost = baseline_energy_cost + standing_charge + float(pso_levy_annual)
        total_cons_kwh = float(cons.sum())
        annual_co2_grid_only = total_cons_kwh * _grid_co2_factor()

        baseline_row = {
            "Tariff": tname,
            "Scenario": "Grid only",
            "PV (kWp)": 0,
            "Battery (kWh)": 0,
            COL_ANNUAL_ELECTRICITY_BILL_EUR: baseline_cost,
            COL_ANNUAL_GRID_IMPORT_COST_EUR: baseline_energy_cost,
            COL_GRID_IMPORT_KWH: total_cons_kwh,
            COL_BATTERY_CHARGE_KWH: 0.0,
            COL_BATTERY_DISCHARGE_KWH: 0.0,
            COL_EXPORT_TO_GRID_KWH: 0.0,
            COL_SELF_CONSUMED_PV_KWH: 0.0,
            "Total annual PV generation (kWh)": 0.0,
            "Annual savings (€)": 0.0,
            "CAPEX (€)": 0.0,
            "Payback (yrs)": float("inf"),
            "NPV (€)": 0.0,
            "IRR (%)": float("nan"),
            COL_ANNUAL_GRID_CO2_EMISSIONS_KG: annual_co2_grid_only,
            COL_LIFETIME_CO2_KG: annual_co2_grid_only * float(ly),
            COL_ANNUAL_CO2_REDUCTION_KG: 0.0,
            "CO2 reduction (%)": 0.0,
            "Self-sufficiency (%)": 0.0,
            "Self-consumption ratio (%)": 0.0,
            "Export ratio (% of PV gen)": np.nan,
            "Grid import reduction (kWh)": 0.0,
            _cg: 0.0,
            _cnb: 0.0,
            k_npv_per: float("nan"),
            k_co2_ann_per: float("nan"),
            k_co2_life_per: float("nan"),
            k_asav_per: float("nan"),
            k_gross_per: float("nan"),
            SCENARIO_ROW_KEY_FIELD: compose_scenario_row_key(tcol, "Grid only", 0, 0),
        }
        rows.append(pd.DataFrame([baseline_row]))

    if len(rows) == 0:
        return pd.DataFrame()
    res = pd.concat(rows, ignore_index=True)
    if len(res) > 0:
        # Annual electricity bill reduction (%) = Annual savings (€) vs tariff-specific grid-only annual bill (€).
        # Grid-only annual bill (€) is stored in the synthetic "Grid only" row per tariff.
        grid = (
            res.loc[res["Scenario"] == "Grid only", ["Tariff", COL_ANNUAL_ELECTRICITY_BILL_EUR]]
            .rename(columns={COL_ANNUAL_ELECTRICITY_BILL_EUR: "_grid_only_annual_cost"})
            .copy()
        )
        res = res.merge(grid, on="Tariff", how="left")
        grid_cost = pd.to_numeric(res["_grid_only_annual_cost"], errors="coerce")
        annual_savings = pd.to_numeric(res["Annual savings (€)"], errors="coerce")
        saving_pct = np.where(
            (grid_cost.notna()) & (grid_cost > 0),
            100.0 * annual_savings / grid_cost,
            0.0,
        )
        res[COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT] = saving_pct
        res = res.drop(columns=["_grid_only_annual_cost"])
    if len(res) > 0 and SCENARIO_ROW_KEY_FIELD in res.columns:
        _dup = res[SCENARIO_ROW_KEY_FIELD].duplicated(keep=False)
        if _dup.any():
            warnings.warn(
                f"Consolidated results contain duplicate {SCENARIO_ROW_KEY_FIELD} ({int(_dup.sum())} rows); "
                "row identity may be ambiguous.",
                UserWarning,
                stacklevel=2,
            )
    return res


def _prep_df_for_aggrid(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN so Ag Grid number filters behave sensibly."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def _format_numeric_columns_for_aggrid(
    df: pd.DataFrame,
    *,
    integer_round_cols: frozenset[str] | None = None,
) -> pd.DataFrame:
    """One decimal for numeric KPIs; whole numbers for PV / battery size columns (and optional ``integer_round_cols``)."""
    out = df.copy()
    int_cols = {"PV (kWp)", "Battery (kWh)"} | frozenset(integer_round_cols or ())
    for c in out.columns:
        if c in int_cols:
            num = pd.to_numeric(out[c], errors="coerce")
            out[c] = num.round(0).astype("Int64")
        elif pd.api.types.is_numeric_dtype(out[c]):
            num = pd.to_numeric(out[c], errors="coerce")
            if "per € CAPEX" in str(c):
                out[c] = num.round(3)
            else:
                out[c] = num.round(1)
    return out


def goal_to_tariff_compare_chart_column(
    goal: str,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
    *,
    results_df: pd.DataFrame | None = None,
) -> str:
    """Map **Rank results by** to a numeric column on the consolidated / filtered results table (all-tariff bars)."""
    ly = int(lifetime_years)
    _bill = _df_bill_column(results_df) if results_df is not None else COL_ANNUAL_ELECTRICITY_BILL_EUR
    _co2 = _df_co2_avoided_column(results_df) if results_df is not None else "CO2 savings (kg)"
    m = {
        "Balanced recommendation": "NPV (€)",
        "Best financial value": "NPV (€)",
        "Lowest annual bill": _bill,
        "Fastest payback": "Payback (yrs)",
        "Highest CO₂ saving": _co2,
        "Highest self-consumption": "Self-consumption ratio (%)",
        "Lowest annual electricity cost": _bill,
        "Highest annual savings": "Annual savings (€)",
        "Best payback": "Payback (yrs)",
        "Best self-sufficiency / lowest grid import": "Self-sufficiency (%)",
        "Highest annual CO2 savings": _co2,
        "Best cost–CO2 trade-off": _co2,
        "Best NPV": "NPV (€)",
        "Best IRR": "IRR (%)",
    }
    return m.get(goal, _bill)


def _build_all_tariffs_compare_long_from_filtered_rank_df(
    filtered_rank_df: pd.DataFrame,
    tariff_display_names: List[str],
    scenario_type_ui: str,
    goal: str,
    compare_metric: str,
    *,
    lifetime_years: int,
) -> pd.DataFrame | None:
    """One value per (Tariff, Scenario) = best row for ``goal`` within the **filtered** consolidated universe."""
    if filtered_rank_df is None or len(filtered_rank_df) == 0 or compare_metric not in filtered_rank_df.columns:
        return None
    base = _filter_by_scenario_type(filtered_rank_df, scenario_type_ui)
    if len(base) == 0:
        return None
    allowed = _scenario_allowed_for_filter(scenario_type_ui)
    scenario_order = [s for s in _SCENARIO_TYPES_ALL if s in allowed]
    rows: list[dict[str, object]] = []
    for tname in tariff_display_names:
        for scen in scenario_order:
            sub = base[(base["Tariff"].astype(str) == str(tname)) & (base["Scenario"].astype(str) == scen)]
            if len(sub) == 0:
                rows.append({"Tariff": tname, "Scenario": scen, compare_metric: float("nan")})
                continue
            best = _sort_consolidated_scenarios_for_goal(sub, goal)
            v = pd.to_numeric(best.iloc[0].get(compare_metric, float("nan")), errors="coerce")
            rows.append({"Tariff": tname, "Scenario": scen, compare_metric: float(v) if pd.notna(v) else float("nan")})
    return pd.DataFrame(rows)


def _aggrid_serializable_row_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to plain Python scalars for st_aggrid ``pre_selected_rows``."""
    out: dict = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, str, bytes)):
            out[k] = v
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        elif pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _reconcile_aggrid_row_selection(
    response: object,
    visible_df: pd.DataFrame,
    *,
    selection_session_key: str,
    row_key_field: str,
) -> None:
    """Keep ``selection_session_key`` aligned with the visible (filtered/sorted) grid rows."""
    keys_in_view: set[str] = set()
    if len(visible_df) > 0 and row_key_field in visible_df.columns:
        keys_in_view = set(visible_df[row_key_field].astype(str))
    picked: str | None = None
    sel_rows = None
    try:
        sel_rows = response.get("selected_rows") if hasattr(response, "get") else None
    except Exception:
        sel_rows = None
    # st_aggrid may return selected_rows as a list[dict] or (some versions) a DataFrame — never ``if df:``.
    if isinstance(sel_rows, pd.DataFrame):
        if len(sel_rows) > 0 and row_key_field in sel_rows.columns:
            v = sel_rows.iloc[0][row_key_field]
            if v is not None and not pd.isna(v):
                picked = str(v)
    elif isinstance(sel_rows, list) and len(sel_rows) > 0:
        r0 = sel_rows[0]
        if isinstance(r0, dict):
            if row_key_field in r0 and r0[row_key_field] is not None:
                picked = str(r0[row_key_field])
            else:
                for kk, vv in r0.items():
                    if str(kk) == row_key_field and vv is not None:
                        picked = str(vv)
                        break
        elif isinstance(r0, pd.Series):
            if row_key_field in r0.index and r0[row_key_field] is not None:
                vv = r0[row_key_field]
                if not pd.isna(vv):
                    picked = str(vv)
    if picked and picked in keys_in_view:
        st.session_state[selection_session_key] = picked
    elif len(visible_df) > 0 and row_key_field in visible_df.columns:
        st.session_state[selection_session_key] = str(visible_df.iloc[0][row_key_field])
    else:
        st.session_state[selection_session_key] = None


def render_aggrid_results_table(
    df: pd.DataFrame,
    *,
    grid_key: str,
    height: int = 380,
    caption: str | None = None,
    default_rank_goal: str | None = None,
    rank_goal_table: str | None = None,
    grid_key_extra: str | None = None,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
    selection_session_key: str | None = None,
    selection_row_key_field: str = SCENARIO_ROW_KEY_FIELD,
    display_column_allowlist: list[str] | None = None,
    aggrid_integer_round_cols: frozenset[str] | None = None,
    enable_column_filters: bool = True,
) -> pd.DataFrame:
    """Optional Excel-style per-column filters (client-side). Returns filtered + sorted rows from the grid.

    When ``default_rank_goal`` and ``rank_goal_table`` are set, rows are pre-ordered like **Rank results by**
    (``full`` = consolidated scenario table). Widget ``grid_key`` is suffixed
    with a goal hash so changing the goal remounts the grid with the new default order.
    ``display_column_allowlist`` optionally drops columns **after** goal-sort and **before** the grid; Ag Grid
    filters and sorts then apply only to the remaining columns.

    ``aggrid_integer_round_cols`` lists extra numeric columns to round to whole numbers (nullable integer) for display.

    Set ``enable_column_filters=False`` to hide column filters and the floating filter row (e.g. **Recommended setups**).

    By default this uses Streamlit's ``st.dataframe`` with **single-row selection** (click a row). Set environment
    variable ``REC_USE_AGGRID=1`` to use **st_aggrid** instead (Excel-style column filters; may not render on some hosts).
    """
    if df is None or len(df) == 0:
        st.info("No rows to display.")
        return pd.DataFrame()

    work_df = df
    effective_key = grid_key
    if default_rank_goal and rank_goal_table == "full":
        work_df = _sort_consolidated_scenarios_for_goal(df, default_rank_goal)
        effective_key = f"{grid_key}__g{_aggrid_goal_key_fragment(default_rank_goal)}"
    if grid_key_extra:
        effective_key = f"{effective_key}__x{hashlib.md5(grid_key_extra.encode('utf-8')).hexdigest()[:10]}"
    if not enable_column_filters:
        effective_key = f"{effective_key}__noflt"

    if not ENABLE_BATTERY_UI and "Battery (kWh)" in work_df.columns:
        work_df = work_df.drop(columns=["Battery (kWh)"])

    if display_column_allowlist:
        work_df = _subset_dataframe_display_columns(work_df, display_column_allowlist)

    display_df = _format_numeric_columns_for_aggrid(
        _prep_df_for_aggrid(work_df),
        integer_round_cols=aggrid_integer_round_cols,
    )
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(
        filter=bool(enable_column_filters),
        floatingFilter=bool(enable_column_filters),
        sortable=True,
        resizable=True,
        editable=False,
    )
    use_selection = bool(selection_session_key) and selection_row_key_field in display_df.columns
    pre_sel: list[dict] = []
    if use_selection:
        cur_key = st.session_state.get(selection_session_key)
        if cur_key is not None:
            _m = display_df[display_df[selection_row_key_field].astype(str) == str(cur_key)]
            if len(_m) > 0:
                pre_sel = [_aggrid_serializable_row_dict(_m.iloc[0])]
        gb.configure_selection(
            selection_mode="single",
            use_checkbox=False,
            suppressRowClickSelection=False,
            pre_selected_rows=pre_sel if pre_sel else None,
        )
    else:
        gb.configure_selection(selection_mode="disabled", suppressRowClickSelection=True)
    gb.configure_grid_options(rowHeight=28, headerHeight=34, suppressCellFocus=True)
    go = gb.build()
    go.setdefault("autoSizeStrategy", {"type": "fitGridWidth"})
    if not enable_column_filters:
        # ``from_dataframe`` attaches filter column types (e.g. ``numberColumnFilter``); clear those explicitly.
        _aggrid_col_types_with_filter_ui = frozenset(
            {"numberColumnFilter", "dateColumnFilter", "textColumnFilter", "agNumberColumnFilter", "agTextColumnFilter"}
        )
        for col_def in go.get("columnDefs") or []:
            col_def["filter"] = False
            col_def["floatingFilter"] = False
            t = col_def.get("type")
            if isinstance(t, list):
                col_def["type"] = [x for x in t if x not in _aggrid_col_types_with_filter_ui]
    if selection_row_key_field in display_df.columns:
        for col_def in go.get("columnDefs") or []:
            if col_def.get("field") == selection_row_key_field:
                col_def["hide"] = True
                break

    update_on_events: list[str] = ["sortChanged"]
    if enable_column_filters:
        update_on_events.append("filterChanged")
    if use_selection:
        update_on_events.append("selectionChanged")

    # Default: Streamlit's built-in dataframe + row selection (reliable on Streamlit Cloud). AgGrid is optional.
    # Set environment variable REC_USE_AGGRID=1 to use st_aggrid (Excel-style column filters; may not render on some hosts).
    _use_aggrid = _env_truthy("REC_USE_AGGRID")

    # Optional emergency fallback only (never tied to DEMO_MODE — that hid AgGrid and replaced row-click UX).
    _show_compat_table = _env_truthy("REC_SHOW_COMPAT_TABLE")
    if _show_compat_table:
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=max(260, int(height)),
        )
        if caption:
            st.caption(caption)
        if use_selection and selection_session_key:
            if len(display_df) > 0 and selection_row_key_field in display_df.columns:
                _keys = display_df[selection_row_key_field].astype(str).tolist()
                _cur = st.session_state.get(selection_session_key)
                st.session_state[selection_session_key] = str(_cur) if _cur is not None and str(_cur) in _keys else _keys[0]
            else:
                st.session_state[selection_session_key] = None
        return display_df.copy()

    if not _use_aggrid:
        _show_for_native = (
            display_df.drop(columns=[selection_row_key_field], errors="ignore")
            if selection_row_key_field in display_df.columns
            else display_df
        )
        _df_widget_key = f"nt_{hashlib.md5(str(effective_key).encode('utf-8')).hexdigest()[:26]}"
        if use_selection:
            event = st.dataframe(
                _show_for_native,
                width="stretch",
                hide_index=True,
                height=max(260, int(height)),
                on_select="rerun",
                selection_mode="single-row",
                key=_df_widget_key,
            )
        else:
            st.dataframe(
                _show_for_native,
                width="stretch",
                hide_index=True,
                height=max(260, int(height)),
                key=_df_widget_key,
            )
            event = None
        if caption:
            st.caption(caption)
        if use_selection and selection_session_key and selection_row_key_field in display_df.columns:
            _sel_rows: list[int] = []
            try:
                if event is not None and getattr(event, "selection", None) is not None:
                    _sel_rows = [int(x) for x in event.selection.rows]
            except Exception:
                _sel_rows = []
            _keys = display_df[selection_row_key_field].astype(str).tolist()
            if _sel_rows:
                _ix = _sel_rows[0]
                if 0 <= _ix < len(display_df):
                    st.session_state[selection_session_key] = str(
                        display_df.iloc[_ix][selection_row_key_field]
                    )
            elif len(display_df) > 0:
                _cur = st.session_state.get(selection_session_key)
                if _cur is None or str(_cur) not in _keys:
                    st.session_state[selection_session_key] = str(
                        display_df.iloc[0][selection_row_key_field]
                    )
        return display_df.copy()

    response = AgGrid(
        display_df,
        gridOptions=go,
        update_on=update_on_events,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        height=height,
        key=effective_key,
    )
    if caption:
        st.caption(caption)

    raw = response.get("data")
    if raw is None:
        out = pd.DataFrame()
    elif isinstance(raw, pd.DataFrame):
        out = raw if len(raw) else pd.DataFrame()
    else:
        try:
            out = pd.DataFrame(raw)
            if not len(out):
                out = pd.DataFrame()
        except Exception:
            out = pd.DataFrame()
    if use_selection and selection_session_key:
        _reconcile_aggrid_row_selection(
            response,
            out,
            selection_session_key=selection_session_key,
            row_key_field=selection_row_key_field,
        )
    return out


def build_kpi_guide_table(lifetime_years: int = DEFAULT_LIFETIME_YEARS) -> pd.DataFrame:
    """Single source of truth for KPI definitions (landing page + Settings tab)."""
    ly = int(lifetime_years)
    self_cons_meaning = (
        "PV energy used locally: direct PV to load plus PV-origin battery discharge to load."
        if ENABLE_BATTERY_UI
        else "PV energy used locally (direct PV to load)."
    )
    _npv_pc, _co2a_pc, _co2l_pc, _as_pc, _gr_pc = per_capex_ratio_column_names(ly)
    return pd.DataFrame(
        [
            {
                "KPI": "Analysis horizon (years)",
                "Meaning / formula": (
                    f"**N = {ly}** from **Model setup → Financial assumptions → Analysis horizon (years)**. "
                    f"It is the same **{ly}** used in **{col_npv(ly)}**, **{col_irr(ly)}**, **{col_gross_savings(ly)}**, "
                    f"**{col_net_benefit(ly)}**, and in **{COL_LIFETIME_CO2_KG}** (annual grid-import scenario CO₂ × **{ly}** — "
                    "a straight multi-year scale of year-1 operational emissions, not a year-by-year dynamic forecast)."
                ),
            },
            {
                "KPI": COL_ANNUAL_GRID_IMPORT_COST_EUR,
                "Meaning / formula": "Σ(Grid import kWh × tariff) for the simulated year — import energy cost only (before export income, standing charge, PSO, OPEX). Same quantity as **Annual cost of grid import (€)** in scenario detail tiles.",
            },
            {"KPI": "Annual export earnings (€)", "Meaning / formula": "Exported energy multiplied by export rate over the year."},
            {
                "KPI": COL_NET_IMPORT_EXPORT_COST_EUR,
                "Meaning / formula": "From the hourly simulation only: grid import cost minus export earnings (no standing charge, PSO levy, or OPEX).",
            },
            {"KPI": "Annual electricity cost (€)", "Meaning / formula": "Grid import cost minus export earnings, plus standing charge, PSO levy, and OPEX (detail tiles)."},
            {
                "KPI": COL_ANNUAL_ELECTRICITY_BILL_EUR,
                "Meaning / formula": "Same year-1 total as **Annual electricity cost (€)** — label used in the **Full results** grid and ranking.",
            },
            {"KPI": COL_GRID_IMPORT_KWH, "Meaning / formula": "Total kWh imported from the grid over the simulated year."},
            {
                "KPI": COL_BATTERY_CHARGE_KWH,
                "Meaning / formula": "Total battery charging energy over the simulated year: charge from PV plus charge from grid. Zero for non-battery scenarios.",
            },
            {
                "KPI": COL_BATTERY_DISCHARGE_KWH,
                "Meaning / formula": "Total battery discharge delivered to load over the simulated year. Zero for non-battery scenarios.",
            },
            {"KPI": COL_EXPORT_TO_GRID_KWH, "Meaning / formula": "Total kWh exported to the grid over the simulated year (PV/battery surplus)."},
            {"KPI": COL_SELF_CONSUMED_PV_KWH, "Meaning / formula": self_cons_meaning},
            {
                "KPI": "Annual savings vs grid only (€)",
                "Meaning / formula": "Grid-only annual electricity cost (€) minus scenario annual electricity cost (€) (year 1).",
            },
            {
                "KPI": COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT,
                "Meaning / formula": "Year-1 annual savings (€) divided by the **same tariff’s** grid-only annual electricity bill (€), times 100.",
            },
            {"KPI": "Self-Consumption (kWh)", "Meaning / formula": self_cons_meaning},
            {"KPI": "Self-consumption ratio (%)", "Meaning / formula": "Self-Consumption (kWh) divided by total PV generation, times 100."},
            {
                "KPI": "Export ratio (% of PV gen)",
                "Meaning / formula": "Annual export to grid (kWh) divided by annual PV generation (kWh), times 100. Empty when there is no PV.",
            },
            {"KPI": "Self-sufficiency ratio (%)", "Meaning / formula": "Local renewable energy supplied to load divided by total consumption, times 100."},
            {"KPI": "Payback period (years)", "Meaning / formula": "CAPEX divided by year-1 annual savings (simple payback)."},
            {"KPI": col_npv(ly), "Meaning / formula": f"Present value of {ly}-year savings stream minus CAPEX and discounted replacement costs."},
            {"KPI": col_irr(ly), "Meaning / formula": "Discount rate at which NPV equals zero for the same cashflow assumptions."},
            {"KPI": col_gross_savings(ly), "Meaning / formula": f"Sum of yearly savings over {ly} years (inflated if electricity inflation > 0)."},
            {"KPI": col_net_benefit(ly), "Meaning / formula": f"Gross {ly}-year savings minus CAPEX and nominal replacement costs."},
            {
                "KPI": COL_ANNUAL_GRID_CO2_EMISSIONS_KG,
                "Meaning / formula": "Scenario **operational** CO₂ from **grid imports** for the simulated year (kg): emissions factor × annual import kWh. Not savings vs baseline.",
            },
            {"KPI": "CO2 (kg)", "Meaning / formula": "Same quantity as **Annual grid CO₂ emissions (kg)** in hourly KPI rows."},
            {
                "KPI": COL_LIFETIME_CO2_KG,
                "Meaning / formula": (
                    f"**{COL_ANNUAL_GRID_CO2_EMISSIONS_KG}** × **{ly}** (same **analysis horizon** as NPV/IRR): cumulative **scenario** "
                    "grid-import CO₂ if year-1 operational emissions repeated each year. "
                    f"**Lifetime CO₂ avoided vs grid-only (kg)** = "
                    f"(grid-only annual CO₂ − scenario annual CO₂) × **{ly}**, i.e. **{COL_ANNUAL_CO2_REDUCTION_KG}** × **{ly}** "
                    f"(used in **{_co2l_pc}**)."
                ),
            },
            {
                "KPI": "Annual CO2 savings (kg)",
                "Meaning / formula": f"Grid-only CO2 minus scenario CO2 (not below zero); same kg as **{COL_ANNUAL_CO2_REDUCTION_KG}** in the All scenario grid.",
            },
            {
                "KPI": COL_ANNUAL_CO2_REDUCTION_KG,
                "Meaning / formula": "Year-1 absolute reduction in grid-import CO₂ vs grid-only for the same tariff (not below zero); same values as **Annual CO2 savings (kg)** elsewhere.",
            },
            {"KPI": "CO2 reduction (%)", "Meaning / formula": "Annual CO2 savings divided by grid-only CO2, times 100."},
            {
                "KPI": _npv_pc,
                "Meaning / formula": "**NPV (€)** ÷ **CAPEX (€)** (same horizon as NPV). Shown as **—** when CAPEX = 0.",
            },
            {
                "KPI": _co2a_pc,
                "Meaning / formula": f"**{COL_ANNUAL_CO2_REDUCTION_KG}** ÷ **CAPEX (€)**. Shown as **—** when CAPEX = 0.",
            },
            {
                "KPI": _co2l_pc,
                "Meaning / formula": (
                    f"**{COL_ANNUAL_CO2_REDUCTION_KG}** × **{ly}** ÷ **CAPEX (€)** — lifetime avoided CO₂ vs grid-only at the analysis horizon, per euro CAPEX."
                ),
            },
            {
                "KPI": _as_pc,
                "Meaning / formula": "**Annual savings (€)** ÷ **CAPEX (€)** (year 1). Shown as **—** when CAPEX = 0.",
            },
            {
                "KPI": _gr_pc,
                "Meaning / formula": f"**{col_gross_savings(ly)}** ÷ **CAPEX (€)**. Shown as **—** when CAPEX = 0.",
            },
        ]
    )


def render_full_how_to_use_guide() -> None:
    """Long-form reference: files, KPIs, assumptions. Used in expander (first visit) and Settings tab."""
    st.markdown("### Files you need")
    st.markdown(
        """
- **Consumption CSV:** hourly kWh with a **`date`** column (e.g. `DD/MM/YYYY HH:00`) and consumption (often **`Final_Community_Sum`**). One row per hour.
- **PV CSV (PVGIS-style):** columns **`time`** (e.g. `YYYYMMDD:HH11`) and **`P`** = production in **Wh for 1 kWp**. The app converts Wh to kWh per kWp and scales by PV size.

You can **run without uploads**: defaults load from **`REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV`** / **`REC_FEASIBILITY_DEFAULT_PV_CSV`** (if set), else **`data/local_consumption.csv`** / **`data/local_pv.csv`** if present, else bundled **`data/default_consumption.csv`** and **`data/default_pv.csv`** (same parsers as uploads). Upload your own files to override either one. **Tariffs** use a **matrix** in **Model setup → Data & tariffs**: defaults from **`REC_FEASIBILITY_DEFAULT_TARIFFS_CSV`** / **`data/local_tariffs.csv`** / bundled **`data/default_tariffs.csv`**, optional **Load tariffs from CSV** to replace the table, and **Include** checkboxes so only selected rows run in the optimizer.

When you use custom files, both series should cover the **same hourly period** so rows align after merge.
"""
    )

    st.markdown("### Main KPIs")
    ss_extra = " and PV-origin battery discharge" if ENABLE_BATTERY_UI else ""
    sc_extra = " or via battery from PV" if ENABLE_BATTERY_UI else ""
    st.markdown(
        f"""
- **Annual electricity cost:** what you pay for imports (minus export income), plus standing charge, PSO levy, and OPEX.
- **Annual savings vs grid only:** how much cheaper the scenario is than grid-only in year 1.
- **Payback / NPV / IRR:** how fast the investment pays back and how attractive it is over your **analysis horizon** (years) with your discount rate, inflation, and optional replacements.
- **Self-sufficiency:** share of consumption met by **local renewable** supply (PV to load{ss_extra}).
- **Self-consumption ratio:** share of **PV generation** used on site (directly{sc_extra}).
- **CO2 savings / reduction:** grid-only emissions minus scenario emissions (savings never shown below zero). **Lifetime CO2 (kg)** in results scales year-1 scenario grid-import CO₂ by the same **analysis horizon** (see **KPI formulas** table).
- **Ratios per € CAPEX:** NPV, annual and lifetime CO₂ avoided (vs grid-only), year-1 annual savings, and **gross** lifetime savings — each divided by scenario **CAPEX (€)** (shown as **—** when CAPEX is zero).
"""
    )

    st.markdown("### Key assumptions (main setup, before Run)")
    landing_optimizer_lines = (
        """
- **PV size bounds (kWp):** Slider min/max; only PV sizes in this range are tried.
- **Battery size bounds (kWh):** Slider min/max; only battery sizes in this range are tried.
- **Speed preset (Quick / Fast / Full):** Sets step sizes on those grids—larger steps → fewer combinations and a faster run, but a coarser search. **Quick** = **5** kWp & **5** kWh steps, **Fast** = **10** & **10**, **Full** = **1** & **1**.
"""
        if ENABLE_BATTERY_UI
        else """
- **PV size bounds (kWp):** Slider min/max; only PV sizes in this range are tried.
- **Speed preset (Quick / Fast / Full):** Sets the PV step on the search grid—**Quick** **5** kWp, **Fast** **10**, **Full** **1**; larger steps → fewer sizes and a faster run, but a coarser search.
"""
    )
    landing_battery_finance = (
        f"""
- **Battery CAPEX (€/kWh):** Upfront battery cost per kWh; sets part of each scenario’s CAPEX. Default: **€{BATT_COST_PER_KWH:,.0f}**/kWh.
- **Battery replacement:** One replacement in a chosen year, cost as % of battery CAPEX. Defaults: year **{DEFAULT_BATTERY_REPLACEMENT_YEAR}**, **{DEFAULT_BATTERY_REPLACEMENT_COST_PCT:g}%** of battery CAPEX. Set year **0** to turn off (no battery replacement cash flow).
"""
        if ENABLE_BATTERY_UI
        else ""
    )
    landing_battery_model = (
        """
#### Battery model
- **Round-trip efficiency:** How much energy is lost over a full charge–discharge cycle.
- **Depth of discharge (DoD):** How much of the battery’s capacity you allow to be used.
- **Initial SOC:** Battery state of charge at the start of the simulated year.
- **C-rate:** Caps how fast the battery can charge or discharge relative to its capacity.
- **Charge from surplus PV:** Whether the battery may store excess PV instead of exporting it.
- **Charge from grid at night:** Whether the battery may import power overnight. Default **off** (self-consumption-style default); turn **on** in Advanced for night-grid arbitrage.
- **Discharge schedule:** **Peak only** = 17:00-19:00. **Day+Peak** = that peak band, then 19:00-23:00 if energy remains (not morning or night).
"""
        if ENABLE_BATTERY_UI
        else ""
    )
    st.markdown(
        f"""
#### Costs and finance
- **PV CAPEX (€/kWp):** Upfront PV cost per kWp; sets part of each scenario’s CAPEX. Default: **€{PV_COST_PER_KWP:,.0f}**/kWp.
{landing_battery_finance}- **OPEX %:** Annual operating cost as a percentage of that scenario’s total CAPEX. Default: **{DEFAULT_OPEX_PCT:g}%**.
- **Standing charges:** Fixed €/year per tariff family, included in **annual electricity cost** for grid-only and every scenario. Defaults: Standard **€{DEFAULT_STANDING_CHARGE_STANDARD_EUR:,.2f}**/y, Weekend Saver **€{DEFAULT_STANDING_CHARGE_WEEKEND_EUR:,.2f}**/y, Flat **€{DEFAULT_STANDING_CHARGE_FLAT_EUR:,.2f}**/y.
- **PSO levy (annual, €):** Fixed annual Public Service Obligation charge, same amount for every tariff and scenario (included in **annual electricity cost**). Default: **€{DEFAULT_PSO_LEVY_EUR_PER_YEAR:,.2f}**/y. Escalates with electricity inflation in long-run metrics.
- **Discount rate:** Used to build NPV and IRR from future savings. Default: **{DISCOUNT_RATE * 100:g}%** per year.
- **Electricity inflation:** Grows import costs, export income, standing charges, PSO levy, and OPEX year by year in long-run metrics (CAPEX is not inflated). Default: **{ELECTRICITY_INFLATION_RATE * 100:g}%** per year.
- **Inverter replacement:** One replacement in a chosen year, cost as % of PV CAPEX. Defaults: year **{DEFAULT_INVERTER_REPLACEMENT_YEAR}**, **{DEFAULT_INVERTER_REPLACEMENT_COST_PCT:g}%** of PV CAPEX. Set year **0** to turn off.

#### Optimizer
{landing_optimizer_lines.strip()}
{landing_battery_model}
#### Tariffs (matrix + CSV)
- **Families:** Standard, Weekend Saver, Flat (shown as `standard`, `weekend_saver`, `flat_rate` in the matrix **Type** column).
- **Tariff matrix:** Scrollable table of rows from the default file or an upload; each row is one supplier × family with standing charge, export rate, and import bands. Tick **Include** for rows that should participate in **Run analysis**; **Select all** / **Select none** apply to every row.
- **CSV upload:** In **Model setup → Data & tariffs**, **Load tariffs from CSV** replaces the whole matrix (same column schema as documented in the project README).
- **Export rate:** Per row (€/kWh) in the matrix.

Changing any of these means you should **Run analysis** again so results match your inputs.
"""
    )

    st.markdown("### Full KPI reference table")
    st.dataframe(
        build_kpi_guide_table(int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))),
        width="stretch",
        hide_index=True,
    )


def render_compact_preface_before_first_run() -> None:
    """Minimal text above setup; long guide only inside expander."""
    st.caption(
        "**Run analysis** uses the form below: built-in `data/` samples apply when uploads are empty. "
        "Open **How to use this app** for the full guide (also in **Settings & App guide** after you run)."
    )
    with st.expander("Restore saved run (.zip) — no Run analysis needed", expanded=False):
        st.caption(
            "Upload a bundle exported from this app (**Run your own analysis** → **Saved run — export / import**). "
            "Restores results and frozen inputs **without** running the optimizer."
        )
        render_saved_run_import_controls(include_section_heading=False, widget_key_suffix="_preface")
    with st.expander("How to use this app", expanded=False):
        render_full_how_to_use_guide()


def render_saved_run_bundle_expander(setup: SetupFormValues) -> None:
    """Export / import saved-run ZIP — lives on **Run your own analysis**."""
    with st.expander("Saved run — export / import (.zip)", expanded=False):
        st.caption(
            f"**Saved-run bundle** (schema **v{BUNDLE_SCHEMA_VERSION}**): ZIP with **manifest.json**, **JSON** state, "
            "**Parquet** frames (`prepared_df`, per-tariff `opt_dfs`, optional `full_results_df`), and raw **inputs** "
            "(consumption + PV CSV bytes; optional tariffs CSV; normalized **tariff_profiles.json**). "
            "**No pickle.** **Restore replaces** the current session’s results and frozen last-run inputs with the bundle "
            "(same snapshot as after **Run analysis**). **No optimizer** runs."
        )
        if st.session_state.get("opt_dfs") is None:
            st.info("Complete **Run analysis** once to enable **Download saved run**.")
        else:
            c_bytes, p_bytes, t_csv = _resolve_bundle_export_input_bytes(setup)
            if c_bytes is None or p_bytes is None:
                st.warning(
                    "Could not resolve consumption/PV bytes matching the last run fingerprints. "
                    "Use **Run your own analysis** → **Edit assumptions and rerun** (or re-upload the same files), then **Run analysis** again, or export from the same session right after a successful run."
                )
            else:
                _fr = st.session_state.full_results_df
                if _fr is None or not _full_results_snapshot_is_usable(_fr):
                    _fr = build_full_scenario_results_df(
                        st.session_state.opt_dfs,
                        st.session_state.prepared_df,
                        list(st.session_state.last_tariff_profiles or _default_tariff_profiles()),
                        pv_cost_per_kwp=st.session_state.last_pv_capex,
                        batt_cost_per_kwh=st.session_state.last_batt_capex,
                        electricity_inflation_rate=st.session_state.last_electricity_inflation_rate,
                        battery_replacement_year=st.session_state.last_battery_replacement_year,
                        battery_replacement_pct_of_batt_capex=st.session_state.last_battery_replacement_cost_pct,
                        inverter_replacement_year=st.session_state.last_inverter_replacement_year,
                        inverter_replacement_pct_of_pv_capex=st.session_state.last_inverter_replacement_cost_pct,
                        pso_levy_annual=float(st.session_state.last_pso_levy),
                        lifetime_years=int(st.session_state.last_lifetime_years),
                    )
                    st.session_state.full_results_df = _fr
                try:
                    _zip = build_saved_run_zip_bytes(
                        prepared_df=st.session_state.prepared_df,
                        opt_dfs=st.session_state.opt_dfs,
                        full_results_df=_fr,
                        cons_bytes=c_bytes,
                        pv_bytes=p_bytes,
                        tariff_csv_bytes=t_csv if isinstance(t_csv, (bytes, bytearray)) else None,
                        last_tariff_profiles=list(st.session_state.last_tariff_profiles or []),
                        last_run_json=_last_run_dict_for_bundle(),
                    )
                    st.download_button(
                        label="Download saved run (.zip)",
                        data=_zip,
                        file_name="rec_saved_run.zip",
                        mime="application/zip",
                        key="download_saved_run_zip",
                    )
                except Exception as e:
                    st.error(f"Could not build saved-run bundle: {e}")

        render_saved_run_import_controls(include_section_heading=True, widget_key_suffix="_bundle")


def render_settings_kpi_guide_tab(setup: SetupFormValues) -> None:
    """Full guide content for the Settings & App guide tab (after a run)."""
    st.subheader("Settings and App guide")
    st.caption("Plain-language explanation of model inputs, outputs, and formulas used in the app.")

    st.markdown("### How to use this app (full reference)")
    render_full_how_to_use_guide()
    st.divider()

    render_last_run_tariffs_and_assumptions_section()

    st.markdown("### How to read results in 5 steps")
    _step1 = (
        "1. **Run analysis** with your chosen assumptions in the **Run your own analysis** tab (**Model setup**), or **Edit assumptions and rerun** there after the first run."
        if not DEMO_MODE
        else "1. In the full app, **Run analysis** lives under **Run your own analysis** (**Model setup**). **Disabled in demo** — this build loads bundled **Demo runs** from the sidebar instead."
    )
    st.markdown(
        f"""
{_step1}
2. **Choose Tariff family** (optional), **Scenario type**, and **Rank results by** in the **left sidebar** (after a run). **Recommended setups** shows a **multi-criteria snapshot** and orders the table by that ranking.
3. **Check core economics** first: annual electricity cost, annual savings, payback, NPV, IRR.
4. **Check energy and carbon metrics**: self-sufficiency, self-consumption, CO2 savings/reduction.
5. **Filter the full scenario table** (Ag Grid, Excel-style column filters) and **export CSV** when needed.
"""
    )

    st.markdown("### Run settings (main setup + sidebar filters)")
    capex_bullet = (
        f"- **PV CAPEX / Battery CAPEX:** upfront investment cost per unit size. Defaults: **€{PV_COST_PER_KWP:,.0f}**/kWp and **€{BATT_COST_PER_KWH:,.0f}**/kWh. CAPEX affects payback, NPV, IRR, and net benefit."
        if ENABLE_BATTERY_UI
        else f"- **PV CAPEX:** upfront investment cost per kWp. Default: **€{PV_COST_PER_KWP:,.0f}**/kWp. CAPEX affects payback, NPV, IRR, and net benefit."
    )
    batt_repl_bullet = (
        f"- **Battery replacement:** one cash outflow in a chosen year, % of battery CAPEX. Defaults: year **{DEFAULT_BATTERY_REPLACEMENT_YEAR}**, **{DEFAULT_BATTERY_REPLACEMENT_COST_PCT:g}%**; set year **0** to omit. Applies when the scenario includes a battery.\n"
        if ENABLE_BATTERY_UI
        else ""
    )
    batt_model_bullet = (
        "- **Battery model settings:** efficiency, DoD, initial SOC, C-rate, whether PV/grid can charge, and dispatch rules for discharging to the load.\n"
        if ENABLE_BATTERY_UI
        else ""
    )
    opt_bounds_bullet = (
        "- **Optimizer bounds and speed:** min/max PV and battery sizes; **Quick / Fast / Full** set grid steps (**5/5**, **10/10**, **1/1** kWp & kWh).\n"
        if ENABLE_BATTERY_UI
        else "- **Optimizer bounds and speed:** min/max PV sizes; **Quick / Fast / Full** set PV steps (**5**, **10**, **1** kWp).\n"
    )
    _ly_run = int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))
    st.markdown(
        f"""
{capex_bullet}
- **Analysis horizon (years):** **{_ly_run}** in your **last completed run** — same **N** as in **NPV / IRR / gross & net savings** column names and in **Lifetime CO2 (kg)** (= annual scenario grid-import CO₂ × **N**). Set in **Model setup → Financial assumptions** before **Run analysis** (or comes from a **saved run**).
- **Standing charge:** fixed annual tariff charge per tariff family. Defaults: Standard **€{DEFAULT_STANDING_CHARGE_STANDARD_EUR:,.2f}**/y, Weekend Saver **€{DEFAULT_STANDING_CHARGE_WEEKEND_EUR:,.2f}**/y, Flat **€{DEFAULT_STANDING_CHARGE_FLAT_EUR:,.2f}**/y.
- **PSO levy (annual, €):** fixed annual charge included in every tariff/scenario **annual electricity cost**. Default: **€{DEFAULT_PSO_LEVY_EUR_PER_YEAR:,.2f}**/y; escalates with electricity inflation like standing charge.
- **OPEX (% of CAPEX):** annual operating cost estimated from CAPEX. Default: **{DEFAULT_OPEX_PCT:g}%**.
- **Discount rate:** used to discount future yearly savings in NPV. Default: **{DISCOUNT_RATE * 100:g}%** per year.
- **Electricity inflation:** yearly escalation applied to recurring costs and savings streams, including PSO levy (not to CAPEX). Default: **{ELECTRICITY_INFLATION_RATE * 100:g}%** per year.
{batt_repl_bullet}- **Inverter replacement:** one cash outflow in a chosen year, % of PV CAPEX. Defaults: year **{DEFAULT_INVERTER_REPLACEMENT_YEAR}**, **{DEFAULT_INVERTER_REPLACEMENT_COST_PCT:g}%**; set year **0** to omit. Applies when the scenario includes PV.
{batt_model_bullet}{opt_bounds_bullet}- **Tariff matrix:** default CSV or upload; **Include** checkboxes choose which supplier rows the optimizer evaluates (Standard / Weekend Saver / Flat).
"""
    )

    st.markdown("### Results controls (post-run)")
    scenario_type_help = (
        "(All, Grid only, PV + Grid, PV + Battery + Grid, Battery + Grid)"
        if ENABLE_BATTERY_UI
        else "(All scenarios in this build: Grid only and PV + Grid)"
    )
    st.markdown(
        f"""
- **Rank results by:** chooses which KPI orders scenarios after **Tariff family** and **Scenario type** filtering.
- **Tariff family:** **All tariff types**, or only **Standard**, **Weekend saver**, or **Flat rate** (all named variants in that family); does not rerun optimization.
- **Scenario type:** limits which scenario family rows are eligible {scenario_type_help}; does not rerun optimization.
- **Full results tab:** interactive consolidated table (Ag Grid) with per-column filters like Excel; **click a row** for KPIs, **rank under Rank results by**, and trade-off charts (no filtered-set comparison table or cumulative outlook on this tab); CSV export omits internal row keys.
- **Recommended setups tab:** **Multi-criteria snapshot** (lowest annual bill, highest CO₂ savings, top **Rank results by** pick) for the filtered set, then the recommended grid — same **feasible set** from **Decision constraints** (payback, optional NPV/CO₂ gates, SCR and export caps — defaults **80%** min SCR, **20%** max export). **Recommendation preset** picks the winner among feasible rows per tariff × scenario (lexicographic order + lowest CAPEX). Rows respect sidebar **Scenario type** and **Tariff family** (same rules as **Full results**). Shown in **Ag Grid** (ordered by **Rank results by**); **click a row** for KPIs, charts, comparison, trade-offs, cumulative outlook, and **All tariffs — comparison** at the bottom. Same `opt_dfs` — no second optimizer. Selection is **not** synced with **Full results**. **CSV:** **full consolidated KPI columns** (like Full results) per recommended row — **results only** or **results + assumptions**. Default battery: **night grid charging off** unless enabled in Advanced.
- **Consumption patterns tab:** scenario-independent demand charts from the prepared consumption series; these charts do **not** change with sidebar ranking, tariff, or scenario selection.
- **Production patterns tab:** scenario-independent charts from prepared **`pv_per_kwp`** (kWh per hour per 1 kWp nominal); same independence as consumption patterns.
- **Research results tab:** bundled fixed **Excel** reference (`assets/research/res.xlsx`) — **not** from your current **Run analysis**. Overview table, **all-rules winners** summary, and **grouped bar charts** for each comparison metric (bill, NPV, IRR, CAPEX, CO₂, self-sufficiency, self-consumption ratio (SCR), payback).
"""
    )

    st.markdown("### KPI formulas")
    st.dataframe(
        build_kpi_guide_table(int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))),
        width="stretch",
        hide_index=True,
    )


def render_bundled_research_tab() -> None:
    """Bundled Excel research matrix: overall-comparison table, grouped charts, then overview/winners tables."""
    st.subheader("Research results")
    st.caption(
        "**Bundled reference table** — fixed results shipped with the app, **not** from your current **Run analysis** session."
    )
    if not BUNDLED_RESEARCH_XLSX.is_file():
        st.warning("Bundled research file is missing (`assets/research/res.xlsx`).")
        return
    try:
        raw, scenario_titles, tariff_names, mat = load_bundled_research_xlsx(BUNDLED_RESEARCH_XLSX)
    except Exception as e:
        st.error(f"Could not read bundled research workbook: {e}")
        return

    st.caption(
        "**Winner rules:** each metric uses **max** or **min** per scenario block as in the bundled research definitions "
        "(e.g. lowest bill, best NPV, shortest payback with non-finite payback excluded)."
    )

    st.markdown("##### Overall comparison")
    if RESEARCH_OVERALL_COMPARISON_IMAGE.is_file():
        st.image(str(RESEARCH_OVERALL_COMPARISON_IMAGE), width="stretch")
    else:
        st.warning("Overall comparison image is missing (`assets/research/overall_comparison.png`).")

    disp = format_research_display_dataframe(
        build_research_display_dataframe(raw, scenario_titles, tariff_names)
    )

    st.markdown("##### Comparison bar charts (all metrics)")
    st.caption(
        "Each chart: **scenario** on the horizontal axis, **supplier / tariff** as coloured bars. "
        "Same layout for every metric — no filter needed."
    )
    for rule in RESEARCH_WINNER_RULES:
        st.markdown(f"###### {rule.label}")
        fig_bars = research_metric_grouped_bars(raw, scenario_titles, tariff_names, mat, rule)
        render_plotly_figure(fig_bars, key=f"research_bars_{rule.id}")

    st.markdown("##### Overview table")
    st.dataframe(disp, width="stretch")

    winners_all = build_all_winners_summary_df(raw, scenario_titles, tariff_names, mat)
    st.markdown("##### Winners by scenario (all rules)")
    st.dataframe(winners_all, width="stretch", hide_index=True)


def _kpi_eur_whole_keys(lifetime_years: int) -> frozenset:
    ly = int(lifetime_years)
    return frozenset(
        {
            COL_ANNUAL_ELECTRICITY_COST_EUR,
            COL_ANNUAL_ELECTRICITY_BILL_EUR,
            COL_ANNUAL_GRID_IMPORT_COST_EUR,
            COL_NET_IMPORT_EXPORT_COST_EUR,
            "Export income (€)",
            "Annual savings (€)",
            "CAPEX (€)",
            col_npv(ly),
            col_gross_savings(ly),
            col_net_benefit(ly),
        }
    )
_KPI_KWH_WHOLE_KEYS = frozenset(
    {
        COL_GRID_IMPORT_KWH,
        COL_BATTERY_CHARGE_KWH,
        COL_BATTERY_DISCHARGE_KWH,
        "Grid import reduction (kWh)",
        "Total annual PV generation (kWh)",
        COL_SELF_CONSUMED_PV_KWH,
        COL_EXPORT_TO_GRID_KWH,
    }
)
_KPI_CO2_KG_WHOLE_KEYS = frozenset(
    {
        COL_ANNUAL_GRID_CO2_EMISSIONS_KG,
        COL_LIFETIME_CO2_KG,
        "CO2 (kg)",
        "CO2 savings (kg)",
        COL_ANNUAL_CO2_REDUCTION_KG,
    }
)
_KPI_PCT_ONE_DECIMAL_KEYS = frozenset(
    {
        COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT,
        "CO2 reduction (%)",
        "Self-sufficiency ratio (%)",
        "Self-consumption ratio (%)",
        "Export ratio (% of PV gen)",
    }
)


def _format_kpi_tile_value(
    col_key: str,
    val,
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> str:
    """Whole euros for money KPIs; one decimal for payback / IRR / %; whole units for kWh and kg CO₂."""
    if val is None:
        return "—"
    if isinstance(val, (int, float, np.floating)):
        if not np.isfinite(val):
            return "inf" if val > 0 else "—"
        x = float(val)
        if col_key in _kpi_eur_whole_keys(lifetime_years):
            return f"€{x:,.0f}"
        if col_key == "Payback period (years)":
            return f"{x:,.1f}"
        if col_key == col_irr(int(lifetime_years)):
            return f"{x:,.1f}%"
        if col_key in _KPI_PCT_ONE_DECIMAL_KEYS:
            return f"{x:,.1f}%"
        if col_key in per_capex_ratio_column_names(int(lifetime_years)):
            if "kg/€" in str(col_key):
                return f"{x:,.2f}"
            return f"{x:,.3f}"
        if col_key in _KPI_KWH_WHOLE_KEYS or col_key in _KPI_CO2_KG_WHOLE_KEYS:
            return f"{x:,.0f}"
        return f"{x:,.1f}"
    return str(val)


def render_compact_kpi_tile_grid(
    row: pd.Series,
    kpi_labels: List[Tuple[str, str]],
    *,
    n_columns: int = 4,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> None:
    """Dense card-style KPI tiles for the Results dashboard (no extra vertical chrome)."""
    tile_css = (
        "background:#f8fafc;border:1px solid #e8edf3;border-radius:7px;padding:7px 10px 8px;"
        "box-shadow:0 1px 2px rgba(15,23,42,0.04);"
    )
    lbl_css = "font-size:11px;font-weight:500;color:#64748b;line-height:1.2;margin:0 0 3px 0;"
    val_css = (
        "font-size:22px;font-weight:700;color:#0f172a;line-height:1.18;"
        "font-variant-numeric:tabular-nums;letter-spacing:-0.02em;"
    )
    nc = max(1, min(int(n_columns), 6))
    parts: List[str] = [
        f'<div style="display:grid;grid-template-columns:repeat({nc},minmax(0,1fr));gap:6px 8px;margin:2px 0 10px 0;">'
    ]
    for label, col_key in kpi_labels:
        val = row.get(col_key, "—")
        val_disp = "—" if val == "—" else _format_kpi_tile_value(col_key, val, lifetime_years=int(lifetime_years))
        esc_l = html.escape(label)
        esc_v = html.escape(val_disp)
        parts.append(
            f'<div style="{tile_css}"><div style="{lbl_css}">{esc_l}</div>'
            f'<div style="{val_css}">{esc_v}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _maybe_scroll_to_results_top() -> None:
    """After a successful optimization, force the browser back to the top."""
    if not st.session_state.get("_scroll_results_top", False):
        return
    try:
        # Scroll to an explicit anchor rendered at the top of the Results tab.
        # Use a small timeout so the DOM for the tab is mounted before scrolling.
        _scroll_html = """<!DOCTYPE html><html><head><meta charset="utf-8"/></head><body>
<script>
  setTimeout(() => {
    const el = document.getElementById('results-top-anchor');
    if (el) { el.scrollIntoView({behavior: 'auto', block: 'start'}); }
    else { window.scrollTo(0,0); }
  }, 80);
</script></body></html>"""
        st.iframe(_scroll_html, height=1)
    except Exception:
        pass
    st.session_state["_scroll_results_top"] = False


def _results_scenario_label(r: pd.Series) -> str:
    tar = str(r.get("Tariff", "") or "")
    scen = str(r.get("Scenario", ""))
    pv = int(pd.to_numeric(r.get("PV (kWp)", 0), errors="coerce") or 0)
    batt = int(pd.to_numeric(r.get("Battery (kWh)", 0), errors="coerce") or 0)
    if ENABLE_BATTERY_UI and batt > 0:
        base = f"{scen} · {pv} kWp · {batt} kWh"
    elif pv > 0 or scen != "Grid only":
        base = f"{scen} · {pv} kWp"
    else:
        base = scen
    return f"{tar}: {base}" if tar else base


def _results_row_key(r: pd.Series) -> tuple[str, int, int, str]:
    return (
        str(r.get("Scenario", "")),
        int(pd.to_numeric(r.get("PV (kWp)", 0), errors="coerce") or 0),
        int(pd.to_numeric(r.get("Battery (kWh)", 0), errors="coerce") or 0),
        str(r.get("Tariff", "") or ""),
    )


def _tradeoff_sel_mask(plot_df: pd.DataFrame, sel_key: str | tuple[str, int, int, str]) -> pd.Series:
    """Boolean mask for the selected scenario point in trade-off scatter data."""
    if isinstance(sel_key, str) and SCENARIO_ROW_KEY_FIELD in plot_df.columns:
        return plot_df[SCENARIO_ROW_KEY_FIELD].astype(str) == str(sel_key)
    if isinstance(sel_key, tuple):
        sk = sel_key
        return (
            (plot_df["Scenario"].astype(str) == sk[0])
            & (pd.to_numeric(plot_df["PV (kWp)"], errors="coerce").fillna(0).astype(int) == sk[1])
            & (pd.to_numeric(plot_df["Battery (kWh)"], errors="coerce").fillna(0).astype(int) == sk[2])
            & (plot_df["Tariff"].astype(str) == sk[3])
        )
    return pd.Series(False, index=plot_df.index)


def render_filtered_rank_summary_and_multi_winner_cards(
    hard_filtered_rank_df: pd.DataFrame,
    ranked: list[tuple[str, pd.Series]],
    goal: str,
    *,
    payback_max: float | None,
) -> None:
    """Filter row-count / payback sanity + three winner cards (lowest bill, highest CO₂, top **Rank results by** row)."""
    st.caption(f"Rows after filters: **{len(hard_filtered_rank_df):,}**")
    if len(hard_filtered_rank_df) == 0:
        st.warning("No scenarios meet the current constraints. Relax one or more **Decision constraints**.")
        if len(ranked) == 0:
            st.markdown(
                '<div style="font-size:0.78rem;color:#666;margin-top:0.5rem;">No ranked rows for this scenario type.</div>',
                unsafe_allow_html=True,
            )
        return
    if payback_max is not None and "Payback (yrs)" in hard_filtered_rank_df.columns:
        pb_vals = pd.to_numeric(hard_filtered_rank_df["Payback (yrs)"], errors="coerce")
        pb_vals = pb_vals[pd.Series(np.isfinite(pb_vals), index=pb_vals.index)]
        if len(pb_vals) > 0:
            st.caption(
                f"Max payback in filtered set: **{float(pb_vals.max()):.2f}** years (limit **{float(payback_max):.2f}**) "
            )

    if len(ranked) == 0:
        st.markdown(
            '<div style="font-size:0.78rem;color:#666;margin-top:0.5rem;">No ranked rows for this scenario type.</div>',
            unsafe_allow_html=True,
        )

    _bcf_rank = _df_bill_column(hard_filtered_rank_df)
    _ccf_rank = _df_co2_avoided_column(hard_filtered_rank_df)
    _cs = pd.to_numeric(hard_filtered_rank_df[_bcf_rank], errors="coerce")
    _gs = pd.to_numeric(hard_filtered_rank_df[_ccf_rank], errors="coerce")
    _r_cost = (
        hard_filtered_rank_df.loc[_cs.idxmin()]
        if _cs.notna().any()
        else hard_filtered_rank_df.iloc[0]
    )
    _r_co2 = (
        hard_filtered_rank_df.loc[_gs.idxmax()]
        if _gs.notna().any()
        else hard_filtered_rank_df.iloc[0]
    )
    _r_top = ranked[0][1] if len(ranked) > 0 else _r_cost
    _co2_has_positive = False
    if _gs.notna().any():
        try:
            _co2_has_positive = float(pd.to_numeric(_gs, errors="coerce").max()) > 0.0
        except Exception:
            _co2_has_positive = False
    render_results_multi_winner_cards(
        _r_cost,
        _r_co2,
        _r_top,
        goal_label=str(goal),
        co2_has_positive=_co2_has_positive,
    )


def render_recommended_snapshot_cards_from_table(
    rec_df_grid: pd.DataFrame,
    *,
    goal: str,
    payback_max: float | None,
) -> None:
    """Snapshot cards sourced from the Recommended-table universe."""
    st.caption(f"Rows after filters: **{len(rec_df_grid):,}**")
    if rec_df_grid is None or len(rec_df_grid) == 0:
        st.warning("No rows available in **Recommended setups** for current filters.")
        return

    work = rec_df_grid.copy()
    if "Scenario" not in work.columns and "Scenario family" in work.columns:
        work["Scenario"] = work["Scenario family"]
    if "CO2 savings (kg)" not in work.columns and "CO₂ savings (kg)" in work.columns:
        work["CO2 savings (kg)"] = pd.to_numeric(work["CO₂ savings (kg)"], errors="coerce")

    if payback_max is not None and "Payback (yrs)" in work.columns:
        pb_vals = pd.to_numeric(work["Payback (yrs)"], errors="coerce")
        pb_vals = pb_vals[pd.Series(np.isfinite(pb_vals), index=pb_vals.index)]
        if len(pb_vals) > 0:
            st.caption(
                f"Max payback in filtered set: **{float(pb_vals.max()):.2f}** years (limit **{float(payback_max):.2f}**) "
            )

    if SCENARIO_ROW_KEY_FIELD in work.columns:
        _is_infeasible = work[SCENARIO_ROW_KEY_FIELD].astype(str).str.startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX)
        source = work[~_is_infeasible].copy()
    else:
        source = work
    if len(source) == 0:
        source = work
    if len(source) == 0:
        st.warning("No ranked rows for this scenario type.")
        return

    _bcf = _df_bill_column(source)
    _co2_col = "CO2 savings (kg)" if "CO2 savings (kg)" in source.columns else _df_co2_avoided_column(source)
    _cs = pd.to_numeric(source[_bcf], errors="coerce") if _bcf in source.columns else pd.Series(dtype=float)
    _gs = pd.to_numeric(source[_co2_col], errors="coerce") if _co2_col in source.columns else pd.Series(dtype=float)

    _r_cost = source.loc[_cs.idxmin()] if len(_cs) and _cs.notna().any() else source.iloc[0]
    _r_co2 = source.loc[_gs.idxmax()] if len(_gs) and _gs.notna().any() else source.iloc[0]
    _r_top = source.iloc[0]
    _co2_has_positive = bool(len(_gs) and _gs.notna().any() and float(_gs.max()) > 0.0)

    render_results_multi_winner_cards(
        _r_cost,
        _r_co2,
        _r_top,
        goal_label=str(goal),
        co2_has_positive=_co2_has_positive,
    )


def render_results_multi_winner_cards(
    r_cost: pd.Series,
    r_co2: pd.Series,
    r_ranked: pd.Series,
    *,
    goal_label: str,
    co2_has_positive: bool = True,
) -> None:
    """Highlight sidebar **Rank results by** #1 first, then lowest bill and highest CO₂ (same filter universe)."""
    card = (
        "border:1px solid #e2e8f0;border-radius:10px;padding:10px 12px;background:#fafafa;"
        "min-height:4.5rem;"
    )
    tit = "font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;"
    body = "font-size:14px;font-weight:600;color:#0f172a;line-height:1.35;margin-top:6px;"
    sub = "font-size:12px;color:#475569;margin-top:4px;"
    lc = html.escape(_results_scenario_label(r_cost))
    lg = html.escape(_results_scenario_label(r_co2))
    lr = html.escape(_results_scenario_label(r_ranked))
    ec = html.escape(f"€{float(r_cost.get(_df_bill_column(r_cost), 0)):,.0f}")
    eg = html.escape(f"{float(r_co2.get(_df_co2_avoided_column(r_co2), 0)):,.0f} kg")
    esc_goal = html.escape(goal_label)
    co2_sub = (
        f"{eg} / yr"
        if co2_has_positive
        else "No positive CO₂ savings in filtered set (rows tie at 0 kg/yr)."
    )
    st.markdown(
        f'<p style="margin:0.5rem 0 0.35rem 0;font-size:0.9rem;color:#334155;">'
        f"<b>Multi-criteria snapshot</b> — same rows as after your filters; "
        f"<b>Rank results by</b> first, then lowest annual cost and highest CO₂ savings.</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div style="{card}"><div style="{tit}">Top pick ({esc_goal})</div>'
            f'<div style="{body}">{lr}</div><div style="{sub}">Sidebar <b>Best</b> in ranked list</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="{card}"><div style="{tit}">Best for lowest annual electricity cost</div>'
            f'<div style="{body}">{lc}</div><div style="{sub}">{ec} / yr</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="{card}"><div style="{tit}">Best for highest CO₂ savings</div>'
            f'<div style="{body}">{lg}</div><div style="{sub}">{html.escape(co2_sub)}</div></div>',
            unsafe_allow_html=True,
        )


def render_results_tradeoff_scatters(
    plot_df: pd.DataFrame,
    *,
    sel_key: str | tuple[str, int, int, str],
    tariff_display_name: str | None = None,
    plotly_chart_key_prefix: str = "tradeoff",
) -> None:
    """Scatter plots for cost / CAPEX / payback vs CO₂ and self-metrics (filtered scenario universe)."""
    if plot_df is None or len(plot_df) == 0:
        return
    df = plot_df.copy()
    if tariff_display_name is not None and str(tariff_display_name).strip() != "":
        df = df[df["Tariff"].astype(str) == str(tariff_display_name)].copy()
    if len(df) == 0:
        return
    df["_label"] = df.apply(_results_scenario_label, axis=1)
    sel_mask = _tradeoff_sel_mask(df, sel_key)

    def _scatter(x_col: str, y_col: str, x_title: str, y_title: str, title: str) -> go.Figure:
        xd = pd.to_numeric(df[x_col], errors="coerce")
        yd = pd.to_numeric(df[y_col], errors="coerce")
        fig = go.Figure()
        base = df[~sel_mask]
        sel = df[sel_mask]
        if len(base) > 0:
            fig.add_trace(
                go.Scatter(
                    x=xd[~sel_mask],
                    y=yd[~sel_mask],
                    mode="markers",
                    marker=dict(size=8, color="#94a3b8", line=dict(width=0)),
                    text=base["_label"],
                    hovertemplate="%{text}<br>" + x_title + ": %{x:,.1f}<br>" + y_title + ": %{y:,.1f}<extra></extra>",
                    name="Scenarios",
                )
            )
        if len(sel) > 0:
            fig.add_trace(
                go.Scatter(
                    x=xd[sel_mask],
                    y=yd[sel_mask],
                    mode="markers",
                    marker=dict(size=14, color="#16a34a", symbol="diamond", line=dict(width=1, color="#14532d")),
                    text=sel["_label"],
                    hovertemplate="%{text}<br><b>Selected</b><br>" + x_title + ": %{x:,.1f}<br>" + y_title + ": %{y:,.1f}<extra></extra>",
                    name="Selected",
                )
            )
        fig.update_layout(
            title=title,
            margin=dict(l=8, r=8, t=40, b=8),
            height=280,
            showlegend=False,
            xaxis_title=x_title,
            yaxis_title=y_title,
        )
        return fig

    st.markdown("##### Trade-off charts")
    st.caption("Each point is one scenario size in the filtered set. **Green diamond** = scenario shown in the KPI block above.")
    _bill_c = _df_bill_column(df)
    _co2_av_c = _df_co2_avoided_column(df)
    r1c1, r1c2 = st.columns(2)
    _pk = str(plotly_chart_key_prefix)
    with r1c1:
        render_plotly_figure(
            _scatter(
                _bill_c,
                _co2_av_c,
                _bill_c,
                "CO₂ avoided (kg)",
                "Annual electricity bill vs CO₂ avoided",
            ),
            key=f"{_pk}_tf_bill_co2",
        )
    with r1c2:
        render_plotly_figure(
            _scatter("CAPEX (€)", _co2_av_c, "CAPEX (€)", "CO₂ avoided (kg)", "CAPEX vs CO₂ avoided"),
            key=f"{_pk}_tf_capex_co2",
        )
    r2c1, r2c2 = st.columns(2)
    _pb_series = pd.to_numeric(df["Payback (yrs)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df_pb = df.assign(_pb=_pb_series).dropna(subset=["_pb"])
    if len(df_pb) > 0:
        _co2_pb = _df_co2_avoided_column(df_pb)

        def _scatter_pb() -> go.Figure:
            dfp = df_pb.copy()
            sm = _tradeoff_sel_mask(dfp, sel_key)
            fig = go.Figure()
            if (~sm).any():
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_numeric(dfp.loc[~sm, "Payback (yrs)"], errors="coerce"),
                        y=pd.to_numeric(dfp.loc[~sm, _co2_pb], errors="coerce"),
                        mode="markers",
                        marker=dict(size=8, color="#94a3b8"),
                        text=dfp.loc[~sm, "_label"],
                        hovertemplate="%{text}<br>Payback (y): %{x:.2f}<br>CO₂ kg: %{y:,.0f}<extra></extra>",
                        name="Scenarios",
                    )
                )
            if sm.any():
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_numeric(dfp.loc[sm, "Payback (yrs)"], errors="coerce"),
                        y=pd.to_numeric(dfp.loc[sm, _co2_pb], errors="coerce"),
                        mode="markers",
                        marker=dict(size=14, color="#16a34a", symbol="diamond", line=dict(width=1, color="#14532d")),
                        text=dfp.loc[sm, "_label"],
                        hovertemplate="%{text}<br><b>Selected</b><extra></extra>",
                        name="Selected",
                    )
                )
            fig.update_layout(
                title="Payback vs CO₂ savings (finite payback only)",
                margin=dict(l=8, r=8, t=40, b=8),
                height=280,
                showlegend=False,
                xaxis_title="Payback (years)",
                yaxis_title="CO₂ avoided (kg)",
            )
            return fig

        with r2c1:
            render_plotly_figure(
                _scatter_pb(), key=f"{_pk}_tf_payback_co2"
            )
    else:
        with r2c1:
            st.caption("No finite payback rows to plot.")

    with r2c2:
        render_plotly_figure(
            _scatter("Self-sufficiency (%)", "Self-consumption ratio (%)", "Self-sufficiency (%)", "Self-consumption (%)", "Self-sufficiency vs self-consumption"),
            key=f"{_pk}_tf_ss_scr",
        )


def _rank_position_for_consolidated_row(
    ranked: list[tuple[str, pd.Series]],
    cons_row: pd.Series,
) -> tuple[int | None, int]:
    """1-based position under **Rank results by** sort order, and total rows ranked."""
    n = len(ranked)
    if n == 0 or cons_row is None or len(cons_row.index) == 0:
        return None, n
    for i, (_sn, r) in enumerate(ranked, start=1):
        if SCENARIO_ROW_KEY_FIELD in cons_row.index and SCENARIO_ROW_KEY_FIELD in r.index:
            ck = cons_row.get(SCENARIO_ROW_KEY_FIELD)
            rk = r.get(SCENARIO_ROW_KEY_FIELD)
            if ck is not None and rk is not None and not (pd.isna(ck) or pd.isna(rk)):
                if str(ck) == str(rk):
                    return i, n
        if _results_row_key(cons_row) == _results_row_key(r):
            return i, n
    return None, n


def render_filtered_scenario_comparison_styled_table(
    hard_filtered_rank_df: pd.DataFrame,
    cons_row_highlight: pd.Series,
    *,
    selection_caption: str,
) -> None:
    """Compact styled comparison over the current filtered universe; highlights ``cons_row_highlight``."""
    if hard_filtered_rank_df is None or len(hard_filtered_rank_df) == 0:
        st.info("No rows in the filtered set for comparison.")
        return
    st.markdown("##### Scenario comparison (filtered set)")
    if selection_caption:
        st.caption(selection_caption)
    _comp_cols = [
        "Scenario",
        "Tariff",
        "PV (kWp)",
        COL_ANNUAL_ELECTRICITY_BILL_EUR,
        "Annual savings (€)",
        COL_ANNUAL_CO2_REDUCTION_KG,
        "Self-sufficiency (%)",
        "CAPEX (€)",
        "Payback (yrs)",
    ]
    if ENABLE_BATTERY_UI:
        _comp_cols.insert(2, "Battery (kWh)")
    _cmp = hard_filtered_rank_df[[c for c in _comp_cols if c in hard_filtered_rank_df.columns]].copy()
    _sel_key_tuple = _results_row_key(cons_row_highlight)
    _sel_key_str = (
        str(cons_row_highlight[SCENARIO_ROW_KEY_FIELD])
        if SCENARIO_ROW_KEY_FIELD in cons_row_highlight.index and pd.notna(cons_row_highlight.get(SCENARIO_ROW_KEY_FIELD))
        else ""
    )
    _fmt = {
        COL_ANNUAL_ELECTRICITY_BILL_EUR: "{:,.0f}",
        "Annual savings (€)": "{:,.0f}",
        COL_ANNUAL_CO2_REDUCTION_KG: "{:,.0f}",
        "Self-sufficiency (%)": "{:,.1f}",
        "CAPEX (€)": "{:,.0f}",
        "Payback (yrs)": "{:,.2f}",
    }
    try:
        _sty = _cmp.style.format(_fmt, na_rep="—")
        if COL_ANNUAL_ELECTRICITY_BILL_EUR in _cmp.columns:
            _sty = _sty.highlight_min(subset=[COL_ANNUAL_ELECTRICITY_BILL_EUR], color="#bbf7d0")
        if COL_ANNUAL_CO2_REDUCTION_KG in _cmp.columns:
            _sty = _sty.highlight_max(subset=[COL_ANNUAL_CO2_REDUCTION_KG], color="#bae6fd")

        def _highlight_selected_row(s: pd.Series) -> list[str]:
            idx = s.name
            if _sel_key_str and idx is not None and SCENARIO_ROW_KEY_FIELD in hard_filtered_rank_df.columns:
                try:
                    if str(hard_filtered_rank_df.loc[idx, SCENARIO_ROW_KEY_FIELD]) == _sel_key_str:
                        return ["background-color: #fef9c3; font-weight: 600"] * len(s)
                except Exception:
                    pass
            if _results_row_key(s) == _sel_key_tuple:
                return ["background-color: #fef9c3; font-weight: 600"] * len(s)
            return [""] * len(s)

        _sty = _sty.apply(_highlight_selected_row, axis=1)
        st.dataframe(_sty, width="stretch", hide_index=True, height=min(420, 56 + 28 * len(_cmp)))
    except Exception:
        st.dataframe(_cmp, width="stretch", hide_index=True, height=min(420, 56 + 28 * len(_cmp)))


def _goal_higher_better_for_tariff_bar_rank(goal: str) -> bool:
    """Whether **Rank results by** prefers larger metric values when ranking tariffs for the bar chart."""
    _lower_better = frozenset(
        {
            "Lowest annual electricity cost",
            "Lowest annual bill",
            "Best payback",
            "Fastest payback",
        }
    )
    return goal not in _lower_better


def _top_tariff_display_names_for_compare_chart(
    all_cmp_df: pd.DataFrame,
    compare_metric: str,
    goal: str,
    k: int,
) -> list[str]:
    if len(all_cmp_df) == 0 or compare_metric not in all_cmp_df.columns:
        return []
    higher = _goal_higher_better_for_tariff_bar_rank(goal)
    names = [str(x) for x in all_cmp_df["Tariff"].dropna().unique()]
    scored: list[tuple[str, float]] = []
    for t in names:
        sub = all_cmp_df[all_cmp_df["Tariff"].astype(str) == t]
        v = pd.to_numeric(sub[compare_metric], errors="coerce")
        arr = v.to_numpy(dtype=float)
        m = np.isfinite(arr)
        if not m.any():
            continue
        agg = float(arr[m].max() if higher else arr[m].min())
        scored.append((t, agg))
    scored.sort(key=lambda x: x[1], reverse=higher)
    kk = max(0, int(k))
    return [t for t, _ in scored[:kk]]


def _distinct_qualitative_hex_colors(n: int) -> list[str]:
    """``n`` soft pastel hex colours, well separated in hue (All tariffs grouped bars — easy on the eye)."""
    if n <= 0:
        return []
    phi = 0.618033988749895
    out: list[str] = []
    for i in range(n):
        h = ((i * phi) + 0.06 * (i // 4)) % 1.0
        # HLS: high lightness + moderate saturation → pastel fills that stay distinguishable.
        light = 0.74 + 0.15 * ((i % 7) / 6.0)
        sat = 0.34 + 0.18 * ((i % 5) / 4.0)
        r, g, b = colorsys.hls_to_rgb(h, min(float(light), 0.9), min(float(sat), 0.52))
        r = max(0.0, min(1.0, float(r)))
        g = max(0.0, min(1.0, float(g)))
        b = max(0.0, min(1.0, float(b)))
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _tariff_profile_family_key(p: object) -> str:
    """``standard`` | ``weekend`` | ``flat`` for matrix profile dicts (or name-prefix fallback)."""
    if not isinstance(p, dict):
        return "standard"
    raw = str(p.get("kind") or p.get("family") or "").strip().lower()
    if raw in ("standard", "weekend", "flat"):
        return raw
    name = str(p.get("name", "") or "")
    if name.startswith("Weekend"):
        return "weekend"
    if name.startswith("Flat"):
        return "flat"
    return "standard"


def _tariff_compare_grouped_bar_colors(profiles_ordered: List[Dict]) -> list[str]:
    """
    All-tariffs comparison chart: **one colour family per tariff type**, **distinct variants per row**.

    Earlier versions only slid lightness/saturation at a fixed hue, so many greens/blues looked alike.
    Here we **spread hue within the family** (still read as Standard / Weekend / Flat) and use a
    wider lightness range at **higher saturation** so neighbouring bars are easier to tell apart.
    """
    if not profiles_ordered:
        return []
    # Anchor hue (HLS 0–1) + half-width of hue sweep for that family (radians on colour wheel)
    hue_anchor = {"standard": 0.02, "weekend": 0.58, "flat": 0.36}
    hue_width = {"standard": 0.09, "weekend": 0.11, "flat": 0.13}
    fams = [_tariff_profile_family_key(p) for p in profiles_ordered]
    n_per = Counter(fams)
    idx_run = {"standard": 0, "weekend": 0, "flat": 0}
    out: list[str] = []
    for fam in fams:
        h0 = hue_anchor.get(fam, hue_anchor["standard"])
        hw = hue_width.get(fam, 0.10)
        ni = int(n_per[fam])
        ii = idx_run[fam]
        idx_run[fam] = ii + 1
        if ni <= 1:
            t = 0.5
        else:
            t = ii / float(ni - 1)
        # Move along hue (wrap) so e.g. five “flat” tariffs are yellow-green → emerald, not five mints.
        h = (h0 + (t - 0.5) * 2.0 * hw) % 1.0
        # Wider lightness + strong saturation; zig-zag sat slightly so mid-bars do not plateau.
        light = float(np.clip(0.30 + 0.52 * t + (0.04 if (ii % 2 == 1) else 0.0), 0.26, 0.84))
        sat = float(np.clip(0.62 + 0.14 * (1.0 - abs(t - 0.5) * 1.4), 0.52, 0.82))
        r, g, b = colorsys.hls_to_rgb(h, light, sat)
        r = max(0.0, min(1.0, float(r)))
        g = max(0.0, min(1.0, float(g)))
        b = max(0.0, min(1.0, float(b)))
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _dataframe_compare_two_scenario_rows(a: pd.Series, b: pd.Series, ly: int) -> pd.DataFrame:
    """Side-by-side metrics for two consolidated rows (same columns where possible)."""
    cn = col_npv(ly)
    ci = col_irr(ly)
    keys: list[tuple[str, str]] = [
        ("Tariff", "Tariff"),
        ("Scenario", "Scenario"),
        ("PV (kWp)", "PV (kWp)"),
        ("Battery (kWh)", "Battery (kWh)"),
        (COL_ANNUAL_ELECTRICITY_BILL_EUR, "Annual electricity bill (€)"),
        ("Annual savings (€)", "Annual savings (€)"),
        (COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT, COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT),
        ("CAPEX (€)", "CAPEX (€)"),
        ("Payback (yrs)", "Payback (yrs)"),
        (cn, cn),
        (ci, ci),
        *[(k, k) for k in per_capex_ratio_column_names(ly)],
        (COL_ANNUAL_CO2_REDUCTION_KG, "Annual CO₂ reduction (kg)"),
        ("Self-consumption ratio (%)", "Self-consumption (%)"),
        ("Export ratio (% of PV gen)", "Export ratio (% of PV gen)"),
    ]
    rows_out: list[dict[str, object]] = []
    for col, label in keys:
        if col not in a.index and col not in b.index:
            continue
        va = a[col] if col in a.index else None
        vb = b[col] if col in b.index else None
        rows_out.append({"Metric": label, "Scenario A": va, "Scenario B": vb})
    return pd.DataFrame(rows_out)


def _format_saved_run_import_user_message(exc: BaseException) -> str:
    low = str(exc).lower()
    if "unsupported bundle schema" in low:
        return "This bundle was saved with a different schema version than this app supports."
    if "checksum" in low or "sha256" in low:
        return "The bundle contents do not match the manifest checksum (file may be corrupted or edited)."
    if "unsafe zip member" in low or "disallowed" in low:
        return "This ZIP does not look like a valid saved-run bundle (unexpected paths or files)."
    if "uncompressed size" in low or "exceeds limit" in low:
        return "This bundle is too large to load safely."
    if "zip member set" in low:
        return "The ZIP contents do not match the manifest (incomplete or modified bundle)."
    if "missing manifest" in low or "invalid bundle" in low:
        return "This file is missing saved-run contents (e.g. manifest or data files)."
    return "The file could not be read as a saved-run bundle."


def _tariff_internal_col_for_display_name(tariff_profiles: List[Dict], tariff_display: str) -> str | None:
    td = str(tariff_display or "").strip()
    for p in tariff_profiles or []:
        if str(p.get("name", "") or "").strip() == td:
            c = str(p.get("col", "") or "").strip()
            return c if c else None
    return None


def _hourly_dispatch_for_consolidated_scenario(
    prepared_df: pd.DataFrame,
    *,
    scenario_name: str,
    pv_kwp: int,
    batt_kwh: int,
    tariff_col: str,
    battery_settings: BatterySettings,
) -> pd.DataFrame | None:
    if prepared_df is None or len(prepared_df) == 0 or not tariff_col or tariff_col not in prepared_df.columns:
        return None
    df = prepared_df.copy()
    try:
        if scenario_name == "Grid only":
            return run_scenario_grid_only(df, tariff_col)
        if scenario_name == "PV + Grid":
            return run_scenario_pv_grid(df, int(pv_kwp), tariff_col)
        if scenario_name == "PV + Battery + Grid":
            return run_scenario_pv_battery_grid(df, int(pv_kwp), int(batt_kwh), tariff_col, battery_settings)
        if scenario_name == "Battery + Grid":
            return run_scenario_battery_grid(df, int(batt_kwh), tariff_col, battery_settings)
    except Exception:
        return None
    return None


def _plotly_notebook_monthly_layout(
    fig: go.Figure,
    *,
    title: str,
    yaxis_title: str,
    xaxis_title: str = "",
    height: int = 300,
    barmode: str | None = None,
    legend: dict[str, object] | None = None,
) -> None:
    ly: dict[str, object] = dict(
        title=dict(text=title, font=dict(size=15)),
        margin=dict(l=48, r=20, t=52, b=44),
        height=int(height),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=11, color="#1e293b"),
        xaxis=dict(
            title=xaxis_title if xaxis_title else None,
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            rangemode="tozero",
        ),
    )
    if barmode:
        ly["barmode"] = barmode
    if legend is not None:
        ly["legend"] = legend
    fig.update_layout(**ly)


def render_recommended_monthly_notebook_style_charts(
    cons_row: pd.Series,
    prepared_df: pd.DataFrame,
    tariff_profiles: List[Dict],
    battery_settings: BatterySettings,
    *,
    plotly_chart_key_prefix: str,
    widget_key_suffix: str,
) -> None:
    """Monthly charts in the style of the household notebook: load vs PV, dispatch stack, time bands, SSCR/SCR, CO₂."""
    if prepared_df is None or len(prepared_df) == 0:
        st.caption("Monthly charts need prepared hourly data — run **Run analysis** first.")
        return
    tariff_name = str(cons_row.get("Tariff", "") or "").strip()
    tcol = _tariff_internal_col_for_display_name(tariff_profiles, tariff_name)
    if not tcol:
        st.caption("Could not resolve the tariff column for monthly charts.")
        return
    if tcol not in prepared_df.columns:
        st.caption(f"Tariff column **{tcol}** is missing from prepared data.")
        return
    scen = str(cons_row.get("Scenario", "Grid only") or "Grid only").strip()
    pv_i = int(round(float(cons_row.get("PV (kWp)", 0) or 0)))
    bt_i = int(round(float(cons_row.get("Battery (kWh)", 0) or 0)))
    d = _hourly_dispatch_for_consolidated_scenario(
        prepared_df,
        scenario_name=scen,
        pv_kwp=pv_i,
        batt_kwh=bt_i,
        tariff_col=tcol,
        battery_settings=battery_settings,
    )
    if d is None or len(d) == 0:
        st.warning("Could not build hourly dispatch for this scenario — monthly charts are unavailable.")
        return
    d_dt = pd.to_datetime(d["date"])
    d = d.assign(month=d_dt.dt.month, hour=d_dt.dt.hour)
    xs = _MONTH_ABB
    g = d.groupby("month", sort=True)
    agg = g.agg(
        load=("consumption", "sum"),
        pv_gen=("pv_generation", "sum"),
        self_use=("self_consumed_pv", "sum"),
        grid_imp=("grid_import", "sum"),
        export=("feed_in", "sum"),
        local_ren=("local_renewable_to_load", "sum"),
    )
    agg = agg.reindex(range(1, 13), fill_value=0.0)
    load_y = agg["load"].to_numpy(dtype=float)
    pv_y = agg["pv_gen"].to_numpy(dtype=float)
    exp_y = agg["export"].to_numpy(dtype=float)
    self_y = agg["self_use"].to_numpy(dtype=float)
    grid_y = agg["grid_imp"].to_numpy(dtype=float)

    st.markdown(
        "<p style='font-size:1.05rem;margin:2px 0 14px 0;'><b>Monthly energy balance</b></p>",
        unsafe_allow_html=True,
    )

    _leg_nb_right = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        x=1.02,
        xanchor="left",
        font=dict(size=10),
    )
    _margin_nb_legend_right = dict(l=48, r=118, t=52, b=44)
    c1, c2 = st.columns(2)
    with c1:
        fig_lpv = go.Figure(
            data=[
                go.Bar(name="Load (kWh)", x=xs, y=load_y, marker_color=_NOTEBOOK_CHART_LOAD_BLUE),
                go.Bar(name="PV gen (kWh)", x=xs, y=pv_y, marker_color=_NOTEBOOK_CHART_PV_ORANGE),
            ]
        )
        _plotly_notebook_monthly_layout(
            fig_lpv,
            title="Load vs PV generation",
            yaxis_title="kWh",
            barmode="group",
            height=300,
            legend=_leg_nb_right,
        )
        fig_lpv.update_layout(bargap=0.12, margin=_margin_nb_legend_right)
        render_plotly_figure(fig_lpv, key=f"{plotly_chart_key_prefix}_rec_nb_lpv_{widget_key_suffix}", apply_bar_value_labels=False)
    with c2:
        fig_st = go.Figure(
            data=[
                go.Bar(name="Export", x=xs, y=exp_y, marker_color=_NOTEBOOK_CHART_STACK_EXPORT),
                go.Bar(name="Self-use", x=xs, y=self_y, marker_color=_NOTEBOOK_CHART_STACK_SELF),
                go.Bar(name="Grid import", x=xs, y=grid_y, marker_color=_NOTEBOOK_CHART_STACK_GRID),
            ]
        )
        _plotly_notebook_monthly_layout(
            fig_st,
            title="Energy dispatch (monthly kWh)",
            yaxis_title="kWh",
            barmode="stack",
            height=300,
            legend=_leg_nb_right,
        )
        fig_st.update_layout(bargap=0.15, margin=_margin_nb_legend_right)
        render_plotly_figure(fig_st, key=f"{plotly_chart_key_prefix}_rec_nb_st_{widget_key_suffix}", apply_bar_value_labels=False)

    pv_m = agg["pv_gen"].to_numpy(dtype=float)
    self_m = agg["self_use"].to_numpy(dtype=float)
    load_m = agg["load"].to_numpy(dtype=float)
    loc_m = agg["local_ren"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sscr = np.where(pv_m > 1e-9, 100.0 * self_m / pv_m, 0.0)
        scr = np.where(load_m > 1e-9, 100.0 * loc_m / load_m, 0.0)
    sscr = np.nan_to_num(sscr, nan=0.0, posinf=0.0, neginf=0.0)
    scr = np.nan_to_num(scr, nan=0.0, posinf=0.0, neginf=0.0)

    d0 = run_scenario_grid_only(prepared_df.copy(), tcol)
    d0["month"] = pd.to_datetime(d0["date"]).dt.month
    co2_f = _grid_co2_factor()
    grid_only_kwh_m = (
        d0.groupby("month", sort=True)["grid_import"].sum().reindex(range(1, 13), fill_value=0.0).to_numpy(dtype=float)
    )
    scen_grid_kwh_m = agg["grid_imp"].to_numpy(dtype=float)
    co2_save_m = np.maximum((grid_only_kwh_m - scen_grid_kwh_m) * co2_f, 0.0)

    c3, c4 = st.columns(2)
    _leg_sscr_right = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        x=1.02,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=10),
    )
    with c3:
        fig_ln = go.Figure()
        fig_ln.add_trace(
            go.Scatter(
                x=xs,
                y=sscr,
                name="SSCR (%)",
                mode="lines+markers",
                line=dict(color=_NOTEBOOK_CHART_LOAD_BLUE, width=2),
                marker=dict(symbol="circle", size=8, color=_NOTEBOOK_CHART_LOAD_BLUE),
                hovertemplate="%{x}<br>SSCR %{y:.1f}%<extra></extra>",
            )
        )
        fig_ln.add_trace(
            go.Scatter(
                x=xs,
                y=scr,
                name="SCR (%)",
                mode="lines+markers",
                line=dict(color=_NOTEBOOK_CHART_PV_ORANGE, width=2, dash="dash"),
                marker=dict(symbol="square", size=7, color=_NOTEBOOK_CHART_PV_ORANGE),
                hovertemplate="%{x}<br>SCR %{y:.1f}%<extra></extra>",
            )
        )
        _plotly_notebook_monthly_layout(
            fig_ln,
            title="Monthly SSCR & SCR",
            yaxis_title="%",
            height=300,
            legend=_leg_sscr_right,
        )
        fig_ln.update_layout(margin=_margin_nb_legend_right)
        fig_ln.update_yaxes(range=[0, 100], rangemode="normal")
        render_plotly_figure(fig_ln, key=f"{plotly_chart_key_prefix}_rec_nb_sscr_{widget_key_suffix}", apply_bar_value_labels=False)
    with c4:
        fig_co2 = go.Figure(data=[go.Bar(x=xs, y=co2_save_m, marker_color=_NOTEBOOK_CHART_LOAD_BLUE, name="CO₂ saving")])
        _plotly_notebook_monthly_layout(
            fig_co2,
            title="Monthly CO₂ saving vs grid-only",
            yaxis_title="kg CO₂",
            height=300,
        )
        render_plotly_figure(fig_co2, key=f"{plotly_chart_key_prefix}_rec_nb_co2_{widget_key_suffix}", apply_bar_value_labels=False)

    if bt_i > 0 and scen in ("PV + Battery + Grid", "Battery + Grid") and "battery_soc_kwh" in d.columns:
        render_recommended_battery_week_dispatch_soc(
            d,
            scenario_name=scen,
            pv_kwp=pv_i,
            batt_kwh=bt_i,
            battery_settings=battery_settings,
            plotly_chart_key_prefix=plotly_chart_key_prefix,
            widget_key_suffix=widget_key_suffix,
        )

    st.caption(
        "**SSCR** = self-consumed PV ÷ monthly PV generation. **SCR** = renewable-to-load ÷ monthly load "
        "(same definitions as annual KPIs, by calendar month). **CO₂ saving** = "
        "(grid-only import kWh − scenario grid import kWh) × emission factor, by month."
        + (
            " Battery rows add a **sample summer week** (168 h from **21 June** when possible, else **July** or **1 June**—"
            "see the chart caption for the exact calendar window) with dispatch areas and end-of-hour **state of charge** "
            "vs min/max energy limits."
            if bt_i > 0 and scen in ("PV + Battery + Grid", "Battery + Grid")
            else ""
        )
    )


def _battery_energy_limits_kwh(batt_kwh: int, battery_settings: BatterySettings) -> tuple[float, float]:
    """Usable SOC window (kWh) — same rules as dispatch simulation."""
    h = float(batt_kwh)
    if h <= 0:
        return 0.0, 0.0
    soc_min_dod = h * max(0.0, 1.0 - float(battery_settings.dod))
    soc_min_user = h * max(0.0, min(1.0, float(battery_settings.min_soc)))
    soc_max_user = h * max(0.0, min(1.0, float(battery_settings.max_soc)))
    soc_min = max(0.0, soc_min_dod, min(soc_min_user, soc_max_user))
    soc_max = min(h, max(soc_min_user, soc_max_user))
    return float(soc_min), float(soc_max)


def _slice_sample_summer_week_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Up to 168 consecutive hours, anchored in northern-hemisphere summer (not early June by default).

    Preference order for the start index: 21 June 00:00 (solstice day), any hour from 21–30 June,
    1 July 00:00, first hour in July, 1 June 00:00, first hour in June, else first row of the series.
    """
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True)
    dt = pd.to_datetime(out["date"])

    def _first_where(mask: pd.Series) -> int | None:
        if not mask.any():
            return None
        return int(mask.idxmax())

    i0: int | None = None
    for m in (
        (dt.dt.month == 6) & (dt.dt.day == 21) & (dt.dt.hour == 0),
        (dt.dt.month == 6) & (dt.dt.day >= 21),
        (dt.dt.month == 7) & (dt.dt.day == 1) & (dt.dt.hour == 0),
        dt.dt.month == 7,
        (dt.dt.month == 6) & (dt.dt.day == 1) & (dt.dt.hour == 0),
        dt.dt.month == 6,
    ):
        i0 = _first_where(m)
        if i0 is not None:
            break
    if i0 is None:
        i0 = 0
    return out.iloc[i0 : i0 + 168].copy()


def render_recommended_battery_week_dispatch_soc(
    d: pd.DataFrame,
    *,
    scenario_name: str,
    pv_kwp: int,
    batt_kwh: int,
    battery_settings: BatterySettings,
    plotly_chart_key_prefix: str,
    widget_key_suffix: str,
) -> None:
    """Weekly-style dispatch (168 h) + battery SoC with min/max reference lines — battery scenarios only."""
    if "battery_soc_kwh" not in d.columns or batt_kwh <= 0:
        return
    wdf = _slice_sample_summer_week_hourly(d)
    if wdf is None or len(wdf) < 48:
        st.caption("Not enough hourly data to plot a sample week of battery behaviour.")
        return
    hx = np.arange(len(wdf), dtype=float)
    load = pd.to_numeric(wdf["consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pv = pd.to_numeric(wdf["pv_generation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dch = pd.to_numeric(wdf["battery_discharge_to_load_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    soc = pd.to_numeric(wdf["battery_soc_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    soc_lo, soc_hi = _battery_energy_limits_kwh(int(batt_kwh), battery_settings)

    _w0 = pd.to_datetime(wdf["date"].iloc[0])
    _w1 = pd.to_datetime(wdf["date"].iloc[-1])
    st.markdown(
        "<p style='font-size:1.02rem;margin:8px 0 10px 0;'><b>Sample summer week</b> &nbsp;|&nbsp; "
        f"Dispatch & battery SoC (hours 0–{len(wdf) - 1} of slice)</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"**Calendar window shown:** {_w0:%d %b %Y, %H:00} → {_w1:%d %b %Y, %H:00} "
        f"({len(wdf)} hourly steps). The slice starts at the first available anchor in the series: "
        "**21 Jun 00:00**, else **21–30 Jun**, else **1 Jul 00:00**, else first **July** hour, "
        "else **1 Jun 00:00**, else first **June** hour, else the start of the uploaded series."
    )

    fig_w = go.Figure()
    fig_w.add_trace(
        go.Scatter(
            x=hx,
            y=load,
            name="Load (kWh/h)",
            mode="lines",
            fill="tozeroy",
            line=dict(color="rgba(37,99,235,0.95)", width=1.2),
            fillcolor="rgba(37,99,235,0.22)",
            hovertemplate="Hour %{x:.0f}<br>Load %{y:.2f} kWh/h<extra></extra>",
        )
    )
    if scenario_name != "Battery + Grid" and float(pv_kwp) > 0 and float(np.max(pv)) > 1e-9:
        fig_w.add_trace(
            go.Scatter(
                x=hx,
                y=pv,
                name="PV gen (kWh/h)",
                mode="lines",
                fill="tozeroy",
                line=dict(color="rgba(234,88,12,0.95)", width=1.2),
                fillcolor="rgba(234,88,12,0.18)",
                hovertemplate="Hour %{x:.0f}<br>PV %{y:.2f} kWh/h<extra></extra>",
            )
        )
    if float(np.max(dch)) > 1e-9:
        fig_w.add_trace(
            go.Scatter(
                x=hx,
                y=dch,
                name="Battery → load (kWh/h)",
                mode="lines",
                line=dict(color="#0f766e", width=2, dash="dot"),
                hovertemplate="Hour %{x:.0f}<br>Discharge %{y:.2f} kWh/h<extra></extra>",
            )
        )
    _leg_bat_wk_right = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        x=1.02,
        xanchor="left",
        font=dict(size=9),
    )
    _margin_bat_wk_legend = dict(l=48, r=132, t=52, b=44)
    _plotly_notebook_monthly_layout(
        fig_w,
        title="Weekly dispatch (sample week)",
        yaxis_title="kWh/h",
        height=300,
        legend=_leg_bat_wk_right,
    )
    fig_w.update_layout(xaxis_title="Hour of week", margin=_margin_bat_wk_legend)
    render_plotly_figure(fig_w, key=f"{plotly_chart_key_prefix}_rec_bat_wk_{widget_key_suffix}", apply_bar_value_labels=False)

    fig_s = go.Figure()
    fig_s.add_trace(
        go.Scatter(
            x=hx,
            y=soc,
            name="Battery SoC (kWh)",
            mode="lines",
            line=dict(color="#1d4ed8", width=2.2),
            hovertemplate="Hour %{x:.0f}<br>SoC %{y:.2f} kWh<extra></extra>",
        )
    )
    fig_s.add_trace(
        go.Scatter(
            x=hx,
            y=np.full_like(hx, soc_hi, dtype=float),
            name=f"Max SoC ({soc_hi:.1f} kWh)",
            mode="lines",
            line=dict(color="#93c5fd", width=1.5, dash="dash"),
            hovertemplate=f"Max {soc_hi:.1f} kWh<extra></extra>",
        )
    )
    fig_s.add_trace(
        go.Scatter(
            x=hx,
            y=np.full_like(hx, soc_lo, dtype=float),
            name=f"Min SoC ({soc_lo:.1f} kWh)",
            mode="lines",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
            hovertemplate=f"Min {soc_lo:.1f} kWh<extra></extra>",
        )
    )
    y_top = max(float(batt_kwh) * 1.06, soc_hi * 1.02, float(np.nanmax(soc)) if len(soc) else 1.0, 0.1)
    _leg_bat_soc_right = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        x=1.02,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=9),
    )
    _margin_bat_soc_legend = dict(l=48, r=128, t=52, b=44)
    _plotly_notebook_monthly_layout(
        fig_s,
        title="Battery state of charge",
        yaxis_title="kWh",
        height=300,
        legend=_leg_bat_soc_right,
    )
    fig_s.update_layout(xaxis_title="Hour of week", margin=_margin_bat_soc_legend)
    fig_s.update_yaxes(range=[0.0, y_top], tickformat=".1f", rangemode="normal")
    render_plotly_figure(fig_s, key=f"{plotly_chart_key_prefix}_rec_bat_soc_{widget_key_suffix}", apply_bar_value_labels=False)


def render_consolidated_selection_detail_block(
    cons_row: pd.Series,
    *,
    full_table_rank: pd.DataFrame,
    hard_filtered_rank_df: pd.DataFrame,
    ranked: list[tuple[str, pd.Series]],
    goal: str,
    ly: int,
    tradeoff_expander_title: str,
    comparison_selection_caption: str,
    plotly_chart_key_prefix: str = "consolidated_detail",
    prominent_header: bool = False,
    show_secondary_detail: bool = True,
    show_filtered_scenario_comparison: bool = True,
    show_cumulative_outlook: bool = True,
    show_tradeoff_expanded_inline: bool = False,
    show_cumulative_expanded_inline: bool = False,
) -> None:
    """KPI tiles, charts, and optionally trade-offs, comparison table, cumulative — shared by tabs."""
    _rt = str(cons_row.get("Tariff", "Standard"))
    _pv = int(cons_row["PV (kWp)"]) if "PV (kWp)" in cons_row.index else 0
    _bt = int(cons_row["Battery (kWh)"]) if "Battery (kWh)" in cons_row.index else 0
    _hdr = f"Viewing details for: {_rt} · {str(cons_row.get('Scenario', '—'))} · {_pv} kWp · {_bt} kWh"
    if prominent_header:
        st.markdown(
            (
                "<div style='margin:2px 0 8px 0;padding:10px 12px;border-radius:8px;"
                "background:#ecfdf5;border:1px solid #86efac;'>"
                "<div style='font-size:1.06rem;font-weight:700;color:#14532d;'>"
                f"{html.escape(_hdr)}"
                "</div></div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{_hdr}**")
    _rk, _nt = _rank_position_for_consolidated_row(ranked, cons_row)
    if _nt > 0:
        if _rk is not None:
            st.caption(
                f"**Rank** under sidebar **Rank results by** ({goal}): **#{_rk:,}** of **{_nt:,}** in the filtered set."
            )
        else:
            st.caption(
                "Could not match this row to the current ranked list; KPIs still reflect the **grid** selection."
            )
    row_m = _render_decision_kpi_through_charts_for_consolidated_row(
        cons_row,
        full_table_rank=full_table_rank,
        ly=ly,
        bill_compare_tariff_names=[_rt] if _rt else None,
        plotly_chart_key_prefix=plotly_chart_key_prefix,
    )
    if show_secondary_detail:
        if show_tradeoff_expanded_inline:
            st.markdown(f"##### {tradeoff_expander_title}")
            if SCENARIO_ROW_KEY_FIELD in cons_row.index and pd.notna(cons_row.get(SCENARIO_ROW_KEY_FIELD)):
                render_results_tradeoff_scatters(
                    hard_filtered_rank_df,
                    sel_key=str(cons_row[SCENARIO_ROW_KEY_FIELD]),
                    tariff_display_name=_rt,
                    plotly_chart_key_prefix=plotly_chart_key_prefix,
                )
            else:
                render_results_tradeoff_scatters(
                    hard_filtered_rank_df,
                    sel_key=_results_row_key(cons_row),
                    tariff_display_name=_rt,
                    plotly_chart_key_prefix=plotly_chart_key_prefix,
                )
        else:
            with st.expander(tradeoff_expander_title, expanded=False):
                if SCENARIO_ROW_KEY_FIELD in cons_row.index and pd.notna(cons_row.get(SCENARIO_ROW_KEY_FIELD)):
                    render_results_tradeoff_scatters(
                        hard_filtered_rank_df,
                        sel_key=str(cons_row[SCENARIO_ROW_KEY_FIELD]),
                        tariff_display_name=_rt,
                        plotly_chart_key_prefix=plotly_chart_key_prefix,
                    )
                else:
                    render_results_tradeoff_scatters(
                        hard_filtered_rank_df,
                        sel_key=_results_row_key(cons_row),
                        tariff_display_name=_rt,
                        plotly_chart_key_prefix=plotly_chart_key_prefix,
                    )
        if show_filtered_scenario_comparison:
            render_filtered_scenario_comparison_styled_table(
                hard_filtered_rank_df,
                cons_row,
                selection_caption=comparison_selection_caption,
            )
        if show_cumulative_outlook and row_m is not None and not row_m.empty:
            _render_cumulative_outlook_expander_for_row(
                row_m,
                _pv,
                _bt,
                ly,
                plotly_chart_key_prefix=plotly_chart_key_prefix,
                expanded_inline=show_cumulative_expanded_inline,
            )


def _render_cumulative_outlook_expander_for_row(
    row: pd.Series,
    pv_kwp: int,
    batt_kwh: int,
    ly: int,
    *,
    plotly_chart_key_prefix: str = "cumulative",
    expanded_inline: bool = False,
) -> None:
    if expanded_inline:
        st.markdown("##### Cumulative outlook (selected scenario)")
        st.caption(f"{ly}-year views for the **selected** scenario only (same cash-flow assumptions as elsewhere).")
        years = np.arange(0, ly + 1, dtype=int)
        t = years[1:]
        infl = float(st.session_state.last_electricity_inflation_rate)
        r = float(st.session_state.active_discount_rate)
        annual_savings_y1 = float(row.get("Annual savings (€)", 0.0))
        annual_co2_savings_y1 = _annual_co2_savings_kg_from_consolidated_row(row)
        capex = float(row.get("CAPEX (€)", 0.0))
        batt_repl_nominal = 0.0
        if st.session_state.last_battery_replacement_year is not None and 1 <= int(st.session_state.last_battery_replacement_year) <= ly:
            batt_repl_nominal = (batt_kwh * float(st.session_state.last_batt_capex)) * (
                float(st.session_state.last_battery_replacement_cost_pct) / 100.0
            )
        inv_repl_nominal = 0.0
        if st.session_state.last_inverter_replacement_year is not None and 1 <= int(st.session_state.last_inverter_replacement_year) <= ly:
            inv_repl_nominal = (pv_kwp * float(st.session_state.last_pv_capex)) * (
                float(st.session_state.last_inverter_replacement_cost_pct) / 100.0
            )
        if infl <= 0:
            savings_stream = annual_savings_y1 * np.ones_like(t, dtype=float)
        else:
            savings_stream = annual_savings_y1 * (1.0 + infl) ** (t - 1)
        cumulative_savings = np.concatenate([[0.0], np.cumsum(savings_stream)])
        cumulative_co2 = annual_co2_savings_y1 * years.astype(float)
        cashflow_disc = np.zeros_like(years, dtype=float)
        cashflow_disc[0] = -capex
        disc_den = (1.0 + r) ** t if r != -1 else np.inf
        savings_disc = savings_stream / disc_den
        replacement_disc = np.zeros_like(t, dtype=float)
        if st.session_state.last_battery_replacement_year is not None:
            by = int(st.session_state.last_battery_replacement_year)
            if 1 <= by <= ly:
                replacement_disc += batt_repl_nominal * (by == t).astype(float) / disc_den
        if st.session_state.last_inverter_replacement_year is not None:
            iy = int(st.session_state.last_inverter_replacement_year)
            if 1 <= iy <= ly:
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
                xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, ly]),
                yaxis=dict(rangemode="tozero"),
                showlegend=False,
            )
            _apply_yaxis_range_from_values(fig_sav, cumulative_savings)
            render_plotly_figure(
                fig_sav,
                key=f"{plotly_chart_key_prefix}_cum_savings",
            )
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
                xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, ly]),
                yaxis=dict(rangemode="tozero"),
                showlegend=False,
            )
            _apply_yaxis_range_from_values(fig_co2, cumulative_co2)
            render_plotly_figure(
                fig_co2,
                key=f"{plotly_chart_key_prefix}_cum_co2",
            )
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
                xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, ly]),
                yaxis=dict(rangemode="tozero"),
                showlegend=False,
            )
            _apply_yaxis_range_from_values(fig_cf, cumulative_disc_net_cashflow)
            render_plotly_figure(
                fig_cf,
                key=f"{plotly_chart_key_prefix}_cum_dcf",
            )
    else:
        with st.expander("Cumulative outlook (selected scenario)", expanded=False):
            st.caption(f"{ly}-year views for the **selected** scenario only (same cash-flow assumptions as elsewhere).")
            years = np.arange(0, ly + 1, dtype=int)
            t = years[1:]
            infl = float(st.session_state.last_electricity_inflation_rate)
            r = float(st.session_state.active_discount_rate)
            annual_savings_y1 = float(row.get("Annual savings (€)", 0.0))
            annual_co2_savings_y1 = _annual_co2_savings_kg_from_consolidated_row(row)
            capex = float(row.get("CAPEX (€)", 0.0))
            batt_repl_nominal = 0.0
            if st.session_state.last_battery_replacement_year is not None and 1 <= int(st.session_state.last_battery_replacement_year) <= ly:
                batt_repl_nominal = (batt_kwh * float(st.session_state.last_batt_capex)) * (
                    float(st.session_state.last_battery_replacement_cost_pct) / 100.0
                )
            inv_repl_nominal = 0.0
            if st.session_state.last_inverter_replacement_year is not None and 1 <= int(st.session_state.last_inverter_replacement_year) <= ly:
                inv_repl_nominal = (pv_kwp * float(st.session_state.last_pv_capex)) * (
                    float(st.session_state.last_inverter_replacement_cost_pct) / 100.0
                )
            if infl <= 0:
                savings_stream = annual_savings_y1 * np.ones_like(t, dtype=float)
            else:
                savings_stream = annual_savings_y1 * (1.0 + infl) ** (t - 1)
            cumulative_savings = np.concatenate([[0.0], np.cumsum(savings_stream)])
            cumulative_co2 = annual_co2_savings_y1 * years.astype(float)
            cashflow_disc = np.zeros_like(years, dtype=float)
            cashflow_disc[0] = -capex
            disc_den = (1.0 + r) ** t if r != -1 else np.inf
            savings_disc = savings_stream / disc_den
            replacement_disc = np.zeros_like(t, dtype=float)
            if st.session_state.last_battery_replacement_year is not None:
                by = int(st.session_state.last_battery_replacement_year)
                if 1 <= by <= ly:
                    replacement_disc += batt_repl_nominal * (by == t).astype(float) / disc_den
            if st.session_state.last_inverter_replacement_year is not None:
                iy = int(st.session_state.last_inverter_replacement_year)
                if 1 <= iy <= ly:
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
                    xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, ly]),
                    yaxis=dict(rangemode="tozero"),
                    showlegend=False,
                )
                _apply_yaxis_range_from_values(fig_sav, cumulative_savings)
                render_plotly_figure(
                    fig_sav,
                    key=f"{plotly_chart_key_prefix}_cum_savings",
                )
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
                    xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, ly]),
                    yaxis=dict(rangemode="tozero"),
                    showlegend=False,
                )
                _apply_yaxis_range_from_values(fig_co2, cumulative_co2)
                render_plotly_figure(
                    fig_co2,
                    key=f"{plotly_chart_key_prefix}_cum_co2",
                )
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
                    xaxis=dict(tickmode="linear", tick0=0, dtick=5, range=[0, ly]),
                    yaxis=dict(rangemode="tozero"),
                    showlegend=False,
                )
                _apply_yaxis_range_from_values(fig_cf, cumulative_disc_net_cashflow)
                render_plotly_figure(
                    fig_cf,
                    key=f"{plotly_chart_key_prefix}_cum_dcf",
                )


def render_all_tariffs_comparison_grouped_bars(
    *,
    hard_filtered_rank_df: pd.DataFrame,
    scenario_type_ui: str,
    tariff_family_ui: str,
    goal: str,
    ly: int,
    selected_kpi_tariff: str | None,
    radio_session_key: str,
) -> None:
    """Grouped-bar comparison of every tariff across scenario types (e.g. Recommended setups)."""
    compare_metric = goal_to_tariff_compare_chart_column(goal, ly, results_df=hard_filtered_rank_df)
    st.markdown("##### All tariffs — comparison (grouped bars)")
    st.caption(
        f"Grouped bars use **{compare_metric}**, matching **Rank results by** on the **same filtered table** "
        "as ranking (sidebar hard constraints + scenario type)."
    )
    st.markdown("")
    _scope_opt = (
        "All tariffs in sidebar filter",
        "Top 5 tariffs (by rank metric)",
        "Selected KPI tariff only",
    )
    _tariff_cmp_scope = st.radio(
        "Show tariffs",
        _scope_opt,
        index=0,
        horizontal=True,
        key=radio_session_key,
        help="**Top 5** ranks tariffs by best value of the chart metric within each tariff (same direction as **Rank results by**). "
        "**Selected KPI tariff** uses the tariff from your **Recommended setups** table selection (top **Rank results by** row when applicable).",
    )
    _t_at0 = time.perf_counter()
    _tps_cmp = list(st.session_state.get("last_tariff_profiles") or _default_tariff_profiles())
    _cmp_kind = _tariff_family_ui_to_kind(tariff_family_ui)
    if _cmp_kind is not None:
        _tps_cmp = [
            p
            for p in _tps_cmp
            if str(p.get("kind", p.get("family", "standard"))).strip().lower() == _cmp_kind
        ]
    _cmp_tariff_names = [str(p.get("name", "") or p.get("col", "Tariff")) for p in _tps_cmp]
    all_cmp_df = _build_all_tariffs_compare_long_from_filtered_rank_df(
        hard_filtered_rank_df,
        _cmp_tariff_names,
        scenario_type_ui,
        goal,
        compare_metric,
        lifetime_years=ly,
    )
    if all_cmp_df is None:
        all_cmp_df = pd.DataFrame()
    if _perf_profiling_enabled():
        _dt_at = time.perf_counter() - _t_at0
        st.session_state["_perf_all_tariffs_eval_s"] = _dt_at
        _perf_record("all_tariffs_compare_from_filtered_df", _dt_at)
    if len(all_cmp_df) == 0 or compare_metric not in all_cmp_df.columns:
        st.info("Nothing to chart for the current tariff / scenario-type settings.")
    else:
        _tps_plot = list(_tps_cmp)
        if _tariff_cmp_scope == _scope_opt[1]:
            _top_names = set(
                _top_tariff_display_names_for_compare_chart(all_cmp_df, compare_metric, goal, 5)
            )
            _tps_plot = [p for p in _tps_cmp if str(p.get("name", "") or "") in _top_names]
        elif _tariff_cmp_scope == _scope_opt[2]:
            _sel_t = selected_kpi_tariff
            if _sel_t:
                _tps_plot = [p for p in _tps_cmp if str(p.get("name", "") or "") == str(_sel_t)]
            else:
                _tps_plot = []
        if len(_tps_plot) == 0:
            st.info(
                "No tariffs to show for this scope (pick **All** or select a scenario row for **Selected KPI tariff**)."
            )
        else:
            _cmp_trace_specs: list[tuple[str, object, np.ndarray, Dict]] = []
            for p in _tps_plot:
                tname = str(p.get("name", "") or p.get("col", "Tariff"))
                sub = all_cmp_df[all_cmp_df["Tariff"] == tname]
                if len(sub) == 0:
                    continue
                yv = pd.to_numeric(sub[compare_metric], errors="coerce").to_numpy(dtype=float)
                yv = np.where(np.isfinite(yv), yv, np.nan)
                _cmp_trace_specs.append((tname, sub["Scenario"], yv, p))
            _cmp_colors = _tariff_compare_grouped_bar_colors([s[3] for s in _cmp_trace_specs])
            fig_cmp = go.Figure()
            for i, (tname, x_sc, yv, _) in enumerate(_cmp_trace_specs):
                fig_cmp.add_trace(
                    go.Bar(
                        name=tname,
                        x=x_sc,
                        y=yv,
                        marker_color=_cmp_colors[i],
                        marker_line=dict(width=1.1, color="rgba(15,23,42,0.38)"),
                    )
                )
            fig_cmp.update_layout(
                barmode="group",
                title=compare_metric,
                xaxis_title="Scenario",
                yaxis_title=compare_metric,
                height=320,
                margin=dict(l=20, r=20, t=44, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            _cmp_y_arrays = [s[2] for s in _cmp_trace_specs if len(s[2]) > 0]
            if _cmp_y_arrays:
                _cmp_y_flat = np.concatenate(_cmp_y_arrays).astype(float, copy=False)
                _cmp_y_flat = _cmp_y_flat[np.isfinite(_cmp_y_flat)]
                if _cmp_y_flat.size > 0:
                    _apply_yaxis_range_from_values(fig_cmp, _cmp_y_flat)
            render_plotly_figure(
                fig_cmp,
                key=f"{radio_session_key}_all_tariffs_grouped_bars",
            )


def _render_decision_kpi_through_charts_for_consolidated_row(
    cons_row: pd.Series,
    *,
    full_table_rank: pd.DataFrame,
    ly: int,
    bill_compare_tariff_names: Optional[List[str]] = None,
    plotly_chart_key_prefix: str = "decision_kpi_detail",
) -> pd.Series:
    """Render KPI tiles and scenario detail charts for one consolidated results row; returns metrics row for follow-ups."""
    scenario_name = str(cons_row.get("Scenario", ""))
    pv_kwp = int(cons_row["PV (kWp)"]) if "PV (kWp)" in cons_row.index else 0
    batt_kwh = int(cons_row["Battery (kWh)"]) if "Battery (kWh)" in cons_row.index else 0
    row_tariff_name = str(cons_row.get("Tariff", "Standard"))
    _tps = list(st.session_state.get("last_tariff_profiles") or _default_tariff_profiles())
    _by_name = {str(p.get("name", "")): p for p in _tps}
    _tp = _by_name.get(row_tariff_name) or (
        _tps[0] if len(_tps) else {"col": "tariff_standard_0", "standing_charge": 0.0, "export_rate": DEFAULT_EXPORT_RATE}
    )
    chosen_tcol = str(_tp.get("col", "tariff_standard_0"))
    chosen_export_rate = float(_tp.get("export_rate", DEFAULT_EXPORT_RATE))
    cn = col_npv(ly)
    ci = col_irr(ly)
    row, hourly_df = metrics_and_hourly_for_scenario_at_sizes(
        st.session_state.prepared_df,
        chosen_tcol,
        scenario_name,
        pv_kwp,
        batt_kwh,
        chosen_export_rate,
        float(_tp.get("standing_charge", 0.0) or 0.0),
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
        lifetime_years=ly,
    )
    if "Payback (yrs)" in cons_row.index:
        row["Payback period (years)"] = float(cons_row["Payback (yrs)"])
    if "NPV (€)" in cons_row.index:
        row[cn] = float(cons_row["NPV (€)"])
    if "IRR (%)" in cons_row.index:
        irrv = cons_row["IRR (%)"]
        row[ci] = float(irrv) if pd.notna(irrv) else row.get(ci, float("nan"))
    if COL_ANNUAL_ELECTRICITY_BILL_EUR in cons_row.index:
        _bv = float(cons_row[COL_ANNUAL_ELECTRICITY_BILL_EUR])
        row[COL_ANNUAL_ELECTRICITY_BILL_EUR] = _bv
        row[COL_ANNUAL_ELECTRICITY_COST_EUR] = _bv
    elif COL_ANNUAL_ELECTRICITY_COST_EUR in cons_row.index:
        _bv = float(cons_row[COL_ANNUAL_ELECTRICITY_COST_EUR])
        row[COL_ANNUAL_ELECTRICITY_COST_EUR] = _bv
        row[COL_ANNUAL_ELECTRICITY_BILL_EUR] = _bv
    if "Annual savings (€)" in cons_row.index:
        row["Annual savings (€)"] = float(cons_row["Annual savings (€)"])
    if "CAPEX (€)" in cons_row.index:
        row["CAPEX (€)"] = float(cons_row["CAPEX (€)"])
    if COL_ANNUAL_CO2_REDUCTION_KG in cons_row.index:
        row["CO2 savings (kg)"] = float(cons_row[COL_ANNUAL_CO2_REDUCTION_KG])
    elif "CO2 savings (kg)" in cons_row.index:
        row["CO2 savings (kg)"] = float(cons_row["CO2 savings (kg)"])
    if "CO2 reduction (%)" in cons_row.index:
        row["CO2 reduction (%)"] = float(cons_row["CO2 reduction (%)"])
    if "Grid import reduction (kWh)" in cons_row.index:
        row["Grid import reduction (kWh)"] = float(cons_row["Grid import reduction (kWh)"])
    try:
        _grid_only = full_table_rank[
            (full_table_rank["Tariff"] == row_tariff_name) & (full_table_rank["Scenario"] == "Grid only")
        ]
        _bgc = _df_bill_column(full_table_rank)
        _base_cost = (
            float(pd.to_numeric(_grid_only[_bgc], errors="coerce").iloc[0]) if len(_grid_only) > 0 else 0.0
        )
        _annual_savings = float(pd.to_numeric(cons_row.get("Annual savings (€)", 0.0), errors="coerce"))
        row[COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT] = (
            0.0 if (not np.isfinite(_base_cost) or _base_cost <= 0) else float(100.0 * _annual_savings / _base_cost)
        )
    except Exception:
        row[COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT] = 0.0
    _pc_names = per_capex_ratio_column_names(ly)
    for _k in _pc_names:
        if _k in cons_row.index:
            _v = cons_row[_k]
            if pd.notna(_v):
                row[_k] = float(_v)
    st.markdown("##### A. Economic impact")
    econ_kpi = [
        ("Annual electricity bill (€)", COL_ANNUAL_ELECTRICITY_BILL_EUR),
        ("Annual savings vs grid-only (€)", "Annual savings (€)"),
        (
            COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT,
            COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT,
        ),
        ("CAPEX (€)", "CAPEX (€)"),
        ("Payback (years)", "Payback period (years)"),
        (cn, cn),
        (ci, ci),
        (f"NPV per € CAPEX ({ly}y, €/€)", _pc_names[0]),
        ("Annual savings per € CAPEX (€/€)", _pc_names[3]),
        (f"Gross savings per € CAPEX ({ly}y, €/€)", _pc_names[4]),
    ]
    render_compact_kpi_tile_grid(row, econ_kpi, n_columns=3, lifetime_years=ly)
    row_ext = row.copy()
    _export_ratio_col = "Export ratio (% of PV gen)"
    _pv_gen_y1 = float(pd.to_numeric(row_ext.get("Total annual PV generation (kWh)", 0.0), errors="coerce") or 0.0)
    if scenario_name in ("Grid only", "Battery + Grid") or (not np.isfinite(_pv_gen_y1)) or _pv_gen_y1 <= 1e-9:
        row_ext[_export_ratio_col] = float("nan")
    elif _export_ratio_col in cons_row.index and pd.notna(cons_row.get(_export_ratio_col)):
        row_ext[_export_ratio_col] = float(cons_row[_export_ratio_col])
    else:
        _exp_tile = float(pd.to_numeric(row_ext.get(COL_EXPORT_TO_GRID_KWH, 0.0), errors="coerce") or 0.0)
        row_ext[_export_ratio_col] = max(0.0, 100.0 * _exp_tile / _pv_gen_y1)
    st.markdown("##### B. Community impact")
    env_kpi = [
        ("Annual grid CO₂ emissions (kg)", "CO2 (kg)"),
        ("Annual CO2 reduction vs grid-only (kg)", "CO2 savings (kg)"),
        ("CO2 reduction vs grid-only (%)", "CO2 reduction (%)"),
        ("Annual CO₂ reduction per € CAPEX (kg/€)", _pc_names[1]),
        (f"Lifet. CO₂ avoided per € CAPEX ({ly}y, kg/€)", _pc_names[2]),
        ("Self-sufficiency (%)", "Self-sufficiency ratio (%)"),
        ("Self-consumption (%)", "Self-consumption ratio (%)"),
        ("Export ratio (% of PV gen)", _export_ratio_col),
        ("Annual PV production (kWh)", "Total annual PV generation (kWh)"),
        ("Grid import reduction (kWh)", "Grid import reduction (kWh)"),
        ("Battery charge (kWh)", COL_BATTERY_CHARGE_KWH),
        ("Battery discharge (kWh)", COL_BATTERY_DISCHARGE_KWH),
    ]
    render_compact_kpi_tile_grid(row_ext, env_kpi, n_columns=3, lifetime_years=ly)
    _render_selected_scenario_detail_charts(
        prepared_df=st.session_state.prepared_df,
        hourly_df=hourly_df,
        chosen_tcol=chosen_tcol,
        export_rate=chosen_export_rate,
        full_results_df=st.session_state.full_results_df,
        scenario_name=scenario_name,
        pv_kwp=pv_kwp,
        batt_kwh=batt_kwh,
        row_tariff_name=row_tariff_name,
        capex_eur=float(row.get("CAPEX (€)", 0) or 0),
        opex_pct=float(st.session_state.last_opex_pct),
        bill_compare_tariff_names=bill_compare_tariff_names,
        plotly_chart_key_prefix=plotly_chart_key_prefix,
    )
    return row


def _render_selected_scenario_detail_charts(
    *,
    prepared_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    chosen_tcol: str,
    export_rate: float,
    full_results_df: pd.DataFrame,
    scenario_name: str,
    pv_kwp: int,
    batt_kwh: int,
    row_tariff_name: str,
    capex_eur: float,
    opex_pct: float,
    bill_compare_tariff_names: Optional[List[str]] = None,
    plotly_chart_key_prefix: str = "sel_scenario_charts",
) -> None:
    """UI-only: decision-relevant charts for the selected KPI scenario.

    These charts are derived from the selected scenario hourly dispatch (no changes to optimization math).
    """
    if hourly_df is None or len(hourly_df) == 0:
        return

    h = hourly_df.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["hour"] = h["date"].dt.hour
    h["month"] = h["date"].dt.month
    h["dow"] = h["date"].dt.dayofweek
    h["is_weekend"] = h["dow"] >= 5
    h["time_band"] = h["hour"].apply(_consumption_time_band)

    # Baseline grid-only series for the same tariff.
    base = run_scenario_grid_only(prepared_df, chosen_tcol)
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"])
    base["hour"] = base["date"].dt.hour
    base["month"] = base["date"].dt.month
    base["dow"] = base["date"].dt.dayofweek
    base["is_weekend"] = base["dow"] >= 5
    base["time_band"] = base["hour"].apply(_consumption_time_band)

    # 1) Avg hourly consumption (dotted) vs avg hourly PV generation (solid)
    avg_by_hour = h.groupby("hour").agg(
        cons_avg=("consumption", "mean"),
        pv_avg=("pv_generation", "mean"),
    )
    avg_by_hour = avg_by_hour.reindex(range(24)).fillna(0.0)
    hours_x = list(range(24))

    fig_avg = go.Figure()
    fig_avg.add_trace(
        go.Scatter(
            x=hours_x,
            y=avg_by_hour["cons_avg"].to_numpy(dtype=float),
            mode="lines",
            name="Average consumption",
            line=dict(color="#94a3b8", dash="dot", width=3),
            hovertemplate="Hour %{x}<br>Avg consumption: %{y:.2f} kWh/h<extra></extra>",
        )
    )
    fig_avg.add_trace(
        go.Scatter(
            x=hours_x,
            y=avg_by_hour["pv_avg"].to_numpy(dtype=float),
            mode="lines",
            name="Average PV production",
            line=dict(color="#0ea5e9", width=3),
            hovertemplate="Hour %{x}<br>Avg PV: %{y:.2f} kWh/h<extra></extra>",
        )
    )
    fig_avg.update_layout(
        title="Selected scenario: average hourly demand vs PV production",
        height=280,
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
        yaxis_title="kWh/h",
        showlegend=True,
    )
    _apply_yaxis_range_from_values(
        fig_avg,
        np.concatenate(
            [
                avg_by_hour["cons_avg"].to_numpy(dtype=float),
                avg_by_hour["pv_avg"].to_numpy(dtype=float),
            ]
        ),
    )

    # 2) Savings by month (energy cost savings with OPEX allocated evenly across months)
    tariff_series_h = pd.to_numeric(h[chosen_tcol], errors="coerce").to_numpy(dtype=float)
    tariff_series_b = pd.to_numeric(base[chosen_tcol], errors="coerce").to_numpy(dtype=float)
    export = float(export_rate)

    h_grid = pd.to_numeric(h["grid_import"], errors="coerce").to_numpy(dtype=float)
    h_feed = pd.to_numeric(h["feed_in"], errors="coerce").to_numpy(dtype=float)
    b_grid = pd.to_numeric(base["grid_import"], errors="coerce").to_numpy(dtype=float)

    scen_cost_hour = h_grid * tariff_series_h - h_feed * export
    base_cost_hour = b_grid * tariff_series_b  # feed_in=0 for grid-only in this app
    energy_savings_hour = base_cost_hour - scen_cost_hour

    h = h.assign(energy_savings_cost=energy_savings_hour)
    month_savings_energy = (
        h.groupby("month")["energy_savings_cost"].sum().reindex(range(1, 13)).fillna(0.0)
    )

    opex_total = float(capex_eur) * (float(opex_pct) / 100.0) if np.isfinite(capex_eur) else 0.0
    opex_month = opex_total / 12.0
    month_savings_total = month_savings_energy.to_numpy(dtype=float) - opex_month

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig_month = go.Figure()
    fig_month.add_trace(
        go.Bar(
            x=month_names,
            y=month_savings_total,
            marker_color="#22c55e",
            hovertemplate="Month %{x}<br>Monthly savings: %{y:,.0f} €<extra></extra>",
        )
    )
    fig_month.update_layout(
        title="Savings by month vs grid-only (tariff matched)",
        height=280,
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis_title="Month",
        yaxis_title="€",
        yaxis=dict(rangemode="tozero"),
        xaxis=dict(tickmode="array", tickvals=month_names),
        showlegend=False,
    )
    _apply_yaxis_range_from_values(fig_month, month_savings_total)

    # 3) Donuts: savings share by time band + weekday/weekend
    h["energy_savings_pos"] = np.maximum(h["energy_savings_cost"].to_numpy(dtype=float), 0.0)
    total_pos = float(h["energy_savings_pos"].sum())

    by_band = h.groupby("time_band")["energy_savings_pos"].sum()
    by_band = by_band.reindex(["Night", "Day", "Peak"]).fillna(0.0)
    band_vals = by_band.to_numpy(dtype=float)
    band_pct = (band_vals / total_pos * 100.0) if total_pos > 0 else np.zeros_like(band_vals)

    fig_band = go.Figure(
        data=[
            go.Pie(
                labels=["Night", "Day", "Peak"],
                values=band_pct,
                hole=0.55,
                sort=False,
                marker=dict(colors=[TIME_BAND_CHART_COLORS[b] for b in ["Night", "Day", "Peak"]]),
                hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
            )
        ]
    )
    fig_band.update_layout(
        title="Share of positive savings by time band (losses excluded)",
        height=260,
        margin=dict(l=8, r=8, t=40, b=8),
    )

    weekday_sum = float(h.loc[~h["is_weekend"], "energy_savings_pos"].sum())
    weekend_sum = float(h.loc[h["is_weekend"], "energy_savings_pos"].sum())
    wk_total = weekday_sum + weekend_sum
    wk_pct_weekend = (weekend_sum / wk_total * 100.0) if wk_total > 0 else 0.0
    wk_pct_weekday = 100.0 - wk_pct_weekend if wk_total > 0 else 0.0

    fig_week = go.Figure(
        data=[
            go.Pie(
                labels=["Weekdays", "Weekend"],
                values=[wk_pct_weekday, wk_pct_weekend],
                hole=0.55,
                sort=False,
                marker=dict(colors=["#60a5fa", "#fbbf24"]),
                hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
            )
        ]
    )
    fig_week.update_layout(
        title="Share of positive savings: weekdays vs weekend (losses excluded)",
        height=260,
        margin=dict(l=8, r=8, t=40, b=8),
    )

    # 4) Annual electricity bill (€): grid-only vs selected scenario (default = selected tariff only for readability)
    if bill_compare_tariff_names is not None:
        tariff_names = [str(x) for x in bill_compare_tariff_names if str(x).strip() != ""]
    else:
        tariff_names = [str(p.get("name", "")) for p in (st.session_state.get("last_tariff_profiles") or [])]
        if len(tariff_names) == 0:
            tariff_names = [str(p.get("name", "")) for p in _default_tariff_profiles()]
    costs_grid = []
    costs_sel = []

    pv_col = "PV (kWp)"
    batt_col = "Battery (kWh)"
    pv_num = pd.to_numeric(full_results_df[pv_col], errors="coerce")
    batt_num = pd.to_numeric(full_results_df[batt_col], errors="coerce")
    tar_series = full_results_df["Tariff"].astype(str)
    scen_series = full_results_df["Scenario"].astype(str)
    _bill_tar = _df_bill_column(full_results_df)
    full_cost = pd.to_numeric(full_results_df[_bill_tar], errors="coerce")

    for tname in tariff_names:
        m_grid = (tar_series == tname) & (scen_series == "Grid only") & (pv_num == 0) & (batt_num == 0)
        costs_grid.append(float(full_cost[m_grid].iloc[0]) if bool(m_grid.any()) else 0.0)

        m_sel = (
            (tar_series == tname)
            & (scen_series == str(scenario_name))
            & (pv_num == int(pv_kwp))
            & (batt_num == int(batt_kwh))
        )
        costs_sel.append(float(full_cost[m_sel].iloc[0]) if bool(m_sel.any()) else np.nan)

    fig_tariff = go.Figure()
    fig_tariff.add_trace(
        go.Bar(
            name="Grid only",
            x=tariff_names,
            y=costs_grid,
            marker_color="#93c5fd",
            hovertemplate="%{x}<br>€%{y:,.0f}<extra></extra>",
        )
    )

    is_grid_only_selected = str(scenario_name) == "Grid only" and int(pv_kwp) == 0 and int(batt_kwh) == 0
    if not is_grid_only_selected:
        fig_tariff.add_trace(
            go.Bar(
                name="Selected scenario (same PV/Batt sizes)",
                x=tariff_names,
                y=costs_sel,
                marker_color="#fde68a",
                hovertemplate="%{x}<br>€%{y:,.0f}<extra></extra>",
            )
        )
    _bill_title = (
        "Annual electricity bill (selected tariff)"
        if len(tariff_names) <= 1
        else (
            "Annual electricity bill by tariff"
            if is_grid_only_selected
            else "Annual electricity bill by tariff (grid-only vs selected scenario)"
        )
    )
    fig_tariff.update_layout(
        title=_bill_title,
        height=320,
        margin=dict(l=8, r=8, t=40, b=8),
        barmode="group",
        xaxis_title="Tariff",
        yaxis_title=_bill_tar,
        showlegend=True,
    )
    _apply_yaxis_range_from_values(fig_tariff, np.asarray(costs_grid + [c for c in costs_sel if np.isfinite(c)], dtype=float))

    # Layout in 3 rows (2 charts, then 2 charts, then full width).
    _p = str(plotly_chart_key_prefix)
    st.markdown("##### Selected scenario charts (demand, savings, tariff costs)")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        render_plotly_figure(fig_avg, key=f"{_p}_avg_hour")
    with r1c2:
        render_plotly_figure(fig_month, key=f"{_p}_savings_month")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        render_plotly_figure(fig_band, key=f"{_p}_donut_band")
    with r2c2:
        render_plotly_figure(fig_week, key=f"{_p}_donut_week")

    render_plotly_figure(fig_tariff, key=f"{_p}_bill_tariff")


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

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 26rem !important;
            width: 26rem !important;
        }
        section[data-testid="stSidebar"] input[type="number"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #94a3b8 !important;
            border-radius: 0.375rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "prepared_df" not in st.session_state:
    st.session_state.prepared_df = None
if "opt_dfs" not in st.session_state:
    st.session_state.opt_dfs = None
if "prepared_meta" not in st.session_state:
    st.session_state.prepared_meta = None
if "show_setup_after_run" not in st.session_state:
    st.session_state.show_setup_after_run = False
if "active_export_rate" not in st.session_state:
    st.session_state.active_export_rate = DEFAULT_EXPORT_RATE
if "active_discount_rate" not in st.session_state:
    st.session_state.active_discount_rate = DISCOUNT_RATE
if "view_goal" not in st.session_state:
    st.session_state.view_goal = RECOMMENDED_WINNER_PRESETS[0][1]
_vg = st.session_state.get("view_goal")
if isinstance(_vg, str) and _vg in _LEGACY_RANK_GOAL_TO_PRESET_LABEL:
    st.session_state.view_goal = _LEGACY_RANK_GOAL_TO_PRESET_LABEL[_vg]
if st.session_state.get("view_goal") == "Most CO2 savings":
    st.session_state.view_goal = "Highest CO₂ saving"
if "view_scenario_type" not in st.session_state:
    st.session_state.view_scenario_type = "All scenarios"
if "view_tariff_family" not in st.session_state:
    st.session_state.view_tariff_family = "All tariff types"
if "view_filter_tariff" not in st.session_state:
    st.session_state.view_filter_tariff = "Standard"

# Sidebar **Decision constraints** widget keys (numeric inputs are disabled until their checkbox is ON).
if "hard_capex_max_en" not in st.session_state:
    st.session_state.hard_capex_max_en = False
if "hard_capex_max_eur" not in st.session_state:
    st.session_state.hard_capex_max_eur = 0.0
if "hard_npv_min_en" not in st.session_state:
    st.session_state.hard_npv_min_en = True
if "hard_npv_min_eur" not in st.session_state:
    st.session_state.hard_npv_min_eur = 0.0
if "hard_payback_max_en" not in st.session_state:
    st.session_state.hard_payback_max_en = True
if "hard_payback_max_years" not in st.session_state:
    st.session_state.hard_payback_max_years = 10.0
if "hard_irr_min_en" not in st.session_state:
    st.session_state.hard_irr_min_en = False
if "hard_irr_min_pct" not in st.session_state:
    st.session_state.hard_irr_min_pct = 0.0
if "hard_ss_min_en" not in st.session_state:
    st.session_state.hard_ss_min_en = False
if "hard_ss_min_pct" not in st.session_state:
    st.session_state.hard_ss_min_pct = 0.0
if "hard_co2_min_en" not in st.session_state:
    st.session_state.hard_co2_min_en = True
if "hard_co2_min_pct" not in st.session_state:
    st.session_state.hard_co2_min_pct = 0.0
if "hard_ann_cost_max_en" not in st.session_state:
    st.session_state.hard_ann_cost_max_en = False
if "hard_ann_cost_max_eur" not in st.session_state:
    st.session_state.hard_ann_cost_max_eur = 0.0
if "hard_ann_cost_saving_min_en" not in st.session_state:
    st.session_state.hard_ann_cost_saving_min_en = False
if "hard_ann_cost_saving_min_pct" not in st.session_state:
    st.session_state.hard_ann_cost_saving_min_pct = 0.0
if "hard_self_cons_min_en" not in st.session_state:
    st.session_state.hard_self_cons_min_en = True
if "hard_self_cons_min_pct" not in st.session_state:
    st.session_state.hard_self_cons_min_pct = 80.0
if "hard_export_max_en" not in st.session_state:
    st.session_state.hard_export_max_en = True
if "hard_export_max_pct" not in st.session_state:
    st.session_state.hard_export_max_pct = 20.0

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
if "last_export_rate" not in st.session_state:
    st.session_state.last_export_rate = DEFAULT_EXPORT_RATE
if "last_input_hashes" not in st.session_state:
    st.session_state.last_input_hashes = {"cons_sha": None, "pv_sha": None}
if "last_opt_cfg" not in st.session_state:
    st.session_state.last_opt_cfg = {
        "pv_min": 5,
        "pv_max": 60,
        "batt_min": 0,
        "batt_max": 40,
        "pv_step": 5,
        "batt_step": 5,
        "speed_preset": "Quick (PV step 5, battery step 5)",
    }
if "active_tariff_profiles" not in st.session_state:
    st.session_state.active_tariff_profiles = []
if "last_tariff_profiles" not in st.session_state:
    st.session_state.last_tariff_profiles = []
if "last_tariff_matrix_source_label" not in st.session_state:
    st.session_state.last_tariff_matrix_source_label = ""
if "last_pso_levy" not in st.session_state:
    st.session_state.last_pso_levy = float(DEFAULT_PSO_LEVY_EUR_PER_YEAR)
if "last_co2_factor" not in st.session_state:
    st.session_state.last_co2_factor = float(DEFAULT_CO2_FACTOR)
if "last_lifetime_years" not in st.session_state:
    st.session_state.last_lifetime_years = int(DEFAULT_LIFETIME_YEARS)

def _default_tariff_profiles() -> List[Dict]:
    # One "Market average" variant per family.
    return [
        {
            "family": "standard",
            "variant": "Market average",
            "col": "tariff_standard_0",
            "name": "Standard — Market average",
            "kind": "standard",
            "rates": {"standard": dict(DEFAULT_TARIFFS["standard"])},
            "standing_charge": float(DEFAULT_STANDING_CHARGE_STANDARD_EUR),
            "export_rate": float(DEFAULT_EXPORT_RATE),
        },
        {
            "family": "weekend",
            "variant": "Market average",
            "col": "tariff_weekend_0",
            "name": "Weekend Saver — Market average",
            "kind": "weekend",
            "rates": {"weekend": DEFAULT_TARIFFS["weekend"]},
            "standing_charge": float(DEFAULT_STANDING_CHARGE_WEEKEND_EUR),
            "export_rate": float(DEFAULT_EXPORT_RATE),
        },
        {
            "family": "flat",
            "variant": "Market average",
            "col": "tariff_flat_0",
            "name": "Flat — Market average",
            "kind": "flat",
            "rates": {"flat": {"flat": float(DEFAULT_TARIFFS["flat"])}},
            "standing_charge": float(DEFAULT_STANDING_CHARGE_FLAT_EUR),
            "export_rate": float(DEFAULT_EXPORT_RATE),
        },
    ]


def _normalize_tariff_csv_col(c: str) -> str:
    s = str(c).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_tariff_variants_csv_bytes(csv_bytes: bytes) -> Dict[str, List[Dict]]:
    """
    Parse a CSV defining multiple named tariff variants under the 3 families:
      - family: standard | weekend | flat
      - variant: company/supplier name (column name: `variant` or `name`)
      - standing_charge: optional (€/year)
      - export_rate: optional (€/kWh), per variant

    Expected rate columns (family-specific):
      - Standard:
          standard_day, standard_peak, standard_night
      - Weekend Saver:
          weekend_weekday_day, weekend_weekday_peak, weekend_weekday_night
          weekend_weekend_day, weekend_weekend_peak, weekend_weekend_night
      - Flat:
          flat_rate (or rate)
    """
    import io as _io

    df = pd.read_csv(_io.BytesIO(csv_bytes))
    if df is None or len(df) == 0:
        return {"standard": [], "weekend": [], "flat": []}

    df.columns = [_normalize_tariff_csv_col(c) for c in df.columns]

    family_col = "family" if "family" in df.columns else ("tariff_type" if "tariff_type" in df.columns else None)
    if family_col is None:
        raise ValueError("Tariff CSV must contain a 'family' column (standard|weekend|flat).")

    name_col = None
    for cand in ("variant", "name", "company", "supplier"):
        if cand in df.columns:
            name_col = cand
            break
    if not name_col:
        raise ValueError("Tariff CSV must contain a 'variant' (or 'name') column for the company/supplier.")

    sc_col = None
    for cand in ("standing_charge", "standing_charge_eur_year", "standing_charge_eur_yr"):
        if cand in df.columns:
            sc_col = cand
            break

    out: Dict[str, List[Dict]] = {"standard": [], "weekend": [], "flat": []}

    def _get_float(row: pd.Series, cols: List[str], label: str) -> float:
        actual = None
        for c in cols:
            if c in df.columns:
                actual = c
                break
        if actual is None:
            raise ValueError(f"Missing required column '{label}' for this tariff family.")
        val = row[actual]
        if pd.isna(val):
            raise ValueError(f"Column '{actual}' has empty value for row {row.name}.")
        return float(val)

    for _, row in df.iterrows():
        # Skip blank/empty rows (common when CSVs have trailing commas / empty lines).
        fam_raw = row.get(family_col, None)
        if fam_raw is None or (isinstance(fam_raw, float) and np.isnan(fam_raw)) or pd.isna(fam_raw):
            continue

        fam = str(fam_raw).strip().lower()
        if not fam or fam == "nan":
            continue
        if fam in ("standard", "std"):
            fam = "standard"
        elif fam in ("weekend", "weekend_saver", "weekendsaver", "weekend_saver_tariff"):
            fam = "weekend"
        elif fam in ("flat", "single", "flat_rate", "flatrate"):
            fam = "flat"
        else:
            raise ValueError(f"Unknown family '{row[family_col]}' (expected standard|weekend|flat).")

        variant_raw = row.get(name_col, None)
        if variant_raw is None or pd.isna(variant_raw):
            continue
        variant_name = str(variant_raw).strip()
        if not variant_name:
            raise ValueError(f"Empty variant/name in row {row.name}.")

        default_sc = {
            "standard": float(DEFAULT_STANDING_CHARGE_STANDARD_EUR),
            "weekend": float(DEFAULT_STANDING_CHARGE_WEEKEND_EUR),
            "flat": float(DEFAULT_STANDING_CHARGE_FLAT_EUR),
        }[fam]
        standing_charge = float(row[sc_col]) if sc_col is not None and not pd.isna(row[sc_col]) else default_sc
        export_col = "export_rate" if "export_rate" in df.columns else ("export_rate_eur_kwh" if "export_rate_eur_kwh" in df.columns else None)
        export_rate = float(row[export_col]) if export_col is not None and not pd.isna(row[export_col]) else float(DEFAULT_EXPORT_RATE)

        if fam == "standard":
            rates = {
                "standard": {
                    "day": _get_float(row, ["standard_day", "weekday_day"], "standard_day"),
                    "peak": _get_float(row, ["standard_peak", "weekday_peak"], "standard_peak"),
                    "night": _get_float(row, ["standard_night", "weekday_night"], "standard_night"),
                }
            }
            out[fam].append({"variant": variant_name, "standing_charge": standing_charge, "export_rate": export_rate, "rates": rates})
        elif fam == "weekend":
            rates = {
                "weekend": {
                    "weekday": {
                        "day": _get_float(row, ["weekend_weekday_day", "weekday_day"], "weekday_day"),
                        "peak": _get_float(row, ["weekend_weekday_peak", "weekday_peak"], "weekday_peak"),
                        "night": _get_float(row, ["weekend_weekday_night", "weekday_night"], "weekday_night"),
                    },
                    "weekend": {
                        "day": _get_float(row, ["weekend_weekend_day", "weekend_day"], "weekend_day"),
                        "peak": _get_float(row, ["weekend_weekend_peak", "weekend_peak"], "weekend_peak"),
                        "night": _get_float(row, ["weekend_weekend_night", "weekend_night"], "weekend_night"),
                    },
                }
            }
            out[fam].append({"variant": variant_name, "standing_charge": standing_charge, "export_rate": export_rate, "rates": rates})
        else:  # flat
            flat_rate = None
            if "flat_rate" in df.columns and not pd.isna(row.get("flat_rate")):
                flat_rate = float(row.get("flat_rate"))
            elif "rate" in df.columns and not pd.isna(row.get("rate")):
                flat_rate = float(row.get("rate"))
            elif "weekday_day" in df.columns and not pd.isna(row.get("weekday_day")):
                flat_rate = float(row.get("weekday_day"))
            if flat_rate is None:
                raise ValueError(f"Missing required flat import rate (expected 'flat_rate' or 'rate') for row {row.name}.")
            # Keep `rates` shape consistent with the UI loader, which expects:
            #   v["rates"]["flat"]["flat"] (and then calls r.get("flat", ...)).
            rates = {"flat": {"flat": float(flat_rate)}}
            out[fam].append({"variant": variant_name, "standing_charge": standing_charge, "export_rate": export_rate, "rates": rates})

    return out


def _tariff_type_display_label(family: str) -> str:
    """CSV-style type label for the tariff matrix table."""
    return {"standard": "standard", "weekend": "weekend_saver", "flat": "flat_rate"}.get(str(family).strip(), str(family))


def _tariff_matrix_profiles_from_parsed(
    parsed: Dict[str, List[Dict]], *, max_per_family: int = 20
) -> List[Dict]:
    """Flatten parsed tariff CSV into profile dicts with unique ``col`` keys ``tariff_sel_0`` …."""
    out: List[Dict] = []
    idx = 0
    for fam in ("standard", "weekend", "flat"):
        for item in list(parsed.get(fam) or [])[: int(max_per_family)]:
            if not isinstance(item, dict):
                continue
            v = str(item.get("variant", "Unknown")).strip()
            sc = float(item.get("standing_charge", 0.0) or 0.0)
            er = float(item.get("export_rate", DEFAULT_EXPORT_RATE))
            rates = item.get("rates") or {}
            if fam == "standard":
                std = rates.get("standard", {})
                out.append(
                    {
                        "family": "standard",
                        "variant": v,
                        "col": f"tariff_sel_{idx}",
                        "name": f"Standard — {v}",
                        "kind": "standard",
                        "rates": {"standard": dict(std)},
                        "standing_charge": sc,
                        "export_rate": er,
                    }
                )
            elif fam == "weekend":
                out.append(
                    {
                        "family": "weekend",
                        "variant": v,
                        "col": f"tariff_sel_{idx}",
                        "name": f"Weekend Saver — {v}",
                        "kind": "weekend",
                        "rates": dict(rates) if rates else {},
                        "standing_charge": sc,
                        "export_rate": er,
                    }
                )
            else:
                out.append(
                    {
                        "family": "flat",
                        "variant": v,
                        "col": f"tariff_sel_{idx}",
                        "name": f"Flat — {v}",
                        "kind": "flat",
                        "rates": dict(rates) if rates else {},
                        "standing_charge": sc,
                        "export_rate": er,
                    }
                )
            idx += 1
    return out


def _tariff_matrix_from_builtin_defaults() -> List[Dict]:
    """Fallback when no tariff CSV is available: three built-in Market average profiles."""
    out: List[Dict] = []
    for i, p in enumerate(_default_tariff_profiles()):
        q = dict(p)
        q["col"] = f"tariff_sel_{i}"
        out.append(q)
    return out


def _tariff_rates_summary_for_matrix(profile: Dict) -> str:
    """One-line rate summary for the scrollable tariff matrix."""
    kind = str(profile.get("kind", ""))
    rates = profile.get("rates") or {}
    if kind == "standard":
        s = rates.get("standard", {})
        return f"d={float(s.get('day', 0)):.4f} p={float(s.get('peak', 0)):.4f} n={float(s.get('night', 0)):.4f}"
    if kind == "flat":
        fe = rates.get("flat", DEFAULT_TARIFFS["flat"])
        r = float(fe.get("flat", DEFAULT_TARIFFS["flat"])) if isinstance(fe, dict) else float(fe)
        return f"flat={r:.4f}"
    if kind == "weekend":
        wk = rates.get("weekend", {})
        wd = wk.get("weekday", {})
        we = wk.get("weekend", {})
        return (
            f"wd {float(wd.get('day', 0)):.3f}/{float(wd.get('peak', 0)):.3f}/{float(wd.get('night', 0)):.3f} · "
            f"we {float(we.get('day', 0)):.3f}/{float(we.get('peak', 0)):.3f}/{float(we.get('night', 0)):.3f}"
        )
    return "—"


def _load_tariff_matrix_profiles_initial() -> Tuple[List[Dict], str]:
    """Load matrix from env / local_tariffs.csv / bundled default_tariffs.csv, else three built-in profiles."""
    env = os.environ.get("REC_FEASIBILITY_DEFAULT_TARIFFS_CSV", "").strip()
    candidates: List[Tuple[Path, str]] = []
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            candidates.append((p, f"default ({p.name})"))
    if LOCAL_OVERRIDE_TARIFFS_CSV.is_file():
        candidates.append((LOCAL_OVERRIDE_TARIFFS_CSV, "default (local_tariffs.csv)"))
    if BUILTIN_DEFAULT_TARIFFS_CSV.is_file():
        candidates.append((BUILTIN_DEFAULT_TARIFFS_CSV, "default (bundled tariffs)"))
    for path, label in candidates:
        try:
            parsed = _parse_tariff_variants_csv_bytes(path.read_bytes())
            profs = _tariff_matrix_profiles_from_parsed(parsed)
            if profs:
                return profs, label
        except Exception:
            continue
    return _tariff_matrix_from_builtin_defaults(), "built-in (3× Market average)"


# Season definitions (Northern hemisphere) using equinox/solstice-style date boundaries.
# Spring:  Mar 21 – Jun 20
# Summer:  Jun 21 – Sep 22
# Autumn:  Sep 23 – Dec 20
# Winter:  Dec 21 – Mar 20
def _date_to_nh_season(d: pd.Timestamp) -> str:
    if pd.isna(d):
        return "All Year"
    mmdd = int(d.month) * 100 + int(d.day)
    if 321 <= mmdd <= 620:
        return "Spring"
    if 621 <= mmdd <= 922:
        return "Summer"
    if 923 <= mmdd <= 1220:
        return "Autumn"
    # Winter spans year end: Dec 21–Dec 31 OR Jan 1–Mar 20
    return "Winter"


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
    out["season"] = out["date"].map(_date_to_nh_season)
    out["time_band"] = out["hour"].map(_consumption_time_band)
    return out


def render_community_consumption_patterns(prepared_df: pd.DataFrame) -> None:
    """Scenario-independent demand charts: how the community consumes electricity."""
    if prepared_df is None or len(prepared_df) == 0:
        st.info("Load consumption data to see community demand patterns.")
        return
    h = _community_consumption_features(prepared_df)
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
    seasonal_order = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = h.groupby("season", sort=False)["consumption"].sum().reindex(seasonal_order).fillna(0)

    band_labels_share = ["Night", "Day", "Peak"]

    bp_m = h.groupby(["month", "time_band"], sort=True)["consumption"].sum().unstack(fill_value=0.0)
    for _b in ("Night", "Day", "Peak"):
        if _b not in bp_m.columns:
            bp_m[_b] = 0.0
    bp_m = bp_m.reindex(range(1, 13), fill_value=0.0)[["Night", "Day", "Peak"]]
    monthly_bar_totals = bp_m.sum(axis=1).to_numpy(dtype=float)

    # First chart row: match **Production patterns** `render_production_patterns_per_kwp` first row
    # (height 220, margins l/r/t/b 8 except extra right margin when a legend is shown).
    _leg_tb = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        x=1.02,
        xanchor="left",
        font=dict(size=10),
    )
    _cp_top_h = 220
    _cp_top_margin_std = dict(l=8, r=8, t=36, b=8)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_tb = go.Figure()
        for band in ("Night", "Day", "Peak"):
            fig_tb.add_trace(
                go.Bar(
                    name=band,
                    x=month_names,
                    y=bp_m[band].to_numpy(dtype=float),
                    marker_color=TIME_BAND_CHART_COLORS.get(band, "#64748b"),
                )
            )
        fig_tb.update_layout(
            barmode="stack",
            title="Monthly community consumption by time band (kWh)",
            margin={**_cp_top_margin_std, "r": 108},
            height=_cp_top_h,
            legend=_leg_tb,
            bargap=0.12,
            xaxis_title="",
            yaxis_title="kWh",
            yaxis=dict(rangemode="tozero"),
        )
        _apply_yaxis_range_from_values(fig_tb, monthly_bar_totals)
        render_plotly_figure(
            fig_tb,
            key="consumption_patterns_monthly_timeband",
            apply_bar_value_labels=False,
        )
    with r1c2:
        fig_s = go.Figure(data=[go.Bar(x=seasonal.index.tolist(), y=seasonal.values, marker_color="#6366f1")])
        fig_s.update_layout(
            title="Seasonal consumption (kWh)",
            margin=_cp_top_margin_std,
            height=_cp_top_h,
            showlegend=False,
            xaxis_title="",
            yaxis_title="kWh",
            yaxis=dict(rangemode="tozero"),
        )
        _apply_yaxis_range_from_values(fig_s, seasonal.values)
        render_plotly_figure(fig_s, key="consumption_patterns_seasonal", apply_bar_value_labels=False)

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
        _apply_yaxis_range_from_values(fig_d, daily.values)
        render_plotly_figure(fig_d)
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
                    marker_color=TIME_BAND_CHART_COLORS[band],
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
        _apply_yaxis_range_from_values(fig_av, hr_all.to_numpy(dtype=float))
        render_plotly_figure(fig_av)

    def _time_band_share_fig(title: str, sub: pd.DataFrame) -> go.Figure:
        if len(sub) == 0:
            vals = [0.0, 0.0, 0.0]
        else:
            sums = sub.groupby("time_band")["consumption"].sum()
            vals = [float(sums.get(b, 0.0)) for b in band_labels_share]
        if sum(vals) <= 0:
            vals = [1.0, 1.0, 1.0]
        colors = [TIME_BAND_CHART_COLORS[b] for b in band_labels_share]
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
            render_plotly_figure(
                _time_band_share_fig(ttl, sub),
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
    _apply_yaxis_range_from_values(fig_wd, np.concatenate([hr_wd.to_numpy(dtype=float), hr_we.to_numpy(dtype=float)]))

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
    _apply_yaxis_range_from_values(
        fig_season_h,
        np.concatenate(
            [
                h[h["season"] == s].groupby("hour")["consumption"].mean().reindex(range(24)).fillna(0.0).to_numpy(dtype=float)
                for s in seasonal_order
            ]
        )
        if len(seasonal_order) > 0
        else np.asarray([], dtype=float),
    )

    line_wd_col, line_season_col = st.columns(2)
    with line_wd_col:
        render_plotly_figure(fig_wd)
    with line_season_col:
        render_plotly_figure(fig_season_h)

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
        render_plotly_figure(fig_hm)
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
        render_plotly_figure(fig_heat_dow)


def _pv_per_kwp_pattern_features(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly rows: PV production per nominal kWp (kWh/kWp per hour), with calendar features."""
    out = prepared_df[["date", "pv_per_kwp"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["pv_kwh_per_kwp"] = pd.to_numeric(out["pv_per_kwp"], errors="coerce").fillna(0.0)
    out["month"] = out["date"].dt.month
    out["hour"] = out["date"].dt.hour
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["dow"] >= 5
    out["day"] = out["date"].dt.normalize()
    out["season"] = out["date"].map(_date_to_nh_season)
    out["time_band"] = out["hour"].map(_consumption_time_band)
    return out


def render_production_patterns_per_kwp(prepared_df: pd.DataFrame) -> None:
    """Scenario-independent PV profile charts: hourly production per 1 kWp from prepared data."""
    if prepared_df is None or len(prepared_df) == 0:
        st.info("Load PV data to see production patterns.")
        return
    if "pv_per_kwp" not in prepared_df.columns:
        st.warning("Prepared data has no **pv_per_kwp** column — production patterns are unavailable for this session.")
        return
    h = _pv_per_kwp_pattern_features(prepared_df)
    ycol = "pv_kwh_per_kwp"
    annual_yield = float(h[ycol].sum())
    n_hours = len(h)
    daily_totals = h.groupby("day")[ycol].sum()
    n_days = max(len(daily_totals), 1)
    avg_daily_yield = float(daily_totals.mean()) if len(daily_totals) else 0.0
    k0a, k0b, k0c = st.columns(3)
    with k0a:
        st.metric(
            "Annual specific yield (kWh/kWp)",
            f"{annual_yield:,.2f}",
            help=f"Sum of hourly kWh per 1 kWp over {n_hours:,} hours (energy one 1 kWp unit would produce).",
        )
    with k0b:
        st.metric(
            "Average daily yield (kWh/kWp/day)",
            f"{avg_daily_yield:,.2f}",
            help=f"Mean daily sum of kWh/kWp over {n_days:,} calendar days.",
        )
    with k0c:
        st.metric(
            "Average hourly yield (kWh/kWp/h)",
            f"{annual_yield / max(n_hours, 1):,.3f}",
            help="Mean kWh per kWp per clock hour over the series.",
        )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = h.groupby("month", sort=True)[ycol].sum()
    seasonal_order = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = h.groupby("season", sort=False)[ycol].sum().reindex(seasonal_order).fillna(0)

    band_labels_share = ["Night", "Day", "Peak"]

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_m = go.Figure(
            data=[go.Bar(x=[month_names[m - 1] for m in monthly.index], y=monthly.values, marker_color="#f59e0b")]
        )
        fig_m.update_layout(
            title="Monthly production (kWh/kWp)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=220,
            showlegend=False,
            xaxis_title="",
            yaxis_title="kWh/kWp",
            yaxis=dict(rangemode="tozero"),
        )
        _apply_yaxis_range_from_values(fig_m, monthly.values)
        render_plotly_figure(fig_m)
    with r1c2:
        fig_s = go.Figure(data=[go.Bar(x=seasonal.index.tolist(), y=seasonal.values, marker_color="#ea580c")])
        fig_s.update_layout(
            title="Seasonal production (kWh/kWp)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=220,
            showlegend=False,
            xaxis_title="",
            yaxis_title="kWh/kWp",
            yaxis=dict(rangemode="tozero"),
        )
        _apply_yaxis_range_from_values(fig_s, seasonal.values)
        render_plotly_figure(fig_s)

    r4c1, r4c2 = st.columns(2)
    daily = daily_totals.sort_index()
    with r4c1:
        fig_d = go.Figure(data=[go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(color="#d97706", width=1))])
        fig_d.update_layout(
            title="Daily production (kWh/kWp)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=240,
            xaxis_title="Date",
            yaxis_title="kWh/kWp",
            xaxis=dict(tickformat="%b %Y", nticks=10),
            yaxis=dict(rangemode="tozero"),
        )
        _apply_yaxis_range_from_values(fig_d, daily.values)
        render_plotly_figure(fig_d)
    with r4c2:
        hr_all = h.groupby("hour")[ycol].mean().reindex(range(24)).fillna(0.0)
        fig_av = go.Figure()
        for band in band_labels_share:
            xs = [hr for hr in range(24) if _consumption_time_band(hr) == band]
            ys = [float(hr_all.iloc[hr]) for hr in xs]
            fig_av.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=band,
                    marker_color=TIME_BAND_CHART_COLORS[band],
                    marker_line_width=0,
                    hovertemplate="Hour %{x}<br>Mean kWh/kWp/h %{y:.3f}<extra></extra>",
                )
            )
        fig_av.update_layout(
            title="Average daily profile (kWh/kWp per hour)",
            margin=dict(l=8, r=120, t=36, b=40),
            height=240,
            xaxis_title="Hour",
            yaxis_title="kWh/kWp/h",
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
        _apply_yaxis_range_from_values(fig_av, hr_all.to_numpy(dtype=float))
        render_plotly_figure(fig_av)

    def _time_band_share_fig_pv(title: str, sub: pd.DataFrame) -> go.Figure:
        if len(sub) == 0:
            vals = [0.0, 0.0, 0.0]
        else:
            sums = sub.groupby("time_band")[ycol].sum()
            vals = [float(sums.get(b, 0.0)) for b in band_labels_share]
        if sum(vals) <= 0:
            vals = [1.0, 1.0, 1.0]
        colors = [TIME_BAND_CHART_COLORS[b] for b in band_labels_share]
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
                    hovertemplate="<b>%{label}</b><br>%{value:,.0f} kWh/kWp<br>(%{percent})<extra></extra>",
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

    st.markdown("##### Share of production by time band (kWh/kWp)")
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
            render_plotly_figure(
                _time_band_share_fig_pv(ttl, sub),
            )

    st.markdown("")
    hr_wd = h[~h["is_weekend"]].groupby("hour")[ycol].mean().reindex(range(24)).fillna(0.0)
    hr_we = h[h["is_weekend"]].groupby("hour")[ycol].mean().reindex(range(24)).fillna(0.0)
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
        title=dict(text="Average hourly yield — weekday vs weekend (kWh/kWp/h)", y=0.97, x=0, xanchor="left"),
        margin=dict(l=8, r=120, t=48, b=40),
        height=260,
        xaxis_title="Hour",
        yaxis_title="kWh/kWp/h",
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
    _apply_yaxis_range_from_values(fig_wd, np.concatenate([hr_wd.to_numpy(dtype=float), hr_we.to_numpy(dtype=float)]))

    season_line_colors = {"Winter": "#38bdf8", "Spring": "#4ade80", "Summer": "#fbbf24", "Autumn": "#fb923c"}
    fig_season_h = go.Figure()
    for s in seasonal_order:
        sub = h[h["season"] == s]
        if len(sub) == 0:
            continue
        hr = sub.groupby("hour")[ycol].mean().reindex(range(24)).fillna(0.0)
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
        title=dict(text="Average hourly yield by season (kWh/kWp/h)", y=0.97, x=0, xanchor="left"),
        margin=dict(l=8, r=120, t=48, b=40),
        height=260,
        xaxis_title="Hour",
        yaxis_title="kWh/kWp/h",
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
    _apply_yaxis_range_from_values(
        fig_season_h,
        np.concatenate(
            [
                h[h["season"] == s].groupby("hour")[ycol].mean().reindex(range(24)).fillna(0.0).to_numpy(dtype=float)
                for s in seasonal_order
            ]
        )
        if len(seasonal_order) > 0
        else np.asarray([], dtype=float),
    )

    line_wd_col, line_season_col = st.columns(2)
    with line_wd_col:
        render_plotly_figure(fig_wd)
    with line_season_col:
        render_plotly_figure(fig_season_h)

    hm_left, hm_right = st.columns(2)
    pivot_hm = h.pivot_table(values=ycol, index="hour", columns="month", aggfunc="mean")
    pivot_hm = pivot_hm.reindex(index=range(24), columns=range(1, 13)).fillna(0)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot_dow = h.pivot_table(values=ycol, index="hour", columns="dow", aggfunc="mean")
    pivot_dow = pivot_dow.reindex(index=range(24), columns=range(7)).fillna(0)
    with hm_left:
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot_hm.values,
                x=[month_names[m - 1] for m in pivot_hm.columns],
                y=pivot_hm.index,
                colorscale="YlOrRd",
                colorbar=dict(title="kWh/kWp/h", len=0.45),
            )
        )
        fig_hm.update_layout(
            title="Mean yield — hour × month (kWh/kWp/h)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=260,
            xaxis_title="Month",
            yaxis_title="Hour",
        )
        render_plotly_figure(fig_hm)
    with hm_right:
        fig_heat_dow = go.Figure(
            data=go.Heatmap(
                z=pivot_dow.values,
                x=dow_names,
                y=pivot_dow.index,
                colorscale="YlOrRd",
                colorbar=dict(title="kWh/kWp/h", len=0.45),
            )
        )
        fig_heat_dow.update_layout(
            title="Mean yield — hour × day of week (kWh/kWp/h)",
            margin=dict(l=8, r=8, t=36, b=8),
            height=260,
            xaxis_title="",
            yaxis_title="Hour",
        )
        render_plotly_figure(fig_heat_dow)


RANK_GOAL_OPTIONS = [p[1] for p in RECOMMENDED_WINNER_PRESETS]

TARIFF_FAMILY_FILTER_OPTIONS = [
    "All tariff types",
    "Standard",
    "Weekend saver",
    "Flat rate",
]

# Sidebar "Decision constraints" enable flags (number inputs share these keys for disabled state).
HARD_FILTER_ENABLE_KEYS: tuple[str, ...] = (
    "hard_capex_max_en",
    "hard_npv_min_en",
    "hard_payback_max_en",
    "hard_irr_min_en",
    "hard_ss_min_en",
    "hard_co2_min_en",
    "hard_ann_cost_saving_min_en",
    "hard_ann_cost_max_en",
    "hard_self_cons_min_en",
    "hard_export_max_en",
)


def _sidebar_active_hard_filter_count() -> int:
    return sum(1 for k in HARD_FILTER_ENABLE_KEYS if st.session_state.get(k))


def _sidebar_clear_decision_constraints() -> None:
    for k in HARD_FILTER_ENABLE_KEYS:
        st.session_state[k] = False


def _sidebar_apply_recommended_decision_constraints_defaults() -> None:
    """Apply the default Recommended constraints set."""
    _sidebar_clear_decision_constraints()
    st.session_state.hard_payback_max_en = True
    st.session_state.hard_payback_max_years = float(RECOMMENDED_SETUP_MAX_PAYBACK_YEARS)
    st.session_state.hard_self_cons_min_en = True
    st.session_state.hard_self_cons_min_pct = float(RECOMMENDED_SETUP_DEFAULT_MIN_SELF_CONSUMPTION_PCT)
    st.session_state.hard_export_max_en = True
    st.session_state.hard_export_max_pct = float(RECOMMENDED_SETUP_DEFAULT_MAX_EXPORT_RATIO_PCT)
    st.session_state.hard_npv_min_en = True
    st.session_state.hard_npv_min_eur = 0.0
    st.session_state.hard_co2_min_en = True
    st.session_state.hard_co2_min_pct = 0.0


def _sidebar_reset_all_result_filters() -> None:
    st.session_state.view_goal = RANK_GOAL_OPTIONS[0]
    st.session_state.view_scenario_type = "All scenarios"
    st.session_state.view_tariff_family = TARIFF_FAMILY_FILTER_OPTIONS[0]
    _sidebar_apply_recommended_decision_constraints_defaults()


@dataclass
class SetupFormValues:
    """Widget outputs from the model setup form (main page / expander after run)."""

    cons_file: Optional[object]
    pv_file: Optional[object]
    pv_capex: float
    batt_capex: float
    tariff_profiles: List[Dict]
    pso_levy: float
    opex_pct: float
    discount_rate: float
    electricity_inflation_rate: float
    battery_replacement_year: int | None
    battery_replacement_cost_pct: float
    inverter_replacement_year: int | None
    inverter_replacement_cost_pct: float
    speed_preset: str
    opt_pv_step: int
    opt_batt_step: int
    pv_min: int
    pv_max: int
    batt_min: int
    batt_max: int
    rt_eff_pct: float
    dod_pct: float
    init_soc_pct: float
    min_soc_pct: float
    max_soc_pct: float
    c_rate: float
    charge_from_pv: bool
    charge_from_grid_at_night: bool
    discharge_schedule: str
    run_button: bool
    co2_factor: float = DEFAULT_CO2_FACTOR
    lifetime_years: int = DEFAULT_LIFETIME_YEARS


def render_setup_form() -> SetupFormValues:
    """Compact top: data + core finance + one Run row; three expanders for the rest."""
    st.caption(
        "Top: uploads and core costs → **Run analysis**. Below: **Data & tariffs**, **Financial assumptions**, **Advanced / optimizer settings**."
    )
    if "stop_run_requested" not in st.session_state:
        st.session_state.stop_run_requested = False

    col_data, col_core = st.columns(2)
    with col_data:
        st.markdown("##### Data")
        cons_file = st.file_uploader(
            "Consumption CSV",
            type=["csv"],
            key="cons",
            help="DD/MM/YYYY HH:00 + consumption. Empty → env/local/bundled default (see data/local_consumption.csv or REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV).",
        )
        pv_file = st.file_uploader(
            "PV CSV (PVGIS)",
            type=["csv"],
            key="pv",
            help="time + P (Wh per 1 kWp). Empty → env/local/bundled default (see data/local_pv.csv or REC_FEASIBILITY_DEFAULT_PV_CSV).",
        )
        _cl = "uploaded" if cons_file is not None else "built-in sample"
        _pl = "uploaded" if pv_file is not None else "built-in sample"
        st.caption(
            f"Next run: consumption **{_cl}** · PV **{_pl}** · tariffs **from setup** (matrix: default CSV / upload, **Include** checkboxes)."
        )

    with col_core:
        st.markdown("##### Core costs")
        pv_capex = st.number_input(
            "PV CAPEX (€/kWp)",
            min_value=0.0,
            value=float(PV_COST_PER_KWP),
            step=50.0,
        )
        if ENABLE_BATTERY_UI:
            batt_capex = st.number_input(
                "Battery CAPEX (€/kWh)",
                min_value=0.0,
                value=float(BATT_COST_PER_KWH),
                step=50.0,
            )
        else:
            batt_capex = float(BATT_COST_PER_KWH)
        pso_levy = st.number_input(
            "PSO levy (annual, €)",
            min_value=0.0,
            value=float(DEFAULT_PSO_LEVY_EUR_PER_YEAR),
            step=0.05,
            help="Included in annual electricity cost; escalates with inflation in long-run metrics.",
        )
        opex_pct = st.number_input(
            "OPEX (annual, % of CAPEX)",
            min_value=0.0,
            value=float(DEFAULT_OPEX_PCT),
            step=0.5,
            help="Annual operating cost as % of scenario total CAPEX.",
        )
        discount_rate_pct = st.number_input(
            "Discount rate for NPV (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(DISCOUNT_RATE * 100),
            step=0.5,
            help="Annual discount rate for NPV.",
        )
        discount_rate = discount_rate_pct / 100.0
        electricity_inflation_pct = st.number_input(
            "Electricity inflation (% per year)",
            min_value=0.0,
            max_value=15.0,
            value=float(ELECTRICITY_INFLATION_RATE * 100),
            step=0.1,
            help="Escalates import/export, standing charge, PSO, OPEX (not CAPEX).",
        )
        electricity_inflation_rate = electricity_inflation_pct / 100.0
        if "setup_grid_co2_factor" not in st.session_state:
            st.session_state.setup_grid_co2_factor = float(
                st.session_state.get("last_co2_factor", DEFAULT_CO2_FACTOR)
            )
        grid_co2_factor = st.number_input(
            "Grid CO₂ factor (kg/kWh)",
            min_value=0.0,
            max_value=2.0,
            step=0.0001,
            format="%.4f",
            key="setup_grid_co2_factor",
            help="kg CO₂ per kWh of grid electricity imported — drives scenario CO₂ and savings vs grid-only.",
        )

    with st.expander("Data & tariffs", expanded=False):
        st.caption(
            "Default tariffs load from **data/default_tariffs.csv** (override with **data/local_tariffs.csv** or "
            "**REC_FEASIBILITY_DEFAULT_TARIFFS_CSV**). Tick **Include** for each row the optimizer should evaluate."
        )

        st.markdown("###### Tariff rates CSV (optional)")
        tariffs_csv_file = st.file_uploader(
            "Upload tariff rates CSV",
            type=["csv"],
            key="tariffs_csv",
            help="One row per variant. Required: family (standard|weekend|flat), variant (company name). Optional: standing_charge (€/year), export_rate (€/kWh). "
            "Rates columns: standard_day/standard_peak/standard_night; weekend_weekday_day/… + weekend_weekend_day/…; flat_rate (or rate). "
            "Also accepted aliases: tariff_type/company, weekday_day|peak|night and weekend_day|peak|night.",
        )
        load_tariffs_clicked = st.button("Load tariffs from CSV", disabled=tariffs_csv_file is None, key="load_tariffs_csv")

        if "tariff_matrix_version" not in st.session_state:
            st.session_state.tariff_matrix_version = 0
        if "tariff_matrix_profiles" not in st.session_state:
            _tm_init, _tm_lbl = _load_tariff_matrix_profiles_initial()
            st.session_state.tariff_matrix_profiles = _tm_init
            st.session_state.tariff_matrix_source_label = _tm_lbl

        # Keep defaults in sync with disk/env overrides when no upload is active.
        # This lets "Edit assumptions and rerun" pick up reduced tariff CSV rows
        # from data/local_tariffs.csv (or REC_FEASIBILITY_DEFAULT_TARIFFS_CSV)
        # without requiring a fresh browser session.
        _tm_src_cur = str(st.session_state.get("tariff_matrix_source_label", "") or "")
        if tariffs_csv_file is None and _tm_src_cur != "uploaded CSV":
            _tm_fresh, _tm_lbl_fresh = _load_tariff_matrix_profiles_initial()
            if (
                list(st.session_state.get("tariff_matrix_profiles") or []) != list(_tm_fresh or [])
                or _tm_src_cur != str(_tm_lbl_fresh)
            ):
                st.session_state.tariff_matrix_profiles = list(_tm_fresh or [])
                st.session_state.tariff_matrix_source_label = str(_tm_lbl_fresh)
                st.session_state.tariff_matrix_version = int(st.session_state.get("tariff_matrix_version", 0)) + 1

        if load_tariffs_clicked and tariffs_csv_file is not None:
            try:
                parsed = _parse_tariff_variants_csv_bytes(tariffs_csv_file.getvalue())
                profs = _tariff_matrix_profiles_from_parsed(parsed)
                if not profs:
                    st.error("No valid tariff rows found in CSV.")
                else:
                    st.session_state.tariff_matrix_profiles = profs
                    st.session_state.tariff_matrix_version = int(st.session_state.tariff_matrix_version) + 1
                    st.session_state.tariff_matrix_source_label = "uploaded CSV"
                    n_std = len(parsed.get("standard") or [])
                    n_wk = len(parsed.get("weekend") or [])
                    n_fl = len(parsed.get("flat") or [])
                    st.success(
                        f"Loaded **{len(profs)}** tariff row(s) ({n_std} standard, {n_wk} weekend saver, {n_fl} flat). "
                        "All rows are included by default — uncheck any you want to skip."
                    )
            except Exception as e:
                st.error(f"Could not load tariff CSV. {e}")

        _tm_profs = list(st.session_state.get("tariff_matrix_profiles") or [])
        _tm_ver = int(st.session_state.get("tariff_matrix_version", 0))
        _tm_src = str(st.session_state.get("tariff_matrix_source_label", ""))

        st.markdown("###### Tariff matrix (include in optimizer)")
        _n_sel = sum(
            1
            for i in range(len(_tm_profs))
            if bool(st.session_state.get(f"tm_inc_{_tm_ver}_{i}", True))
        )
        st.caption(f"Source: **{_tm_src}** · **{_n_sel}** / **{len(_tm_profs)}** rows selected for the next run.")

        _b_all, _b_none, _ = st.columns([1, 1, 4])
        if _b_all.button("Select all", key="tariff_matrix_select_all"):
            for i in range(len(_tm_profs)):
                st.session_state[f"tm_inc_{_tm_ver}_{i}"] = True
            st.rerun()
        if _b_none.button("Select none", key="tariff_matrix_select_none"):
            for i in range(len(_tm_profs)):
                st.session_state[f"tm_inc_{_tm_ver}_{i}"] = False
            st.rerun()

        with st.container(height=400):
            _h0, _h1, _h2, _h3, _h4, _h5 = st.columns([0.55, 1.15, 1.35, 1.05, 0.85, 2.6])
            with _h0:
                st.caption("**✓**")
            with _h1:
                st.caption("**Type**")
            with _h2:
                st.caption("**Supplier**")
            with _h3:
                st.caption("**Standing €/y**")
            with _h4:
                st.caption("**Export**")
            with _h5:
                st.caption("**Rates (€/kWh)**")

            for _i, _p in enumerate(_tm_profs):
                _r0, _r1, _r2, _r3, _r4, _r5 = st.columns([0.55, 1.15, 1.35, 1.05, 0.85, 2.6])
                with _r0:
                    st.checkbox(
                        "Include",
                        value=True,
                        key=f"tm_inc_{_tm_ver}_{_i}",
                        label_visibility="collapsed",
                    )
                with _r1:
                    st.text(_tariff_type_display_label(str(_p.get("family", ""))))
                with _r2:
                    st.text(str(_p.get("variant", "")))
                with _r3:
                    st.text(f"{float(_p.get('standing_charge', 0.0) or 0.0):,.2f}")
                with _r4:
                    st.text(f"{float(_p.get('export_rate', DEFAULT_EXPORT_RATE)):.4f}")
                with _r5:
                    st.text(_tariff_rates_summary_for_matrix(_p))

        profiles: List[Dict] = []
        for _i, _p in enumerate(_tm_profs):
            if bool(st.session_state.get(f"tm_inc_{_tm_ver}_{_i}", True)):
                profiles.append(_p)

        if not profiles:
            st.warning("No tariff rows selected. Check at least one **Include** before **Run analysis**.")

    with st.expander("Financial assumptions", expanded=False):
        st.caption("NPV, IRR, gross/net savings, and cumulative charts use the horizon below.")
        _ly_default = int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))
        _ly_clamped = min(30, max(15, _ly_default))
        lifetime_years = int(
            st.number_input(
                "Analysis horizon (years)",
                min_value=15,
                max_value=30,
                value=_ly_clamped,
                step=1,
                help="Discounted cashflows, gross/net savings, and replacement-year upper bounds.",
                key="setup_lifetime_years_input",
            )
        )
        st.caption("Replacement cashflows (NPV/IRR); optional.")
        if ENABLE_BATTERY_UI:
            battery_replacement_year_input = st.number_input(
                "Battery replacement year (0 = none)",
                min_value=0,
                max_value=lifetime_years,
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
        else:
            battery_replacement_year_input = 0
            battery_replacement_cost_pct = 0.0
        inverter_replacement_year_input = st.number_input(
            "Inverter replacement year (0 = none)",
            min_value=0,
            max_value=lifetime_years,
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

    with st.expander("Advanced / optimizer settings", expanded=False):
        if ENABLE_BATTERY_UI:
            st.markdown("###### Battery model")
            rt_eff_pct = st.slider(
                "Round-trip efficiency (%)",
                min_value=50,
                max_value=100,
                value=95,
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
            min_soc_pct = st.slider(
                "Minimum SOC floor (%)",
                min_value=0,
                max_value=80,
                value=10,
                step=1,
                help="Battery will not discharge below this SOC.",
            )
            max_soc_pct = st.slider(
                "Maximum SOC ceiling (%)",
                min_value=int(min_soc_pct) + 1,
                max_value=100,
                value=max(90, int(min_soc_pct) + 1),
                step=1,
                help="Battery will not charge above this SOC.",
            )
            c_rate = st.slider(
                "Max charge/discharge power as C-rate",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
            charge_from_pv = st.checkbox("Allow battery charging from PV surplus", value=True)
            if "setup_battery_charge_from_grid_night" not in st.session_state:
                st.session_state.setup_battery_charge_from_grid_night = False
            charge_from_grid_at_night = st.checkbox(
                "Allow battery charging from grid during night",
                key="setup_battery_charge_from_grid_night",
            )
            discharge_schedule = st.selectbox(
                "Battery discharging schedule",
                ["Peak only", "Day+Peak"],
                index=0,
                help="**Peak only:** discharge 17:00-19:00. **Day+Peak:** same peak window, then 19:00-23:00 if SOC remains (no discharge 23:00-08:00 or 08:00-17:00).",
            )
        else:
            rt_eff_pct = 95.0
            dod_pct = 90.0
            init_soc_pct = 0.0
            min_soc_pct = 10.0
            max_soc_pct = 90.0
            c_rate = 0.5
            charge_from_pv = True
            charge_from_grid_at_night = False
            discharge_schedule = "Peak only"

        st.markdown("###### Optimizer search")
        st.caption("PV" + (" and battery" if ENABLE_BATTERY_UI else "") + " size grid for the search.")
        if ENABLE_BATTERY_UI:
            speed_preset = st.selectbox(
                "Optimizer speed preset",
                [
                    "Quick (PV step 5, battery step 5)",
                    "Fast (PV step 10, battery step 10)",
                    "Full (PV step 1, battery step 1)",
                ],
                index=2,
            )
            if speed_preset == "Quick (PV step 5, battery step 5)":
                opt_pv_step, opt_batt_step = 5, 5
            elif speed_preset == "Fast (PV step 10, battery step 10)":
                opt_pv_step, opt_batt_step = 10, 10
            else:
                opt_pv_step, opt_batt_step = 1, 1
        else:
            speed_preset = st.selectbox(
                "Optimizer speed preset",
                [
                    "Quick (PV step 5)",
                    "Fast (PV step 10)",
                    "Full (PV step 1)",
                ],
                index=2,
            )
            if speed_preset == "Quick (PV step 5)":
                opt_pv_step, opt_batt_step = 5, 5
            elif speed_preset == "Fast (PV step 10)":
                opt_pv_step, opt_batt_step = 10, 10
            else:
                opt_pv_step, opt_batt_step = 1, 1

        pv_range = st.slider(
            "PV size range (kWp)",
            min_value=0,
            max_value=150,
            value=(5, 60),
            step=5,
            key="pv_range",
        )
        pv_min, pv_max = pv_range[0], pv_range[1]
        if ENABLE_BATTERY_UI:
            batt_range = st.slider(
                "Battery size range (kWh)",
                min_value=0,
                max_value=300,
                value=(0, 40),
                step=5,
                key="batt_range",
            )
            batt_min, batt_max = batt_range[0], batt_range[1]
        else:
            batt_min, batt_max = 0, 0

    run_c1, run_c2 = st.columns([2, 1])
    with run_c1:
        run_clicked = st.button("Run analysis", type="primary", key="setup_run_analysis")
    with run_c2:
        if st.button("Stop run", help="Stop at next optimizer checkpoint.", key="setup_stop_run"):
            st.session_state.stop_run_requested = True

    run_button = bool(run_clicked)

    return SetupFormValues(
        cons_file=cons_file,
        pv_file=pv_file,
        pv_capex=float(pv_capex),
        batt_capex=float(batt_capex),
        tariff_profiles=profiles,
        pso_levy=float(pso_levy),
        opex_pct=float(opex_pct),
        discount_rate=float(discount_rate),
        electricity_inflation_rate=float(electricity_inflation_rate),
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_pct=float(battery_replacement_cost_pct),
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_pct=float(inverter_replacement_cost_pct),
        speed_preset=str(speed_preset),
        opt_pv_step=int(opt_pv_step),
        opt_batt_step=int(opt_batt_step),
        pv_min=int(pv_min),
        pv_max=int(pv_max),
        batt_min=int(batt_min),
        batt_max=int(batt_max),
        rt_eff_pct=float(rt_eff_pct),
        dod_pct=float(dod_pct),
        init_soc_pct=float(init_soc_pct),
        min_soc_pct=float(min_soc_pct),
        max_soc_pct=float(max_soc_pct),
        c_rate=float(c_rate),
        charge_from_pv=bool(charge_from_pv),
        charge_from_grid_at_night=bool(charge_from_grid_at_night),
        discharge_schedule=str(discharge_schedule),
        run_button=bool(run_button),
        co2_factor=float(grid_co2_factor),
        lifetime_years=int(lifetime_years),
    )


def _full_results_snapshot_is_usable(df: pd.DataFrame | None) -> bool:
    if df is None or len(df) == 0:
        return False
    need = {"Grid import reduction (kWh)", "CO2 reduction (%)", "Export ratio (% of PV gen)"}
    return need.issubset(df.columns)


def _last_run_dict_for_bundle() -> Dict[str, object]:
    ss = st.session_state
    pm = dict(ss.prepared_meta or {})
    for k in ("cons_sha", "pv_sha", "cons_source", "pv_source"):
        if k not in pm:
            pm[k] = None
    return {
        "last_pv_capex": float(ss.last_pv_capex),
        "last_batt_capex": float(ss.last_batt_capex),
        "last_opex_pct": float(ss.last_opex_pct),
        "last_discount_rate": float(ss.last_discount_rate),
        "last_electricity_inflation_rate": float(ss.last_electricity_inflation_rate),
        "last_battery_replacement_year": ss.last_battery_replacement_year,
        "last_battery_replacement_cost_pct": float(ss.last_battery_replacement_cost_pct),
        "last_inverter_replacement_year": ss.last_inverter_replacement_year,
        "last_inverter_replacement_cost_pct": float(ss.last_inverter_replacement_cost_pct),
        "last_pso_levy": float(ss.last_pso_levy),
        "last_co2_factor": float(ss.last_co2_factor),
        "last_lifetime_years": int(ss.last_lifetime_years),
        "last_export_rate": float(ss.last_export_rate),
        "last_opt_cfg": dict(ss.last_opt_cfg or {}),
        "last_battery_settings": battery_settings_to_json_dict(ss.last_battery_settings),
        "last_input_hashes": dict(ss.last_input_hashes or {}),
        "prepared_meta": pm,
        "last_tariff_matrix_source_label": str(ss.last_tariff_matrix_source_label or ""),
        "bundle_schema_version": int(BUNDLE_SCHEMA_VERSION),
    }


def _setup_form_values_from_last_run_for_cache() -> SetupFormValues:
    """Rebuild cached setup values from frozen last-run fields (after import)."""
    ss = st.session_state
    loc = ss.last_opt_cfg or {}
    bs: BatterySettings = ss.last_battery_settings
    profiles = list(ss.last_tariff_profiles or [])
    br_y = ss.last_battery_replacement_year
    ir_y = ss.last_inverter_replacement_year
    return SetupFormValues(
        cons_file=None,
        pv_file=None,
        pv_capex=float(ss.last_pv_capex),
        batt_capex=float(ss.last_batt_capex),
        tariff_profiles=profiles,
        pso_levy=float(ss.last_pso_levy),
        opex_pct=float(ss.last_opex_pct),
        discount_rate=float(ss.last_discount_rate),
        electricity_inflation_rate=float(ss.last_electricity_inflation_rate),
        battery_replacement_year=int(br_y) if br_y is not None and int(br_y) > 0 else None,
        battery_replacement_cost_pct=float(ss.last_battery_replacement_cost_pct),
        inverter_replacement_year=int(ir_y) if ir_y is not None and int(ir_y) > 0 else None,
        inverter_replacement_cost_pct=float(ss.last_inverter_replacement_cost_pct),
        speed_preset=str(loc.get("speed_preset", "Quick (PV step 5, battery step 5)")),
        opt_pv_step=int(loc.get("pv_step", 5)),
        opt_batt_step=int(loc.get("batt_step", 5)),
        pv_min=int(loc.get("pv_min", 5)),
        pv_max=int(loc.get("pv_max", 60)),
        batt_min=int(loc.get("batt_min", 0)),
        batt_max=int(loc.get("batt_max", 40)),
        rt_eff_pct=float(bs.eff_round_trip) * 100.0,
        dod_pct=float(bs.dod) * 100.0,
        init_soc_pct=float(bs.init_soc) * 100.0,
        min_soc_pct=float(bs.min_soc) * 100.0,
        max_soc_pct=float(bs.max_soc) * 100.0,
        c_rate=float(bs.c_rate),
        charge_from_pv=bool(bs.charge_from_pv),
        charge_from_grid_at_night=bool(bs.charge_from_grid_at_night),
        discharge_schedule=str(bs.discharge_schedule),
        run_button=False,
        co2_factor=float(ss.last_co2_factor),
        lifetime_years=int(ss.last_lifetime_years),
    )


def _setup_form_values_demo_preflight_placeholder() -> SetupFormValues:
    """When ``DEMO_MODE`` hides Model setup: valid defaults before embedded load populates session (``run_button`` always False)."""
    _tm_profs, _ = _load_tariff_matrix_profiles_initial()
    profiles = list(_tm_profs or [])
    if ENABLE_BATTERY_UI:
        _br_in = int(DEFAULT_BATTERY_REPLACEMENT_YEAR)
        battery_replacement_year = _br_in if _br_in > 0 else None
        _ir_in = int(DEFAULT_INVERTER_REPLACEMENT_YEAR)
        inverter_replacement_year = _ir_in if _ir_in > 0 else None
        return SetupFormValues(
            cons_file=None,
            pv_file=None,
            pv_capex=float(PV_COST_PER_KWP),
            batt_capex=float(BATT_COST_PER_KWH),
            tariff_profiles=profiles,
            pso_levy=float(DEFAULT_PSO_LEVY_EUR_PER_YEAR),
            opex_pct=float(DEFAULT_OPEX_PCT),
            discount_rate=float(DISCOUNT_RATE),
            electricity_inflation_rate=float(ELECTRICITY_INFLATION_RATE),
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_pct=float(DEFAULT_BATTERY_REPLACEMENT_COST_PCT),
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_pct_of_pv_capex=float(DEFAULT_INVERTER_REPLACEMENT_COST_PCT),
            speed_preset="Full (PV step 1, battery step 1)",
            opt_pv_step=1,
            opt_batt_step=1,
            pv_min=5,
            pv_max=60,
            batt_min=0,
            batt_max=40,
            rt_eff_pct=95.0,
            dod_pct=90.0,
            init_soc_pct=0.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            c_rate=0.5,
            charge_from_pv=True,
            charge_from_grid_at_night=False,
            discharge_schedule="Peak only",
            run_button=False,
            co2_factor=float(DEFAULT_CO2_FACTOR),
            lifetime_years=int(min(30, max(15, int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))))),
        )
    return SetupFormValues(
        cons_file=None,
        pv_file=None,
        pv_capex=float(PV_COST_PER_KWP),
        batt_capex=float(BATT_COST_PER_KWH),
        tariff_profiles=profiles,
        pso_levy=float(DEFAULT_PSO_LEVY_EUR_PER_YEAR),
        opex_pct=float(DEFAULT_OPEX_PCT),
        discount_rate=float(DISCOUNT_RATE),
        electricity_inflation_rate=float(ELECTRICITY_INFLATION_RATE),
        battery_replacement_year=None,
        battery_replacement_cost_pct=0.0,
        inverter_replacement_year=None,
        inverter_replacement_pct_of_pv_capex=float(DEFAULT_INVERTER_REPLACEMENT_COST_PCT),
        speed_preset="Full (PV step 1)",
        opt_pv_step=1,
        opt_batt_step=5,
        pv_min=5,
        pv_max=60,
        batt_min=0,
        batt_max=0,
        rt_eff_pct=95.0,
        dod_pct=90.0,
        init_soc_pct=0.0,
        min_soc_pct=10.0,
        max_soc_pct=90.0,
        c_rate=0.5,
        charge_from_pv=True,
        charge_from_grid_at_night=False,
        discharge_schedule="Peak only",
        run_button=False,
        co2_factor=float(DEFAULT_CO2_FACTOR),
        lifetime_years=int(min(30, max(15, int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))))),
    )


def _resolve_bundle_export_input_bytes(setup: SetupFormValues) -> Tuple[Optional[bytes], Optional[bytes], Optional[bytes]]:
    """Return consumption, PV, optional tariff CSV bytes matching ``last_input_hashes`` when possible."""
    ss = st.session_state
    h = ss.last_input_hashes or {}
    lc = ss.get("last_bundle_cons_bytes")
    lp = ss.get("last_bundle_pv_bytes")
    if isinstance(lc, (bytes, bytearray)) and isinstance(lp, (bytes, bytearray)):
        if hashlib.sha256(lc).hexdigest() == h.get("cons_sha") and hashlib.sha256(lp).hexdigest() == h.get("pv_sha"):
            _tc = ss.get("last_bundle_tariff_csv_bytes")
            return bytes(lc), bytes(lp), _tc if isinstance(_tc, (bytes, bytearray)) else None
    try:
        c, _ = resolve_consumption_csv_bytes(setup.cons_file)
        p, _ = resolve_pv_csv_bytes(setup.pv_file)
    except FileNotFoundError:
        return None, None, None
    if hashlib.sha256(c).hexdigest() != h.get("cons_sha") or hashlib.sha256(p).hexdigest() != h.get("pv_sha"):
        return None, None, None
    _tc = ss.get("last_bundle_tariff_csv_bytes")
    return c, p, _tc if isinstance(_tc, (bytes, bytearray)) else None


def _apply_imported_run_bundle_payload(payload: Dict[str, object]) -> None:
    """Hydrate session state from :func:`saved_run_bundle.load_bundle_from_zip` (no optimizer)."""
    ss = st.session_state
    lr = payload["last_run"]  # type: ignore[assignment]
    if not isinstance(lr, dict):
        raise ValueError("Invalid bundle payload")

    profiles = list(payload["tariff_profiles"])  # type: ignore[arg-type]
    bs_raw = lr.get("last_battery_settings")
    if not isinstance(bs_raw, dict):
        raise ValueError("Invalid last_battery_settings")
    bs = BatterySettings(**bs_raw)

    ss.prepared_df = payload["prepared_df"]  # type: ignore[assignment]
    ss.opt_dfs = payload["opt_dfs"]  # type: ignore[assignment]
    ss.prepared_meta = dict(lr.get("prepared_meta") or {})
    ss.last_input_hashes = dict(lr.get("last_input_hashes") or {})

    ss.last_pv_capex = float(lr["last_pv_capex"])
    ss.last_batt_capex = float(lr["last_batt_capex"])
    ss.last_opex_pct = float(lr["last_opex_pct"])
    ss.last_discount_rate = float(lr["last_discount_rate"])
    ss.last_electricity_inflation_rate = float(lr["last_electricity_inflation_rate"])
    ss.last_battery_replacement_year = lr.get("last_battery_replacement_year")
    ss.last_battery_replacement_cost_pct = float(lr["last_battery_replacement_cost_pct"])
    ss.last_inverter_replacement_year = lr.get("last_inverter_replacement_year")
    ss.last_inverter_replacement_cost_pct = float(lr["last_inverter_replacement_cost_pct"])
    ss.last_pso_levy = float(lr["last_pso_levy"])
    ss.last_co2_factor = float(lr["last_co2_factor"])
    ss.last_lifetime_years = int(lr["last_lifetime_years"])
    ss.last_export_rate = float(lr.get("last_export_rate", DEFAULT_EXPORT_RATE))
    ss.last_opt_cfg = dict(lr.get("last_opt_cfg") or {})
    ss.last_battery_settings = bs
    ss.battery_settings = bs
    ss.last_tariff_profiles = profiles
    ss.active_tariff_profiles = profiles
    ss.last_tariff_matrix_source_label = str(lr.get("last_tariff_matrix_source_label", "") or "")
    ss.active_discount_rate = float(ss.last_discount_rate)

    ss.tariff_matrix_profiles = profiles
    ss.tariff_matrix_source_label = ss.last_tariff_matrix_source_label
    ss.tariff_matrix_version = int(ss.get("tariff_matrix_version", 0)) + 1

    ss.setup_grid_co2_factor = float(ss.last_co2_factor)
    ss.setup_battery_charge_from_grid_night = bool(bs.charge_from_grid_at_night)

    # Always rebuild consolidated results from canonical imported optimizer tables.
    # Some historical bundles may contain a `full_results_df` snapshot with stale/legacy
    # metric values (notably CO2 fields). Recomputing from `opt_dfs` + frozen assumptions
    # guarantees that Full results, Recommended table, and snapshot cards share one source.
    ss.full_results_df = build_full_scenario_results_df(
        ss.opt_dfs,
        ss.prepared_df,
        profiles,
        pv_cost_per_kwp=ss.last_pv_capex,
        batt_cost_per_kwh=ss.last_batt_capex,
        electricity_inflation_rate=ss.last_electricity_inflation_rate,
        battery_replacement_year=ss.last_battery_replacement_year,
        battery_replacement_pct_of_batt_capex=ss.last_battery_replacement_cost_pct,
        inverter_replacement_year=ss.last_inverter_replacement_year,
        inverter_replacement_pct_of_pv_capex=ss.last_inverter_replacement_cost_pct,
        pso_levy_annual=float(ss.last_pso_levy),
        lifetime_years=int(ss.last_lifetime_years),
    )

    ss["last_bundle_cons_bytes"] = payload["cons_bytes"]
    ss["last_bundle_pv_bytes"] = payload["pv_bytes"]
    tc = payload.get("tariff_csv_bytes")
    ss["last_bundle_tariff_csv_bytes"] = tc if isinstance(tc, (bytes, bytearray)) else None

    ss["_setup_form_values_cache"] = replace(_setup_form_values_from_last_run_for_cache(), run_button=False)
    ss.show_setup_after_run = False
    # Defer to next rerun before sidebar widgets instantiate (avoids Streamlit widget-state mutation errors).
    ss["_pending_apply_recommended_constraints_defaults"] = True
    ss.pop("selected_explorer_row_key", None)
    ss.pop("selected_recommended_row_key", None)
    ss["_scroll_results_top"] = True


def _embedded_saved_runs_available() -> dict[str, Path]:
    """Embedded `.zip` saved-run demos stored in `assets/saved_runs/`."""
    out: dict[str, Path] = {}
    if EMBEDDED_RUN_NIGHT_CHARGING_OFF_ZIP.is_file():
        out["Battery night charging: OFF"] = EMBEDDED_RUN_NIGHT_CHARGING_OFF_ZIP
    if EMBEDDED_RUN_NIGHT_CHARGING_ON_ZIP.is_file():
        out["Battery night charging: ON"] = EMBEDDED_RUN_NIGHT_CHARGING_ON_ZIP
    return out


def _load_embedded_saved_run(zip_path: Path, *, label: str) -> None:
    """Load an embedded saved-run ZIP by applying the same payload hydrator as uploads."""
    if not zip_path.is_file():
        st.warning(f"Embedded demo run not found: {zip_path}")
        return
    _raw = zip_path.read_bytes()
    _manifest, _payload = load_bundle_from_zip(_raw)
    _apply_imported_run_bundle_payload(_payload)
    st.session_state["_embedded_saved_run_label"] = label
    st.rerun()


def render_embedded_saved_runs_picker(*, section_label: str = "Demo runs") -> None:
    """Roll-down menu to load embedded battery-scenario runs (no upload needed)."""
    if not embedded_saved_runs_active():
        return
    available = _embedded_saved_runs_available()
    if not available:
        return

    labels = list(available.keys())
    # Default view: ON (as requested).
    default_label = "Battery night charging: ON" if "Battery night charging: ON" in labels else labels[0]

    if st.session_state.get("_embedded_saved_run_choice") not in labels:
        st.session_state["_embedded_saved_run_choice"] = default_label

    st.markdown(f"### {section_label}")
    st.caption("Replaces the current session’s results (no optimizer rerun).")
    choice = st.selectbox(
        "Battery night charging",
        labels,
        index=labels.index(str(st.session_state["_embedded_saved_run_choice"])),
        key="embedded_saved_run_choice",
        label_visibility="visible",
    )
    # Auto-load on change, without requiring a separate confirmation click.
    desired = str(choice)
    already_loaded = st.session_state.get("_embedded_saved_run_choice_loaded")
    if already_loaded is None:
        # First render: auto-load only when the session has no results yet.
        # This avoids overriding a just-restored uploaded run (or a completed analysis run)
        # merely because the demo picker is visible in the sidebar.
        has_results = (
            st.session_state.get("prepared_df") is not None
            and st.session_state.get("opt_dfs") is not None
        )
        st.session_state["_embedded_saved_run_choice_loaded"] = desired
        if not has_results:
            _load_embedded_saved_run(available[desired], label=desired)
    elif already_loaded != desired:
        # Subsequent change: load the newly selected choice.
        st.session_state["_embedded_saved_run_choice_loaded"] = desired
        _load_embedded_saved_run(available[desired], label=desired)


def render_saved_run_import_controls(
    *, include_section_heading: bool = True, widget_key_suffix: str = "_bundle"
) -> None:
    """Upload + restore saved-run ZIP (no optimizer). Safe to call before first Run analysis.

    ``widget_key_suffix`` must differ when this block is rendered twice on the same page (e.g. preface + main expander)
    so Streamlit widget keys stay unique.
    """
    # Show success once per restore (main **Saved run** block); avoid consuming the flag before the second import UI.
    if widget_key_suffix == "_bundle" and st.session_state.pop("_saved_run_import_ok", False):
        st.success(
            "Saved run restored from **.zip** — this session’s results and last-run inputs now match the bundle "
            f"(**no optimizer** run). Schema **v{BUNDLE_SCHEMA_VERSION}**."
        )
    if include_section_heading:
        st.markdown("##### Import saved run")
    st.warning(
        "**Restore replaces** everything from your current run in this browser tab with the bundle "
        "(results, inputs snapshot, and frozen settings). **No optimizer** runs."
    )
    _up = st.file_uploader(
        "Upload saved run (.zip)", type=["zip"], key=f"saved_run_import_zip{widget_key_suffix}"
    )
    if _up is not None:
        try:
            _man_prev = read_manifest_from_zip(_up.getvalue())
            _ts = str(_man_prev.get("export_timestamp_utc", "—"))
            _av = str(_man_prev.get("app_version", "—"))
            _sch = int(_man_prev.get("schema_version", -1))
            _hfr = bool(_man_prev.get("has_full_results", False))
            st.info(
                f"**Bundle preview:** exported **{_ts}** · build **{_av}** · schema **v{_sch}** · "
                f"full results in bundle: **{'yes' if _hfr else 'no'}**. Use **Restore** below to apply."
            )
        except Exception as e:
            st.warning(f"Could not read this file as a saved-run bundle (manifest): {e}")
    if st.button("Restore saved run from upload", key=f"saved_run_import_apply{widget_key_suffix}"):
        if _up is None:
            st.warning("Select a **.zip** file in the field above, then click **Restore** again.")
        else:
            # Defer hydrate to the top of the next run so the main column can show the loading message first.
            # under the banner (same pattern as embedded demo loads).
            st.session_state["_pending_saved_run_zip_bytes"] = _up.getvalue()
            st.rerun()


def render_sidebar_postrun_filters() -> None:
    """Post-run only: ranking, scenario type, optional decision constraints (no model rerun)."""
    if st.session_state.pop("_pending_apply_recommended_constraints_defaults", False):
        _sidebar_apply_recommended_decision_constraints_defaults()
    st.sidebar.markdown("### Results filters")
    st.sidebar.caption(
        "View and rank **existing** results only — no optimizer rerun. "
        "Open **Decision constraints** to narrow rows with thresholds."
    )

    if st.session_state.view_goal not in RANK_GOAL_OPTIONS:
        st.session_state.view_goal = RANK_GOAL_OPTIONS[0]
    st.sidebar.markdown("##### Rank results by")
    _rank_help = RECOMMENDED_WINNER_PRESET_HELP
    st.sidebar.selectbox(
        "Choose",
        RANK_GOAL_OPTIONS,
        index=RANK_GOAL_OPTIONS.index(st.session_state.view_goal),
        key="view_goal",
        label_visibility="visible",
        help=_rank_help,
    )
    with st.sidebar.expander("How Rank results by works (full detail)", expanded=False):
        st.markdown(_rank_help)

    _scenario_opts = scenario_type_ui_options()
    if st.session_state.view_scenario_type not in _scenario_opts:
        st.session_state.view_scenario_type = "All scenarios"
    st.sidebar.markdown("##### Scenario type")
    st.sidebar.selectbox(
        "Scenario type",
        _scenario_opts,
        index=_scenario_opts.index(st.session_state.view_scenario_type),
        key="view_scenario_type",
        label_visibility="collapsed",
        help="Which scenario rows are ranked (same filter as the **Full results** table).",
    )

    if st.session_state.view_tariff_family not in TARIFF_FAMILY_FILTER_OPTIONS:
        st.session_state.view_tariff_family = TARIFF_FAMILY_FILTER_OPTIONS[0]
    st.sidebar.markdown("##### Tariff family")
    st.sidebar.selectbox(
        "Tariff family",
        TARIFF_FAMILY_FILTER_OPTIONS,
        index=TARIFF_FAMILY_FILTER_OPTIONS.index(st.session_state.view_tariff_family),
        key="view_tariff_family",
        label_visibility="collapsed",
        help=(
            "Restrict the **Recommended setups** / **Full results** table, ranking, KPI block, and **All tariffs — comparison** to tariffs "
            "in this family. **All tariff types** keeps every variant. **Standard** / **Weekend saver** / **Flat rate** "
            "include every named variant under that family from Model setup."
        ),
    )

    n_active = _sidebar_active_hard_filter_count()
    st.sidebar.markdown(
        f'<p style="margin:0.75rem 0 0.35rem 0;font-size:0.82rem;color:#475569;">'
        f"<b>{n_active}</b> decision constraint(s) active</p>",
        unsafe_allow_html=True,
    )

    rb1, rb2 = st.sidebar.columns(2)
    with rb1:
        if st.button(
            "Reset filters",
            key="sidebar_btn_reset_all_filters",
            width="stretch",
            help="Restore ranking defaults (goal, scenario type, tariff family, threshold) and reset **Decision constraints** to the Recommended-setups defaults.",
        ):
            _sidebar_reset_all_result_filters()
            st.rerun()
    with rb2:
        if st.button(
            "Clear constraints",
            key="sidebar_btn_clear_constraints",
            width="stretch",
            help="Turn off all decision constraint checkboxes (keeps ranking, tariff family, scenario type).",
        ):
            _sidebar_clear_decision_constraints()
            st.rerun()

    def _en(k: str) -> bool:
        return bool(st.session_state.get(k, False))

    with st.sidebar.container():
        st.caption("Enable a row, then set its threshold. Units are in the help tooltips.")

        r1a, r1b = st.columns([1.35, 1.0])
        with r1a:
            st.checkbox(
                "CAPEX max",
                key="hard_capex_max_en",
                help="Cap total scenario CAPEX. Enter the limit in **full euros** (e.g. 25000 for €25k), not in thousands.",
            )
        with r1b:
            st.number_input(
                "CAPEX max value",
                min_value=0.0,
                step=1000.0,
                format="%.0f",
                disabled=not _en("hard_capex_max_en"),
                label_visibility="collapsed",
                key="hard_capex_max_eur",
                help="Full euros (e.g. 21000 for €21k). Step 1000 is only for convenience — the value is not in €k.",
            )
        if _en("hard_capex_max_en"):
            _cm = float(st.session_state.get("hard_capex_max_eur") or 0.0)
            st.caption(
                "CAPEX max is in **full euros**. Example: a €21k budget → enter **21000** (not **21**)."
            )
            if 0.0 < _cm < float(DECISION_CONSTRAINT_CAPEX_WARN_BELOW_EUR):
                st.warning(
                    f"CAPEX max is **€{_cm:,.0f}**. Values below **€{DECISION_CONSTRAINT_CAPEX_WARN_BELOW_EUR:,.0f}** are often a "
                    "**thousands** mistake (e.g. typing **21** instead of **21000**). Confirm this matches your real cap."
                )

        r2a, r2b = st.columns([1.35, 1.0])
        with r2a:
            st.checkbox(
                "NPV min",
                key="hard_npv_min_en",
                help=f"Minimum NPV over {int(st.session_state.get('last_lifetime_years', DEFAULT_LIFETIME_YEARS))} years (€).",
            )
        with r2b:
            st.number_input(
                "NPV min value",
                min_value=-1_000_000_000.0,
                max_value=1_000_000_000.0,
                step=1000.0,
                disabled=not _en("hard_npv_min_en"),
                label_visibility="collapsed",
                key="hard_npv_min_eur",
                help="€",
            )

        r3a, r3b = st.columns([1.35, 1.0])
        with r3a:
            st.checkbox("Payback max", key="hard_payback_max_en", help="Maximum simple payback (years).")
        with r3b:
            st.number_input(
                "Payback max value",
                min_value=0.0,
                step=0.5,
                disabled=not _en("hard_payback_max_en"),
                label_visibility="collapsed",
                key="hard_payback_max_years",
                help="years",
            )

        r4a, r4b = st.columns([1.35, 1.0])
        with r4a:
            st.checkbox(
                "IRR min",
                key="hard_irr_min_en",
                help=f"Minimum IRR over {int(st.session_state.get('last_lifetime_years', DEFAULT_LIFETIME_YEARS))} years (%).",
            )
        with r4b:
            st.number_input(
                "IRR min value",
                min_value=-100.0,
                max_value=100.0,
                step=0.5,
                disabled=not _en("hard_irr_min_en"),
                label_visibility="collapsed",
                key="hard_irr_min_pct",
                help="%",
            )

        r5a, r5b = st.columns([1.35, 1.0])
        with r5a:
            st.checkbox("Self-sufficiency min", key="hard_ss_min_en", help="Minimum self-sufficiency ratio (%).")
        with r5b:
            st.number_input(
                "Self-sufficiency min value",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                disabled=not _en("hard_ss_min_en"),
                label_visibility="collapsed",
                key="hard_ss_min_pct",
                help="%",
            )

        r6a, r6b = st.columns([1.35, 1.0])
        with r6a:
            st.checkbox(
                "CO2 reduction min (%)",
                key="hard_co2_min_en",
                help="Minimum annual CO2 reduction vs grid-only (percentage).",
            )
        with r6b:
            st.number_input(
                "CO2 reduction min (%)",
                min_value=0.0,
                step=1.0,
                disabled=not _en("hard_co2_min_en"),
                label_visibility="collapsed",
                key="hard_co2_min_pct",
                help="%",
            )

        r7a, r7b = st.columns([1.35, 1.0])
        with r7a:
            st.checkbox(
                "Annual electricity cost max",
                key="hard_ann_cost_max_en",
                help="Maximum annual electricity cost (€, year 1; same column as the results table).",
            )
        with r7b:
            st.number_input(
                "Annual electricity cost max value",
                min_value=0.0,
                step=1000.0,
                format="%.0f",
                disabled=not _en("hard_ann_cost_max_en"),
                label_visibility="collapsed",
                key="hard_ann_cost_max_eur",
                help="Full euros per year (year-1 **annual electricity cost (€)** in the table).",
            )

        r8a, r8b = st.columns([1.35, 1.0])
        with r8a:
            st.checkbox(
                "Annual electricity bill reduction min (%)",
                key="hard_ann_cost_saving_min_en",
                help="Minimum annual electricity bill reduction vs the Grid-only scenario (%).",
            )
        with r8b:
            st.number_input(
                "Annual electricity bill reduction min (%) value",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                disabled=not _en("hard_ann_cost_saving_min_en"),
                label_visibility="collapsed",
                key="hard_ann_cost_saving_min_pct",
                help="%",
            )

        r9a, r9b = st.columns([1.35, 1.0])
        with r9a:
            st.checkbox(
                "Self-consumption min",
                key="hard_self_cons_min_en",
                help="Minimum self-consumption ratio (% of PV generation).",
            )
        with r9b:
            st.number_input(
                "Self-consumption min value",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                disabled=not _en("hard_self_cons_min_en"),
                label_visibility="collapsed",
                key="hard_self_cons_min_pct",
                help="%",
            )

        r10a, r10b = st.columns([1.35, 1.0])
        with r10a:
            st.checkbox(
                "Export ratio max",
                key="hard_export_max_en",
                help="Maximum share of annual PV generation exported to the grid (%). Grid-only / no-PV rows are not filtered.",
            )
        with r10b:
            st.number_input(
                "Export ratio max value",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                disabled=not _en("hard_export_max_en"),
                label_visibility="collapsed",
                key="hard_export_max_pct",
                help="% of PV generation",
            )


_has_completed_run = (
    st.session_state.get("prepared_df") is not None and st.session_state.get("opt_dfs") is not None
)

if HEADER_BANNER_IMAGE.is_file():
    _render_header_banner_strip(HEADER_BANNER_IMAGE, max_height_px=112)

st.title(
    "REC Feasibility Analyzer (PV + Battery + Tariffs)"
    if ENABLE_BATTERY_UI
    else "REC Feasibility Analyzer (PV + Tariffs)"
)
if DEMO_MODE:
    st.caption(
        "**Demo** — **Run your own analysis** (Model setup, Run analysis, saved-run upload) is **Disabled in demo**. "
        "Use **Demo runs** in the sidebar (or wait for the initial load)."
    )

_LOADING_RESULTS_MSG = "The results are loading. Please wait."
_pending_saved_zip = st.session_state.get("_pending_saved_run_zip_bytes")
_embedded_will_autoload = embedded_saved_runs_active() and (not _has_completed_run) and any(
    p.is_file() for p in (EMBEDDED_RUN_NIGHT_CHARGING_OFF_ZIP, EMBEDDED_RUN_NIGHT_CHARGING_ON_ZIP)
)

# Saved-run restore: defer heavy ZIP work to here (after title) so this message can render first; then ``st.rerun()``.
if _pending_saved_zip is not None:
    st.info(_LOADING_RESULTS_MSG)
    try:
        with st.spinner("Restoring saved run…"):
            _manifest_sr, _payload_sr = load_bundle_from_zip(bytes(_pending_saved_zip))
            _apply_imported_run_bundle_payload(_payload_sr)
        st.session_state.pop("_pending_saved_run_zip_bytes", None)
        st.session_state["_saved_run_import_ok"] = True
        st.rerun()
    except Exception as e:
        st.session_state.pop("_pending_saved_run_zip_bytes", None)
        st.error(_format_saved_run_import_user_message(e))
        with st.expander("Technical details (for debugging)"):
            st.code(str(e), language=None)
elif _embedded_will_autoload:
    # Embedded demo runs auto-load from the sidebar on first open; that work can take a noticeable moment
    # (ZIP read + hydrate + sometimes rebuilding the consolidated table). Show feedback before ``st.rerun()``.
    st.info(_LOADING_RESULTS_MSG)

with st.sidebar:
    # If embedded demo runs exist, let the user switch between them without uploading.
    # These are stored as `.zip` bundles under `assets/saved_runs/`.
    if embedded_saved_runs_active() and any(
        p.is_file() for p in (EMBEDDED_RUN_NIGHT_CHARGING_OFF_ZIP, EMBEDDED_RUN_NIGHT_CHARGING_ON_ZIP)
    ):
        render_embedded_saved_runs_picker(section_label="Demo runs (battery night charging)")
    if _has_completed_run:
        render_sidebar_postrun_filters()
    else:
        st.caption(
            "Results filters appear after a successful **Run analysis**. To load a bundle without running, use "
            "**Run your own analysis** → **Saved run — export / import**"
            + (
                " (**Disabled in demo** — use **Demo runs** above.)"
                if DEMO_MODE
                else "."
            )
        )

_show_setup_panel = (not _has_completed_run) or bool(st.session_state.show_setup_after_run)

if DEMO_MODE:
    (
        tab_recommended,
        tab_explorer,
        tab_consumption,
        tab_production,
        tab_research,
        tab_explainer,
    ) = st.tabs(
        [
            "Recommended setups",
            "Full results",
            "Consumption patterns",
            "Production patterns",
            "Research results",
            "Settings & App guide",
        ]
    )
else:
    (
        tab_recommended,
        tab_explorer,
        tab_consumption,
        tab_production,
        tab_run_own,
        tab_research,
        tab_explainer,
    ) = st.tabs(
        [
            "Recommended setups",
            "Full results",
            "Consumption patterns",
            "Production patterns",
            "Run your own analysis",
            "Research results",
            "Settings & App guide",
        ]
    )

if not DEMO_MODE:
    with tab_run_own:
        st.subheader("Run your own analysis")
        st.caption(
            "Configure **Model setup**, click **Run analysis**, and use **Saved run** to export or restore a session (.zip)."
        )
        if not _has_completed_run:
            render_compact_preface_before_first_run()
        elif not _show_setup_panel:
            st.caption("Open **Edit assumptions** to change inputs, then **Run analysis** again.")
            if st.button("Edit assumptions and rerun", key="show_setup_panel_btn"):
                st.session_state.show_setup_after_run = True
                st.rerun()
        if _show_setup_panel:
            st.markdown("### Model setup")
            setup = render_setup_form()
            st.session_state["_setup_form_values_cache"] = replace(setup, run_button=False)
        else:
            _cached = st.session_state.get("_setup_form_values_cache")
            if _cached is None:
                st.session_state.show_setup_after_run = True
                st.markdown("### Model setup")
                setup = render_setup_form()
                st.session_state["_setup_form_values_cache"] = replace(setup, run_button=False)
            else:
                setup = replace(_cached, run_button=False)
        st.divider()
        render_saved_run_bundle_expander(setup)

if DEMO_MODE:
    _demo_cache = st.session_state.get("_setup_form_values_cache")
    if _demo_cache is not None:
        setup = replace(_demo_cache, run_button=False)
    elif _has_completed_run:
        setup = _setup_form_values_from_last_run_for_cache()
        st.session_state["_setup_form_values_cache"] = replace(setup, run_button=False)
    else:
        setup = _setup_form_values_demo_preflight_placeholder()


def _include_flags_for_scenario_type(scenario_type_ui: str) -> tuple[bool, bool]:
    """Map Results Controls scenario type to optimizer include_pv, include_battery."""
    if scenario_type_ui == "All scenarios":
        return True, ENABLE_BATTERY_UI
    if scenario_type_ui == "Grid only":
        return False, False
    if scenario_type_ui == "PV + Grid":
        return True, False
    if scenario_type_ui == "PV + Battery + Grid":
        return True, True
    if scenario_type_ui == "Battery + Grid":
        return False, True
    return True, ENABLE_BATTERY_UI


def _filter_by_scenario_type(res_df: pd.DataFrame, scenario_type_ui: str) -> pd.DataFrame:
    """Keep rows allowed by Scenario type; drop degenerate non–Grid-only 0/0 clones."""
    if res_df is None or len(res_df) == 0:
        return pd.DataFrame()
    out = res_df.copy()
    allowed = _scenario_allowed_for_filter(scenario_type_ui)
    out = out[out["Scenario"].isin(allowed)].copy()
    deg_mask = (
        (out["Scenario"] != "Grid only")
        & (pd.to_numeric(out["PV (kWp)"], errors="coerce").fillna(0) <= 0)
        & (pd.to_numeric(out["Battery (kWh)"], errors="coerce").fillna(0) <= 0)
    )
    out = out[~deg_mask].copy()
    return out


def _tariff_family_ui_to_kind(family_ui: str) -> str | None:
    """Map sidebar label to profile ``kind``; ``None`` = no filter."""
    return {
        "All tariff types": None,
        "Standard": "standard",
        "Weekend saver": "weekend",
        "Flat rate": "flat",
    }.get(family_ui)


def _tariff_display_name_to_kind(profiles: List[Dict]) -> Dict[str, str]:
    """Tariff column label → ``kind`` (standard | weekend | flat)."""
    out: Dict[str, str] = {}
    for p in profiles or []:
        name = str(p.get("name", "") or p.get("col", "") or "")
        if not name:
            continue
        raw = p.get("kind", p.get("family", "standard"))
        out[name] = str(raw).strip().lower()
    return out


def _filter_tariff_profiles_by_family_ui(profiles: List[Dict], family_ui: str) -> List[Dict]:
    """Subset tariff profiles for the **Tariff family** sidebar filter (same logic as the consolidated table)."""
    want_kind = _tariff_family_ui_to_kind(family_ui)
    if want_kind is None:
        return list(profiles or [])
    tmap = _tariff_display_name_to_kind(profiles)
    out: List[Dict] = []
    for p in profiles or []:
        name = str(p.get("name", "") or p.get("col", "") or "")
        if tmap.get(name, "") == want_kind:
            out.append(p)
    return out


def _filter_by_tariff_family(res_df: pd.DataFrame, family_ui: str, profiles: List[Dict]) -> pd.DataFrame:
    """Keep rows whose ``Tariff`` belongs to the selected family (all variants in that family remain)."""
    if res_df is None or len(res_df) == 0:
        return pd.DataFrame()
    want_kind = _tariff_family_ui_to_kind(family_ui)
    if want_kind is None:
        return res_df.copy()
    if "Tariff" not in res_df.columns:
        return res_df.copy()
    tmap = _tariff_display_name_to_kind(profiles)
    kinds = res_df["Tariff"].astype(str).map(lambda t: tmap.get(t, ""))
    return res_df[kinds == want_kind].copy()


def _apply_hard_filters_to_results_df(
    df: pd.DataFrame,
    *,
    capex_max_eur: float | None = None,
    payback_max_years: float | None = None,
    npv_min_eur: float | None = None,
    irr_min_pct: float | None = None,
    self_sufficiency_min_pct: float | None = None,
    annual_co2_savings_min_kg: float | None = None,
    annual_electricity_cost_saving_min_pct: float | None = None,
    annual_electricity_cost_max_eur: float | None = None,
    self_consumption_ratio_min_pct: float | None = None,
    export_ratio_max_pct: float | None = None,
    annual_co2_reduction_min_pct: float | None = None,
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

    # Prefer CO2 reduction (%) if provided; it is more directly aligned with decision intent
    # for renewable communities than raw kg savings.
    if annual_co2_reduction_min_pct is not None:
        if "CO2 reduction (%)" in out.columns:
            co2r = _col_numeric_finite("CO2 reduction (%)")
            out = out[co2r >= float(annual_co2_reduction_min_pct)].copy()
    elif annual_co2_savings_min_kg is not None:
        _co2f = _df_co2_avoided_column(out)
        if _co2f in out.columns:
            co2 = _col_numeric_finite(_co2f)
            out = out[co2 >= float(annual_co2_savings_min_kg)].copy()

    if annual_electricity_cost_saving_min_pct is not None:
        if COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT in out.columns:
            cs = _col_numeric_finite(COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT)
            out = out[cs >= float(annual_electricity_cost_saving_min_pct)].copy()

    if annual_electricity_cost_max_eur is not None:
        _bcf = _df_bill_column(out)
        if _bcf in out.columns:
            ac = _col_numeric_finite(_bcf)
            out = out[ac <= float(annual_electricity_cost_max_eur)].copy()

    if self_consumption_ratio_min_pct is not None:
        if "Self-consumption ratio (%)" in out.columns:
            scr = _col_numeric_finite("Self-consumption ratio (%)")
            out = out[scr >= float(self_consumption_ratio_min_pct)].copy()

    if export_ratio_max_pct is not None:
        _ercol = "Export ratio (% of PV gen)"
        if _ercol in out.columns:
            ex = _col_numeric_finite(_ercol)
            # No PV → NaN export ratio: keep row (constraint does not apply).
            out = out[ex.isna() | (ex <= float(export_ratio_max_pct) + 1e-9)].copy()

    return out


def _dataframe_cost_co2_balance_score(cand: pd.DataFrame) -> pd.Series:
    """Lower is better: equal weight on normalized annual electricity cost (€) vs inverted CO2 savings (year 1)."""
    cost = pd.to_numeric(cand[_df_bill_column(cand)], errors="coerce").astype(float)
    co2 = pd.to_numeric(cand[_df_co2_avoided_column(cand)], errors="coerce").fillna(0.0).astype(float)
    cmin, cmax = float(cost.min()), float(cost.max())
    gmin, gmax = float(co2.min()), float(co2.max())
    cr = (cost - cmin) / (cmax - cmin + 1e-9)
    gr = (co2 - gmin) / (gmax - gmin + 1e-9)
    return 0.5 * cr + 0.5 * (1.0 - gr)


def _rank_scenarios_from_consolidated_table(
    df: pd.DataFrame,
    goal: str,
) -> list[tuple[str, pd.Series]]:
    """Rank rows from ``build_full_scenario_results_df`` (same universe as the All scenario grid).

    Ordering matches the All-scenarios grid: uses :func:`_sort_consolidated_scenarios_for_goal` only.
    """
    if df is None or len(df) == 0:
        return []
    cand = _sort_consolidated_scenarios_for_goal(df, goal)
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
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Dispatch + KPI row (evaluate_for_tariff-shaped) and hourly dataframe for one scenario size."""
    ly = int(lifetime_years)
    cn = col_npv(ly)
    ci = col_irr(ly)
    cg = col_gross_savings(ly)
    cnb = col_net_benefit(ly)
    d_base = run_scenario_grid_only(df, tcol)
    baseline_co2 = float(d_base["grid_import"].to_numpy(dtype=float).sum()) * _grid_co2_factor()
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
            COL_ANNUAL_ELECTRICITY_COST_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual,
            COL_ANNUAL_ELECTRICITY_BILL_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual,
            "CAPEX (€)": 0.0,
            "Annual savings (€)": 0.0,
            cg: 0.0,
            cnb: 0.0,
            "Payback period (years)": float("inf"),
            cn: 0.0,
            ci: float("nan"),
            "CO2 savings (kg)": 0.0,
            "CO2 reduction (%)": 0.0,
        }
        return pd.Series(row), d

    if scenario_name == "PV + Grid":
        d = run_scenario_pv_grid(df, int(pv_kwp), tcol)
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        capex = int(pv_kwp) * pv_cost_eff
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k[COL_NET_IMPORT_EXPORT_COST_EUR] - opex
        batt_repl = 0.0
        inv_repl = (
            capex * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= ly)
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
            lifetime_years=ly,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            n_years=ly,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl,
        )
        gross = _gross_savings_lifetime(annual_savings, electricity_inflation_rate, ly)
        row = {
            "Scenario": "PV + Grid",
            "PV (kWp)": int(pv_kwp),
            "Battery (kWh)": 0,
            **k,
            COL_ANNUAL_ELECTRICITY_COST_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
            COL_ANNUAL_ELECTRICITY_BILL_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
            "CAPEX (€)": capex,
            "Annual savings (€)": annual_savings,
            cg: gross,
            cnb: gross - capex - inv_repl,
            "Payback period (years)": payback,
            cn: npv,
            ci: 100.0 * irr if np.isfinite(irr) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
        return pd.Series(row), d

    if scenario_name == "PV + Battery + Grid":
        d = run_scenario_pv_battery_grid(df, int(pv_kwp), int(batt_kwh), tcol, battery_settings)
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        capex = int(pv_kwp) * pv_cost_eff + int(batt_kwh) * batt_cost_eff
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k[COL_NET_IMPORT_EXPORT_COST_EUR] - opex
        batt_repl = (
            (int(batt_kwh) * batt_cost_eff) * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= ly)
            else 0.0
        )
        inv_repl = (
            (int(pv_kwp) * pv_cost_eff) * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
            if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= ly)
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
            lifetime_years=ly,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            n_years=ly,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl,
        )
        gross = _gross_savings_lifetime(annual_savings, electricity_inflation_rate, ly)
        row = {
            "Scenario": "PV + Battery + Grid",
            "PV (kWp)": int(pv_kwp),
            "Battery (kWh)": int(batt_kwh),
            **k,
            COL_ANNUAL_ELECTRICITY_COST_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
            COL_ANNUAL_ELECTRICITY_BILL_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
            "CAPEX (€)": capex,
            "Annual savings (€)": annual_savings,
            cg: gross,
            cnb: gross - capex - batt_repl - inv_repl,
            "Payback period (years)": payback,
            cn: npv,
            ci: 100.0 * irr if np.isfinite(irr) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
        return pd.Series(row), d

    if scenario_name == "Battery + Grid":
        d = run_scenario_battery_grid(df, int(batt_kwh), tcol, battery_settings)
        k = compute_kpis_for_scenario(d, tcol, export_rate)
        capex = int(batt_kwh) * batt_cost_eff
        opex = capex * (opex_pct / 100.0)
        annual_savings = baseline_cost - k[COL_NET_IMPORT_EXPORT_COST_EUR] - opex
        batt_repl = (
            capex * (float(battery_replacement_pct_of_batt_capex) / 100.0)
            if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= ly)
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
            lifetime_years=ly,
        )
        irr = compute_irr(
            capex,
            annual_savings,
            n_years=ly,
            electricity_inflation_rate=electricity_inflation_rate,
            battery_replacement_year=battery_replacement_year,
            battery_replacement_cost_eur=batt_repl,
            inverter_replacement_year=inverter_replacement_year,
            inverter_replacement_cost_eur=inv_repl,
        )
        gross = _gross_savings_lifetime(annual_savings, electricity_inflation_rate, ly)
        row = {
            "Scenario": "Battery + Grid",
            "PV (kWp)": 0,
            "Battery (kWh)": int(batt_kwh),
            **k,
            COL_ANNUAL_ELECTRICITY_COST_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
            COL_ANNUAL_ELECTRICITY_BILL_EUR: k[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex,
            "CAPEX (€)": capex,
            "Annual savings (€)": annual_savings,
            cg: gross,
            cnb: gross - capex - batt_repl,
            "Payback period (years)": payback,
            cn: npv,
            ci: 100.0 * irr if np.isfinite(irr) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
        return pd.Series(row), d

    raise ValueError(f"Unknown scenario_name for metrics/hourly: {scenario_name!r}")


def _aggrid_goal_key_fragment(goal: str) -> str:
    """Stable short id for Streamlit widget keys when goal-based grid order changes."""
    return hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]


def _sort_consolidated_for_winner_preset(cand: pd.DataFrame, preset_id: str) -> pd.DataFrame:
    """Lexicographic order on consolidated KPI columns (matches :func:`_sort_feasible_for_recommended_winner_preset`)."""
    if len(cand) == 0:
        return cand
    work = cand.copy()
    bill_c = _df_bill_column(work)
    co2_c = _df_co2_avoided_column(work)
    work["_b"] = pd.to_numeric(work[bill_c], errors="coerce")
    work["_co2"] = pd.to_numeric(work[co2_c], errors="coerce")
    work["_npv"] = pd.to_numeric(work["NPV (€)"], errors="coerce")
    work["_sav"] = pd.to_numeric(work["Annual savings (€)"], errors="coerce")
    work["_pb"] = pd.to_numeric(work["Payback (yrs)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work["_scr"] = pd.to_numeric(work["Self-consumption ratio (%)"], errors="coerce").fillna(0.0)
    work["_ssr"] = pd.to_numeric(work["Self-sufficiency (%)"], errors="coerce")
    work["_exr"] = pd.to_numeric(work["Export ratio (% of PV gen)"], errors="coerce").fillna(0.0)
    work["_capex"] = pd.to_numeric(work["CAPEX (€)"], errors="coerce")
    pid = preset_id if preset_id in RECOMMENDED_WINNER_PRESET_LABEL_BY_ID else RECOMMENDED_WINNER_PRESET_DEFAULT
    if pid == "balanced":
        by = ["_npv", "_co2", "_scr", "_sav", "_capex"]
        asc = [False, False, False, False, True]
    elif pid == "financial":
        by = ["_npv", "_sav", "_pb", "_b", "_capex"]
        asc = [False, False, True, True, True]
    elif pid == "lowest_bill":
        by = ["_b", "_sav", "_npv", "_co2", "_capex"]
        asc = [True, False, False, False, True]
    elif pid == "fast_payback":
        by = ["_pb", "_npv", "_sav", "_b", "_capex"]
        asc = [True, False, False, True, True]
    elif pid == "highest_co2":
        by = ["_co2", "_ssr", "_npv", "_scr", "_capex"]
        asc = [False, False, False, False, True]
    elif pid == "highest_scr":
        by = ["_scr", "_exr", "_npv", "_sav", "_capex"]
        asc = [False, True, False, False, True]
    else:
        by = ["_npv", "_co2", "_scr", "_sav", "_capex"]
        asc = [False, False, False, False, True]
    out = work.sort_values(by=by, ascending=asc, na_position="last", kind="mergesort")
    return out.drop(columns=[c for c in ("_b", "_co2", "_npv", "_sav", "_pb", "_scr", "_ssr", "_exr", "_capex") if c in out.columns])


def _sort_consolidated_scenarios_for_goal(
    df: pd.DataFrame,
    goal: str,
) -> pd.DataFrame:
    """Single source of truth for goal-based row order on the consolidated all-scenarios table (AgGrid + ranked pick)."""
    if df is None or len(df) == 0:
        return df
    cand = df.copy()
    if goal in RECOMMENDED_WINNER_PRESET_ID_BY_LABEL:
        return _sort_consolidated_for_winner_preset(cand, RECOMMENDED_WINNER_PRESET_ID_BY_LABEL[goal])
    if goal == "Lowest annual electricity cost":
        return cand.sort_values(_df_bill_column(cand), ascending=True, kind="mergesort")
    if goal == "Highest annual savings":
        return cand.sort_values("Annual savings (€)", ascending=False, kind="mergesort")
    if goal == "Best payback":
        pb = pd.to_numeric(cand["Payback (yrs)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return cand.assign(_pb=pb).sort_values("_pb", ascending=True, na_position="last", kind="mergesort").drop(
            columns=["_pb"]
        )
    if goal == "Best self-sufficiency / lowest grid import":
        return cand.sort_values("Self-sufficiency (%)", ascending=False, kind="mergesort")
    if goal == "Highest annual CO2 savings":
        return cand.sort_values(_df_co2_avoided_column(cand), ascending=False, kind="mergesort")
    if goal == "Best cost–CO2 trade-off":
        return cand.assign(_bal=_dataframe_cost_co2_balance_score(cand)).sort_values(
            "_bal", ascending=True, kind="mergesort"
        ).drop(columns=["_bal"])
    if goal == "Best NPV":
        return cand.sort_values("NPV (€)", ascending=False, kind="mergesort")
    if goal == "Best IRR":
        irr = pd.to_numeric(cand["IRR (%)"], errors="coerce").fillna(-1.0)
        return cand.assign(_irr=irr).sort_values("_irr", ascending=False, kind="mergesort").drop(columns=["_irr"])
    return cand.sort_values("Annual savings (€)", ascending=False, kind="mergesort")


def evaluate_for_tariff(
    df: pd.DataFrame,
    opt_dfs: Dict[str, pd.DataFrame],
    tcol: str,
    tname: str,
    goal: str,
    include_pv: bool,
    include_battery: bool,
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
    *,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    ly = int(lifetime_years)
    cn = col_npv(ly)
    ci = col_irr(ly)
    cg = col_gross_savings(ly)
    cnb = col_net_benefit(ly)
    opt_df = opt_dfs[tcol]

    pv_cost_eff = float(PV_COST_PER_KWP if pv_cost_per_kwp is None else pv_cost_per_kwp)
    batt_cost_eff = float(BATT_COST_PER_KWH if batt_cost_per_kwh is None else batt_cost_per_kwh)

    def has_config(config_name: str) -> bool:
        return len(opt_df[opt_df["config"] == config_name]) > 0

    pv_only_kwp = 0
    pv_batt_kwp = 0
    batt_kwp = 0
    batt_only_kwh = 0

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
    baseline_co2 = baseline_grid_import * _grid_co2_factor()
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
            COL_ANNUAL_ELECTRICITY_COST_EUR: k1[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual,
            "CAPEX (€)": 0.0,
            "Annual savings (€)": 0.0,
            cg: 0.0,
            cnb: 0.0,
            "Payback period (years)": float("inf"),
            cn: 0.0,
            ci: float("nan"),
            "CO2 savings (kg)": 0.0,
            "CO2 reduction (%)": 0.0,
        }
    )

    # PV + Grid
    d2 = run_scenario_pv_grid(df, pv_only_kwp, tcol)
    k2 = compute_kpis_for_scenario(d2, tcol, export_rate)
    capex2 = pv_only_kwp * pv_cost_eff
    opex2 = capex2 * (opex_pct / 100.0)
    annual_savings2 = baseline_cost - k2[COL_NET_IMPORT_EXPORT_COST_EUR] - opex2
    batt_repl2 = 0.0
    inv_repl2 = (
        capex2 * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= ly)
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
        lifetime_years=ly,
    )
    irr2 = compute_irr(
        capex2,
        annual_savings2,
        n_years=ly,
        electricity_inflation_rate=electricity_inflation_rate,
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_eur=batt_repl2,
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_eur=inv_repl2,
    )
    gross2 = _gross_savings_lifetime(annual_savings2, electricity_inflation_rate, ly)
    scenarios.append(
        {
            "Scenario": "PV + Grid",
            "PV (kWp)": pv_only_kwp,
            "Battery (kWh)": 0,
            **k2,
            COL_ANNUAL_ELECTRICITY_COST_EUR: k2[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex2,
            "CAPEX (€)": capex2,
            "Annual savings (€)": annual_savings2,
            cg: gross2,
            cnb: gross2 - capex2 - inv_repl2,
            "Payback period (years)": payback2,
            cn: npv2,
            ci: 100.0 * irr2 if np.isfinite(irr2) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k2["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k2["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
    )

    # PV + Battery + Grid
    d3 = run_scenario_pv_battery_grid(df, pv_batt_kwp, batt_kwp, tcol, battery_settings)
    k3 = compute_kpis_for_scenario(d3, tcol, export_rate)
    capex3 = pv_batt_kwp * pv_cost_eff + batt_kwp * batt_cost_eff
    opex3 = capex3 * (opex_pct / 100.0)
    annual_savings3 = baseline_cost - k3[COL_NET_IMPORT_EXPORT_COST_EUR] - opex3
    batt_repl3 = (
        (batt_kwp * batt_cost_eff) * (float(battery_replacement_pct_of_batt_capex) / 100.0)
        if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= ly)
        else 0.0
    )
    inv_repl3 = (
        (pv_batt_kwp * pv_cost_eff) * (float(inverter_replacement_pct_of_pv_capex) / 100.0)
        if (inverter_replacement_year is not None and 1 <= int(inverter_replacement_year) <= ly)
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
        lifetime_years=ly,
    )
    irr3 = compute_irr(
        capex3,
        annual_savings3,
        n_years=ly,
        electricity_inflation_rate=electricity_inflation_rate,
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_eur=batt_repl3,
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_eur=inv_repl3,
    )
    gross3 = _gross_savings_lifetime(annual_savings3, electricity_inflation_rate, ly)
    scenarios.append(
        {
            "Scenario": "PV + Battery + Grid",
            "PV (kWp)": pv_batt_kwp,
            "Battery (kWh)": batt_kwp,
            **k3,
            COL_ANNUAL_ELECTRICITY_COST_EUR: k3[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex3,
            "CAPEX (€)": capex3,
            "Annual savings (€)": annual_savings3,
            cg: gross3,
            cnb: gross3 - capex3 - batt_repl3 - inv_repl3,
            "Payback period (years)": payback3,
            cn: npv3,
            ci: 100.0 * irr3 if np.isfinite(irr3) else float("nan"),
            "CO2 savings (kg)": max(0.0, baseline_co2 - k3["CO2 (kg)"]),
            "CO2 reduction (%)": (100.0 * max(0.0, baseline_co2 - k3["CO2 (kg)"]) / baseline_co2) if baseline_co2 > 0 else 0.0,
        }
    )

    # Battery + Grid (no PV)
    d4 = run_scenario_battery_grid(df, batt_only_kwh, tcol, battery_settings)
    k4 = compute_kpis_for_scenario(d4, tcol, export_rate)
    capex4 = batt_only_kwh * batt_cost_eff
    opex4 = capex4 * (opex_pct / 100.0)
    annual_savings4 = baseline_cost - k4[COL_NET_IMPORT_EXPORT_COST_EUR] - opex4
    batt_repl4 = (
        capex4 * (float(battery_replacement_pct_of_batt_capex) / 100.0)
        if (battery_replacement_year is not None and 1 <= int(battery_replacement_year) <= ly)
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
        lifetime_years=ly,
    )
    irr4 = compute_irr(
        capex4,
        annual_savings4,
        n_years=ly,
        electricity_inflation_rate=electricity_inflation_rate,
        battery_replacement_year=battery_replacement_year,
        battery_replacement_cost_eur=batt_repl4,
        inverter_replacement_year=inverter_replacement_year,
        inverter_replacement_cost_eur=inv_repl4,
    )
    gross4 = _gross_savings_lifetime(annual_savings4, electricity_inflation_rate, ly)
    scenarios.append(
        {
            "Scenario": "Battery + Grid",
            "PV (kWp)": 0,
            "Battery (kWh)": batt_only_kwh,
            **k4,
            COL_ANNUAL_ELECTRICITY_COST_EUR: k4[COL_NET_IMPORT_EXPORT_COST_EUR] + standing_charge + pso_levy_annual + opex4,
            "CAPEX (€)": capex4,
            "Annual savings (€)": annual_savings4,
            cg: gross4,
            cnb: gross4 - capex4 - batt_repl4,
            "Payback period (years)": payback4,
            cn: npv4,
            ci: 100.0 * irr4 if np.isfinite(irr4) else float("nan"),
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


if setup.run_button:
    if _perf_profiling_enabled():
        st.session_state["_perf_log"] = []
        st.session_state.pop("_perf_optimizer_by_tariff", None)
    st.session_state.stop_run_requested = False
    if not (setup.tariff_profiles or []):
        st.error("Select at least one tariff row (**Include** checkbox) in **Data & tariffs** before running.")
        st.stop()
    try:
        cons_bytes, cons_src = resolve_consumption_csv_bytes(setup.cons_file)
        pv_bytes, pv_src = resolve_pv_csv_bytes(setup.pv_file)
    except FileNotFoundError as e:
        st.error(str(e))
    else:
        st.session_state.prepared_meta = {
            "cons_sha": hashlib.sha256(cons_bytes).hexdigest(),
            "pv_sha": hashlib.sha256(pv_bytes).hexdigest(),
            "cons_source": cons_src,
            "pv_source": pv_src,
        }

        with st.spinner("Loading and aligning data..."):
            profiles = list(setup.tariff_profiles or _default_tariff_profiles())
            st.session_state.active_tariff_profiles = profiles
            st.session_state.active_export_rate = float(DEFAULT_EXPORT_RATE)
            tck = _tariff_profiles_cache_key(profiles)
            try:
                _t_ld = time.perf_counter()
                st.session_state.prepared_df = load_and_prepare_data(cons_bytes, pv_bytes, profiles, tck)
                if _perf_profiling_enabled():
                    _dt_ld = time.perf_counter() - _t_ld
                    st.session_state["_perf_load_data_s"] = _dt_ld
                    _perf_record("load_and_prepare_data", _dt_ld)
            except Exception as e:
                st.error(f"Could not parse input files. {e}")
                st.stop()

        _tariff_note = f"Tariffs: **{len(st.session_state.active_tariff_profiles):,} profiles** (from Model setup)"
        _data_note = (
            "Using built-in sample data where uploads were not provided. "
            if (str(cons_src).startswith("default") or str(pv_src).startswith("default"))
            else ""
        )
        st.success(
            f"{_data_note}"
            f"Loaded — Consumption: **{cons_src}** · PV: **{pv_src}** · {_tariff_note}"
        )

        # Update global CAPEX assumptions (used inside optimizer and KPI calculations)
        PV_COST_PER_KWP = float(setup.pv_capex)
        BATT_COST_PER_KWH = float(setup.batt_capex)

        # Battery dispatch settings
        st.session_state.battery_settings = BatterySettings(
            eff_round_trip=float(setup.rt_eff_pct) / 100.0,
            dod=float(setup.dod_pct) / 100.0,
            init_soc=float(setup.init_soc_pct) / 100.0,
            min_soc=float(setup.min_soc_pct) / 100.0,
            max_soc=float(setup.max_soc_pct) / 100.0,
            c_rate=float(setup.c_rate),
            charge_from_pv=bool(setup.charge_from_pv),
            charge_from_grid_at_night=bool(setup.charge_from_grid_at_night),
            discharge_schedule=str(setup.discharge_schedule),
        )

        st.session_state.active_discount_rate = setup.discount_rate
        opt_cfg = OptimizerConfig(
            pv_min=setup.pv_min,
            pv_max=setup.pv_max,
            batt_min=setup.batt_min,
            batt_max=setup.batt_max,
            pv_step=setup.opt_pv_step,
            batt_step=setup.opt_batt_step,
        )
        tariffs_to_run = [str(p.get("col", "")) for p in (setup.tariff_profiles or []) if str(p.get("col", ""))]
        opt_dfs = {}
        tcol_to_profile = {str(p.get("col")): p for p in (setup.tariff_profiles or [])}
        total_evals_per_tariff = count_optimizer_evaluations(opt_cfg)
        total_work = max(1, total_evals_per_tariff * len(tariffs_to_run))
        completed_work_ref = {"value": 0}
        current_tariff_name = "Standard"
        # Ensure optimizer CO₂ calculations use the *current* setup's grid CO₂ factor.
        # `optimize()` uses `_grid_co2_factor()`, which reads `st.session_state.last_co2_factor`.
        st.session_state.last_co2_factor = float(setup.co2_factor)
        progress_overlay = st.empty()
        with st.status("Optimizer running…", expanded=True) as run_status:
            run_status.update(label="Starting…", state="running")
            progress_bar = st.progress(0, text="Starting optimizer... 0%")
            st.caption(
                "Tip: Use **Stop current run** in **Model setup** on the **Run your own analysis** tab "
                "(open **Edit assumptions and rerun** there first) to stop at the next checkpoint."
            )

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
            _perf_by_tcol: dict[str, float] = {}
            _t_opt0 = time.perf_counter()
            for tcol in tariffs_to_run:
                p = tcol_to_profile.get(tcol, {})
                current_tariff_name = str(p.get("name", "") or tcol)
                t_export_rate = float(p.get("export_rate", DEFAULT_EXPORT_RATE))
                _t_one = time.perf_counter()
                opt_dfs[tcol] = optimize(
                    st.session_state.prepared_df,
                    tcol,
                    opt_cfg,
                    st.session_state.battery_settings,
                    t_export_rate,
                    standing_charge=float(p.get("standing_charge", 0.0) or 0.0),
                    opex_pct=float(setup.opex_pct),
                    discount_rate=setup.discount_rate,
                    electricity_inflation_rate=setup.electricity_inflation_rate,
                    battery_replacement_year=setup.battery_replacement_year,
                    battery_replacement_pct_of_batt_capex=float(setup.battery_replacement_cost_pct),
                    inverter_replacement_year=setup.inverter_replacement_year,
                    inverter_replacement_pct_of_pv_capex=float(setup.inverter_replacement_cost_pct),
                    pso_levy_annual=float(setup.pso_levy),
                    lifetime_years=int(setup.lifetime_years),
                    progress_callback=_progress_tick,
                    stop_requested=_stop_requested,
                )
                if _perf_profiling_enabled():
                    _perf_by_tcol[str(tcol)] = time.perf_counter() - _t_one
                if _stop_requested():
                    aborted = True
                    break

            if _perf_profiling_enabled() and not aborted:
                st.session_state["_perf_optimizer_total_s"] = time.perf_counter() - _t_opt0
                st.session_state["_perf_optimizer_by_tariff"] = _perf_by_tcol
                _perf_record("optimizer_total", float(st.session_state["_perf_optimizer_total_s"]))

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
        st.session_state.last_pso_levy = float(setup.pso_levy)
        st.session_state.last_co2_factor = float(setup.co2_factor)
        st.session_state.last_opex_pct = float(setup.opex_pct)
        # Freeze discount rate used for NPV/IRR calculations
        st.session_state.last_discount_rate = setup.discount_rate
        # Freeze electricity inflation rate used for NPV/IRR and lifetime-horizon metrics
        st.session_state.last_electricity_inflation_rate = setup.electricity_inflation_rate
        st.session_state.last_battery_replacement_year = setup.battery_replacement_year
        st.session_state.last_battery_replacement_cost_pct = float(setup.battery_replacement_cost_pct)
        st.session_state.last_inverter_replacement_year = setup.inverter_replacement_year
        st.session_state.last_inverter_replacement_cost_pct = float(setup.inverter_replacement_cost_pct)
        st.session_state.last_lifetime_years = int(setup.lifetime_years)
        st.session_state.last_battery_settings = st.session_state.battery_settings
        st.session_state.last_tariff_profiles = list(setup.tariff_profiles or [])
        _tm_snap = str(st.session_state.get("tariff_matrix_source_label", "") or "").strip()
        if not _tm_snap:
            _, _tm_snap = _load_tariff_matrix_profiles_initial()
        st.session_state.last_tariff_matrix_source_label = str(_tm_snap)
        st.session_state.last_export_rate = float(DEFAULT_EXPORT_RATE)
        st.session_state.last_input_hashes = dict(st.session_state.prepared_meta)
        st.session_state.last_opt_cfg = {
            "pv_min": int(setup.pv_min),
            "pv_max": int(setup.pv_max),
            "batt_min": int(setup.batt_min),
            "batt_max": int(setup.batt_max),
            "pv_step": int(setup.opt_pv_step),
            "batt_step": int(setup.opt_batt_step),
            "speed_preset": str(setup.speed_preset),
        }
        # Freeze CAPEX assumptions used during post-run evaluation
        st.session_state.last_pv_capex = float(setup.pv_capex)
        st.session_state.last_batt_capex = float(setup.batt_capex)
        st.session_state["_setup_form_values_cache"] = replace(setup, run_button=False)
        st.session_state.show_setup_after_run = False
        st.session_state["last_bundle_cons_bytes"] = cons_bytes
        st.session_state["last_bundle_pv_bytes"] = pv_bytes
        _tcsv_widget = st.session_state.get("tariffs_csv")
        if _tcsv_widget is not None:
            try:
                st.session_state["last_bundle_tariff_csv_bytes"] = _tcsv_widget.getvalue()
            except Exception:
                st.session_state["last_bundle_tariff_csv_bytes"] = None
        else:
            st.session_state["last_bundle_tariff_csv_bytes"] = None
        # Defer to next rerun before sidebar widgets instantiate (avoids Streamlit widget-state mutation errors).
        st.session_state["_pending_apply_recommended_constraints_defaults"] = True
        # UI elements (setup form + sidebar filters) may have been rendered earlier in this same run.
        # Trigger a fresh rerender so the successful-run state takes effect immediately.
        st.session_state["_scroll_results_top"] = True
        st.rerun()


# If data is prepared, evaluate instantly when dropdowns change (Research tab uses bundled Excel only — no run required.)
_has_results = st.session_state.prepared_df is not None and st.session_state.opt_dfs is not None
if _has_results:
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
        eff_round_trip=float(setup.rt_eff_pct) / 100.0,
        dod=float(setup.dod_pct) / 100.0,
        init_soc=float(setup.init_soc_pct) / 100.0,
        min_soc=float(setup.min_soc_pct) / 100.0,
        max_soc=float(setup.max_soc_pct) / 100.0,
        c_rate=float(setup.c_rate),
        charge_from_pv=bool(setup.charge_from_pv),
        charge_from_grid_at_night=bool(setup.charge_from_grid_at_night),
        discharge_schedule=str(setup.discharge_schedule),
    )
    current_tariff_profiles = list(setup.tariff_profiles or [])
    # After a run, Model setup is often collapsed so file uploaders are not rendered. Cached
    # ``UploadedFile`` objects on ``SetupFormValues`` can then yield empty ``getvalue()`` while
    # ``st.session_state["cons"]`` / ``["pv"]`` (widget keys) still hold the bytes — use those first
    # so we do not false-trigger the stale-results banner.
    _cons_for_hash = st.session_state.get("cons")
    _cons_for_hash = setup.cons_file if _cons_for_hash is None else _cons_for_hash
    _pv_for_hash = st.session_state.get("pv")
    _pv_for_hash = setup.pv_file if _pv_for_hash is None else _pv_for_hash
    _cmp_cons, _cmp_pv = _resolve_csv_bytes_for_comparison(_cons_for_hash, _pv_for_hash)
    current_cons_sha = hashlib.sha256(_cmp_cons).hexdigest() if _cmp_cons else None
    current_pv_sha = hashlib.sha256(_cmp_pv).hexdigest() if _cmp_pv else None

    # Guard against Streamlit uploaders yielding empty buffers when the setup panel is collapsed.
    # If we cannot re-hash the current input, treat it as unchanged to avoid false stale warnings.
    last_cons_sha = st.session_state.last_input_hashes.get("cons_sha")
    last_pv_sha = st.session_state.last_input_hashes.get("pv_sha")
    if current_cons_sha is None:
        current_cons_sha = last_cons_sha
    if current_pv_sha is None:
        current_pv_sha = last_pv_sha

    # True when the setup form no longer matches the last completed optimizer run (stale KPIs/charts).
    assumptions_changed = (
        (st.session_state.last_input_hashes.get("cons_sha") != current_cons_sha)
        or (st.session_state.last_input_hashes.get("pv_sha") != current_pv_sha)
        or (abs(float(st.session_state.last_opex_pct) - float(setup.opex_pct)) > 1e-12)
        or (abs(float(st.session_state.last_discount_rate) - float(setup.discount_rate)) > 1e-12)
        or (abs(float(st.session_state.last_electricity_inflation_rate) - float(setup.electricity_inflation_rate)) > 1e-12)
        or (int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS)) != int(setup.lifetime_years))
        or (st.session_state.last_battery_replacement_year != setup.battery_replacement_year)
        or (abs(float(st.session_state.last_battery_replacement_cost_pct) - float(setup.battery_replacement_cost_pct)) > 1e-12)
        or (st.session_state.last_inverter_replacement_year != setup.inverter_replacement_year)
        or (abs(float(st.session_state.last_inverter_replacement_cost_pct) - float(setup.inverter_replacement_cost_pct)) > 1e-12)
        or (list(st.session_state.get("last_tariff_profiles") or []) != current_tariff_profiles)
        or (int(st.session_state.last_opt_cfg.get("pv_min", -1)) != int(setup.pv_min))
        or (int(st.session_state.last_opt_cfg.get("pv_max", -1)) != int(setup.pv_max))
        or (int(st.session_state.last_opt_cfg.get("batt_min", -1)) != int(setup.batt_min))
        or (int(st.session_state.last_opt_cfg.get("batt_max", -1)) != int(setup.batt_max))
        or (int(st.session_state.last_opt_cfg.get("pv_step", -1)) != int(setup.opt_pv_step))
        or (int(st.session_state.last_opt_cfg.get("batt_step", -1)) != int(setup.opt_batt_step))
        or (str(st.session_state.last_opt_cfg.get("speed_preset", "")) != str(setup.speed_preset))
        or (abs(float(st.session_state.last_pv_capex) - float(setup.pv_capex)) > 1e-12)
        or (abs(float(st.session_state.last_batt_capex) - float(setup.batt_capex)) > 1e-12)
        or (abs(float(st.session_state.last_pso_levy) - float(setup.pso_levy)) > 1e-12)
        or (abs(float(st.session_state.last_co2_factor) - float(setup.co2_factor)) > 1e-12)
    )
    mismatch_reasons: list[str] = []
    if st.session_state.last_input_hashes.get("cons_sha") != current_cons_sha:
        mismatch_reasons.append("consumption CSV")
    if st.session_state.last_input_hashes.get("pv_sha") != current_pv_sha:
        mismatch_reasons.append("PV CSV")
    if abs(float(st.session_state.last_opex_pct) - float(setup.opex_pct)) > 1e-12:
        mismatch_reasons.append("OPEX %")
    if abs(float(st.session_state.last_discount_rate) - float(setup.discount_rate)) > 1e-12:
        mismatch_reasons.append("discount rate")
    if abs(float(st.session_state.last_electricity_inflation_rate) - float(setup.electricity_inflation_rate)) > 1e-12:
        mismatch_reasons.append("electricity inflation")
    if int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS)) != int(setup.lifetime_years):
        mismatch_reasons.append("lifetime years")
    if st.session_state.last_battery_replacement_year != setup.battery_replacement_year:
        mismatch_reasons.append("battery replacement year")
    if abs(float(st.session_state.last_battery_replacement_cost_pct) - float(setup.battery_replacement_cost_pct)) > 1e-12:
        mismatch_reasons.append("battery replacement %")
    if st.session_state.last_inverter_replacement_year != setup.inverter_replacement_year:
        mismatch_reasons.append("inverter replacement year")
    if abs(float(st.session_state.last_inverter_replacement_cost_pct) - float(setup.inverter_replacement_cost_pct)) > 1e-12:
        mismatch_reasons.append("inverter replacement %")
    if list(st.session_state.get("last_tariff_profiles") or []) != current_tariff_profiles:
        mismatch_reasons.append("tariff profiles")
    if int(st.session_state.last_opt_cfg.get("pv_min", -1)) != int(setup.pv_min):
        mismatch_reasons.append("pv_min")
    if int(st.session_state.last_opt_cfg.get("pv_max", -1)) != int(setup.pv_max):
        mismatch_reasons.append("pv_max")
    if int(st.session_state.last_opt_cfg.get("batt_min", -1)) != int(setup.batt_min):
        mismatch_reasons.append("batt_min")
    if int(st.session_state.last_opt_cfg.get("batt_max", -1)) != int(setup.batt_max):
        mismatch_reasons.append("batt_max")
    if int(st.session_state.last_opt_cfg.get("pv_step", -1)) != int(setup.opt_pv_step):
        mismatch_reasons.append("pv_step")
    if int(st.session_state.last_opt_cfg.get("batt_step", -1)) != int(setup.opt_batt_step):
        mismatch_reasons.append("batt_step")
    if str(st.session_state.last_opt_cfg.get("speed_preset", "")) != str(setup.speed_preset):
        mismatch_reasons.append("speed preset")
    if abs(float(st.session_state.last_pv_capex) - float(setup.pv_capex)) > 1e-12:
        mismatch_reasons.append("PV capex")
    if abs(float(st.session_state.last_batt_capex) - float(setup.batt_capex)) > 1e-12:
        mismatch_reasons.append("battery capex")
    if abs(float(st.session_state.last_pso_levy) - float(setup.pso_levy)) > 1e-12:
        mismatch_reasons.append("PSO levy")
    if abs(float(st.session_state.last_co2_factor) - float(setup.co2_factor)) > 1e-12:
        mismatch_reasons.append("CO2 factor")
    # While "Edit assumptions and rerun" is open, the form is *supposed* to differ until the user runs again —
    # a banner here is redundant and reads like an error. Only warn when the setup panel is collapsed (results-only
    # view) but something still drifted vs the frozen last run (unexpected).
    _editing_assumptions = bool(st.session_state.get("show_setup_after_run", False))
    if assumptions_changed and not _editing_assumptions and not DEMO_MODE:
        _stale_tail = (
            "Open the **Run your own analysis** tab → **Edit assumptions and rerun**, then **Run analysis** to refresh."
            if not DEMO_MODE
            else "**Run your own analysis** is **Disabled in demo** — switch **Demo runs** in the sidebar to reload a consistent snapshot."
        )
        st.warning(
            "**Displayed results don’t match the current Model setup fields.** KPIs and charts still reflect the "
            f"**last completed Run analysis**. {_stale_tail}"
        )
        if mismatch_reasons:
            st.caption(f"Stale-check mismatch reason(s): {', '.join(mismatch_reasons[:6])}{'…' if len(mismatch_reasons) > 6 else ''}")

    _tariff_opts = [str(p.get("name", "")) for p in (st.session_state.get("last_tariff_profiles") or _default_tariff_profiles())]
    if st.session_state.view_filter_tariff not in _tariff_opts:
        st.session_state.view_filter_tariff = _tariff_opts[0]
    # Filtered ranking universe shared by Recommended setups and Full results (compute once per run).
    ly = int(st.session_state.get("last_lifetime_years", DEFAULT_LIFETIME_YEARS))
    capex_max = float(st.session_state.hard_capex_max_eur) if st.session_state.get("hard_capex_max_en") else None
    payback_max = float(st.session_state.hard_payback_max_years) if st.session_state.get("hard_payback_max_en") else None
    npv_min = float(st.session_state.hard_npv_min_eur) if st.session_state.get("hard_npv_min_en") else None
    irr_min = float(st.session_state.hard_irr_min_pct) if st.session_state.get("hard_irr_min_en") else None
    ss_min = float(st.session_state.hard_ss_min_pct) if st.session_state.get("hard_ss_min_en") else None
    co2_reduction_min_pct = (
        float(st.session_state.hard_co2_min_pct) if st.session_state.get("hard_co2_min_en") else None
    )
    ann_cost_max = float(st.session_state.hard_ann_cost_max_eur) if st.session_state.get("hard_ann_cost_max_en") else None
    ann_cost_saving_min_pct = (
        float(st.session_state.hard_ann_cost_saving_min_pct)
        if st.session_state.get("hard_ann_cost_saving_min_en")
        else None
    )
    scr_min = float(st.session_state.hard_self_cons_min_pct) if st.session_state.get("hard_self_cons_min_en") else None
    export_max = float(st.session_state.hard_export_max_pct) if st.session_state.get("hard_export_max_en") else None
    goal = st.session_state.view_goal
    scenario_type_ui = st.session_state.view_scenario_type
    tariff_family_ui = str(st.session_state.view_tariff_family)
    _profiles_rank = list(st.session_state.get("last_tariff_profiles") or _default_tariff_profiles())
    _full_pre = st.session_state.full_results_df
    if _full_pre is not None and (
        not {"Grid import reduction (kWh)", "CO2 reduction (%)"}.issubset(_full_pre.columns)
        or "Export ratio (% of PV gen)" not in _full_pre.columns
    ):
        st.session_state.full_results_df = None
    if st.session_state.full_results_df is None:
        _t_bf = time.perf_counter()
        st.session_state.full_results_df = build_full_scenario_results_df(
            st.session_state.opt_dfs,
            st.session_state.prepared_df,
            _profiles_rank,
            pv_cost_per_kwp=st.session_state.last_pv_capex,
            batt_cost_per_kwh=st.session_state.last_batt_capex,
            electricity_inflation_rate=st.session_state.last_electricity_inflation_rate,
            battery_replacement_year=st.session_state.last_battery_replacement_year,
            battery_replacement_pct_of_batt_capex=st.session_state.last_battery_replacement_cost_pct,
            inverter_replacement_year=st.session_state.last_inverter_replacement_year,
            inverter_replacement_pct_of_pv_capex=st.session_state.last_inverter_replacement_cost_pct,
            pso_levy_annual=float(st.session_state.last_pso_levy),
            lifetime_years=ly,
        )
        if _perf_profiling_enabled():
            _dt_bf = time.perf_counter() - _t_bf
            st.session_state["_perf_build_full_s"] = _dt_bf
            _perf_record("build_full_scenario_results_df", _dt_bf)
    full_table_rank = st.session_state.full_results_df
    slice_rank = full_table_rank.copy()
    slice_rank = _filter_by_scenario_type(slice_rank, scenario_type_ui)
    slice_rank = _filter_by_tariff_family(slice_rank, tariff_family_ui, _profiles_rank)
    hard_filtered_rank_df = _apply_hard_filters_to_results_df(
        slice_rank,
        capex_max_eur=capex_max,
        payback_max_years=payback_max,
        npv_min_eur=npv_min,
        irr_min_pct=irr_min,
        self_sufficiency_min_pct=ss_min,
        annual_co2_reduction_min_pct=co2_reduction_min_pct,
        annual_electricity_cost_saving_min_pct=ann_cost_saving_min_pct,
        annual_electricity_cost_max_eur=ann_cost_max,
        self_consumption_ratio_min_pct=scr_min,
        export_ratio_max_pct=export_max,
    )
    ranked = _rank_scenarios_from_consolidated_table(hard_filtered_rank_df, goal)

    render_sidebar_performance_panel()

with tab_recommended:
    if not _has_results:
        st.info(
            "Run **Run analysis** or load a **saved run** / **Demo runs** to use **Recommended setups**. "
            "**Research results** below is always available (bundled workbook, no run required)."
        )
    else:
        st.markdown('<div id="results-top-anchor"></div>', unsafe_allow_html=True)
        _maybe_scroll_to_results_top()
        _rec_df = render_recommended_setups_tab_section(
            subheader="Recommended setups",
            grid_key="recommended_setups_aggrid",
            selection_session_key="selected_recommended_row_key",
            download_key_plain="recommended_setups_dl_plain",
            download_key_full="recommended_setups_dl",
            csv_filename_plain="recommended_setups_full_kpi_results_only.csv",
            csv_filename_full="recommended_setups_full_kpi_with_assumptions.csv",
            plotly_chart_key_prefix="rec_detail",
            selection_detail_heading="Recommended selection",
            tradeoff_expander_title="Advanced trade-off charts (Recommended selection)",
            download_help_full="Last run + sidebar Decision constraints (Recommended), blank line, then full-KPI table.",
            constraints_label="Decision constraints",
            goal=goal,
            ly=ly,
            full_table_rank=full_table_rank,
            hard_filtered_rank_df=hard_filtered_rank_df,
            ranked=ranked,
        )

with tab_explorer:
    if not _has_results:
        st.info(
            "Run **Run analysis** or load a **saved run** / **Demo runs** to use **Full results**. "
            "**Research results** is always available from the bundled workbook."
        )
    else:
        st.subheader("Full results")
        st.caption(
            "Same **sidebar** scenario universe as **Recommended setups**. Use the **filters below the column mode** to narrow rows, "
            "then **click a row** to sync KPIs and charts. If your pick disappears, the first visible row is selected."
        )
        filtered_view_ex = pd.DataFrame()
        full_table_df_ex = full_table_rank
        if full_table_df_ex is None or len(full_table_df_ex) == 0:
            st.warning("No consolidated scenario table available.")
        else:
            base_table_df_ex = hard_filtered_rank_df
            n_all_ex = len(base_table_df_ex)
            _col_mode_ex = st.radio(
                "Table columns",
                ["Core columns", "All columns"],
                index=0,
                horizontal=True,
                key="explorer_aggrid_column_mode",
                help="**Core** shows main KPIs and identifiers. **All** shows every column in the consolidated table.",
            )
            _allow_ex = (
                _scenario_explorer_core_display_columns(ly)
                if _col_mode_ex == "Core columns"
                else None
            )
            _tar_opts = (
                sorted(base_table_df_ex["Tariff"].astype(str).unique().tolist())
                if "Tariff" in base_table_df_ex.columns
                else []
            )
            _scen_opts = (
                sorted(base_table_df_ex["Scenario"].astype(str).unique().tolist())
                if "Scenario" in base_table_df_ex.columns
                else []
            )
            with st.expander("Filter this table (tariff, scenario, search, ranges)", expanded=True):
                _f1, _f2 = st.columns(2)
                with _f1:
                    _sel_tar_ex = (
                        st.multiselect(
                            "Tariff",
                            options=_tar_opts,
                            default=_tar_opts,
                            key="explorer_ms_tariff",
                            disabled=len(_tar_opts) == 0,
                        )
                        if _tar_opts
                        else []
                    )
                with _f2:
                    _sel_scen_ex = (
                        st.multiselect(
                            "Scenario",
                            options=_scen_opts,
                            default=_scen_opts,
                            key="explorer_ms_scenario",
                            disabled=len(_scen_opts) == 0,
                        )
                        if _scen_opts
                        else []
                    )
                _search_ex = st.text_input(
                    "Search in any column",
                    "",
                    key="explorer_txt_search",
                    help="Keeps rows where any cell contains this text (case-insensitive).",
                )
            _pv_lo_b, _pv_hi_b = _numeric_column_bounds(base_table_df_ex, "PV (kWp)")
            _bt_lo_b, _bt_hi_b = _numeric_column_bounds(base_table_df_ex, "Battery (kWh)")
            _np_lo_b, _np_hi_b = _numeric_column_bounds(base_table_df_ex, "NPV (€)")
            _pb_lo_b, _pb_hi_b = _numeric_column_bounds(base_table_df_ex, "Payback (yrs)")
            with st.expander("Numeric range filters (optional)", expanded=False):
                _r1, _r2 = st.columns(2)
                with _r1:
                    if "PV (kWp)" in base_table_df_ex.columns and _pv_hi_b > _pv_lo_b:
                        _pv_rng_ex = st.slider(
                            "PV (kWp)",
                            min_value=float(_pv_lo_b),
                            max_value=float(_pv_hi_b),
                            value=(float(_pv_lo_b), float(_pv_hi_b)),
                            key="explorer_sl_pv",
                        )
                    else:
                        _pv_rng_ex = (_pv_lo_b, _pv_hi_b)
                    if "Battery (kWh)" in base_table_df_ex.columns and _bt_hi_b > _bt_lo_b:
                        _bt_rng_ex = st.slider(
                            "Battery (kWh)",
                            min_value=float(_bt_lo_b),
                            max_value=float(_bt_hi_b),
                            value=(float(_bt_lo_b), float(_bt_hi_b)),
                            key="explorer_sl_batt",
                        )
                    else:
                        _bt_rng_ex = (_bt_lo_b, _bt_hi_b)
                with _r2:
                    if "NPV (€)" in base_table_df_ex.columns and _np_hi_b > _np_lo_b:
                        _np_rng_ex = st.slider(
                            "NPV (€)",
                            min_value=float(_np_lo_b),
                            max_value=float(_np_hi_b),
                            value=(float(_np_lo_b), float(_np_hi_b)),
                            key="explorer_sl_npv",
                        )
                    else:
                        _np_rng_ex = (_np_lo_b, _np_hi_b)
                    if "Payback (yrs)" in base_table_df_ex.columns and _pb_hi_b > _pb_lo_b:
                        _pb_rng_ex = st.slider(
                            "Payback (yrs)",
                            min_value=float(_pb_lo_b),
                            max_value=float(_pb_hi_b),
                            value=(float(_pb_lo_b), float(_pb_hi_b)),
                            key="explorer_sl_payback",
                        )
                    else:
                        _pb_rng_ex = (_pb_lo_b, _pb_hi_b)
            explorer_table_df = _apply_full_results_explorer_table_filters(
                base_table_df_ex,
                tariff_pick=_sel_tar_ex,
                tariff_universe=_tar_opts,
                scenario_pick=_sel_scen_ex,
                scenario_universe=_scen_opts,
                search_text=_search_ex,
                pv_min=float(_pv_rng_ex[0]),
                pv_max=float(_pv_rng_ex[1]),
                batt_min=float(_bt_rng_ex[0]),
                batt_max=float(_bt_rng_ex[1]),
                npv_min=float(_np_rng_ex[0]),
                npv_max=float(_np_rng_ex[1]),
                payback_min=float(_pb_rng_ex[0]),
                payback_max=float(_pb_rng_ex[1]),
            )
            _filter_sig_ex = "|".join(
                [
                    ",".join(_sel_tar_ex),
                    ",".join(_sel_scen_ex),
                    _search_ex.strip(),
                    f"{_pv_rng_ex[0]:.8g}:{_pv_rng_ex[1]:.8g}",
                    f"{_bt_rng_ex[0]:.8g}:{_bt_rng_ex[1]:.8g}",
                    f"{_np_rng_ex[0]:.8g}:{_np_rng_ex[1]:.8g}",
                    f"{_pb_rng_ex[0]:.8g}:{_pb_rng_ex[1]:.8g}",
                ]
            )
            _grid_extra_ex = (
                f"cols{'core' if _col_mode_ex == 'Core columns' else 'all'}"
                f"|f:{hashlib.md5(_filter_sig_ex.encode('utf-8')).hexdigest()[:14]}"
            )
            filtered_view_ex = render_aggrid_results_table(
                explorer_table_df,
                grid_key="scenario_explorer_aggrid",
                height=420,
                default_rank_goal=goal,
                rank_goal_table="full",
                lifetime_years=ly,
                selection_session_key="selected_explorer_row_key",
                grid_key_extra=_grid_extra_ex,
                display_column_allowlist=_allow_ex,
                enable_column_filters=_env_truthy("REC_USE_AGGRID"),
            )
            n_f_ex = len(filtered_view_ex)
            if _env_truthy("REC_USE_AGGRID"):
                st.caption(
                    f"**{n_f_ex:,}** / **{n_all_ex:,}** rows in this table after **Full results** filters (AgGrid column filters also apply). "
                    "**CSV:** choose **results only** (plain table) or **+ assumptions** (Setting/Value block, blank line, then data; internal row keys omitted)."
                )
            else:
                st.caption(
                    f"**{n_f_ex:,}** / **{n_all_ex:,}** rows after **Full results** filters · **click a row** to update KPIs and charts. "
                    "**CSV:** **results only** or **+ assumptions** (Setting/Value block, blank line, then data; internal row keys omitted)."
                )
            _ass_scenarios_ex = last_run_assumptions_snapshot_df()
            eg1, eg2 = st.columns(2)
            with eg1:
                st.caption("**Filtered table view** (rows after Full results filters)")
                g1a, g1b = st.columns(2)
                with g1a:
                    st.download_button(
                        "Results only (CSV)",
                        data=(
                            _export_results_df_for_csv(filtered_view_ex).to_csv(index=False).encode("utf-8-sig")
                            if n_f_ex
                            else b"\xef\xbb\xbf"
                        ),
                        file_name="scenario_grid_view_results_only.csv",
                        mime="text/csv",
                        key="dl_scenarios_grid_csv_res_only",
                        disabled=n_f_ex == 0,
                    )
                with g1b:
                    st.download_button(
                        "Results + assumptions (CSV)",
                        data=(
                            encode_csv_assumptions_block_then_results_df(
                                _ass_scenarios_ex, _export_results_df_for_csv(filtered_view_ex)
                            )
                            if n_f_ex
                            else b"\xef\xbb\xbf"
                        ),
                        file_name="scenario_grid_view_with_assumptions.csv",
                        mime="text/csv",
                        key="dl_scenarios_grid_csv_explorer",
                        disabled=n_f_ex == 0,
                    )
            with eg2:
                st.caption("**Full filtered table** (all rows in sidebar universe)")
                g2a, g2b = st.columns(2)
                with g2a:
                    st.download_button(
                        "Results only (CSV)",
                        data=_export_results_df_for_csv(base_table_df_ex).to_csv(index=False).encode("utf-8-sig"),
                        file_name="scenario_full_results_only.csv",
                        mime="text/csv",
                        key="dl_scenarios_all_csv_res_only",
                    )
                with g2b:
                    st.download_button(
                        "Results + assumptions (CSV)",
                        data=encode_csv_assumptions_block_then_results_df(
                            _ass_scenarios_ex, _export_results_df_for_csv(base_table_df_ex)
                        ),
                        file_name="scenario_full_with_assumptions.csv",
                        mime="text/csv",
                        key="dl_scenarios_all_csv_explorer",
                    )

        ex_key = st.session_state.get("selected_explorer_row_key")
        ex_row = None
        if (
            ex_key is not None
            and len(filtered_view_ex) > 0
            and SCENARIO_ROW_KEY_FIELD in filtered_view_ex.columns
        ):
            _hit_ex = filtered_view_ex[filtered_view_ex[SCENARIO_ROW_KEY_FIELD].astype(str) == str(ex_key)]
            if len(_hit_ex) > 0:
                ex_row = _hit_ex.iloc[0]
        if ex_row is not None:
            st.divider()
            st.markdown("##### Selected row — KPIs and charts")
            render_consolidated_selection_detail_block(
                ex_row,
                full_table_rank=full_table_rank,
                hard_filtered_rank_df=hard_filtered_rank_df,
                ranked=ranked,
                goal=goal,
                ly=ly,
                tradeoff_expander_title="Advanced trade-off charts (explorer selection)",
                comparison_selection_caption=(
                    "Lowest **annual electricity bill (€)** and highest **annual CO₂ reduction (kg)** are highlighted; "
                    "**yellow row** = the scenario selected in the **Full results** table above."
                ),
                plotly_chart_key_prefix="explorer_detail",
                prominent_header=True,
                show_filtered_scenario_comparison=False,
                show_cumulative_outlook=False,
            )
        elif len(filtered_view_ex) > 0:
            st.caption(
                "Select a row in the grid above to load **KPIs** and **trade-off** charts for that scenario "
                "(or use filters — the first visible row is selected when your old pick drops out)."
            )

with tab_consumption:
    if not _has_results:
        st.info(
            "Run **Run analysis** or load a **saved run** / **Demo runs** to view **Consumption patterns**. "
            "**Research results** is always available from the bundled workbook."
        )
    else:
        st.subheader("Consumption patterns")
        render_community_consumption_patterns(st.session_state.prepared_df)

with tab_production:
    if not _has_results:
        st.info(
            "Run **Run analysis** or load a **saved run** / **Demo runs** to view **Production patterns**. "
            "**Research results** is always available from the bundled workbook."
        )
    else:
        st.subheader("Production patterns")
        render_production_patterns_per_kwp(st.session_state.prepared_df)

with tab_research:
    render_bundled_research_tab()

with tab_explainer:
    render_settings_kpi_guide_tab(setup)

