"""Parse bundled research Excel (fixed reference, not from live Run analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SHEET_NAME = "Overall comparisson"
BUNDLED_RELATIVE_PATH = Path("assets") / "research" / "res.xlsx"

WinnerDirection = Literal["min", "max"]


@dataclass(frozen=True)
class ResearchWinnerRule:
    id: str
    label: str
    description: str
    metric_row_substr: str
    direction: WinnerDirection


# Row match is case-insensitive substring on first-column labels (after normalizing € / unicode subscripts).
RESEARCH_WINNER_RULES: tuple[ResearchWinnerRule, ...] = (
    ResearchWinnerRule(
        id="lowest_bill",
        label="Lowest electricity bill",
        description="minimum **Annual electricity bill (€)** in each scenario block",
        metric_row_substr="annual electricity bill",
        direction="min",
    ),
    ResearchWinnerRule(
        id="best_npv",
        label="Best NPV",
        description="maximum **NPV (€)** in each scenario block",
        metric_row_substr="npv (",
        direction="max",
    ),
    ResearchWinnerRule(
        id="best_irr",
        label="Best IRR",
        description="maximum **IRR (%)** in each scenario block",
        metric_row_substr="irr",
        direction="max",
    ),
    ResearchWinnerRule(
        id="lowest_capex",
        label="Lowest CAPEX",
        description="minimum **CAPEX (€)** in each scenario block",
        metric_row_substr="capex (",
        direction="min",
    ),
    ResearchWinnerRule(
        id="best_co2_reduction",
        label="Best CO₂ reduction",
        description="maximum **Annual CO2 reduction (kg)** in each scenario block",
        metric_row_substr="annual co2 reduction",
        direction="max",
    ),
    ResearchWinnerRule(
        id="best_self_sufficiency",
        label="Best self-sufficiency",
        description="maximum **Self-sufficiency (%)** in each scenario block",
        metric_row_substr="self-sufficiency",
        direction="max",
    ),
    ResearchWinnerRule(
        id="best_self_consumption_ratio",
        label="Best self-consumption ratio",
        description="maximum **Self-consumption ratio (%)** in each scenario block",
        metric_row_substr="self-consumption ratio",
        direction="max",
    ),
    ResearchWinnerRule(
        id="shortest_payback",
        label="Shortest payback",
        description="minimum **Payback (yrs)** in each scenario block (non-finite values ignored)",
        metric_row_substr="payback",
        direction="min",
    ),
)


def _norm_metric_label(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("€", "EUR").replace("₂", "2").replace("CO₂", "CO2")
    return s.strip().lower()


def _find_metric_row_index(row_labels: list[str], substr: str) -> int:
    sub = substr.lower()
    for i, lab in enumerate(row_labels):
        if sub in _norm_metric_label(lab):
            return i
    raise ValueError(f"No metric row matching {substr!r}")


def load_bundled_research_xlsx(path: Path) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray]:
    """Return raw headerless frame, scenario block titles, tariff names (one block), and numeric data array (rows x 25)."""
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None)
    ncols = raw.shape[1]
    if ncols < 6:
        raise ValueError("Unexpected research table width")
    n_blocks = (ncols - 1) // 5
    scenario_titles: list[str] = []
    for b in range(n_blocks):
        c0 = 1 + b * 5
        h = raw.iloc[0, c0]
        scenario_titles.append(str(h).strip() if pd.notna(h) else f"Scenario {b + 1}")

    tariff_names: list[str] = []
    for j in range(1, 6):
        t = raw.iloc[1, j]
        tariff_names.append(str(t).replace("\n", " ").strip() if pd.notna(t) else f"Col{j}")

    data_region = raw.iloc[2:, 1: 1 + n_blocks * 5].apply(pd.to_numeric, errors="coerce")
    mat = data_region.to_numpy(dtype=float)
    return raw, scenario_titles, tariff_names, mat


def build_research_display_dataframe(
    raw: pd.DataFrame,
    scenario_titles: list[str],
    tariff_names: list[str],
) -> pd.DataFrame:
    """Wide table: Metric | (scenario,tariff) columns."""
    n_blocks = len(scenario_titles)
    row_labels = [raw.iloc[i, 0] for i in range(2, len(raw))]
    cols: dict[tuple[str, str], Any] = {}
    for b, sname in enumerate(scenario_titles):
        for j, tname in enumerate(tariff_names):
            cidx = 1 + b * 5 + j
            col_key = (sname, tname)
            cols[col_key] = [raw.iloc[i, cidx] for i in range(2, len(raw))]
    idx = pd.Index([str(x) if pd.notna(x) else "" for x in row_labels], name="Metric")
    out = pd.DataFrame(cols, index=idx)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def _research_metric_display_format(metric_label: str) -> str:
    """How to format numeric cells for the research overview table."""
    ml = _norm_metric_label(metric_label)
    if "capex" in ml:
        return "int_money"
    if "optimum pv size" in ml or ("optimum" in ml and "kwp" in ml and "pv" in ml):
        return "int_size"
    if "optimum battery" in ml or ("battery size" in ml and "kwh" in ml):
        return "int_size"
    return "float1"


def format_research_display_dataframe(display_df: pd.DataFrame) -> pd.DataFrame:
    """Format numbers for display: 1 decimal by default; PV/battery sizes and CAPEX as integers."""
    out = display_df.copy().astype(object)
    for metric_label in out.index:
        fmt = _research_metric_display_format(str(metric_label))
        for col in out.columns:
            v = out.loc[metric_label, col]
            if pd.isna(v):
                out.loc[metric_label, col] = "—"
                continue
            if v is None or (isinstance(v, str) and not str(v).strip()):
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x):
                out.loc[metric_label, col] = "—"
                continue
            if fmt == "int_size":
                out.loc[metric_label, col] = str(int(round(x)))
            elif fmt == "int_money":
                out.loc[metric_label, col] = str(int(round(x)))
            else:
                out.loc[metric_label, col] = f"{x:.1f}"
    return out


def build_overall_comparison_highest_npv_df(
    raw: pd.DataFrame,
    scenario_titles: list[str],
    tariff_names: list[str],
    mat: np.ndarray,
) -> pd.DataFrame:
    """One KPI table where each scenario column uses the tariff with the highest NPV."""
    row_labels = [raw.iloc[i, 0] for i in range(2, len(raw))]
    row_labels_str = [str(x) if pd.notna(x) else "" for x in row_labels]
    npv_ridx = _find_metric_row_index(row_labels_str, "npv (")
    n_blocks = len(scenario_titles)

    # Winner tariff index per scenario block based on highest NPV.
    winner_idx_per_block: list[int] = []
    for b in range(n_blocks):
        vals = mat[npv_ridx, b * 5 : (b + 1) * 5].astype(float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            winner_idx_per_block.append(0)
            continue
        candidates = np.where(mask)[0]
        sub = vals[mask]
        winner_idx_per_block.append(int(candidates[int(np.argmax(sub))]))

    out_rows: list[dict[str, object]] = []
    tariff_row: dict[str, object] = {"KPI": "Tariff"}
    for b, sname in enumerate(scenario_titles):
        j = winner_idx_per_block[b]
        tariff_row[str(sname)] = str(tariff_names[j]).replace("\n", " ").strip()
    out_rows.append(tariff_row)

    for i, metric_label in enumerate(row_labels):
        kpi = str(metric_label) if pd.notna(metric_label) else ""
        row: dict[str, object] = {"KPI": kpi}
        fmt = _research_metric_display_format(kpi)
        for b, sname in enumerate(scenario_titles):
            j = winner_idx_per_block[b]
            v = raw.iloc[2 + i, 1 + b * 5 + j]
            if pd.isna(v):
                row[str(sname)] = "—"
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                row[str(sname)] = v
                continue
            if not np.isfinite(x):
                row[str(sname)] = "inf" if np.isinf(x) else "—"
                continue
            if fmt in ("int_size", "int_money"):
                row[str(sname)] = str(int(round(x)))
            else:
                row[str(sname)] = f"{x:.1f}"
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def compute_winners_for_rule(
    raw: pd.DataFrame,
    scenario_titles: list[str],
    tariff_names: list[str],
    mat: np.ndarray,
    rule: ResearchWinnerRule,
) -> pd.DataFrame:
    """One row per scenario block: winning tariff, value, tie note."""
    row_labels = [str(raw.iloc[i, 0]) if pd.notna(raw.iloc[i, 0]) else "" for i in range(2, raw.shape[0])]
    try:
        ridx = _find_metric_row_index(row_labels, rule.metric_row_substr)
    except ValueError:
        return pd.DataFrame(
            {"Scenario": scenario_titles, "Winner tariff": ["—"] * len(scenario_titles), "Value": [np.nan] * len(scenario_titles), "Note": ["metric row not found"] * len(scenario_titles)}
        )

    n_blocks = len(scenario_titles)
    rows_out: list[dict[str, Any]] = []
    for b in range(n_blocks):
        vals = mat[ridx, b * 5 : (b + 1) * 5].astype(float)
        mask = np.isfinite(vals)
        if rule.direction == "min" and rule.metric_row_substr == "payback":
            mask = mask & (vals > 0) & (vals < 1e6)
        if not np.any(mask):
            rows_out.append(
                {
                    "Scenario": scenario_titles[b],
                    "Winner tariff": "—",
                    "Value": np.nan,
                    "Note": "no finite values",
                }
            )
            continue
        candidates = np.where(mask)[0]
        sub = vals[mask]
        if rule.direction == "max":
            best_j = candidates[int(np.argmax(sub))]
        else:
            best_j = candidates[int(np.argmin(sub))]
        best_val = float(vals[best_j])
        tol = 1e-6 * max(1.0, abs(best_val))
        tied = [
            tariff_names[j]
            for j in range(5)
            if mask[j] and abs(float(vals[j]) - best_val) <= tol
        ]
        note = f"tied: {', '.join(tied)}" if len(tied) > 1 else ""
        rows_out.append(
            {
                "Scenario": scenario_titles[b],
                "Winner tariff": tariff_names[best_j],
                "Value": best_val,
                "Note": note,
            }
        )
    return pd.DataFrame(rows_out)


def build_all_winners_summary_df(
    raw: pd.DataFrame,
    scenario_titles: list[str],
    tariff_names: list[str],
    mat: np.ndarray,
) -> pd.DataFrame:
    """Wide table: one row per scenario block; columns for each rule (winner tariff + value)."""
    out = pd.DataFrame({"Scenario": scenario_titles})
    for rule in RESEARCH_WINNER_RULES:
        w = compute_winners_for_rule(raw, scenario_titles, tariff_names, mat, rule)
        out[f"{rule.label} — tariff"] = w["Winner tariff"].values
        out[f"{rule.label} — value"] = w["Value"].values
    return out


def research_metric_grouped_bars(
    raw: pd.DataFrame,
    scenario_titles: list[str],
    tariff_names: list[str],
    mat: np.ndarray,
    rule: ResearchWinnerRule,
) -> go.Figure:
    """Grouped bars: x = scenario block, one bar series per tariff (same metric as winner rule)."""
    row_labels = [str(raw.iloc[i, 0]) if pd.notna(raw.iloc[i, 0]) else "" for i in range(2, raw.shape[0])]
    try:
        ridx = _find_metric_row_index(row_labels, rule.metric_row_substr)
    except ValueError:
        fig = go.Figure()
        fig.add_annotation(text="Metric not found", showarrow=False)
        return fig

    n_blocks = len(scenario_titles)
    short_titles = [_short_scenario_title(s) for s in scenario_titles]
    metric_title = row_labels[ridx]
    fig = go.Figure()
    for j, tname in enumerate(tariff_names):
        ys = [float(mat[ridx, b * 5 + j]) if np.isfinite(mat[ridx, b * 5 + j]) else None for b in range(n_blocks)]
        _tn = str(tname).replace("\n", " ")
        fig.add_trace(
            go.Bar(
                name=_tn,
                x=short_titles,
                y=ys,
                hovertemplate=f"{_tn}<br>%{{y:,.1f}}<extra></extra>",
            )
        )
    _vals = [
        float(mat[ridx, b * 5 + j])
        for b in range(n_blocks)
        for j in range(len(tariff_names))
        if np.isfinite(mat[ridx, b * 5 + j])
    ]
    # Let Plotly choose y limits from the traces (with its own padding). A fixed manual ``range`` was clipping
    # some grouped bars when padding was tight or data span edge cases; autorange always fits all bar tops.
    _rangemode: Literal["tozero", "normal"] = (
        "tozero" if _vals and min(_vals) >= -1e-9 else "normal"
    )
    _yaxis = dict(
        title=metric_title[:60],
        automargin=True,
        fixedrange=False,
        autorange=True,
        rangemode=_rangemode,
    )

    fig.update_layout(
        barmode="group",
        margin=dict(l=8, r=8, t=56, b=8),
        title=dict(text=str(metric_title), font=dict(size=14)),
        xaxis=dict(title="Scenario", tickangle=0, automargin=True),
        yaxis=_yaxis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(440, 360 + 28 * n_blocks),
        # ``x unified`` shows every tariff in one tooltip; ``closest`` shows the single bar under the cursor.
        hovermode="closest",
    )
    return fig


def _short_scenario_title(s: str) -> str:
    s = s.strip()
    if len(s) <= 42:
        return s
    return s[:39] + "…"


def research_rule_by_id(rule_id: str) -> ResearchWinnerRule:
    for r in RESEARCH_WINNER_RULES:
        if r.id == rule_id:
            return r
    return RESEARCH_WINNER_RULES[0]
