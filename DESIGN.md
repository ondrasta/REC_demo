# REC Financial Feasibility Web App — Design Document

## 1. Overview

Streamlit web application for analyzing financial feasibility of Renewable Energy Communities (REC), comparing grid-only and PV scenarios (and optionally battery scenarios), and running optimal system sizing based on configurable goals.

**UI vs model:** `ENABLE_BATTERY_UI` is enabled in this app; battery setup controls and scenario types (**Battery + Grid**, **PV + Battery + Grid**) are active in the UI and optimizer.

---

## 2. Data Requirements

### 2.1 Input CSVs

| File | Purpose | Expected structure |
|------|---------|-------------------|
| **Consumption** | REC hourly demand (kWh) | `date` (DD/MM/YYYY HH:00), consumption column |
| **PV profile** | PVGIS hourly `P` in **Wh for 1 kWp** (parsed to kWh/kWp by dividing by 1000); column `time` aligned to consumption | Same period as consumption after alignment in `load_and_prepare_data` |
| **Tariffs (default)** | Multiple named import tariffs + standing/export | Same schema as optional upload; see §3.2 |

**Year filter:** In `_parse_consumption_csv` and `_parse_pv_timeseries_csv`, if **2020** appears in the series, rows are restricted to **that calendar year** (legacy alignment with the bundled sample / notebook workflow). Multi-year uploads are silently subset—document in user-facing README.

**Optional defaults:** If the Streamlit uploaders are empty, bytes are loaded from (in order): environment overrides (`REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV` / `REC_FEASIBILITY_DEFAULT_PV_CSV`), then `data/local_consumption.csv` / `data/local_pv.csv` (if present), else bundled `data/default_consumption.csv` and `data/default_pv.csv`. The same `_parse_consumption_csv` / `_parse_pv_timeseries_csv` and `load_and_prepare_data` apply.

**Default tariffs CSV:** If present, tariff rows load from (in order): `REC_FEASIBILITY_DEFAULT_TARIFFS_CSV`, then `data/local_tariffs.csv`, else bundled `data/default_tariffs.csv`. Parsed with `_parse_tariff_variants_csv_bytes`, flattened to profile list via `_tariff_matrix_profiles_from_parsed` (columns `tariff_sel_*`). If no file parses successfully, fallback is `_tariff_matrix_from_builtin_defaults()` (three Market-average profiles). The UI matrix supports **Include** checkboxes; only checked profiles are passed to `load_and_prepare_data` and the optimizer.

---

## 3. Tariff Structure

### 3.1 Time Bands

- **Peak:** 17:00–19:00  
- **Night:** 23:00–08:00  
- **Day:** 08:00–17:00, 19:00–23:00  

### 3.2 Tariff Variants

Tariffs are modeled as profile variants under fixed families:
- `standard`
- `weekend`
- `flat`

Each variant has:
- display name (`family — variant`)
- import rates (family-specific)
- standing charge (€/year)
- export rate (€/kWh, per variant)
- hourly column name `tariff_sel_{i}` when flattened into the prepared dataframe

Profiles come from the **tariff matrix** in **Model setup → Data & tariffs**: defaults or optional CSV upload (`_parse_tariff_variants_csv_bytes`). The flattened list is built by `_tariff_matrix_profiles_from_parsed`, which walks families in order **standard → weekend → flat** (up to **20** rows per family) and assigns sequential `tariff_sel_*` columns. **Include** checkboxes (per row) filter which profiles are passed to `load_and_prepare_data` and the optimizer; unchecked rows are ignored.

- Ignores fully blank trailing rows in the CSV
- Optional **Load tariffs from CSV** replaces the entire matrix from an uploaded file

CSV parser supports both canonical columns (`family`, `variant`, `standing_charge`, `export_rate`, family-specific rate columns) and simplified aliases (`tariff_type`, `company`, `standing_charge (EUR/year)`, `export_rate (EUR/kWh)`, `weekday_*`, `weekend_*`).

---

## 4. Scenarios

| Scenario | Description |
|----------|-------------|
| **Grid only** | All demand from grid. Baseline. |
| **PV + Grid** | PV offsets demand; surplus exported. |
| **PV + Battery + Grid** | PV + battery arbitrage + export (optimizer + UI when `ENABLE_BATTERY_UI`). |
| **Battery + Grid** | No PV; battery may charge from grid at night; discharges per **Peak only** or **Day+Peak** schedule (`_battery_discharge_ok_hour`). |

The consolidated results table and charts show only the scenarios exposed in the UI for the selected **Scenario type** filter.

---

## 5. Battery Model

### 5.1 Round-trip Efficiency (Symmetric)

- `charge_eff = sqrt(round_trip_eff)`
- `discharge_eff = sqrt(round_trip_eff)`
- **Charging:** Energy drawn from source `E` → SOC increases by `E * charge_eff`
- **Discharging:** Load served `D` limited by `soc * discharge_eff`; SOC decreases by `D / discharge_eff`

### 5.2 Default Dispatch Rule

**Peak only** — battery discharges during peak (17:00–19:00) to reduce grid import.

**Day+Peak** — discharge in the same peak window; if SOC is still above the floor, discharge is also allowed in **19:00–23:00**. No discharge in **23:00–08:00** (night) or **08:00–17:00** (earlier daytime), so chronology yields peak use first, then evening, for the same day.

Implementation: `_battery_discharge_ok_hour()` in `app.py`; used by `run_scenario_pv_battery_grid` and `run_scenario_battery_grid`.

### 5.3 Charging sources (defaults)

- **From PV surplus:** on by default (`charge_from_pv`).
- **From grid at night:** **off** by default (`charge_from_grid_at_night`); user may enable in Advanced (arbitrage-style runs).

### 5.4 SOC Operating Window

Battery operation is constrained by:
- **Minimum SOC floor** (`min_soc`) — battery cannot discharge below this bound
- **Maximum SOC ceiling** (`max_soc`) — battery cannot charge above this bound

Defaults in UI:
- Initial SOC: `0%`
- Min SOC floor: `10%`
- Max SOC ceiling: `90%`
- DoD: `90%` (effective floor is the stricter of DoD floor and Min SOC floor)

---

## 6. Self-sufficiency and Self-consumption

### 6.1 Self-sufficiency

`self_sufficiency = local_renewable_to_load / total_consumption * 100`

- **PV + Grid:** local renewable = direct PV to load (self-consumed PV)
- **PV + Battery + Grid:** local renewable = direct PV to load + battery discharge to load that originated from PV
- **Battery + Grid:** 0% — grid-charged battery is not local renewable

### 6.2 Self-Consumption (kWh)

**Self-Consumption (kWh)** = direct PV to load + PV-origin battery discharge to load.

- **PV + Grid:** per hour, `min(PV production, consumption)`; annual sum.
- **PV + Battery + Grid:** per hour, direct PV used by load + battery discharge to load that originated from PV (excludes round-trip losses and PV still in battery at end of simulation).

### 6.3 Self-consumption ratio (%)

`self_consumption_ratio = Self-Consumption (kWh) / total PV generation * 100`

(Do not use `pv_prod - feed_in` for PV + Battery + Grid, as that overstates self-consumption.)

---

## 7. Financial Metrics

**Analysis horizon:** The UI **Analysis horizon (years)** in **Financial assumptions** (default **20**, range **15–30**) sets the number of years *T* for NPV, IRR, gross/net savings, cumulative charts, and the valid range for optional replacement years. Column labels use *T* (e.g. `NPV (Ty, €)`).

| Metric | Formula |
|--------|---------|
| Annual electricity cost (year 1) | Σ(GridImport × Tariff) − Σ(Export × ExportRate) + standing_charge + **PSO levy** + OPEX |
| Annual electricity bill (€), consolidated grid | **Same numeric value** as annual electricity cost (year 1); column name in **Full results** AgGrid and ranking for “lowest annual electricity cost”. |
| Annual grid import cost (€) | Σ(GridImport × Tariff) for the simulated year — import energy spend **before** export income and **before** standing / PSO / OPEX. |
| Annual savings (year 1) | Baseline cost − Scenario cost |
| **Annual electricity bill reduction (%)** (consolidated grid) | Year‑1 **Annual savings (€)** ÷ that tariff’s **grid-only** annual bill (€) × 100; column `COL_ANNUAL_ELECTRICITY_BILL_REDUCTION_PCT` in `app.py`. |
| Gross savings over *T* years | Σ savings_t where savings_t = savings₁ × (1 + inflation)^(t−1), t=1..*T* |
| Net benefit over *T* years | Gross savings over *T* years − CAPEX − nominal replacement costs |
| Simple payback | CAPEX / Annual savings (year 1) |
| NPV (*T* years) | −CAPEX + Σ(savings_t / (1+r)^t), t=1..*T*, savings_t inflated |
| IRR (*T* years) | Discount rate r where NPV = 0 (solved numerically). Shown as %. |

**Per € CAPEX (consolidated grid only):** For each scenario row with CAPEX > 0, the app adds ratio columns (labels include *T* where applicable): **NPV per € CAPEX**, **Annual CO₂ reduction per € CAPEX**, **Lifetime CO₂ avoided per € CAPEX** (= annual CO₂ reduction × *T*, vs grid-only, ÷ CAPEX), **Annual savings per € CAPEX**, **Gross savings per € CAPEX** (gross *T*-year savings ÷ CAPEX). There is **no** “net benefit per € CAPEX” column. Ratios are **NaN** when CAPEX = 0. Detail KPI tiles: economic ratios under **A. Economic impact**; CO₂ ratios under **B. Community impact**.

**Electricity inflation:** Optional input (default 0% in the UI) that escalates each year: electricity costs (baseline + scenario), export income, standing charge, **PSO levy**, and OPEX. CAPEX is not inflated. With inflation rate *i*, savings in year *t* = savings₁ × (1+i)^(t−1).

**Replacements (optional):**
- Battery replacement year + replacement cost (% of battery CAPEX)
- Inverter replacement year + replacement cost (% of PV CAPEX)
- Replacement costs are **not inflated**
- Replacement costs are discounted in NPV/IRR
- Net benefit over *T* years subtracts replacement costs in nominal terms

OPEX = CAPEX × (opex_pct / 100) (default **1%** of scenario CAPEX in the UI). Standing charge and **PSO levy** (default **€19.10**/year in the UI) apply to both baseline and scenarios. OPEX applies only where there is CAPEX (baseline Grid only has zero CAPEX and therefore zero OPEX); savings net them out.

---

## 8. CO₂ Metrics

| Metric | Formula |
|--------|---------|
| Grid CO₂ (scenario) | Σ(GridImport × **emission factor** kg/kWh) for the scenario. Shown in the consolidated grid as **Annual grid CO₂ emissions (kg)** — **operational** emissions from **imports**, not “savings”. |
| Lifetime CO₂ (kg) | **Annual grid CO₂ emissions (kg)** × **analysis horizon (years)** — shortcut assuming the same import-related emissions every year. |
| CO₂ savings / **Annual CO₂ reduction (kg)** | CO2_GridOnly − CO2_Scenario (clamped ≥ 0). The **Full results** consolidated grid uses the column **Annual CO₂ reduction (kg)**; elsewhere (optimizer rows, tiles, recommended setups) the same quantity may appear as **CO₂ savings (kg)**. |
| Grid CO₂ factor | **User-editable** in **Model setup** as **Grid CO₂ factor (kg/kWh)**; default **`DEFAULT_CO2_FACTOR` = 0.2462**. At runtime, `_grid_co2_factor()` returns the value frozen from the **last completed Run analysis** (`last_co2_factor`). Tests may still import **`CO2_FACTOR`** as an alias of `DEFAULT_CO2_FACTOR`. |
| Export ratio (% of PV gen) | For consolidated results: **Export to grid (kWh) ÷ total annual PV generation (kWh) × 100** when annual PV generation > 0; empty / N/A when there is no PV. Used for **Decision constraints → Export ratio max** and **Recommended setups** logic. |

---

## 9. Optimizer

- Respects `pv_min`, `pv_max`, `batt_min`, `batt_max`, `pv_step`, `batt_step`
- **Speed presets** (UI): **Quick** (e.g. PV 5 / battery 5 kWh steps), **Fast** (10 / 10), **Full** ( **1 / 1** — finest search; slowest).
- PV + Battery + Grid loop: if `batt_min` > 0 start from `batt_min`, else from `batt_step` to avoid duplicating PV + Grid cases
- Uses the same finance stack as displayed KPIs: import/export energy costs, standing charge, **PSO levy**, OPEX, discount rate, electricity inflation, **analysis horizon** *T* (UI **15–30** years, default **20**), optional replacements (`optimize()` in `app.py`)

### 9.1 Recommended setups tab

Post-processes the **same** `opt_dfs` as the main results (**no second optimizer**). Optional **constraint thresholds** define a feasible set: **max payback** (default **10 y**, `RECOMMENDED_SETUP_MAX_PAYBACK_YEARS`), optional **NPV min (€)** and **CO₂ reduction min (%)** when their checkboxes are on (defaults **0** for both — numeric **≥** floors, aligned with **Full results** hard filters; not a hidden strict “> 0” gate when the threshold is zero), **min self-consumption** (default **80%**, `RECOMMENDED_SETUP_DEFAULT_MIN_SELF_CONSUMPTION_PCT`), **max export ratio** (default **20%**, `RECOMMENDED_SETUP_DEFAULT_MAX_EXPORT_RATIO_PCT`). Export ratio for filtering: **annual export to grid ÷ annual PV generation × 100** when `pv_gen_kwh > 0`, else **0**. Among feasible rows for each tariff × scenario family, the **winner** is chosen by **`winner_preset`** (mapped from sidebar **Rank results by** via the same labels as `RECOMMENDED_WINNER_PRESETS`), using **`_sort_feasible_for_recommended_winner_preset`** — a **preset-specific** lexicographic order on optimizer columns (NPV, CO₂, SCR, savings, payback, bill, etc., depending on preset), with a final tie-break **lowest CAPEX (€)**. This is **not** always “max NPV first”; see `RECOMMENDED_WINNER_PRESET_HELP` / `_sort_feasible_for_recommended_winner_preset`. When `ENABLE_BATTERY_UI` and `opt_dfs` exists (completed run), the summary table includes **Battery charging (last run)** from frozen `last_battery_settings.charge_from_grid_at_night` (`PV only charging` vs `PV + night-grid charging`); the column is omitted before the first successful run. Default battery setting: **night grid charging off** (checkbox bound to `st.session_state.setup_battery_charge_from_grid_night`, initialized **False** on first use) unless the user enables it in Advanced.

**UI:** **`render_recommended_snapshot_cards_from_table`** draws the multi-criteria snapshot (top pick, lowest bill, highest CO₂) from the **Recommended grid’s own rows** so cards agree with the table. **`_inject_recommended_metrics_from_consolidated`** overwrites displayed KPI columns from the canonical **`full_table_rank`** / `build_full_scenario_results_df` by `_scenario_row_key`. The summary table is rendered in **Ag Grid** with **single-row selection** (`selection_session_key="selected_recommended_row_key"`). Before display, `augment_recommended_df_with_scenario_row_keys` adds **`_scenario_row_key`**: for rows with valid PV/battery sizing, the key matches `compose_scenario_row_key(tcol, Scenario family, pv, batt)` so the selection can join to `build_full_scenario_results_df`; rows **without** feasible sizing use a dedicated placeholder prefix (`RECOMMENDED_NO_SIZING_KEY_PREFIX`) so they never collide with real consolidated keys — those rows show **`Note`** in the detail area instead of the full KPI stack. The shared renderer (`render_consolidated_selection_detail_block`) drives KPIs/trade-offs and, for this tab, also the filtered-set comparison table, cumulative charts, and **`render_all_tariffs_comparison_grouped_bars`** (`radio_session_key="recommended_all_tariffs_bar_scope"`) when a consolidated match exists. **No cross-tab sync** with `selected_explorer_row_key`.

**CSV:** Primary buttons export **`recommended_setups_join_consolidated_kpis_df`** output — one row per recommended sizing with the **same columns as the consolidated / Full results** table (internal key stripped); last column **`Recommended setups note`** for infeasible or unmatched rows. Buttons are **all rows — full KPIs** and **all rows — full KPIs + assumptions**. **Results + assumptions** prepends the same assumptions block as the explorer (`encode_csv_assumptions_block_then_results_df`), including Recommended constraint rows.

On successful **Run analysis**, `last_tariff_matrix_source_label` is set from `tariff_matrix_source_label`, or from `_load_tariff_matrix_profiles_initial()` if that label was empty. **Settings & App guide → Run assumptions (last completed run)** uses **only** `last_tariff_matrix_source_label`, not the live Data & tariffs caption.

---

## 10. Streamlit layout (summary)

- **Before first run:** One short caption + optional **How to use this app** expander (full reference); **Model setup** lives on the **Run your own analysis** tab as a **two-column** strip (data | core costs), then **three expanders** (Data & tariffs; Financial assumptions — **analysis horizon** + optional replacements; Advanced / optimizer), then the **Run analysis / Stop run** row. Core costs include **Grid CO₂ factor (kg/kWh)**. Long KPI/assumption prose is in that expander and in **Settings & App guide**, not stacked above the form.
- **After first run:** The entire setup is inside **Run your own analysis** → **Edit assumptions and rerun** (collapsed); main content is **tabs** (left → right): **Recommended setups**, **Full results**, **Consumption patterns**, **Production patterns**, **Run your own analysis** (Model setup + **Saved run**), **Settings & App guide**; the **sidebar** stacks post-run controls in **execution order** (see §10.3). **`render_sidebar_postrun_filters`** may apply **`_sidebar_apply_recommended_decision_constraints_defaults`** once per rerun when **`_pending_apply_recommended_constraints_defaults`** is set (after a successful **Run analysis** or **saved-run restore**) so default decision constraints apply before widgets bind — avoids Streamlit widget-state mutation errors.
- **Embedded demo ZIPs** (`assets/saved_runs/`) are **disabled by default** (`ENABLE_EMBEDDED_SAVED_RUNS = False`); no sidebar demo picker or auto-load unless re-enabled in code.
- **Shared ranking slice (before result tabs):** `full_table_rank`, `hard_filtered_rank_df`, and `ranked` are built **once** per rerun (after reading sidebar filters / hard constraints), so **Recommended setups** and **Full results** reuse the same filtered universe and ranking metadata without relying on tab execution order.
- **Recommended setups:** **`render_recommended_snapshot_cards_from_table`** first (snapshot from recommended rows + **`per_capex_ratio_column_names`**-aware table). Then Ag Grid over `build_recommended_setups_summary_df` output, **`_inject_recommended_metrics_from_consolidated`**, sort — augmented with **`_scenario_row_key`** for selection (`selected_recommended_row_key`). Detail block = **`render_consolidated_selection_detail_block`** (`plotly_chart_key_prefix="rec_detail"`) — viewing line, compare pins, compare table, KPIs (including per-€-CAPEX tiles in A/B), charts, trade-offs, comparison, cumulative, **`render_all_tariffs_comparison_grouped_bars`** (`radio_session_key="recommended_all_tariffs_bar_scope"`) — when the key matches `full_table_rank`; placeholder keys show **Note** only. **Independent** of **Full results** selection.
- **Full results:** Ag Grid over the **same hard-filtered** consolidated table as ranking (`hard_filtered_rank_df`). **Table columns** radio: **Core columns** (default) applies `_scenario_explorer_core_display_columns()` then `_subset_dataframe_display_columns` **inside** `render_aggrid_results_table` **before** the grid is built — dropped columns are **not** present in Ag Grid, so **per-column filters and client-side grid sorting** apply only to the **kept** columns. The **initial** goal-based row order (`_sort_consolidated_scenarios_for_goal` when **Rank results by** applies) is computed on the full `hard_filtered_rank_df` **before** that column subset. **All columns** passes the full frame to the grid (filters/sorts can use every column). **Single-row selection** (row click); selection is tracked by **`selected_explorer_row_key`** and an internal column **`_scenario_row_key`** (hidden in the grid). After each filter/sort/selection update, if the stored key is not in the visible dataframe, selection falls back to the **first visible** row (or clears if empty). For the selected row: `render_consolidated_selection_detail_block` with **KPI/charts + trade-off** only (`show_filtered_scenario_comparison=False`, `show_cumulative_outlook=False`; `plotly_chart_key_prefix="explorer_detail"`). **`_export_results_df_for_csv`** strips the key before CSV download.
- **Production patterns:** **`render_production_patterns_per_kwp`** on `prepared_df` — PV yield per nominal kWp (**`pv_per_kwp`**, kWh/kWp per hour): KPI strip, monthly/seasonal/daily sums, average hourly profile by tariff time band, band-share donuts, weekday vs weekend and by-season mean hourly curves, hour×month and hour×DOW heatmaps. Scenario- and filter-independent (same spirit as **Consumption patterns**).
- **PV + Grid sweep:** removed from the UI.
- **CSV exports:** **Full results** offers **results only** (plain table CSV) and **results + assumptions**. **Recommended setups** offers **all rows — full KPIs** and **all rows — full KPIs + assumptions**, using **consolidated KPI columns** (join on **`_scenario_row_key`** to `full_results_df` / `full_table_rank`). All use UTF-8-SIG where applicable; **`_scenario_row_key`** is omitted from CSV.
- **Results filters** (**Rank results by** — one of six `RECOMMENDED_WINNER_PRESETS` labels; scenario type; **tariff family** — all types / Standard / Weekend saver / Flat rate; hard filters) are in the **sidebar** after a successful run. **Rank results by** drives consolidated-table ordering, multi-criteria snapshot cards, and the **`winner_preset`** passed into `build_recommended_setups_summary_df` (same label → preset id). Hard filters can include **export ratio max** (% of PV generation), **annual electricity bill reduction min (%)**, annual bill max (€), and others (`_apply_hard_filters_to_results_df`).
- **Stale vs current setup:** If the frozen “last run” snapshot differs from the current form while the setup panel is **collapsed**, the app may show a warning; while **Edit assumptions and rerun** is **open**, that warning is suppressed (editing before re-run is expected).
- **Rank results by → Lowest annual electricity cost** sorts on **Annual electricity bill (€)** in the **Full results** grid: full year-1 total including recurring charges and OPEX (same value as annual electricity cost; see §7). Extra energy-flow and carbon columns (grid import kWh, export kWh, self-consumed PV, grid import cost, emissions, lifetime CO₂, export ratio) live in that grid, not necessarily on the KPI tile strip.

### 10.1 Consolidated row identity (`_scenario_row_key`)

`build_full_scenario_results_df` adds **`_scenario_row_key`**: internal tariff column id (`tcol` from profiles) + **Scenario** + integer **PV (kWp)** + integer **Battery (kWh)**, with **Grid only** rows always encoded as PV **0** / battery **0** (`compose_scenario_row_key`). Keys are expected unique; duplicates emit a **`UserWarning`**. Used for Ag Grid selection reconciliation and trade-off “selected” highlighting (string key when the column is present; legacy tuple fallback otherwise). Not user-facing and not exported in scenario CSVs.

**Recommended setups** reuses the same key formula for rows with real sizing (`augment_recommended_df_with_scenario_row_keys`, mapping **Tariff** display name → `tcol` via profiles). Infeasible / empty-sizing rows use **`RECOMMENDED_NO_SIZING_KEY_PREFIX`** + `tcol` + separator + scenario family label so they never match a consolidated row.

### 10.2 Shared detail block (`render_consolidated_selection_detail_block`)

Single implementation for **Recommended setups** row click and **Full results** row click. Each call passes a distinct **`plotly_chart_key_prefix`** (`rec_detail`, `explorer_detail`) so Streamlit Plotly widgets do not duplicate element ids across tabs.

Flow: **Viewing details for** caption → optional **Pin for compare A/B** (`st.session_state.compare_scenario_key_a|b` stores `_scenario_row_key` strings) → optional **`_dataframe_compare_two_scenario_rows`** expander when both pins resolve in `full_table_rank` → rank-under-goal caption → **`_render_decision_kpi_through_charts_for_consolidated_row`** (**A.** economic tiles including **per € CAPEX** ratios for NPV / annual savings / gross savings; **B.** community tiles including CO₂ **per € CAPEX** ratios and **Export ratio (% of PV gen)** where applicable) → trade-off expander → optional styled comparison table (`show_filtered_scenario_comparison`) → optional cumulative outlook expander (`show_cumulative_outlook`).

Ag Grid **`selected_rows`** may be a **list** or a **pandas DataFrame** depending on `st_aggrid` version; **`_reconcile_aggrid_row_selection`** normalizes without using `if df` boolean checks on frames.

### 10.3 Sidebar layout (post-run)

Streamlit renders **`st.sidebar`** in script order. After a completed run:

1. **`render_sidebar_postrun_filters()`** — **Results filters** (rank goal, scenario type, tariff family, decision constraints, reset buttons).
2. Later in the same run, after `full_table_rank` / `ranked` exist: **`render_sidebar_compare_pins_summary(full_table_rank)`** — heading **Compare scenarios**, captions **A** / **B** resolved via `_results_scenario_label` from `full_table_rank`, or **(not in current table)** if the pin key has no row; **Clear compare pins**.
3. **`render_sidebar_performance_panel()`** — only if **`REC_FEASIBILITY_PROFILE`** is `1` / `true` / `yes` / `on` (see §10.4).

So visible order is **filters → compare summary → performance (optional)**. Compare **pin buttons** remain in **`render_consolidated_selection_detail_block`**; the sidebar does not duplicate those controls.

### 10.4 Performance profiling (optional)

**Env:** `REC_FEASIBILITY_PROFILE` — truthy when trimmed lower-case value is one of `1`, `true`, `yes`, `on` (`_perf_profiling_enabled()`).

**Helpers:** `_perf_record` appends `(name, seconds)` to `st.session_state["_perf_log"]` (trimmed to last 80 entries). Section totals stored on session state: `_perf_load_data_s`, `_perf_optimizer_total_s`, `_perf_optimizer_by_tariff` (dict `tcol → seconds` per tariff optimize), `_perf_build_full_s`, `_perf_all_tariffs_eval_s`.

**Reset:** On **Run analysis** when profiling is on, `_perf_log` is cleared and `_perf_optimizer_by_tariff` dropped so a new run’s timings replace old ones.

**UI:** `render_sidebar_performance_panel()` — table of section labels + seconds; up to eight slowest per-tariff optimizer rows; optional expander **Raw perf log**. If no timings yet, a short caption.

### 10.5 Full results — core display columns

`_scenario_explorer_core_display_columns(lifetime_years)` returns the allowlist passed to `_subset_dataframe_display_columns` when the user selects **Core columns**. It includes **`_scenario_row_key`** (internal), **Tariff**, **Scenario**, **PV (kWp)**, **Battery (kWh)**, annual bill, savings, bill reduction %, payback, NPV, IRR, CAPEX, annual CO₂ reduction, CO₂ reduction %, self-sufficiency, self-consumption ratio, export ratio, total annual PV generation, grid import kWh, battery charge/discharge, annual grid import cost, plus **`per_capex_ratio_column_names(ly)`** (NPV / CO₂ / savings per € CAPEX) — **only columns present** in the frame are kept.

---

## 11. Saved-run bundle (ZIP)

**Source module:** `saved_run_bundle.py` (tests: `tests/test_saved_run_bundle.py`). The Streamlit layer only wires UI + `st.session_state`; do not review export/import correctness from `app.py` alone.

**Purpose:** Serialize a **completed** optimization (inputs + `prepared_df` + `opt_dfs` + frozen `last_run` / profiles + optional `full_results_df`) so another session can **restore** without calling `optimize()`.

**Import hydrate:** `_apply_imported_run_bundle_payload` always recomputes **`full_results_df`** via **`build_full_scenario_results_df`** from imported `opt_dfs` + `prepared_df` + tariff profiles and frozen economics so consolidated KPIs (CO₂, per-€-CAPEX ratios, etc.) stay consistent; the Parquet snapshot of `full_results_df` in the ZIP is not used as the sole source of truth on restore.

**UI:** **Run your own analysis** (and **Settings & App guide**) → expander **Saved run — export / import (.zip)** — download after a successful run; **Restore** hydrates `st.session_state`. Before the first run, a second **Restore** block may appear in the compact preface expander; **`render_saved_run_import_controls`** takes a **`widget_key_suffix`** (`_preface` vs `_bundle`) so upload/button Streamlit keys do not collide. In-app: **warning** that restore replaces the current tab session; after file pick, **`read_manifest_from_zip`** shows **preview** (export time, `app_version`, schema, `has_full_results`); restore failures use **`_format_saved_run_import_user_message`** (short user text + optional **Technical details** expander with the raw exception). **`_saved_run_import_ok`** success toast is popped only in the `_bundle` instance so it is not consumed twice.

**Layout (inside the ZIP):**

| Path | Role |
|------|------|
| `manifest.json` | `schema_version`, `export_timestamp_utc`, `app_version`, `has_full_results`, `opt_tariff_columns`, `file_sha256` map |
| `state/last_run.json` | JSON-serializable last-run snapshot (assumptions, hashes, optimizer config, etc.) |
| `state/tariff_profiles.json` | Tariff profile list used for that run |
| `inputs/consumption.csv`, `inputs/pv.csv` | Raw input bytes |
| `inputs/tariffs.csv` | Optional; present only if a tariff CSV was part of the run |
| `frames/prepared_df.parquet` | Prepared hourly dataframe |
| `frames/full_results_df.parquet` | Optional consolidated results (if exported with full results) |
| `opt/opt__<sanitized_tcol>.parquet` | One Parquet per tariff column in `opt_dfs` |

**Formats:** JSON + Parquet only (**no pickle**). Dependency: **PyArrow** for Parquet I/O (`requirements.txt`).

**Import validation (`saved_run_bundle.load_bundle_from_zip`):**

1. Reject ZIP members with unsafe paths (absolute, empty segments, `.` / `..`).
2. Allow only names under the known prefixes / patterns (coarse filter), then require the **exact** member set implied by the manifest (including optional `inputs/tariffs.csv` when listed in `file_sha256`).
3. Enforce **maximum uncompressed size** per member and **total** uncompressed size (ZIP-bomb style limits; constants in `saved_run_bundle.py`).
4. Check `schema_version` against `BUNDLE_SCHEMA_VERSION`.
5. Verify **SHA-256** for each file against `manifest["file_sha256"]` when hashes are present.

**`app_version`:** Intended as a **build fingerprint**, not only a product label. Resolution order: **`REC_FEASIBILITY_BUILD_ID`** env → `git rev-parse --short HEAD` (labelled `rec-feasibility-app git:<sha>`) → fallback `rec-feasibility-app (unknown build)`.
