# REC Financial Feasibility Analyzer

Analyzes financial feasibility of a Renewable Energy Community (REC) with:

- **Scenarios:** Grid only, PV + Grid, Battery + Grid, PV + Battery + Grid (`ENABLE_BATTERY_UI = True` in this app)
- **Tariffs:** Standard / Weekend Saver / Flat families; a **scrollable tariff matrix** (default **`data/default_tariffs.csv`** or override — see below) with one row per supplier × type, **Include** checkboxes, and optional CSV upload. Only **checked** rows are passed to the optimizer; each row has its own standing charge and export rate
- **Optimizer:** Searches PV sizes (default 5–60 kWp) and battery sizes (default 0–40 kWh). **Speed preset** sets grid steps: **Quick** 5 kWp / 5 kWh, **Fast** 10 / 10, **Full** 1 / 1 (finest, slowest). **Default preset = Full (1 / 1)**
- **Financial horizon:** **Analysis horizon (years)** in **Financial assumptions** (allowed **15–30**, default **20**) drives NPV, IRR, gross/net savings labels, cumulative outlook charts, lifetime grid-CO₂ scaling (annual × horizon), and the upper bound for optional replacement years

Terminology: the **year‑1 all‑in electricity total** is **import − export income + standing charge + PSO levy + OPEX**. In the **Full results** consolidated AgGrid this appears as **Annual electricity bill (€)**; elsewhere (detail charts and the app guide) the same quantity may still be labelled **Annual electricity cost (€)**. The consolidated grid also includes **Annual electricity bill reduction (%)** — year‑1 savings vs that tariff’s **grid-only** annual bill. The same table adds **per € CAPEX** ratios (NPV, annual CO₂ reduction, lifetime CO₂ avoided vs grid-only, year‑1 annual savings, gross lifetime savings — each ÷ **CAPEX (€)**; **—** when CAPEX is 0). See **Settings & App guide** in the app for full definitions.

**Bundled demo runs:** Demo ZIPs under `assets/saved_runs/` are **off by default** (`ENABLE_EMBEDDED_SAVED_RUNS = False` in `app.py`). Use **Run analysis** or **Restore saved run** with your own bundle; enable embedded demos in code only if you need them.

---

## Run locally

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

3. Open the **Run your own analysis** tab and use **Model setup** to upload your consumption and PV CSVs, set assumptions, and click **Run analysis**. After a successful run, **Model setup is hidden by default**; open **Run your own analysis** again and click **Edit assumptions and rerun** to reopen it (and run again if needed).

**Optional — performance profiling (developers):** set **`REC_FEASIBILITY_PROFILE`** to **`1`**, **`true`**, **`yes`**, or **`on`** (case-insensitive) in the environment before `streamlit run`. On each new **Run analysis**, profiling counters reset. After load/optimizer/build/evaluation, the **sidebar** shows **Performance (debug)** with seconds for: **Load & prepare data**, **Optimizer (all tariffs)** (plus up to eight slowest per-tariff optimize slices), **`build_full_scenario_results_df`** (when that table is rebuilt), and the **All-tariffs evaluate_for_tariff loop** (post-build evaluation pass used for grouped tariff comparison charts). An expander holds a short **raw perf log**. If the env var is unset, no extra sidebar block appears.

**Sidebar order after a successful run (top → bottom):** **Results filters** (rank goal, scenario type, tariff family, decision constraints) → **Compare scenarios** (always-visible **A** / **B** labels for pinned rows, **Clear compare pins**) → **Performance (debug)** (only when profiling is enabled). Pin actions stay in the main **detail** area under each grid; the sidebar only mirrors labels so pin state stays visible while scrolling.

---

## Input files

| File | Description |
|------|-------------|
| **Consumption CSV** | Hourly community consumption (column `Final_Community_Sum`). Date format: `DD/MM/YYYY HH:00` |
| **PV production CSV** | PVGIS timeseries: `time` column `YYYYMMDD:HH11`, `P` column in Wh for 1 kWp |
| **Tariffs CSV** (optional) | One row per variant; see [Tariff CSV format](#tariff-csv-format). Defaults load automatically from bundled **`data/default_tariffs.csv`** unless overridden |

**Calendar year:** If the parsed timestamps include any hour in **2020**, the loaders **keep only 2020** (consumption and PV parsers). Multi-year files are therefore truncated to that year without a separate prompt—use a single-year file or strip other years upstream if you need a different period.

### Built-in sample data (no upload required)

If you leave the file uploaders empty, the app reads:

- `REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV` (if set) / `REC_FEASIBILITY_DEFAULT_PV_CSV` (if set)
- otherwise `data/local_consumption.csv` / `data/local_pv.csv` (if present)
- otherwise bundled `data/default_consumption.csv` / `data/default_pv.csv`

These use the **same parsing and validation** as uploaded files (`resolve_consumption_csv_bytes` / `resolve_pv_csv_bytes` → `load_and_prepare_data`). **Uploading a file replaces** the corresponding built-in default for that run.

**Default tariff table (no upload required):** the app loads tariff rows from (in order):

1. **`REC_FEASIBILITY_DEFAULT_TARIFFS_CSV`** (full path), if set and the file exists  
2. Else **`data/local_tariffs.csv`**, if present  
3. Else bundled **`data/default_tariffs.csv`**

The **Data & tariffs** expander shows a **scrollable matrix**: **Type** (e.g. `standard`, `weekend_saver`, `flat_rate`), supplier name, standing charge, export rate, and a short rate summary. Tick **Include** for each row you want in **Run analysis**; unchecked rows are skipped. **Select all** / **Select none** set every checkbox.

**Optional upload:** **Upload tariff rates CSV** → **Load tariffs from CSV** replaces the matrix from your file (same parser as defaults). Blank trailing rows are ignored. Rows are listed in **family order** in the UI: all **standard**, then **weekend saver**, then **flat** (not necessarily the same order as in the file).

---

## Model setup (main page)

On the **Run your own analysis** tab, before the first run, setup is a **compact two-column block** (data uploads | core costs), then **three expanders only**: **Data & tariffs**, **Financial assumptions**, **Advanced / optimizer settings** (battery model if enabled + search bounds), and then the **Run analysis / Stop run** row. Long help text is **not** above the form: use **How to use this app** (expander) or the **Settings & App guide** tab after a run.

After a successful run, the **whole** setup is **hidden by default** so results stay primary; open **Run your own analysis** and click **Edit assumptions and rerun** to reveal the form.

The page header uses a **short banner strip** (cropped) from `assets/banners/banner_rec_residential_02.png` when available. Legacy header logos were removed.

- **Core row:** PV CAPEX, battery CAPEX, PSO, OPEX, discount rate, electricity inflation, **Grid CO₂ factor (kg/kWh)** — default **0.2462**; scales grid-import CO₂ and CO₂ savings vs grid-only
- **Data & tariffs:** **Tariff matrix** — default file or upload, **Include** checkboxes (optimizer runs only on checked rows), standing charge and **export rate** per row; optional **tariff CSV** upload to replace the table
- **Financial assumptions:** **Analysis horizon (years)** — **15–30** (default **20**). This sets how many years feed **NPV**, **IRR**, gross/net savings, and the cumulative charts, and caps the valid years for optional **battery / inverter replacements**
- **Advanced / optimizer:** PV and battery search bounds (battery range slider **step 5** kWh — matches **Quick**’s **5** kWh optimizer step; **Fast** uses **10** kWh steps, **Full** uses **1** kWh), **speed preset** — **Quick** (PV **5** kWp & battery **5** kWh steps), **Fast** (**10** & **10**), **Full** (**1** & **1**, finest search, **default**); plus (if battery UI is on) battery model sliders (efficiency, DoD, SOC limits, C-rate, **charge from PV** and **charge from grid at night** — **night charging default off**), and **Battery discharging schedule**:
  - **Peak only** — discharge to the load only **17:00–19:00** (aligned with the model’s peak import band).
  - **Day+Peak** — **17:00–19:00**, then **19:00–23:00** if energy remains above the SOC floor (no discharge **08:00–17:00** or **23:00–08:00**). The hourly loop runs in time order, so peak is used before the evening band each day.

### Battery dispatch (reference)

| Setting | When the battery may discharge to serve load |
|--------|-----------------------------------------------|
| **Peak only** | Hours **17:00–19:00** only (both **Battery + Grid** and **PV + Battery + Grid**). |
| **Day+Peak** | **17:00–19:00** and **19:00–23:00** only. |

Charging: optional **from PV surplus** and/or **from grid during night** (23:00–08:00), subject to SOC and C-rate limits — see **Advanced / optimizer settings** in the app.

## Results filters (sidebar, after first run)

These do **not** rerun the optimizer; they only change how existing results are ranked and filtered.

- **Rank results by:** Chooses one of six **recommendation-style** goals (same labels as the **Recommendation preset** family): **Balanced recommendation**, **Best financial value**, **Lowest annual bill**, **Fastest payback**, **Highest CO₂ saving**, **Highest self-consumption**. It **reorders** the **Full results** consolidated table (and drives rank captions / multi-criteria cards) using the same ordering rules as in the in-app help — e.g. **Lowest annual bill** sorts on **Annual electricity bill (€)**; **Balanced recommendation** prioritises **NPV (€)** then **CO₂ savings (kg)** among other tie-breaks (see **Settings & App guide**). It does **not** rerun the optimizer.
- **Tariff family:** **All tariff types**, or only **Standard**, **Weekend saver**, or **Flat rate** (every named variant in that family stays included)
- **Scenario type:** In this app (`ENABLE_BATTERY_UI = True`), **All scenarios** includes Grid only, PV + Grid, Battery + Grid, and PV + Battery + Grid.
- **Decision constraints (hard filters):** Optional post-run filters before ranking — e.g. max payback, max CAPEX, min NPV/IRR, min self-sufficiency / self-consumption, **max export ratio (% of PV generation)**, min CO₂ reduction (%), min **annual electricity bill reduction (%)** vs grid-only, max **annual electricity bill (€)** (same cap as “max annual electricity cost” in logic), etc. After each successful **Run analysis** or **saved-run restore**, the app **re-applies the recommended default** constraint set (same defaults as **Reset filters** for thresholds — no extra click).
- **Results tabs (left → right):** **Recommended setups** — **multi-criteria snapshot** (lowest annual bill, highest CO₂ savings, top row for **Rank results by**) at the top, then an Ag Grid (see [below](#recommended-setups-tab)); **click a row** for the full detail block (**KPIs/charts + trade-offs + filtered-set comparison + cumulative outlook + All tariffs — comparison**) with an independent selection key; **Full results** — filtered Ag Grid (**Core columns** or **All columns**) over the consolidated table; **click a row** for a reduced detail block (**KPIs/charts + trade-offs only**; no filtered-set comparison table or cumulative outlook), with independent row selection; **Consumption patterns**; **Production patterns** (PV yield per kWp from **`pv_per_kwp`**); **Run your own analysis** — **Model setup**, **Run analysis**, and **Saved run** export/import; **Settings & App guide**
- **Sidebar (below Results filters):** **Compare scenarios** shows resolved labels for **A** and **B** (from **Pin for compare** under a detail block), or **—** when empty; pinned keys missing from the current table show **(not in current table)**. **Performance (debug)** appears only when **`REC_FEASIBILITY_PROFILE`** is enabled — see [Run locally](#run-locally).
- **Stale setup:** If **Model setup** is collapsed and something no longer matches the last completed run, a warning may appear; it is suppressed while **Edit assumptions and rerun** is open so editing before **Run analysis** does not look like an error

### KPI detail: recommended vs full results

Two **independent** ways to drive the KPI + chart detail stack (no cross-tab sync):

1. **Recommended setups** — **click a row** in that tab’s Ag Grid (separate internal selection). The tab also shows the **multi-criteria snapshot** cards at the top (same filtered universe as ranking).  
2. **Full results** — **click a row** in the consolidated Ag Grid (selection stored internally).

For **Recommended setups** and **Full results**, the shared detail stack starts with **Viewing details for:** `Tariff · Scenario · PV · Battery`, then optional **Pin for compare A / B** and **Clear compare pins** (the **sidebar** repeats **A** / **B** labels — same state). When two different scenarios are pinned and both rows still exist in the consolidated table, **Compare pinned scenarios (A vs B)** shows a side-by-side metric table. **KPI tiles:** **A. Economic impact** includes NPV, IRR, payback, bill, savings, and **per € CAPEX** ratios (NPV, annual savings, gross lifetime savings). **B. Community impact** includes emissions, CO₂ savings/reduction, self-sufficiency / self-consumption, export ratio, energy flows, and **per € CAPEX** CO₂ ratios (annual reduction; lifetime avoided vs grid-only at the analysis horizon). After that: **Recommended setups** keeps the full block (trade-off + filtered-set comparison + cumulative outlook + **All tariffs — comparison**), while **Full results** keeps **trade-off only** beyond KPI/charts.

In **Full results**, **Table columns** defaults to **Core columns** (a compact subset — tariff, scenario, sizes, bill, savings, payback, NPV, IRR, CAPEX, key CO₂ and energy columns, import cost, battery charge/discharge, and the **per € CAPEX** ratio columns). Choose **All columns** for every consolidated column. The app **drops non-core columns before** building the Ag Grid, so in **Core columns** mode the grid’s **column filters and sorts only see the kept columns** — you cannot filter or sort by a metric that was omitted from the frame. **All columns** passes the full table through to the grid (same row universe as ranking; default row order still follows **Rank results by** on the full consolidated rows before any column subset).

The **Full results** AgGrid lists the **same filtered universe** as ranking and includes **additional columns** (visible when **All columns** is selected), for example: **Annual grid import cost (€)**, **Grid import (kWh)**, **Battery charge (kWh)**, **Battery discharge (kWh)**, **Export to grid (kWh)**, **Self-consumed PV (kWh)**, **Export ratio (% of PV gen)**, **Annual grid CO₂ emissions (kg)** (operational emissions from grid imports — not the same as savings vs grid-only), **Lifetime CO₂ (kg)** (annual emissions × analysis horizon, constant-year shortcut), and **Annual CO₂ reduction (kg)** (year‑1 avoided CO₂ vs grid-only for the same tariff). **Self-consumption ratio (%)** and **Self-sufficiency (%)** in the table are the same concepts people sometimes call SCR / SSR; duplicate SCR/SSR columns were **removed** from the grid to avoid two names for one number.

For older restored runs, battery throughput columns may be empty/zero if those fields were not part of the stored optimizer table in that historical snapshot. Re-run analysis to populate them from current logic.

**Recommended setups** shows **one row per tariff × scenario family**: first the **feasible set** from the tab’s **Decision constraints** (and the same **Scenario type** / **Tariff family** filters as elsewhere), then **one winner per cell** chosen by the current **Rank results by** selection (same **preset** semantics as in the app — lexicographic ordering on optimizer KPIs with a final **lowest CAPEX** tie-break; not a single fixed “NPV only” rule for every preset). Fewer columns than **Full results**. Row click resolves to the matching row in the **full consolidated table** for charts when PV/battery sizing exists; rows with **no feasible sizing** show the tab’s **Note** text instead of full KPIs.

Explorer and recommended Ag Grids each carry an **internal-only** stable row id (hidden in the UI) so selection survives filters and sorts.

**Removed from the explorer grid only** (not from calculations): a second money column that duplicated the bill (**Annual electricity cost (€)**), mirror acronym columns (**SCR (%)**, **SSR (%)**), and a duplicate carbon column (**CO₂ savings (kg)** alongside **Annual CO₂ reduction (kg)** — same values). Optimizer rows and **Recommended setups** tie-breaks still use **CO₂ savings (kg)** where noted below.

**CSV downloads (Full results, Recommended setups):** UTF-8 with BOM. You can download **results only** (a single rectangular table) or **results + assumptions** (a **`Setting`,`Value`** block from the **last completed Run analysis**, plus tab-specific rows where applicable — Recommended constraints — then **one blank line**, then the results table). **Recommended setups** full-KPI exports join each recommended row to the consolidated scenario table (same columns as explorer). **Full results** and **Recommended setups** omit the internal row-key column from CSV. The combined layout is easy to read in Excel; analysts often prefer **results only** for re-use in other tools.

---

## Recommended setups (tab)

This is the **first** results tab after a run. It uses the **same optimizer grid** as the last **Run analysis** (`opt_dfs`) — **no second optimizer pass**. Changing Model setup and **Run analysis** again refreshes both results and this tab.

1. **Feasible set:** For each **tariff** × **scenario family** (PV + Grid; and if battery UI is on, PV + Battery + Grid, Battery + Grid), keep optimizer rows that satisfy **all** enabled constraints in the sidebar (max payback, optional **NPV min (€)**, optional **CO₂ reduction min (%)** vs grid-only, min self-consumption, max export ratio — same semantics as **Full results** hard filters; export ratio = **annual export to grid ÷ annual PV generation × 100** when annual PV generation > 0, else **0**).
2. **Chosen row (per cell):** Among feasible points for that cell, pick **one** winner using the **same preset** as sidebar **Rank results by** (e.g. **Balanced recommendation** → NPV-led lexicographic order; **Lowest annual bill** → bill-led; **Highest self-consumption** → SCR-led — see **Settings & App guide** and the sidebar **How Rank results by works** expander). Final tie-break is **lowest CAPEX (€)**.

At the top, **multi-criteria snapshot** cards (top pick, lowest bill, highest CO₂) are derived from the **same Recommended grid rows** as the table so they stay consistent. Table KPI numbers are injected from the **canonical consolidated** table (`build_full_scenario_results_df`) by `_scenario_row_key` where possible.

The summary is shown in an **Ag Grid** (filters/sorts like Excel). **Click a row** to load KPI tiles + scenario charts + **Advanced trade-off** expander + styled **scenario comparison** table + **cumulative** expander (full detail block) — backed by the **full consolidated results** row when sizing exists. **Selection is not synced** with **Full results**. Rows with **no feasible sizing** only show the **Note** column in the detail area (no full KPI block).

If battery UI is on **and** you have completed **Run analysis** at least once (`opt_dfs` present), the table includes **Battery charging (last run)** — **PV only charging** vs **PV + night-grid charging** from **frozen** `last_battery_settings` for that run (not the live form before you rerun). Before the first run, that column is omitted.

Table **row order** follows sidebar **Rank results by** (via the consolidated ranking list): feasible rows appear **before** infeasible rows, ordered by how each recommended row’s `_scenario_row_key` ranks under the current goal. Changing **Rank results by** updates **Full results**, the multi-criteria snapshot, and this table together.

**Default thresholds** after a fresh run or restore (all adjustable; **Reset filters** / **Clear constraints** behave as labeled):

| Control | Default |
|--------|---------|
| Max payback (years) | **10** (`RECOMMENDED_SETUP_MAX_PAYBACK_YEARS`) |
| Min self-consumption ratio (%) | **80** |
| Max export ratio (% of PV gen) | **20** |
| NPV min (€) | **0** when enabled (floor **≥** value, aligned with Full results) |
| CO₂ reduction min (%) | **0** when enabled (floor **≥** value vs grid-only, aligned with Full results) |

**Battery default:** **Charge from grid at night** is **off** by default in Advanced; expert users can enable it for arbitrage-style behaviour.

**CSV:** Under the grid — **full KPI exports** match **Full results** consolidated columns (one row per recommended sizing; internal row key omitted; last column **Recommended setups note** explains infeasible / unmatched rows). Buttons: **all rows — full KPIs** and **all rows — full KPIs + assumptions** only.

Implementation: `build_recommended_setups_summary_df()` and `augment_recommended_df_with_scenario_row_keys()` in `app.py` (internal keys for Ag Grid selection + lookup into `build_full_scenario_results_df`).

---

## Saved run — export / import (`.zip`)

**Import (restore)** — On the **Run your own analysis** tab before any **Run analysis**, expand **Restore saved run (.zip) — no Run analysis needed** and upload a bundle, or use **Saved run — export / import (.zip)** lower on the same tab (also under **Settings & App guide**). The two blocks use the **same** restore workflow (separate Streamlit widget keys so both can appear on one page without duplicate-key errors).

**Export (download)** — After a successful **Run analysis**, open **Run your own analysis** → **Saved run — export / import (.zip)**, or use **Settings & App guide** for the same expander.

- **Download saved run (.zip)** packages a **full snapshot** of the completed run: `manifest.json` (schema version, checksums, metadata), JSON state (`last_run`, tariff profiles), original **input CSV bytes** (consumption, PV, optional tariffs), **Parquet** frames (`prepared_df`, per-tariff optimizer tables, optional `full_results_df`). **No pickle** — only JSON + Parquet inside the ZIP.
- **Restore saved run from upload** replaces the current session’s results and frozen “last run” inputs with the bundle contents. The app does **not** rerun `optimize()`. The **consolidated results table** is **rebuilt from** imported `opt_dfs` + `prepared_df` + frozen assumptions so KPI columns (including CO₂ and **per € CAPEX** ratios) stay consistent — the bundled `full_results_df` snapshot is not trusted alone. After you pick a **.zip**, the UI shows a **preview** (export time, build fingerprint, schema, whether full results are included) before you confirm **Restore**; failures map to short messages, with optional technical details.
- **Compatibility:** The manifest includes `schema_version` (currently **1**). Older or newer bundles with a mismatched schema are rejected with a clear error.

**Build fingerprint (`app_version` in the manifest):** By default the exporter records **`rec-feasibility-app git:<short SHA>`** when `git rev-parse --short HEAD` succeeds in the app directory, otherwise **`rec-feasibility-app (unknown build)`**. Override for CI or reproducible labels with environment variable **`REC_FEASIBILITY_BUILD_ID`** (any non-empty string wins).

**Import safety (defense in depth):** Loaders reject members with unsafe paths (e.g. `..` segments), enforce an **exact file list** matching the manifest (no extra junk files), cap **per-member** and **total** uncompressed sizes (see `saved_run_bundle.py`), and verify **SHA-256** entries from the manifest when present.

**Where to review the implementation:** All bundle I/O is in **`saved_run_bundle.py`** ( **`app.py`** only calls into it and renders the UI). Round-trip and safety checks are covered by **`tests/test_saved_run_bundle.py`** — include those two paths (plus `requirements.txt` for **pyarrow**) when exporting or reviewing only part of the repo.

---

## Tariff CSV format

One row = one tariff variant (one row in the matrix after load). Hourly import columns in the prepared dataframe are named **`tariff_sel_0`**, **`tariff_sel_1`**, … in UI display order (see below).

Required columns (canonical):
- `family` (`standard` | `weekend` | `flat`)
- `variant` (company/label shown in results)

Optional columns:
- `standing_charge` (€/year)
- `export_rate` (€/kWh)

Family-specific import columns:
- `standard`: `standard_day`, `standard_peak`, `standard_night`
- `weekend`: `weekend_weekday_day`, `weekend_weekday_peak`, `weekend_weekday_night`, `weekend_weekend_day`, `weekend_weekend_peak`, `weekend_weekend_night`
- `flat`: `flat_rate` (or `rate`; or `weekday_day` when all bands match)

Also supported aliases (simplified schema):
- `tariff_type` instead of `family`
- `company` instead of `variant`
- `standing_charge (EUR/year)` instead of `standing_charge`
- `export_rate (EUR/kWh)` instead of `export_rate`
- `weekday_day`, `weekday_peak`, `weekday_night`
- `weekend_day`, `weekend_peak`, `weekend_night`

Type aliases:
- `weekend_saver` → `weekend`
- `flat_rate` → `flat`

**After load:** the matrix lists rows as **all standard variants**, then **all weekend saver**, then **all flat** (up to **20** per family). Use **Include** checkboxes to choose which rows participate in **Run analysis**; if the bundled default fails to parse, the app falls back to three built-in **Market average** profiles.

---

## Developer documentation

**`DESIGN.md`** in this folder covers architecture, tariff and battery behaviour, financial and CO₂ metrics, saved-run bundles, and UI implementation notes (including **sidebar order** after a run, **`REC_FEASIBILITY_PROFILE`**, **Full results** core vs all columns, and **compare pins**).

---

## Tests

From the `rec-feasibility-app` directory:

```bash
pip install pytest
python -m pytest
```

Add `-v` for verbose output. Tests live under `tests/` (e.g. `tests/test_app.py`).

---

## Notebook / Colab

For a notebook-based workflow, see `REC_Feasibility_Analysis.ipynb` if present. It is a **simplified** tutorial (not a full port of the Streamlit finance stack); install **matplotlib** and **seaborn** separately if needed (`pip install matplotlib seaborn`).
