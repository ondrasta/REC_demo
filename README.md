# REC Financial Feasibility Analyzer

Analyzes financial feasibility of a Renewable Energy Community (REC) with:

- **Four scenarios:** Grid only, PV + Grid, Battery + Grid, PV + Battery + Grid
- **Three tariffs:** Standard, Weekend Saver, Flat
- **Optimizer:** Finds best PV (0–150 kWp) and battery (0–300 kWh) sizes for multiple goals

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

3. Upload your consumption and PV timeseries CSVs in the sidebar and click **Run analysis**.

---

## Input files

| File | Description |
|------|-------------|
| **Consumption CSV** | Hourly community consumption (column `Final_Community_Sum`). Date format: `DD/MM/YYYY HH:00` |
| **PV production CSV** | PVGIS timeseries: `time` column `YYYYMMDD:HH11`, `P` column in Wh for 1 kWp |

---

## Run setup (sidebar)

- **Costs:** PV CAPEX (€/kWp), Battery CAPEX (€/kWh), standing charge per tariff, **PSO levy (annual €)**, OPEX (% of CAPEX), electricity inflation rate (% per year), one-time replacement assumptions (battery/inverter year and % CAPEX)
- **Optimizer range:** PV and battery size bounds used by the optimizer
- **Battery dispatch:** Peak only (default) or Day+Peak
- **Tariff overrides:** Optional tariff/export-rate edits for the next run

## Results controls (Results tab)

- **Rank results by:** Lowest bill, highest savings, best payback, best self-sufficiency, most CO₂ savings, best NPV, best IRR, and other ranking goals
- **Scenario type:** Post-run filter for which scenario family is eligible (All scenarios, Grid only, PV + Grid, PV + Battery + Grid, Battery + Grid)
- **Show ranked result:** Best, 2nd best, or 3rd best within that filtered set (no rerun)
- **All scenario results (Ag Grid):** Excel-style column filters on the full shortlist; CSV export; **All tariffs** comparison chart follows **Rank results by**

---

## Tests

From the `rec-feasibility-app` directory:

```bash
pip install pytest
python -m pytest -v
```

(Pytest discovers `tests/test_app.py` by default.)

---

## Notebook / Colab

For a notebook-based workflow, see `REC_Feasibility_Analysis.ipynb` if present.
