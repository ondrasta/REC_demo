# REC Financial Feasibility Web App — Design Document

## 1. Overview

Streamlit web application for analyzing financial feasibility of Renewable Energy Communities (REC), comparing four scenarios and running optimal system sizing based on configurable goals.

---

## 2. Data Requirements

### 2.1 Input CSVs

| File | Purpose | Expected structure |
|------|---------|-------------------|
| **Consumption** | REC hourly demand (kWh) | `date` (DD/MM/YYYY HH:00), consumption column |
| **PV profile** | PVGIS hourly `P` in Wh for 1 kWp (parsed to kWh/kWp by dividing by 1000) | Same timestamps as consumption, scalable column |

---

## 3. Tariff Structure

### 3.1 Time Bands

- **Peak:** 17:00–19:00  
- **Night:** 23:00–08:00  
- **Day:** 08:00–17:00, 19:00–23:00  

### 3.2 Default Tariffs (Immutable)

Defaults are in `DEFAULT_TARIFFS` and `DEFAULT_EXPORT_RATE`. The app never mutates these. Override mode builds a separate config via `get_active_tariff_config()`.

---

## 4. Four Scenarios

| Scenario | Description |
|----------|-------------|
| **1. Grid only** | All demand from grid. Baseline. |
| **2. PV + Grid** | PV offsets demand; surplus exported. |
| **3. PV + Battery + Grid** | PV + battery arbitrage + export. |
| **4. Battery + Grid** | No PV; battery charges on low tariffs, discharges at peak. |

---

## 5. Battery Model

### 5.1 Round-trip Efficiency (Symmetric)

- `charge_eff = sqrt(round_trip_eff)`
- `discharge_eff = sqrt(round_trip_eff)`
- **Charging:** Energy drawn from source `E` → SOC increases by `E * charge_eff`
- **Discharging:** Load served `D` limited by `soc * discharge_eff`; SOC decreases by `D / discharge_eff`

### 5.2 Default Dispatch Rule

**Peak only** — battery discharges during peak (17:00–19:00) to reduce grid import. User can select "Day+Peak" if desired.

### 5.3 Initial SOC caveat

With `init_soc > 0`, the model injects starting energy that can be discharged but has no balancing rule (battery need not end at the same SOC). That effectively adds "free" energy and can bias annual economics. With default `init_soc = 0` this is not an issue.

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

| Metric | Formula |
|--------|---------|
| Annual electricity cost (year 1) | Σ(GridImport × Tariff) − Σ(Export × ExportRate) + standing_charge + OPEX |
| Annual savings (year 1) | Baseline cost − Scenario cost |
| Gross savings over 20 years | Σ savings_t where savings_t = savings₁ × (1 + inflation)^(t−1) |
| Net benefit over 20 years | Gross savings over 20 years − CAPEX − nominal replacement costs |
| Simple payback | CAPEX / Annual savings (year 1) |
| NPV (20 years) | −CAPEX + Σ(savings_t / (1+r)^t), t=1..20, savings_t inflated |
| IRR (20 years) | Discount rate r where NPV = 0 (solved numerically). Shown as %. |

**Electricity inflation:** An optional input (default 3.3% in the UI) escalates each year: electricity costs (baseline + scenario), export income, standing charge, **PSO levy** (annual, same for all tariffs), and OPEX. CAPEX is not inflated. With inflation rate *i*, savings in year *t* = savings₁ × (1+i)^(t−1).

**Replacements (optional):**
- Battery replacement year + replacement cost (% of battery CAPEX)
- Inverter replacement year + replacement cost (% of PV CAPEX)
- Replacement costs are **not inflated**
- Replacement costs are discounted in NPV/IRR
- Net benefit over 20 years subtracts replacement costs in nominal terms

OPEX = CAPEX × (opex_pct / 100) (default **1%** of scenario CAPEX in the UI). Standing charge and **PSO levy** (default **€19.10**/year in the UI) apply to both baseline and scenarios. OPEX applies only where there is CAPEX (baseline Grid only has zero CAPEX and therefore zero OPEX); savings net them out.

---

## 8. CO₂ Metrics

| Metric | Formula |
|--------|---------|
| Grid CO₂ | Σ(GridImport × 0.35 kg/kWh) |
| CO₂ savings | CO2_Grid − CO2_Scenario (clamped ≥ 0) |

---

## 9. Optimizer

- Respects `pv_min`, `pv_max`, `batt_min`, `batt_max`, `pv_step`, `batt_step`
- PV + Battery + Grid loop: if `batt_min` > 0 start from `batt_min`, else from `batt_step` to avoid duplicating PV + Grid cases
- Uses same finance logic as displayed KPIs (standing charge, OPEX, export rate)
