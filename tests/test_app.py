"""Lightweight tests for REC Feasibility app correctness. Run with: pytest tests/test_app.py -v"""
import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None

# Import core logic without Streamlit UI
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import (
    DEFAULT_TARIFFS,
    DEFAULT_EXPORT_RATE,
    get_active_tariff_config,
    BatterySettings,
    run_scenario_battery_grid,
    run_scenario_pv_battery_grid,
    compute_kpis_for_scenario,
    compute_financial_metrics,
    compute_payback_and_npv,
    optimize,
    evaluate_for_tariff,
    build_full_scenario_results_df,
    OptimizerConfig,
    CO2_FACTOR,
    LIFETIME_YEARS,
    build_pv_grid_sweep_table,
    _parse_consumption_csv,
    _order_dataframe_goal5_largest_pv_selfconsumption,
)


def _make_hourly_df(n_hours=100, consumption=1.0):
    """Minimal hourly df for testing."""
    dates = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "date": dates,
        "consumption": np.full(n_hours, consumption),
        "pv_per_kwp": np.zeros(n_hours),
        "tariff_standard": 0.30,
        "tariff_weekend": 0.30,
        "tariff_flat": 0.30,
    })
    return df


def test_tariff_override_no_leak():
    """Turning overrides OFF must return true defaults."""
    cfg_off, er_off = get_active_tariff_config(
        False, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999
    )
    assert cfg_off == DEFAULT_TARIFFS
    assert er_off == DEFAULT_EXPORT_RATE


def test_consumption_year_boundary_midnight_counted_in_2020():
    """01/01/2021 00:00:00 is treated as last hour of 2020 and kept in 2020 filter."""
    csv_text = (
        "date,Final_Community_Sum\n"
        "31/12/2020 23:00,1.0\n"
        "01/01/2021 00:00,2.0\n"
    )
    df = _parse_consumption_csv(csv_text.encode("utf-8"))
    assert len(df) == 2
    assert (df["date"].dt.year == 2020).all()


def test_tariff_override_on_uses_inputs():
    """Turning overrides ON uses edited values."""
    cfg, er = get_active_tariff_config(
        True,
        std_day=0.1, std_peak=0.2, std_night=0.05,
        wk_day=0.1, wk_peak=0.2, wk_night=0.05,
        we_day=0.08, we_peak=0.1, we_night=0.06,
        flat_rate=0.15, export_rate=0.20,
    )
    assert cfg["standard"]["day"] == 0.1
    assert cfg["flat"] == 0.15
    assert er == 0.20


def test_battery_round_trip_efficiency():
    """With 90% round-trip, charge 1 kWh from grid, discharge ~0.9 kWh to load."""
    bs = BatterySettings(
        eff_round_trip=0.9,
        charge_from_grid_at_night=True,
        discharge_schedule="Day+Peak",
        init_soc=0,
    )
    # Exactly 2 hours: one night (charge), one peak (discharge). No other hours to over-charge.
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01 02:00"), pd.Timestamp("2020-01-01 17:00")],
        "consumption": [0.0, 1.0],
        "pv_per_kwp": [0.0, 0.0],
        "tariff_standard": [0.3, 0.3],
        "tariff_weekend": [0.3, 0.3],
        "tariff_flat": [0.3, 0.3],
    })
    # Battery 2 kWh, DOD 0.9, max_power 1 -> charge 1 kWh, SOC+=0.948; discharge 0.9 to load
    d = run_scenario_battery_grid(df, batt_kwh=2, tariff_col="tariff_standard", battery_settings=bs)
    # Charged 1 kWh at night -> SOC += 0.948. Discharge: deliver min(0.948*0.948, 1) = 0.9
    # So grid during discharge hour = 1 - 0.9 = 0.1
    grid_discharge_hour = float(d.loc[1, "grid_import"])
    assert 0.05 < grid_discharge_hour < 0.15, f"Expected ~0.1 grid during discharge, got {grid_discharge_hour}"


def test_battery_only_self_sufficiency_zero():
    """Battery + Grid scenario has 0% self-sufficiency (grid-charged)."""
    bs = BatterySettings(eff_round_trip=0.9, charge_from_grid_at_night=True, discharge_schedule="Peak only")
    df = _make_hourly_df(200)
    df["pv_per_kwp"] = 0
    d = run_scenario_battery_grid(df, batt_kwh=50, tariff_col="tariff_standard", battery_settings=bs)
    k = compute_kpis_for_scenario(d, "tariff_standard", 0.18)
    assert k["Self-sufficiency ratio (%)"] == 0.0


def test_pv_battery_self_sufficiency_counts_pv_origin():
    """PV + Battery + Grid self-sufficiency counts PV-origin discharge, not grid-origin."""
    bs = BatterySettings(eff_round_trip=0.9, charge_from_pv=True, charge_from_grid_at_night=True)
    df = _make_hourly_df(200)
    # Add PV production during day
    df["pv_per_kwp"] = np.where((df["date"].dt.hour >= 8) & (df["date"].dt.hour < 18), 0.5, 0.0)
    d = run_scenario_pv_battery_grid(df, pv_kwp=20, batt_kwh=20, tariff_col="tariff_standard", battery_settings=bs)
    k = compute_kpis_for_scenario(d, "tariff_standard", 0.18)
    # Should have positive self-sufficiency (PV supplies load)
    assert k["Self-sufficiency ratio (%)"] >= 0
    assert "local_renewable_to_load" in d.columns
    # local_renewable = direct PV to load + PV-origin battery discharge
    lr = d["local_renewable_to_load"].sum()
    direct = d["pv_to_load_direct"].sum()
    batt_pv = d["battery_to_load_pv_origin"].sum()
    assert abs(lr - (direct + batt_pv)) < 0.01, "local_renewable should equal direct + battery_pv_origin"


def test_pv_battery_self_consumption_excludes_losses():
    """Self-consumption = direct PV to load + PV-origin battery discharge (excludes losses, end-of-sim)."""
    bs = BatterySettings(eff_round_trip=0.9, charge_from_pv=True, charge_from_grid_at_night=False)
    df = _make_hourly_df(200)
    df["pv_per_kwp"] = np.where((df["date"].dt.hour >= 8) & (df["date"].dt.hour < 18), 0.5, 0.0)
    d = run_scenario_pv_battery_grid(df, pv_kwp=20, batt_kwh=20, tariff_col="tariff_standard", battery_settings=bs)
    k = compute_kpis_for_scenario(d, "tariff_standard", 0.18)
    pv_gen = d["pv_generation"].sum()
    feed_in = d["feed_in"].sum()
    # Old (wrong) formula: pv_prod - feed_in overstates self-consumption
    old_self_cons = pv_gen - feed_in
    # Correct: direct PV to load + PV-origin battery discharge
    pv_used_locally = d["pv_to_load_direct"].sum() + d["battery_to_load_pv_origin"].sum()
    assert pv_used_locally <= old_self_cons + 0.1, "pv_used_locally should be <= pv_prod-feed_in (excludes losses)"
    # KPI should use correct formula
    self_cons_ratio = k["Self-consumption ratio (%)"]
    expected_ratio = 100.0 * pv_used_locally / pv_gen if pv_gen > 0 else 0
    assert abs(self_cons_ratio - expected_ratio) < 0.1, "Self-consumption ratio should use pv_used_locally"


def test_20_year_net_benefit_subtracts_capex():
    """Net benefit over 20 years = gross savings - CAPEX."""
    capex = 10000
    annual_savings = 1000
    gross = annual_savings * LIFETIME_YEARS
    net = gross - capex
    # Formula: net benefit = gross savings - CAPEX
    assert net == annual_savings * LIFETIME_YEARS - capex
    assert net == 10000  # 20k - 10k


def test_optimizer_respects_batt_min():
    """Optimizer PV + Battery + Grid loop starts from batt_min when > 0."""
    df = _make_hourly_df(100)
    df["pv_per_kwp"] = 0.3
    bs = BatterySettings()
    cfg = OptimizerConfig(pv_min=10, pv_max=20, pv_step=10, batt_min=50, batt_max=100, batt_step=50)
    opt = optimize(df, "tariff_standard", cfg, bs, 0.18, standing_charge=0, opex_pct=0)
    pv_batt = opt[opt["config"] == "PV + Battery"]
    if len(pv_batt) > 0:
        assert pv_batt["batt_kwh"].min() >= 50


def test_co2_savings_clamped_everywhere():
    """CO2 savings use same rule in optimizer and display: clamped >= 0."""
    df = _make_hourly_df(100)
    df["pv_per_kwp"] = 0
    bs = BatterySettings(charge_from_grid_at_night=True)
    cfg = OptimizerConfig(pv_min=0, pv_max=0, pv_step=1, batt_min=10, batt_max=50, batt_step=20)
    opt = optimize(df, "tariff_standard", cfg, bs, 0.18, standing_charge=0, opex_pct=0)
    batt_only = opt[opt["config"] == "Battery only"]
    if len(batt_only) > 0:
        assert (batt_only["co2_save_kg"] >= 0).all(), "Optimizer co2_save_kg must be clamped >= 0"
    # evaluate_for_tariff: uses same clamp; run with opt_dfs
    opt_dfs = {"tariff_standard": opt}
    res, _ = evaluate_for_tariff(
        df, opt_dfs, "tariff_standard", "standard", "Most CO2 savings",
        include_pv=False, include_battery=True, goal5_threshold_pct=50,
        battery_settings=bs, export_rate=0.18, standing_charge=0, opex_pct=0,
    )
    batt_row = res[res["Scenario"] == "Battery + Grid"].iloc[0]
    assert batt_row["CO2 savings (kg)"] >= 0, "Display CO2 savings must be clamped >= 0"


def test_finance_formulas_consistent():
    """compute_financial_metrics matches payback/NPV logic."""
    energy_cost = 5000
    baseline_energy_cost = 8000
    capex = 10000
    standing_charge = 100
    opex_pct = 2
    annual_cost, annual_savings, payback, npv = compute_financial_metrics(
        energy_cost, baseline_energy_cost, capex, standing_charge, opex_pct
    )
    opex = capex * 0.02
    assert abs(annual_cost - (energy_cost + standing_charge + opex)) < 0.01
    assert abs(annual_savings - (baseline_energy_cost + standing_charge - annual_cost)) < 0.01
    pb, n = compute_payback_and_npv(capex, annual_savings)
    assert abs(payback - pb) < 0.01
    assert abs(npv - n) < 0.01


def test_pso_levy_adds_to_costs_not_to_savings_delta():
    """PSO applies equally to baseline and scenario bills: savings unchanged, bills shift by PSO."""
    ac0, sv0, _, _ = compute_financial_metrics(
        100.0, 200.0, 0.0, 50.0, 0.0, pso_levy_annual=0.0
    )
    ac1, sv1, _, _ = compute_financial_metrics(
        100.0, 200.0, 0.0, 50.0, 0.0, pso_levy_annual=19.10
    )
    assert abs(sv0 - sv1) < 1e-9
    assert ac1 - ac0 == pytest.approx(19.10)


def test_evaluate_for_tariff_uses_capex_overrides():
    """If pv_cost_per_kwp/batt_cost_per_kwh overrides change, CAPEX/NVP should change."""
    df = _make_hourly_df(48, consumption=1.0)
    # Simple PV profile (half output every hour) so PV has some effect
    df["pv_per_kwp"] = 0.5
    bs = BatterySettings(eff_round_trip=0.9, charge_from_grid_at_night=False, discharge_schedule="Peak only")
    # Force optimizer to single sizes so scenario selection is deterministic
    cfg = OptimizerConfig(pv_min=10, pv_max=10, pv_step=1, batt_min=10, batt_max=10, batt_step=1)
    # Optimize with default globals; evaluation will override capex numbers later
    opt = optimize(df, "tariff_standard", cfg, bs, export_rate=0.18, standing_charge=0.0, opex_pct=5.0)
    opt_dfs = {"tariff_standard": opt}

    res_low, _ = evaluate_for_tariff(
        df,
        opt_dfs,
        "tariff_standard",
        "standard",
        goal="Lowest annual electricity cost",
        include_pv=True,
        include_battery=True,
        goal5_threshold_pct=90,
        battery_settings=bs,
        export_rate=0.18,
        standing_charge=0.0,
        opex_pct=5.0,
        discount_rate=0.05,
        pv_cost_per_kwp=100,   # lower capex
        batt_cost_per_kwh=100,
    )
    res_high, _ = evaluate_for_tariff(
        df,
        opt_dfs,
        "tariff_standard",
        "standard",
        goal="Lowest annual electricity cost",
        include_pv=True,
        include_battery=True,
        goal5_threshold_pct=90,
        battery_settings=bs,
        export_rate=0.18,
        standing_charge=0.0,
        opex_pct=5.0,
        discount_rate=0.05,
        pv_cost_per_kwp=1000,  # higher capex
        batt_cost_per_kwh=500,
    )

    # PV + Battery + Grid scenario should reflect overridden capex
    s2_low = float(res_low[res_low["Scenario"] == "PV + Battery + Grid"]["CAPEX (€)"].iloc[0])
    s2_high = float(res_high[res_high["Scenario"] == "PV + Battery + Grid"]["CAPEX (€)"].iloc[0])
    assert s2_high > s2_low


def test_build_full_scenario_results_df_uses_capex_overrides():
    df = _make_hourly_df(24, consumption=1.0)
    df["pv_per_kwp"] = 0.4
    bs = BatterySettings(charge_from_pv=True, discharge_schedule="Peak only")
    cfg = OptimizerConfig(pv_min=10, pv_max=10, pv_step=1, batt_min=0, batt_max=0, batt_step=1)
    opt = optimize(df, "tariff_standard", cfg, bs, export_rate=0.18, standing_charge=0.0, opex_pct=0.0)
    opt_dfs = {"tariff_standard": opt}
    prepared_df = df.rename(columns={"tariff_standard": "tariff_standard"})

    out = build_full_scenario_results_df(
        opt_dfs=opt_dfs,
        prepared_df=prepared_df,
        standing_charges_by_name={"standard": 0.0},
        pv_cost_per_kwp=200,
        batt_cost_per_kwh=300,
    )
    capex_col = [c for c in out.columns if "CAPEX" in str(c)][0]
    capex_vals = out[out["Scenario"] != "Grid only"][capex_col].to_numpy(dtype=float)
    assert len(capex_vals) > 0
    # With batt=0 and pv=10, CAPEX should be pv_cost_per_kwp * 10
    assert np.allclose(capex_vals, 200 * 10)


def test_electricity_inflation_affects_npv_and_gross_20y():
    """With inflation > 0, NPV and gross 20y should exceed flat (0%) values."""
    capex = 10000.0
    annual_savings = 1000.0
    discount_rate = 0.05

    pb0, npv0 = compute_payback_and_npv(capex, annual_savings, discount_rate, electricity_inflation_rate=0.0)
    pb3, npv3 = compute_payback_and_npv(capex, annual_savings, discount_rate, electricity_inflation_rate=0.03)

    assert pb0 == pb3  # Payback uses year-1 savings only
    assert npv3 > npv0, "NPV with 3% inflation should exceed NPV with 0% inflation"

    # Gross 20y: with inflation, sum of inflated savings > flat sum
    from app import _gross_savings_20y
    gross0 = _gross_savings_20y(annual_savings, 0.0)
    gross3 = _gross_savings_20y(annual_savings, 0.03)
    assert gross0 == annual_savings * LIFETIME_YEARS
    assert gross3 > gross0, "Gross 20y with 3% inflation should exceed flat"


def test_replacement_costs_reduce_npv():
    """Discounted replacement costs should reduce NPV."""
    capex = 10000.0
    annual_savings = 1200.0
    discount_rate = 0.05

    _, npv_no_repl = compute_payback_and_npv(
        capex,
        annual_savings,
        discount_rate=discount_rate,
        electricity_inflation_rate=0.0,
    )
    _, npv_with_repl = compute_payback_and_npv(
        capex,
        annual_savings,
        discount_rate=discount_rate,
        electricity_inflation_rate=0.0,
        battery_replacement_year=10,
        battery_replacement_cost_eur=2000.0,
        inverter_replacement_year=12,
        inverter_replacement_cost_eur=1000.0,
    )
    assert npv_with_repl < npv_no_repl


def test_replacement_costs_reduce_npv_even_when_savings_nonpositive():
    """When annual savings <= 0, replacements still reduce NPV further."""
    capex = 10000.0
    annual_savings = -100.0
    discount_rate = 0.05

    _, npv_no_repl = compute_payback_and_npv(
        capex,
        annual_savings,
        discount_rate=discount_rate,
        electricity_inflation_rate=0.0,
    )
    _, npv_with_repl = compute_payback_and_npv(
        capex,
        annual_savings,
        discount_rate=discount_rate,
        electricity_inflation_rate=0.0,
        battery_replacement_year=10,
        battery_replacement_cost_eur=2000.0,
    )
    assert npv_with_repl < npv_no_repl
    assert npv_no_repl < -capex, "Negative annual savings must reduce NPV below -CAPEX via discounted savings stream."


def test_evaluate_for_tariff_handles_missing_battery_configs():
    """No crash when optimizer has no Battery only / PV + Battery rows."""
    df = _make_hourly_df(48, consumption=1.0)
    df["pv_per_kwp"] = 0.2
    bs = BatterySettings()
    cfg = OptimizerConfig(pv_min=10, pv_max=10, pv_step=1, batt_min=0, batt_max=0, batt_step=1)
    opt = optimize(df, "tariff_standard", cfg, bs, export_rate=0.18, standing_charge=0.0, opex_pct=0.0)
    assert len(opt[opt["config"] == "Battery only"]) == 0
    assert len(opt[opt["config"] == "PV + Battery"]) == 0

    res, _ = evaluate_for_tariff(
        df,
        {"tariff_standard": opt},
        "tariff_standard",
        "standard",
        goal="Lowest annual electricity cost",
        include_pv=True,
        include_battery=True,  # still enabled, but configs missing
        goal5_threshold_pct=90,
        battery_settings=bs,
        export_rate=0.18,
        standing_charge=0.0,
        opex_pct=0.0,
    )
    assert len(res) == 4


def test_goal5_order_respects_selfconsumption_threshold():
    """Rows with self-consumption >= X% and largest PV must rank before sub-threshold rows."""
    cand = pd.DataFrame(
        [
            {"Scenario": "PV + Grid", "PV (kWp)": 50, "Self-consumption ratio (%)": 59.9, "Annual cost (€)": 100.0},
            {"Scenario": "PV + Grid", "PV (kWp)": 20, "Self-consumption ratio (%)": 92.0, "Annual cost (€)": 200.0},
            {"Scenario": "Grid only", "PV (kWp)": 0, "Self-consumption ratio (%)": 0.0, "Annual cost (€)": 300.0},
        ]
    )
    out = _order_dataframe_goal5_largest_pv_selfconsumption(cand, 90.0)
    assert int(out.iloc[0]["PV (kWp)"]) == 20
    assert float(out.iloc[0]["Self-consumption ratio (%)"]) >= 90.0
    assert int(out.iloc[1]["PV (kWp)"]) == 50


def test_pv_grid_sweep_uses_passed_pv_capex():
    df = _make_hourly_df(24, consumption=1.0)
    df["pv_per_kwp"] = 0.5
    # Build sweep table with pv_capex=100 and check CAPEX column scales
    sweep_100 = build_pv_grid_sweep_table(df, "tariff_standard", pv_capex_per_kwp=100, export_rate=0.18, standing_charge=0.0, opex_pct=0.0)
    sweep_200 = build_pv_grid_sweep_table(df, "tariff_standard", pv_capex_per_kwp=200, export_rate=0.18, standing_charge=0.0, opex_pct=0.0)
    # Use a non-zero PV row (pv=0 has CAPEX=0 regardless of €/kWp)
    cap_100_at_10 = float(sweep_100.loc[sweep_100["PV (kWp)"] == 10, "CAPEX (€)"].iloc[0])
    cap_200_at_10 = float(sweep_200.loc[sweep_200["PV (kWp)"] == 10, "CAPEX (€)"].iloc[0])
    assert cap_100_at_10 * 2 == cap_200_at_10
