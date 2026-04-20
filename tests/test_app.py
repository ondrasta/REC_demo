"""Lightweight tests for REC Feasibility app correctness. Run with: pytest tests/test_app.py -v"""
import numpy as np
import pandas as pd
from pathlib import Path
try:
    import pytest
except ImportError:
    pytest = None

# Import core logic without Streamlit UI
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import (
    SCENARIO_ROW_KEY_FIELD,
    _pv_per_kwp_pattern_features,
    RECOMMENDED_NO_SIZING_KEY_PREFIX,
    augment_recommended_df_with_scenario_row_keys,
    _sort_recommended_setups_df_by_sidebar_rank,
    compose_scenario_row_key,
    recommended_setups_join_consolidated_kpis_df,
    RECOMMENDED_SETUPS_EXPORT_NOTE_COL,
    _rank_position_for_consolidated_row,
    _rank_scenarios_from_consolidated_table,
    encode_csv_assumptions_block_then_results_df,
    DEFAULT_TARIFFS,
    DEFAULT_EXPORT_RATE,
    get_active_tariff_config,
    resolve_consumption_csv_bytes,
    resolve_pv_csv_bytes,
    load_and_prepare_data,
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
    DEFAULT_LIFETIME_YEARS,
    build_pv_grid_sweep_table,
    _parse_consumption_csv,
    _parse_pv_timeseries_csv,
    _parse_tariff_variants_csv_bytes,
    _battery_discharge_ok_hour,
    _tariff_matrix_profiles_from_parsed,
    _tariff_type_display_label,
    _tariff_rates_summary_for_matrix,
)


def test_encode_csv_assumptions_then_results_inserts_blank_line():
    ass = pd.DataFrame([{"Setting": "PV CAPEX (€/kWp)", "Value": 1000.0}])
    res = pd.DataFrame([{"Scenario": "PV + Grid", "PV (kWp)": 10}])
    raw = encode_csv_assumptions_block_then_results_df(ass, res)
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "Setting,Value" in text
    assert "PV CAPEX" in text
    assert "Scenario" in text
    parts = text.split("\n\n", 1)
    assert len(parts) == 2
    assert "Setting,Value" in parts[0]
    assert parts[1].lstrip().startswith("Scenario")


def test_encode_csv_results_only_when_no_assumptions():
    res = pd.DataFrame([{"A": 1}])
    raw = encode_csv_assumptions_block_then_results_df(None, res)
    text = raw.decode("utf-8-sig")
    assert text.strip().startswith("A")


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


def test_pv_csv_accepts_consumption_style_datetime_with_seconds():
    """PV CSV also accepts DD/MM/YYYY HH:MM:SS (including double spaces) and keeps year-boundary midnight in 2020."""
    csv_text = (
        "time,P\n"
        "31/12/2020  23:00:00,1000\n"
        "01/01/2021  00:00:00,2000\n"
    )
    df = _parse_pv_timeseries_csv(csv_text.encode("utf-8"))
    assert len(df) == 2
    assert (df["date"].dt.year == 2020).all()
    assert list(df["pv_per_kwp"]) == [1.0, 2.0]


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


def test_battery_discharge_schedule_day_peak_bands():
    """Day+Peak = 17-19 and 19-23 only; Peak only = 17-19; no morning or night."""
    assert _battery_discharge_ok_hour(17, "Peak only") and _battery_discharge_ok_hour(18, "Peak only")
    assert not _battery_discharge_ok_hour(19, "Peak only")
    assert _battery_discharge_ok_hour(19, "Day+Peak") and _battery_discharge_ok_hour(22, "Day+Peak")
    assert not _battery_discharge_ok_hour(23, "Day+Peak")
    assert not _battery_discharge_ok_hour(16, "Day+Peak")
    assert not _battery_discharge_ok_hour(7, "Day+Peak")


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


def test_horizon_net_benefit_subtracts_capex():
    """Net benefit over DEFAULT_LIFETIME_YEARS = gross savings - CAPEX."""
    capex = 10000
    annual_savings = 1000
    gross = annual_savings * DEFAULT_LIFETIME_YEARS
    net = gross - capex
    # Formula: net benefit = gross savings - CAPEX
    assert net == annual_savings * DEFAULT_LIFETIME_YEARS - capex
    assert net == 10000  # 20k - 10k at default 20-year horizon


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
        df, opt_dfs, "tariff_standard", "standard", "Highest annual CO2 savings",
        include_pv=False, include_battery=True,
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
        tariff_profiles=[{"col": "tariff_standard", "name": "standard", "standing_charge": 0.0}],
        pv_cost_per_kwp=200,
        batt_cost_per_kwh=300,
    )
    capex_col = [c for c in out.columns if "CAPEX" in str(c)][0]
    capex_vals = out[out["Scenario"] != "Grid only"][capex_col].to_numpy(dtype=float)
    assert len(capex_vals) > 0
    # With batt=0 and pv=10, CAPEX should be pv_cost_per_kwp * 10
    assert np.allclose(capex_vals, 200 * 10)
    assert SCENARIO_ROW_KEY_FIELD in out.columns
    assert out[SCENARIO_ROW_KEY_FIELD].nunique(dropna=True) == len(out)

    ranked = _rank_scenarios_from_consolidated_table(out, "Lowest annual electricity cost")
    assert len(ranked) == len(out)
    _first = ranked[0][1]
    rp, nt = _rank_position_for_consolidated_row(ranked, _first)
    assert nt == len(out) and rp == 1


def test_recommended_setups_join_consolidated_kpis_matches_full_table():
    profiles = [{"name": "Std", "col": "tariff_standard_0"}]
    key = compose_scenario_row_key("tariff_standard_0", "PV + Grid", 10, 0)
    full = pd.DataFrame(
        [
            {
                "Tariff": "Std",
                "Scenario": "PV + Grid",
                "PV (kWp)": 10,
                "Battery (kWh)": 0,
                "NPV (€)": 123.4,
                SCENARIO_ROW_KEY_FIELD: key,
            }
        ]
    )
    rec = pd.DataFrame(
        [{"Tariff": "Std", "Scenario family": "PV + Grid", "PV (kWp)": 10, "Battery (kWh)": 0, "Note": ""}]
    )
    aug = augment_recommended_df_with_scenario_row_keys(rec, profiles)
    out = recommended_setups_join_consolidated_kpis_df(aug, full)
    assert len(out) == 1
    assert float(out.iloc[0]["NPV (€)"]) == 123.4
    assert RECOMMENDED_SETUPS_EXPORT_NOTE_COL in out.columns
    assert out.iloc[0][RECOMMENDED_SETUPS_EXPORT_NOTE_COL] == ""


def test_recommended_setups_join_consolidated_kpis_infeasible_note():
    profiles = [{"name": "Std", "col": "tariff_standard_0"}]
    full = pd.DataFrame(
        [
            {
                "Tariff": "Std",
                "Scenario": "PV + Grid",
                SCENARIO_ROW_KEY_FIELD: compose_scenario_row_key("tariff_standard_0", "PV + Grid", 10, 0),
            }
        ]
    )
    rec = pd.DataFrame(
        [
            {
                "Tariff": "Std",
                "Scenario family": "PV + Grid",
                "PV (kWp)": np.nan,
                "Battery (kWh)": np.nan,
                "Note": "No feasible sizing under constraints.",
            }
        ]
    )
    aug = augment_recommended_df_with_scenario_row_keys(rec, profiles)
    out = recommended_setups_join_consolidated_kpis_df(aug, full)
    assert len(out) == 1
    assert pd.isna(out.iloc[0].get("Scenario")) or out.iloc[0].get("Scenario") == "PV + Grid"
    assert "No feasible" in str(out.iloc[0][RECOMMENDED_SETUPS_EXPORT_NOTE_COL])


def test_augment_recommended_df_maps_keys_and_placeholder_prefix():
    profiles = [{"name": "Std", "col": "tariff_standard_0"}]
    rec = pd.DataFrame(
        [
            {"Tariff": "Std", "Scenario family": "PV + Grid", "PV (kWp)": 5, "Battery (kWh)": 0},
            {"Tariff": "Std", "Scenario family": "PV + Grid", "PV (kWp)": np.nan, "Battery (kWh)": np.nan},
        ]
    )
    out = augment_recommended_df_with_scenario_row_keys(rec, profiles)
    assert out[SCENARIO_ROW_KEY_FIELD].iloc[0] == compose_scenario_row_key(
        "tariff_standard_0", "PV + Grid", 5, 0
    )
    assert str(out[SCENARIO_ROW_KEY_FIELD].iloc[1]).startswith(RECOMMENDED_NO_SIZING_KEY_PREFIX)


def test_electricity_inflation_affects_npv_and_gross_lifetime():
    """With inflation > 0, NPV and gross lifetime savings should exceed flat (0%) values."""
    capex = 10000.0
    annual_savings = 1000.0
    discount_rate = 0.05

    pb0, npv0 = compute_payback_and_npv(capex, annual_savings, discount_rate, electricity_inflation_rate=0.0)
    pb3, npv3 = compute_payback_and_npv(capex, annual_savings, discount_rate, electricity_inflation_rate=0.03)

    assert pb0 == pb3  # Payback uses year-1 savings only
    assert npv3 > npv0, "NPV with 3% inflation should exceed NPV with 0% inflation"

    # Gross lifetime: with inflation, sum of inflated savings > flat sum
    from app import _gross_savings_lifetime
    gross0 = _gross_savings_lifetime(annual_savings, 0.0)
    gross3 = _gross_savings_lifetime(annual_savings, 0.03)
    assert gross0 == annual_savings * DEFAULT_LIFETIME_YEARS
    assert gross3 > gross0, "Gross lifetime savings with 3% inflation should exceed flat"


def test_parse_tariff_variants_csv_bytes_basic():
    cols = [
        "family",
        "variant",
        "standing_charge",
        "standard_day",
        "standard_peak",
        "standard_night",
        "weekend_weekday_day",
        "weekend_weekday_peak",
        "weekend_weekday_night",
        "weekend_weekend_day",
        "weekend_weekend_peak",
        "weekend_weekend_night",
        "flat_rate",
        "export_rate",
    ]
    df = pd.DataFrame(
        [
            ["standard", "Market average", 286.6, 0.32, 0.36, 0.21, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.18],
            ["weekend", "Company A", 338.85, np.nan, np.nan, np.nan, 0.33, 0.38, 0.25, 0.22, 0.23, 0.19, np.nan, 0.175],
            ["flat", "FlatCo", 286.6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.29966, 0.17],
        ],
        columns=cols,
    )
    res = _parse_tariff_variants_csv_bytes(df.to_csv(index=False).encode("utf-8"))
    assert len(res["standard"]) == 1
    assert res["standard"][0]["variant"] == "Market average"
    assert abs(res["standard"][0]["standing_charge"] - 286.6) < 1e-9
    assert res["standard"][0]["rates"]["standard"]["day"] == 0.32
    assert abs(res["standard"][0]["export_rate"] - 0.18) < 1e-9

    assert len(res["weekend"]) == 1
    assert res["weekend"][0]["variant"] == "Company A"
    assert res["weekend"][0]["rates"]["weekend"]["weekday"]["peak"] == 0.38
    assert abs(res["weekend"][0]["export_rate"] - 0.175) < 1e-9

    assert len(res["flat"]) == 1
    assert res["flat"][0]["variant"] == "FlatCo"
    assert abs(res["flat"][0]["rates"]["flat"]["flat"] - 0.29966) < 1e-9
    assert abs(res["flat"][0]["export_rate"] - 0.17) < 1e-9


def test_parse_tariff_variants_csv_bytes_alias_schema():
    cols = [
        "tariff_type",
        "company",
        "standing_charge (EUR/year)",
        "export_rate (EUR/kWh)",
        "weekday_day",
        "weekday_peak",
        "weekday_night",
        "weekend_day",
        "weekend_peak",
        "weekend_night",
    ]
    df = pd.DataFrame(
        [
            ["flat_rate", "c1", 280.0, 0.16, 0.28, np.nan, np.nan, np.nan, np.nan, np.nan],
            ["weekend_saver", "c2", 330.0, 0.18, 0.32, 0.37, 0.24, 0.21, 0.22, 0.19],
            ["standard", "c3", 290.0, 0.17, 0.31, 0.35, 0.20, np.nan, np.nan, np.nan],
        ],
        columns=cols,
    )
    res = _parse_tariff_variants_csv_bytes(df.to_csv(index=False).encode("utf-8"))
    assert res["flat"][0]["variant"] == "c1"
    assert abs(res["flat"][0]["rates"]["flat"]["flat"] - 0.28) < 1e-9
    assert res["weekend"][0]["rates"]["weekend"]["weekend"]["night"] == 0.19
    assert res["standard"][0]["rates"]["standard"]["peak"] == 0.35


def test_tariff_matrix_profiles_from_parsed_unique_cols():
    """Matrix flatten assigns tariff_sel_* columns in family order (standard → weekend → flat)."""
    cols = [
        "family",
        "variant",
        "standing_charge",
        "standard_day",
        "standard_peak",
        "standard_night",
        "weekend_weekday_day",
        "weekend_weekday_peak",
        "weekend_weekday_night",
        "weekend_weekend_day",
        "weekend_weekend_peak",
        "weekend_weekend_night",
        "flat_rate",
        "export_rate",
    ]
    df = pd.DataFrame(
        [
            ["standard", "Market average", 286.6, 0.32, 0.36, 0.21, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.18],
            ["weekend", "Company A", 338.85, np.nan, np.nan, np.nan, 0.33, 0.38, 0.25, 0.22, 0.23, 0.19, np.nan, 0.175],
            ["flat", "FlatCo", 286.6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.29966, 0.17],
        ],
        columns=cols,
    )
    res = _parse_tariff_variants_csv_bytes(df.to_csv(index=False).encode("utf-8"))
    profs = _tariff_matrix_profiles_from_parsed(res)
    assert len(profs) == 3
    assert profs[0]["col"] == "tariff_sel_0" and profs[0]["kind"] == "standard"
    assert profs[1]["col"] == "tariff_sel_1" and profs[1]["kind"] == "weekend"
    assert profs[2]["col"] == "tariff_sel_2" and profs[2]["kind"] == "flat"
    assert _tariff_type_display_label("weekend") == "weekend_saver"
    assert "flat=" in _tariff_rates_summary_for_matrix(profs[2])


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
        battery_settings=bs,
        export_rate=0.18,
        standing_charge=0.0,
        opex_pct=0.0,
    )
    assert len(res) == 4


def test_pv_per_kwp_pattern_features_sums_match_prepared():
    """Production patterns helper uses pv_per_kwp aligned with date features."""
    n = 48
    dates = pd.date_range("2020-06-01", periods=n, freq="h")
    prep = pd.DataFrame(
        {
            "date": dates,
            "consumption": np.ones(n),
            "pv_per_kwp": np.linspace(0.0, 0.5, n),
            "tariff_standard": 0.3,
        }
    )
    h = _pv_per_kwp_pattern_features(prep)
    assert len(h) == n
    assert float(h["pv_kwh_per_kwp"].sum()) == pytest.approx(float(prep["pv_per_kwp"].sum()))
    assert "time_band" in h.columns


def test_builtin_default_csvs_resolve_and_merge():
    """Shipped data/default_*.csv loads through the same path as upload bytes."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "data" / "default_consumption.csv").is_file()
    assert (root / "data" / "default_pv.csv").is_file()
    cb, cs = resolve_consumption_csv_bytes(None)
    pb, ps = resolve_pv_csv_bytes(None)
    assert cs.startswith("default") and ps.startswith("default")
    assert len(cb) > 500 and len(pb) > 500
    df = load_and_prepare_data(cb, pb, DEFAULT_TARIFFS, "pytest-default-csvs")
    assert len(df) >= 8000
    assert df["consumption"].notna().all()
    assert (df["pv_per_kwp"] >= 0).all()


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


def test_recommended_setups_prefers_higher_npv_then_co2():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 200.0,
                "savings": 500.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 20,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 500.0,
                "co2_save_kg": 80.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 2000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 10
    assert float(out.iloc[0]["NPV (€)"]) == 200.0
    assert float(out.iloc[0]["CO₂ savings (kg)"]) == 50.0


def test_recommended_setups_tie_co2_prefers_higher_npv():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 400.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 30,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 250.0,
                "savings": 600.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 3000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 30


def test_recommended_setups_tie_npv_prefers_higher_co2():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 400.0,
                "co2_save_kg": 40.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 25,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 500.0,
                "co2_save_kg": 70.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 2000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 25
    assert float(out.iloc[0]["CO₂ savings (kg)"]) == 70.0


def test_recommended_setups_battery_charging_column_when_battery_ui():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 200.0,
                "savings": 500.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 1000.0,
            },
        ]
    )
    out_off = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=True,
        charge_from_grid_at_night_last_run=False,
    )
    assert "Battery charging (last run)" in out_off.columns
    assert (out_off["Battery charging (last run)"] == "PV only charging").all()

    out_on = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=True,
        charge_from_grid_at_night_last_run=True,
    )
    assert (out_on["Battery charging (last run)"] == "PV + night-grid charging").all()

    out_none = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=True,
        charge_from_grid_at_night_last_run=None,
    )
    assert "Battery charging (last run)" not in out_none.columns


def test_cumulative_co2_uses_annual_co2_reduction_column():
    """Consolidated rows use Annual CO2 reduction (kg); cumulative outlook must not read only CO2 savings (kg)."""
    from app import COL_ANNUAL_CO2_REDUCTION_KG, _annual_co2_savings_kg_from_consolidated_row

    row = pd.Series({COL_ANNUAL_CO2_REDUCTION_KG: 1234.5})
    assert abs(_annual_co2_savings_kg_from_consolidated_row(row) - 1234.5) < 1e-9
    row_legacy = pd.Series({"CO2 savings (kg)": 99.0})
    assert abs(_annual_co2_savings_kg_from_consolidated_row(row_legacy) - 99.0) < 1e-9


def test_recommended_setups_sort_follows_sidebar_rank_order():
    k_a = "tcol\x1ePV + Grid\x1e10\x1e0"
    k_b = "tcol\x1ePV + Grid\x1e20\x1e0"
    k_infeas = f"{RECOMMENDED_NO_SIZING_KEY_PREFIX}tcol\x1ePV + Grid"
    df = pd.DataFrame(
        [
            {"Tariff": "Z", SCENARIO_ROW_KEY_FIELD: k_b},
            {"Tariff": "Y", SCENARIO_ROW_KEY_FIELD: k_infeas},
            {"Tariff": "X", SCENARIO_ROW_KEY_FIELD: k_a},
        ]
    )
    ranked = [
        ("PV + Grid", pd.Series({SCENARIO_ROW_KEY_FIELD: k_a, "Scenario": "PV + Grid"})),
        ("PV + Grid", pd.Series({SCENARIO_ROW_KEY_FIELD: k_b, "Scenario": "PV + Grid"})),
    ]
    out = _sort_recommended_setups_df_by_sidebar_rank(df, ranked)
    assert out[SCENARIO_ROW_KEY_FIELD].tolist() == [k_a, k_b, k_infeas]


def test_recommended_setups_marks_infeasible_when_payback_above_max():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 11.0,
                "npv": 999.0,
                "savings": 500.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 1000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
    )
    assert len(out) == 1
    assert "No PV/battery size in the grid satisfies" in str(out.iloc[0]["Note"])


def test_recommended_setups_excludes_nonpositive_npv_when_required():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": -50.0,
                "savings": 100.0,
                "co2_save_kg": 20.0,
                "self_consumption_ratio_pct": 90.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 20,
                "batt_kwh": 0,
                "payback": 6.0,
                "npv": 10.0,
                "savings": 200.0,
                "co2_save_kg": 30.0,
                "self_consumption_ratio_pct": 50.0,
                "pv_gen_kwh": 2000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
        require_positive_npv=True,
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 20
    assert float(out.iloc[0]["NPV (€)"]) == 10.0


def test_hard_filter_export_ratio_max_keeps_nan_export_rows():
    from app import _apply_hard_filters_to_results_df

    df = pd.DataFrame(
        [
            {"Scenario": "PV + Grid", "Export ratio (% of PV gen)": 40.0},
            {"Scenario": "PV + Grid", "Export ratio (% of PV gen)": 60.0},
            {"Scenario": "Grid only", "Export ratio (% of PV gen)": np.nan},
        ]
    )
    out = _apply_hard_filters_to_results_df(df, export_ratio_max_pct=50.0)
    assert len(out) == 2
    assert out["Scenario"].tolist() == ["PV + Grid", "Grid only"]


def test_recommended_setups_min_scr_constraint():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 200.0,
                "savings": 500.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 20.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 15,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 400.0,
                "co2_save_kg": 40.0,
                "self_consumption_ratio_pct": 80.0,
                "pv_gen_kwh": 1000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
        min_self_consumption_pct=50.0,
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 15


def test_recommended_setups_summary_includes_scr_ssr_bill_and_co2_reduction_pct():
    from app import COL_ANNUAL_ELECTRICITY_BILL_EUR, build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 200.0,
                "savings": 500.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "self_suff_pct": 12.5,
                "cost": 1000.0,
                "pv_gen_kwh": 1000.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
        grid_baseline_annual_co2_kg=500.0,
    )
    assert COL_ANNUAL_ELECTRICITY_BILL_EUR in out.columns
    assert "SCR" in out.columns
    assert "SSR" in out.columns
    assert "CO2 reduction (%)" in out.columns
    assert "Self-consumption ratio (%)" not in out.columns
    assert float(out.iloc[0][COL_ANNUAL_ELECTRICITY_BILL_EUR]) == 1000.0
    assert float(out.iloc[0]["SCR"]) == 90.0
    assert float(out.iloc[0]["SSR"]) == 12.5
    assert abs(float(out.iloc[0]["CO2 reduction (%)"]) - 10.0) < 1e-9


def test_format_numeric_columns_for_aggrid_integer_round_cols():
    from app import _format_numeric_columns_for_aggrid

    df = pd.DataFrame({"NPV (€)": [123.7, np.nan], "Payback (yrs)": [4.56, 8.1]})
    out = _format_numeric_columns_for_aggrid(df, integer_round_cols=frozenset({"NPV (€)"}))
    assert int(out.iloc[0]["NPV (€)"]) == 124
    assert pd.isna(out.iloc[1]["NPV (€)"])
    assert abs(float(out.iloc[0]["Payback (yrs)"]) - 4.6) < 0.01


def test_recommended_preset_financial_prefers_higher_savings_at_same_npv():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 300.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "self_suff_pct": 10.0,
                "cost": 800.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 12,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 100.0,
                "savings": 500.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "self_suff_pct": 10.0,
                "cost": 800.0,
                "pv_gen_kwh": 1200.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
        winner_preset="financial",
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 12


def test_recommended_preset_lowest_bill_prefers_lower_cost():
    from app import build_recommended_setups_summary_df

    opt_df = pd.DataFrame(
        [
            {
                "config": "PV only",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 200.0,
                "savings": 400.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "self_suff_pct": 10.0,
                "cost": 900.0,
                "pv_gen_kwh": 1000.0,
            },
            {
                "config": "PV only",
                "pv_kwp": 11,
                "batt_kwh": 0,
                "payback": 5.0,
                "npv": 200.0,
                "savings": 400.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 90.0,
                "self_suff_pct": 10.0,
                "cost": 700.0,
                "pv_gen_kwh": 1100.0,
            },
        ]
    )
    out = build_recommended_setups_summary_df(
        {"t1": opt_df},
        [{"col": "t1", "name": "Test"}],
        enable_battery_ui=False,
        winner_preset="lowest_bill",
    )
    assert len(out) == 1
    assert int(out.iloc[0]["PV (kWp)"]) == 11


def test_bundled_research_xlsx_loads_and_winners():
    from pathlib import Path

    import sys

    _root = Path(__file__).resolve().parent.parent
    p = _root / "assets" / "research" / "res.xlsx"
    if not p.is_file():
        pytest.skip("bundled research xlsx not present")
    sys.path.insert(0, str(_root))
    from bundled_research import (
        compute_winners_for_rule,
        load_bundled_research_xlsx,
        research_rule_by_id,
    )

    raw, scenario_titles, tariff_names, mat = load_bundled_research_xlsx(p)
    assert len(scenario_titles) >= 1
    assert len(tariff_names) == 5
    rule = research_rule_by_id("lowest_bill")
    w = compute_winners_for_rule(raw, scenario_titles, tariff_names, mat, rule)
    assert len(w) == len(scenario_titles)
    assert "Winner tariff" in w.columns
