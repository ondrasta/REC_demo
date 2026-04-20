"""Round-trip tests for saved-run ZIP bundles (no Streamlit)."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from saved_run_bundle import (
    BUNDLE_SCHEMA_VERSION,
    build_saved_run_zip_bytes,
    load_bundle_from_zip,
    read_manifest_from_zip,
    resolve_app_build_fingerprint,
)


def _minimal_prepared_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=24, freq="h")
    return pd.DataFrame(
        {
            "date": dates,
            "consumption": np.ones(24),
            "pv_per_kwp": np.zeros(24),
            "tariff_standard_0": 0.3,
        }
    )


def _minimal_opt_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config": "pv_grid",
                "pv_kwp": 10,
                "batt_kwh": 0,
                "npv": 1.0,
                "payback": 5.0,
                "savings": 100.0,
                "co2_save_kg": 50.0,
                "self_consumption_ratio_pct": 80.0,
                "pv_gen_kwh": 1000.0,
            }
        ]
    )


def test_bundle_round_trip_zip_json_parquet():
    prepared = _minimal_prepared_df()
    opt = _minimal_opt_df()
    opt_dfs = {"tariff_standard_0": opt}
    cons_b = b"date,x\n2020-01-01 00:00,1\n"
    pv_b = b"time,P\n2020010100,0\n"
    profiles = [
        {
            "family": "standard",
            "variant": "Test",
            "col": "tariff_standard_0",
            "name": "Standard — Test",
            "kind": "standard",
            "rates": {"standard": {"day": 0.3, "peak": 0.3, "night": 0.2}},
            "standing_charge": 100.0,
            "export_rate": 0.1,
        }
    ]
    last_run = {
        "last_pv_capex": 1000.0,
        "last_batt_capex": 300.0,
        "last_opex_pct": 1.0,
        "last_discount_rate": 0.05,
        "last_electricity_inflation_rate": 0.0,
        "last_battery_replacement_year": None,
        "last_battery_replacement_cost_pct": 0.0,
        "last_inverter_replacement_year": None,
        "last_inverter_replacement_cost_pct": 0.0,
        "last_pso_levy": 10.0,
        "last_co2_factor": 0.25,
        "last_lifetime_years": 25,
        "last_export_rate": 0.1886,
        "last_opt_cfg": {"pv_min": 5, "pv_max": 60, "batt_min": 0, "batt_max": 0, "pv_step": 5, "batt_step": 5, "speed_preset": "Quick"},
        "last_battery_settings": {
            "eff_round_trip": 0.9025,
            "dod": 0.9,
            "init_soc": 0.0,
            "min_soc": 0.1,
            "max_soc": 0.9,
            "c_rate": 0.5,
            "charge_from_pv": True,
            "charge_from_grid_at_night": False,
            "discharge_schedule": "Peak only",
        },
        "last_input_hashes": {"cons_sha": "a" * 64, "pv_sha": "b" * 64},
        "prepared_meta": {"cons_sha": "a" * 64, "pv_sha": "b" * 64, "cons_source": "x", "pv_source": "y"},
        "last_tariff_matrix_source_label": "test",
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }
    full = pd.DataFrame(
        [
            {
                "Scenario": "Grid only",
                "Tariff": "Standard — Test",
                "PV (kWp)": 0,
                "Battery (kWh)": 0,
                "Grid import reduction (kWh)": 0.0,
                "CO2 reduction (%)": 0.0,
                "Export ratio (% of PV gen)": float("nan"),
            }
        ]
    )

    z = build_saved_run_zip_bytes(
        prepared_df=prepared,
        opt_dfs=opt_dfs,
        full_results_df=full,
        cons_bytes=cons_b,
        pv_bytes=pv_b,
        tariff_csv_bytes=None,
        last_tariff_profiles=profiles,
        last_run_json=last_run,
        app_version_label="test",
    )
    man, payload = load_bundle_from_zip(z)
    assert man["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert man["app_version"] == "test"
    assert payload["cons_bytes"] == cons_b
    assert payload["pv_bytes"] == pv_b
    assert list(payload["opt_dfs"].keys()) == ["tariff_standard_0"]
    assert len(payload["prepared_df"]) == 24
    assert payload["last_run"]["last_pv_capex"] == 1000.0
    assert payload["tariff_profiles"][0]["col"] == "tariff_standard_0"
    assert payload["full_results_df"] is not None
    assert "Grid import reduction (kWh)" in payload["full_results_df"].columns


def test_bundle_rejects_wrong_schema():
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "manifest.json",
            '{"schema_version": 999, "file_sha256": {}, "opt_tariff_columns": [], "has_full_results": false}',
        )
    with pytest.raises(ValueError, match="Unsupported bundle"):
        load_bundle_from_zip(buf.getvalue())


def test_bundle_rejects_zip_slip_path():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "manifest.json",
            '{"schema_version": 1, "file_sha256": {}, "opt_tariff_columns": [], "has_full_results": false}',
        )
        zf.writestr("../evil.txt", b"x")
    with pytest.raises(ValueError, match="unsafe zip member"):
        load_bundle_from_zip(buf.getvalue())


def test_read_manifest_rejects_unsafe_paths():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version": 1}')
        zf.writestr("foo/../x.json", b"{}")
    with pytest.raises(ValueError, match="unsafe zip member"):
        read_manifest_from_zip(buf.getvalue())


def test_resolve_app_build_fingerprint_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REC_FEASIBILITY_BUILD_ID", "ci-build-42")
    assert resolve_app_build_fingerprint() == "ci-build-42"
