"""
ZIP saved-run snapshots (JSON + Parquet only). No pickle.

Used by the Streamlit app to export/import a completed optimization without rerunning ``optimize()``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Bump when bundle layout or semantics change incompatibly.
BUNDLE_SCHEMA_VERSION = 1

# Import guards (ZIP slip / oversized members). Tune if legitimate bundles exceed this.
MAX_BUNDLE_UNCOMPRESSED_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB
MAX_BUNDLE_SINGLE_MEMBER_BYTES = 256 * 1024 * 1024  # 256 MiB per file

MANIFEST = "manifest.json"
PATH_LAST_RUN = "state/last_run.json"
PATH_TARIFF_PROFILES = "state/tariff_profiles.json"
PATH_PREPARED = "frames/prepared_df.parquet"
PATH_FULL_RESULTS = "frames/full_results_df.parquet"
PATH_CONS = "inputs/consumption.csv"
PATH_PV = "inputs/pv.csv"
PATH_TARIFF_CSV = "inputs/tariffs.csv"
OPT_PREFIX = "opt/"


def resolve_app_build_fingerprint(*, app_root: Optional[Path] = None) -> str:
    """Human-readable build id for bundle manifest.

    Order: env ``REC_FEASIBILITY_BUILD_ID``, else ``git rev-parse --short HEAD`` from ``app_root``
    (defaults to this package directory), else a generic unknown label.
    """
    env = os.environ.get("REC_FEASIBILITY_BUILD_ID", "").strip()
    if env:
        return env
    root = app_root if app_root is not None else Path(__file__).resolve().parent
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        sha = (r.stdout or "").strip()
        if r.returncode == 0 and sha:
            return f"rec-feasibility-app git:{sha}"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "rec-feasibility-app (unknown build)"


def _is_safe_zip_member_path(name: str) -> bool:
    """Reject absolute paths, ``..`` segments, and odd ZIP names (ZIP-slip)."""
    p = str(name).replace("\\", "/").strip()
    if not p or p.startswith("/"):
        return False
    parts = p.split("/")
    for seg in parts:
        if seg in ("", ".", ".."):
            return False
    return True


def _is_allowed_bundle_member(name: str) -> bool:
    """Only manifest, state/*.json, inputs/*.csv, frames/*.parquet, opt/opt__*.parquet."""
    if not _is_safe_zip_member_path(name):
        return False
    if name == MANIFEST:
        return True
    if name.startswith("state/") and name.endswith(".json"):
        return True
    if name.startswith("inputs/") and name.endswith(".csv"):
        return True
    if name.startswith("frames/") and name.endswith(".parquet"):
        return True
    if name.startswith(OPT_PREFIX) and name.endswith(".parquet"):
        return name[len(OPT_PREFIX) :].startswith("opt__")
    return False


def _zip_entry_is_directory(info: zipfile.ZipInfo) -> bool:
    if getattr(info, "is_dir", None):
        try:
            return bool(info.is_dir())
        except Exception:
            pass
    return str(info.filename).replace("\\", "/").endswith("/")


def _validate_zip_structure_and_size(zf: zipfile.ZipFile) -> None:
    infos = zf.infolist()
    total = 0
    for info in infos:
        name = info.filename
        if _zip_entry_is_directory(info) and int(getattr(info, "file_size", 0) or 0) == 0:
            continue
        if not _is_allowed_bundle_member(name):
            raise ValueError(f"Bundle rejected: disallowed or unsafe zip member {name!r}")
        sz = int(getattr(info, "file_size", 0) or 0)
        if sz > MAX_BUNDLE_SINGLE_MEMBER_BYTES:
            raise ValueError(
                f"Bundle rejected: member {name!r} uncompressed size {sz:,} B exceeds limit "
                f"({MAX_BUNDLE_SINGLE_MEMBER_BYTES:,} B)."
            )
        total += sz
    if total > MAX_BUNDLE_UNCOMPRESSED_TOTAL_BYTES:
        raise ValueError(
            f"Bundle rejected: total uncompressed size {total:,} B exceeds limit "
            f"({MAX_BUNDLE_UNCOMPRESSED_TOTAL_BYTES:,} B)."
        )


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_opt_filename(tcol: str) -> str:
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(tcol))
    return s or "tariff"


def _expected_bundle_paths_from_manifest(manifest: Dict[str, Any]) -> set[str]:
    """Exact file member set for a valid bundle (after coarse path allowlist)."""
    paths = {
        MANIFEST,
        PATH_LAST_RUN,
        PATH_TARIFF_PROFILES,
        PATH_PREPARED,
        PATH_CONS,
        PATH_PV,
    }
    if bool(manifest.get("has_full_results")):
        paths.add(PATH_FULL_RESULTS)
    opt_keys: List[str] = list(manifest.get("opt_tariff_columns") or [])
    for tcol in opt_keys:
        paths.add(f"{OPT_PREFIX}opt__{_safe_opt_filename(tcol)}.parquet")
    sha = manifest.get("file_sha256") or {}
    if isinstance(sha, dict) and PATH_TARIFF_CSV in sha:
        paths.add(PATH_TARIFF_CSV)
    return paths


def _zip_file_member_names(zf: zipfile.ZipFile) -> set[str]:
    """Zip member paths for real files (skip zero-byte directory markers)."""
    out: set[str] = set()
    for info in zf.infolist():
        if _zip_entry_is_directory(info) and int(getattr(info, "file_size", 0) or 0) == 0:
            continue
        out.add(info.filename)
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, float)):
        x = float(o)
        if not np.isfinite(x):
            raise ValueError("Cannot serialize non-finite float to bundle JSON")
        return x
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=_json_default).encode("utf-8")


def _json_loads(b: bytes) -> Any:
    return json.loads(b.decode("utf-8"))


def build_saved_run_zip_bytes(
    *,
    prepared_df: pd.DataFrame,
    opt_dfs: Dict[str, pd.DataFrame],
    full_results_df: Optional[pd.DataFrame],
    cons_bytes: bytes,
    pv_bytes: bytes,
    tariff_csv_bytes: Optional[bytes],
    last_tariff_profiles: List[Dict[str, Any]],
    last_run_json: Dict[str, Any],
    app_version_label: str | None = None,
) -> bytes:
    """Assemble a ZIP bundle. ``last_run_json`` must be JSON-serializable (no numpy types)."""
    _ver = str(app_version_label).strip() if app_version_label else resolve_app_build_fingerprint()
    buf = io.BytesIO()
    file_hashes: Dict[str, str] = {}
    opt_keys_sorted = sorted(opt_dfs.keys(), key=str)

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(PATH_CONS, cons_bytes)
        file_hashes[PATH_CONS] = _sha256_bytes(cons_bytes)
        zf.writestr(PATH_PV, pv_bytes)
        file_hashes[PATH_PV] = _sha256_bytes(pv_bytes)

        if tariff_csv_bytes:
            zf.writestr(PATH_TARIFF_CSV, tariff_csv_bytes)
            file_hashes[PATH_TARIFF_CSV] = _sha256_bytes(tariff_csv_bytes)

        tp_json = _json_dumps(last_tariff_profiles)
        zf.writestr(PATH_TARIFF_PROFILES, tp_json)
        file_hashes[PATH_TARIFF_PROFILES] = _sha256_bytes(tp_json)

        lr_json = _json_dumps(last_run_json)
        zf.writestr(PATH_LAST_RUN, lr_json)
        file_hashes[PATH_LAST_RUN] = _sha256_bytes(lr_json)

        pbuf = io.BytesIO()
        prepared_df.to_parquet(pbuf, engine="pyarrow", index=False)
        pbytes = pbuf.getvalue()
        zf.writestr(PATH_PREPARED, pbytes)
        file_hashes[PATH_PREPARED] = _sha256_bytes(pbytes)

        has_full = full_results_df is not None and len(full_results_df) > 0
        if has_full:
            fbuf = io.BytesIO()
            full_results_df.to_parquet(fbuf, engine="pyarrow", index=False)
            fbytes = fbuf.getvalue()
            zf.writestr(PATH_FULL_RESULTS, fbytes)
            file_hashes[PATH_FULL_RESULTS] = _sha256_bytes(fbytes)

        for k in opt_keys_sorted:
            obuf = io.BytesIO()
            opt_dfs[k].to_parquet(obuf, engine="pyarrow", index=False)
            ob = obuf.getvalue()
            name = f"{OPT_PREFIX}opt__{_safe_opt_filename(k)}.parquet"
            zf.writestr(name, ob)
            file_hashes[name] = _sha256_bytes(ob)

        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "app_version": _ver,
            "has_full_results": bool(has_full),
            "opt_tariff_columns": list(opt_keys_sorted),
            "file_sha256": file_hashes,
        }
        zf.writestr(MANIFEST, _json_dumps(manifest))

    return buf.getvalue()


def read_manifest_from_zip(zip_bytes: bytes) -> Dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        _validate_zip_structure_and_size(zf)
        with zf.open(MANIFEST) as f:
            return _json_loads(f.read())


def load_bundle_from_zip(zip_bytes: bytes) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and load bundle contents. Returns ``(manifest, payload)`` where payload holds
    raw bytes / dataframes for session hydration.
    """
    bio = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(bio, mode="r") as zf:
        _validate_zip_structure_and_size(zf)
        names = _zip_file_member_names(zf)
        if MANIFEST not in names:
            raise ValueError("Invalid bundle: missing manifest.json")

        manifest = _json_loads(zf.read(MANIFEST))
        ver = int(manifest.get("schema_version", -1))
        if ver != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported bundle schema_version {ver!r} (this app supports {BUNDLE_SCHEMA_VERSION})."
            )

        expected = manifest.get("file_sha256") or {}
        if not isinstance(expected, dict):
            raise ValueError("Invalid manifest: file_sha256 must be an object")

        expected_paths = _expected_bundle_paths_from_manifest(manifest)
        if names != expected_paths:
            extra = names - expected_paths
            missing = expected_paths - names
            raise ValueError(
                "Invalid bundle: zip member set does not match manifest "
                f"(extra={sorted(extra)!r}, missing={sorted(missing)!r})."
            )

        def _read_check(path: str) -> bytes:
            if path not in names:
                raise ValueError(f"Invalid bundle: missing {path}")
            raw = zf.read(path)
            exp = expected.get(path)
            if isinstance(exp, str) and len(exp) == 64:
                h = _sha256_bytes(raw)
                if h != exp:
                    raise ValueError(f"Checksum mismatch for {path}")
            return raw

        lr_raw = _read_check(PATH_LAST_RUN)
        tp_raw = _read_check(PATH_TARIFF_PROFILES)
        cons_bytes = _read_check(PATH_CONS)
        pv_bytes = _read_check(PATH_PV)
        prepared_raw = _read_check(PATH_PREPARED)

        tariff_csv_bytes: Optional[bytes] = None
        if PATH_TARIFF_CSV in names:
            tariff_csv_bytes = _read_check(PATH_TARIFF_CSV)

        last_run = _json_loads(lr_raw)
        tariff_profiles = _json_loads(tp_raw)

        prepared_df = pd.read_parquet(io.BytesIO(prepared_raw), engine="pyarrow")

        full_results_df: Optional[pd.DataFrame] = None
        if bool(manifest.get("has_full_results")) and PATH_FULL_RESULTS in names:
            full_raw = _read_check(PATH_FULL_RESULTS)
            full_results_df = pd.read_parquet(io.BytesIO(full_raw), engine="pyarrow")

        opt_keys: List[str] = list(manifest.get("opt_tariff_columns") or [])
        opt_dfs: Dict[str, pd.DataFrame] = {}
        for tcol in opt_keys:
            fname = f"{OPT_PREFIX}opt__{_safe_opt_filename(tcol)}.parquet"
            if fname not in names:
                raise ValueError(f"Invalid bundle: missing optimizer parquet for {tcol!r} ({fname})")
            raw = _read_check(fname)
            opt_dfs[tcol] = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")

        if set(opt_dfs.keys()) != set(opt_keys):
            raise ValueError("Optimizer parquet set does not match manifest opt_tariff_columns.")

    payload = {
        "cons_bytes": cons_bytes,
        "pv_bytes": pv_bytes,
        "tariff_csv_bytes": tariff_csv_bytes,
        "prepared_df": prepared_df,
        "opt_dfs": opt_dfs,
        "full_results_df": full_results_df,
        "last_run": last_run,
        "tariff_profiles": tariff_profiles,
        "manifest": manifest,
    }
    return manifest, payload


def battery_settings_to_json_dict(bs: Any) -> Dict[str, Any]:
    """``bs`` is a :class:`BatterySettings` dataclass instance."""
    return asdict(bs)

