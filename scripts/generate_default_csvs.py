"""One-off script to regenerate data/default_consumption.csv and data/default_pv.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")

    cons = 1.5 + 0.8 * (hours.hour / 23.0) ** 0.5
    lines = ["date,Final_Community_Sum"]
    for t, c in zip(hours, cons):
        lines.append(f"{t.strftime('%d/%m/%Y %H:%M')},{c:.6f}")
    (DATA / "default_consumption.csv").write_text("\n".join(lines), encoding="utf-8")

    pv_lines = ["PVGIS sample (synthetic),,", "time,P"]
    for t in hours:
        h = int(t.hour)
        if 7 <= h <= 17:
            wh = max(0.0, 450.0 * (1.0 - abs(h - 12) / 6.0))
        else:
            wh = 0.0
        pv_lines.append(f"{t.strftime('%Y%m%d')}:{t.strftime('%H')}00,{wh:.2f}")
    (DATA / "default_pv.csv").write_text("\n".join(pv_lines), encoding="utf-8")

    print(f"Wrote {len(hours)} hourly rows to {DATA}")


if __name__ == "__main__":
    main()
