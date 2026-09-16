#!/usr/bin/env python
"""Combine a run's dashboards into one multi-page PDF.

Usage:
    python combine_dashboards.py runs/<run_dir> [more run dirs...]

Writes <run_dir>/dashboard_combined.pdf: the full-period dashboard.png first
(if present), then dashboard_YYYY.png in year order. Built from the PNGs so
it works for runs that finished before compare_models started writing
dashboard_combined.pdf itself (and for runs that crashed before the
full-period dashboard was written).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_YEAR = re.compile(r"^dashboard_(\d{4})\.png$")


def pages_for(run_dir: Path) -> list[Path]:
    yearly = sorted(
        (int(m.group(1)), p)
        for p in run_dir.iterdir()
        if (m := _YEAR.match(p.name))
    )
    pages = [p for _, p in yearly]
    full = run_dir / "dashboard.png"
    return ([full] if full.exists() else []) + pages


def combine(run_dir: Path) -> Path | None:
    pages = pages_for(run_dir)
    if not pages:
        print(f"{run_dir}: no dashboard PNGs found, skipped")
        return None
    out = run_dir / "dashboard_combined.pdf"
    with PdfPages(out) as pdf:
        for page in pages:
            image = mpimg.imread(page)
            height, width = image.shape[:2]
            dpi = 180
            figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            axis = figure.add_axes([0, 0, 1, 1])
            axis.imshow(image)
            axis.axis("off")
            pdf.savefig(figure, dpi=dpi)
            plt.close(figure)
    print(f"{run_dir}: {len(pages)} pages -> {out}")
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        combine(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
