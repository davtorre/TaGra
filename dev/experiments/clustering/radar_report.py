#!/usr/bin/env python
"""
Build clustering_radar_report.html — one Bokeh tab per dataset,
one radar polygon per clustering method.

Axes: DBCV, Silhouette, Vm(true), Coverage (1-noise%), Balance (min_cls%)
Normalization: per-dataset min-max so the best method always reaches 1.0.
Tooltips show the actual (raw) metric values.

Usage:
    cd dev/experiments/clustering
    source ../../venv/bin/activate
    python radar_report.py
"""

import os, re, math, glob
import numpy as np
from bokeh.plotting import figure, save
from bokeh.models import (ColumnDataSource, HoverTool, Legend, LegendItem,
                          Tabs, TabPanel, Span)
from bokeh.layouts import row as bk_row
from bokeh.transform import dodge
from bokeh.io import output_file
from bokeh.palettes import Category20

# ── Config ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "clustering_radar_report.html")

AXES_LABELS = ["DBCV", "Silhouette", "Vm(true)", "Coverage\n(1−noise%)", "Balance\n(min_cls%)"]
N_AXES      = len(AXES_LABELS)
# Angles: start from 12 o'clock, go counter-clockwise
ANGLES      = [math.pi / 2 - 2 * math.pi * i / N_AXES for i in range(N_AXES)]

PALETTE     = Category20[20]
GRID_LEVELS = [0.25, 0.5, 0.75, 1.0]


# ── Report parsing ─────────────────────────────────────────────────────────────

def _to_float(s: str):
    """Return float or None for N/A / dash / empty."""
    s = s.strip().rstrip('%')
    if s in ('N/A', '-', ''):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_data_row(line: str):
    """
    Parse one fixed-width data row produced by run_clustering.py.

    Format (from the source):
      "  {name:<52} {n_clusters:>4} {noise_pct:>6.1f}% {mc:>8} {dbcv:>8} {sil:>8} {vm_true:>9} {vm_ref:>8}"
    """
    if len(line) < 58:
        return None
    name = line[2:54].strip()
    if not name:
        return None
    rest = line[54:].split()
    if len(rest) < 6:
        return None
    try:
        n_clusters = int(rest[0])
    except ValueError:
        return None
    return {
        'name':        name,
        'n_clusters':  n_clusters,
        'noise_pct':   _to_float(rest[1]),
        'min_cls_pct': _to_float(rest[2]),
        'dbcv':        _to_float(rest[3]),
        'sil':         _to_float(rest[4]),
        'vm_true':     _to_float(rest[5]),
        'vm_ref':      _to_float(rest[6]) if len(rest) > 6 else None,
    }


def parse_report(path: str) -> dict:
    """
    Parse all '===…===' sections from a report file.
    Returns {section_title: [row_dicts]}.
    """
    sections     = {}
    current_key  = None
    current_rows = None
    in_data      = False

    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip('\n')

        # ── Section boundary ──────────────────────────────────────────────────
        if re.match(r'^={10,}', raw):
            in_data = False
            # Collect title lines until the '---' separator
            title_parts = []
            i += 1
            while i < len(lines):
                l = lines[i].rstrip('\n')
                if re.match(r'^-{10,}', l):
                    in_data = True
                    break
                # Skip the "Method … Cl  Noise% …" column-header line
                if not l.lstrip().startswith('Method'):
                    title_parts.append(l.strip())
                i += 1
            current_key  = ' | '.join(p for p in title_parts if p)
            current_rows = []
            if current_key:
                sections[current_key] = current_rows
            i += 1
            continue

        # ── Data rows ─────────────────────────────────────────────────────────
        if in_data:
            if raw.startswith('  ') and len(raw) > 54:
                row = _parse_data_row(raw)
                if row and current_rows is not None:
                    current_rows.append(row)
            elif raw.strip() == '' or re.match(r'^={10,}', raw):
                in_data = False
                if re.match(r'^={10,}', raw):
                    i -= 1  # re-process this line as a section boundary

        i += 1

    return sections


# ── Normalization ──────────────────────────────────────────────────────────────

# Fixed domain ranges for each axis: (min, max)
AXIS_DOMAINS = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def _domain_norm(val, lo: float, hi: float) -> float:
    """Normalize a value to [0,1] using a fixed domain. None → 0."""
    if val is None:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def normalize_rows(rows: list):
    """
    Add '_norm' (list of 5 normalized [0,1] values) and '_raw' (actual values)
    to each row dict.  Normalization uses fixed metric domains:
      DBCV [-1,1], Silhouette [-1,1], Vm(true) [0,1], Coverage [0,1], Balance [0,1]
    """
    def get(r, key, default=0.0):
        v = r.get(key)
        return default if v is None else v

    for r in rows:
        dbcv_v = get(r, 'dbcv')
        sil_v  = get(r, 'sil')
        vm_v   = get(r, 'vm_true')
        cov_v  = 1.0 - get(r, 'noise_pct', 100.0) / 100.0
        bal_v  = get(r, 'min_cls_pct', 0.0) / 100.0

        r['_norm'] = [
            _domain_norm(dbcv_v, *AXIS_DOMAINS[0]),
            _domain_norm(sil_v,  *AXIS_DOMAINS[1]),
            _domain_norm(vm_v,   *AXIS_DOMAINS[2]),
            _domain_norm(cov_v,  *AXIS_DOMAINS[3]),
            _domain_norm(bal_v,  *AXIS_DOMAINS[4]),
        ]
        r['_raw'] = {
            'dbcv':     dbcv_v,
            'sil':      sil_v,
            'vm_true':  vm_v,
            'coverage': cov_v * 100,
            'balance':  bal_v * 100,
        }


# ── Radar chart helpers ────────────────────────────────────────────────────────

def _radar_xy(values: list, angles: list):
    """Convert list of radii + angles → closed (xs, ys) polygon."""
    xs = [v * math.cos(a) for v, a in zip(values, angles)]
    ys = [v * math.sin(a) for v, a in zip(values, angles)]
    xs.append(xs[0])
    ys.append(ys[0])
    return xs, ys


def _color_map(rows: list) -> dict:
    """Return {method_name: hex_color} using the same palette order as the plots."""
    return {r['name']: PALETTE[idx % len(PALETTE)] for idx, r in enumerate(rows)}


def make_radar_figure(dataset_name: str, rows: list, color_map: dict) -> figure:
    p = figure(
        frame_width=580, frame_height=580,
        title=f"{dataset_name}  –  Clustering Radar  (axes: per-dataset min-max normalized)",
        x_range=(-1.55, 1.55),
        y_range=(-1.55, 1.55),
        toolbar_location="above",
        tools="pan,wheel_zoom,reset,save",
    )
    p.axis.visible  = False
    p.grid.visible  = False
    p.background_fill_color = "#f9f9f9"
    p.title.text_font_size  = "13px"

    # ── Concentric grid circles ───────────────────────────────────────────────
    theta = np.linspace(0, 2 * math.pi, 200)
    for level in GRID_LEVELS:
        p.line(level * np.cos(theta), level * np.sin(theta),
               color="#cccccc", line_dash="dashed", line_width=1)
        # Tick label on the DBCV axis (first axis = top)
        p.text(
            x=[math.cos(ANGLES[0]) * level - 0.04],
            y=[math.sin(ANGLES[0]) * level],
            text=[f"{level:.2f}"],
            text_align="right", text_baseline="middle",
            text_font_size="9px", text_color="#999999",
        )

    # ── Axis spokes ───────────────────────────────────────────────────────────
    for angle, label in zip(ANGLES, AXES_LABELS):
        p.line([0, math.cos(angle)], [0, math.sin(angle)],
               color="#aaaaaa", line_width=1.5)
        lx = 1.22 * math.cos(angle)
        ly = 1.22 * math.sin(angle)
        # Alignment based on position
        if   lx >  0.15: align = "left"
        elif lx < -0.15: align = "right"
        else:            align = "center"
        if   ly >  0.15: baseline = "bottom"
        elif ly < -0.15: baseline = "top"
        else:            baseline = "middle"
        # Multi-line labels: split on \n
        for k, part in enumerate(label.split('\n')):
            offset = k * 0.12
            p.text(
                x=[lx], y=[ly - offset],
                text=[part],
                text_align=align, text_baseline=baseline,
                text_font_size="11px", text_color="#333333",
                text_font_style="bold",
            )

    # ── One polygon per method ────────────────────────────────────────────────
    legend_items  = []
    patch_renderers = []
    for idx, r in enumerate(rows):
        color   = color_map[r['name']]
        xs, ys  = _radar_xy(r['_norm'], ANGLES)
        raw     = r['_raw']

        src = ColumnDataSource(dict(
            xs         = [xs],
            ys         = [ys],
            name       = [r['name']],
            n_clusters = [r['n_clusters']],
            noise_pct  = [r.get('noise_pct') or float('nan')],
            dbcv       = [raw['dbcv']],
            sil        = [raw['sil']],
            vm_true    = [raw['vm_true']],
            coverage   = [raw['coverage']],
            balance    = [raw['balance']],
        ))

        is_dbscan = 'DBSCAN' in r['name']
        renderer = p.patches(
            'xs', 'ys', source=src,
            fill_color=color, fill_alpha=0.15,
            line_color=color, line_width=2,
            visible=is_dbscan,
        )
        patch_renderers.append(renderer)
        legend_items.append(LegendItem(label=r['name'], renderers=[renderer]))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = Legend(
        items=legend_items,
        location="top_left",
        click_policy="hide",
        label_text_font_size="10px",
        spacing=2,
    )
    p.add_layout(legend, 'right')

    # ── Hover (polygons only) ─────────────────────────────────────────────────
    hover = HoverTool(
        renderers=patch_renderers,
        tooltips=[
            ("Method",     "@name"),
            ("Clusters",   "@n_clusters"),
            ("DBCV",       "@dbcv{0.4f}"),
            ("Silhouette", "@sil{0.4f}"),
            ("Vm(true)",   "@vm_true{0.4f}"),
            ("Coverage",   "@coverage{0.1f}%  (=1−noise%)"),
            ("Balance",    "@balance{0.1f}%  (min cluster)"),
        ],
    )
    p.add_tools(hover)

    return p


def make_bar_figure(dataset_name: str, rows: list, color_map: dict) -> figure:
    """
    Grouped bar chart: x = metric, color = method.
    Bar heights are domain-normalized values (same scale as the radar axes).
    Tooltip shows the raw values.
    """
    metric_labels = ["DBCV", "Silhouette", "Vm(true)", "Coverage", "Balance"]
    raw_keys      = ["dbcv", "sil", "vm_true", "coverage", "balance"]
    raw_fmts      = [".4f",  ".4f",  ".4f",    ".1f",      ".1f"   ]
    raw_units     = ["",     "",     "",        "%",        "%"     ]

    n = len(rows)
    bar_w    = 0.7 / n          # width of one bar
    x_offset = -0.35 + bar_w / 2  # offset of first bar in each group

    p = figure(
        x_range=metric_labels,
        frame_width=max(480, n * 28), frame_height=520,
        title=f"{dataset_name}  –  Metric Comparison  (domain-normalized)",
        toolbar_location="above",
        tools="pan,wheel_zoom,reset,save",
        y_range=(0, 1.08),
    )
    p.background_fill_color = "#f9f9f9"
    p.title.text_font_size  = "13px"
    p.xaxis.major_label_text_font_size = "11px"
    p.yaxis.axis_label      = "Normalized value  [0 – 1]"
    p.xgrid.grid_line_color = None

    # Zero reference line
    p.add_layout(Span(location=0, dimension='width',
                      line_color='#888888', line_width=1, line_dash='solid'))

    legend_items = []
    bar_renderers = []

    for i, r in enumerate(rows):
        color  = color_map[r['name']]
        offset = x_offset + i * bar_w
        norm_vals = r['_norm']
        raw       = r['_raw']

        src = ColumnDataSource(dict(
            metrics   = metric_labels,
            values    = norm_vals,
            name      = [r['name']]         * len(metric_labels),
            n_clusters= [r['n_clusters']]   * len(metric_labels),
            **{k: [raw[k]] * len(metric_labels) for k in raw_keys},
        ))

        is_dbscan = 'DBSCAN' in r['name']
        rend = p.vbar(
            x=dodge('metrics', offset, range=p.x_range),
            top='values', bottom=0,
            width=bar_w * 0.88,
            source=src,
            color=color, alpha=0.8,
            line_color='white', line_width=0.5,
            visible=is_dbscan,
        )
        bar_renderers.append(rend)
        legend_items.append(LegendItem(label=r['name'], renderers=[rend]))

    legend = Legend(
        items=legend_items,
        click_policy="hide",
        label_text_font_size="10px",
        spacing=2,
    )
    p.add_layout(legend, 'right')

    hover = HoverTool(
        renderers=bar_renderers,
        tooltips=[
            ("Method",     "@name"),
            ("Metric",     "@metrics"),
            ("Clusters",   "@n_clusters"),
            ("DBCV",       "@dbcv{0.4f}"),
            ("Silhouette", "@sil{0.4f}"),
            ("Vm(true)",   "@vm_true{0.4f}"),
            ("Coverage",   "@coverage{0.1f}%"),
            ("Balance",    "@balance{0.1f}%"),
        ],
    )
    p.add_tools(hover)

    return p


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build clustering radar HTML report")
    parser.add_argument("--quality", action="store_true",
                        help="Plot only quality-filtered results instead of all results")
    args = parser.parse_args()

    section_tag   = "QUALITY-FILTERED" if args.quality else "ALL RESULTS"
    report_title  = "Clustering Radar Report – Quality Filtered" if args.quality \
                    else "Clustering Radar Report"

    report_dirs = sorted(glob.glob(os.path.join(SCRIPT_DIR, "clustering_results_*")))
    if not report_dirs:
        print("No clustering_results_* directories found.")
        return

    tabs = []
    for rdir in report_dirs:
        reports = glob.glob(os.path.join(rdir, "*_report.txt"))
        if not reports:
            continue

        report_path  = reports[0]
        dataset_name = os.path.basename(rdir).replace("clustering_results_", "").upper()

        sections = parse_report(report_path)

        key = next((k for k in sections if section_tag in k), None)
        if key is None or not sections[key]:
            print(f"  [{dataset_name}] no {section_tag} data — skipping")
            continue

        rows = sections[key]
        print(f"  [{dataset_name}] {len(rows)} methods found")

        normalize_rows(rows)
        cmap    = _color_map(rows)
        radar   = make_radar_figure(dataset_name, rows, cmap)
        bar     = make_bar_figure(dataset_name, rows, cmap)
        tabs.append(TabPanel(child=bk_row(radar, bar), title=dataset_name))

    if not tabs:
        print("Nothing to render.")
        return

    output_file(OUTPUT_FILE, title=report_title)
    save(Tabs(tabs=tabs))
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
