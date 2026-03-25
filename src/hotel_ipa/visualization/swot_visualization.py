"""
SWOT Visualization - Static PNG charts + Interactive HTML dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import os
from datetime import datetime

from hotel_ipa.constants import STANDARD_ATTRIBUTES

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

SWOT_COLORS = {'S': '#27ae60', 'W': '#e74c3c', 'O': '#3498db', 'T': '#f39c12'}
SWOT_LABELS = {'S': '優勢 Strength', 'W': '劣勢 Weakness',
               'O': '機會 Opportunity', 'T': '威脅 Threat'}
SWOT_MARKERS = {'S': 'o', 'W': 's', 'O': 'D', 'T': '^'}


# ============================================================================
# Wu (2024) Style Charts
# ============================================================================

def _quadrant_labels(ax):
    """Add Wu (2024) quadrant background labels to a SWOT scatter plot."""
    props = dict(fontweight='bold', fontstyle='italic', ha='center', va='center', zorder=0)
    c = '#d5d5d5'
    ax.text(0.75, 0.75, 'Strength/Threat', transform=ax.transAxes,
            fontsize=15, color=c, **props)
    ax.text(0.25, 0.75, 'Opportunity/Threat', transform=ax.transAxes,
            fontsize=15, color=c, **props)
    ax.text(0.25, 0.25, 'Weakness/Opportunity', transform=ax.transAxes,
            fontsize=15, color=c, **props)
    ax.text(0.75, 0.25, 'Threat/Opportunity', transform=ax.transAxes,
            fontsize=15, color=c, **props)


def _swot_legend(ax):
    """Add marker-shape + color legend for SWOT types."""
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SWOT_COLORS['S'],
                   markersize=9, markeredgecolor='k', label='Strength'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=SWOT_COLORS['T'],
                   markersize=9, markeredgecolor='k', label='Threat'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=SWOT_COLORS['W'],
                   markersize=9, markeredgecolor='k', label='Weakness'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=SWOT_COLORS['O'],
                   markersize=9, markeredgecolor='k', label='Opportunity'),
    ]
    ax.legend(handles=elements, loc='upper right', fontsize=9, framealpha=0.9)


    # (removed _imp_to_size — uniform marker size now)


def plot_swot_performance_bar(focal_perf: pd.DataFrame, comp_perf: pd.DataFrame,
                               focal_name: str, comp_name: str,
                               output_path: str = None):
    """Wu (2024) Fig. 6 style: horizontal bar chart comparing attribute performance."""
    merged = focal_perf[['屬性', '平均績效']].merge(
        comp_perf[['屬性', '平均績效']], on='屬性', suffixes=('_focal', '_comp'))

    attrs = merged['屬性'].tolist()
    f_vals = merged['平均績效_focal'].tolist()
    c_vals = merged['平均績效_comp'].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(attrs))
    h = 0.35

    bars_f = ax.barh(y - h/2, f_vals, h, label=focal_name, color='#555')
    bars_c = ax.barh(y + h/2, c_vals, h, label=comp_name, color='#bbb')

    for bars, fmt_c in [(bars_f, '#333'), (bars_c, '#666')]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{w:.2f}', va='center', fontsize=8.5, color=fmt_c)

    ax.set_yticks(y)
    ax.set_yticklabels(attrs, fontsize=10)
    ax.set_xlabel('Attribute Performance', fontsize=11)
    ax.set_title(f'Attribute Performance: {focal_name} vs. {comp_name}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(f_vals), max(c_vals)) * 1.12)
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ SWOT 績效對比圖: {output_path}")
    plt.close()


def plot_dynamic_swot(period_swot_data: list, focal_name: str, comp_name: str,
                       importance: dict = None, output_path: str = None):
    """
    Wu (2024) Fig. 7-9 style: Dynamic SWOT scatter with arrows.

    - Color = SWOT type (S=green, W=red, O=blue, T=orange) at last period
    - Marker shape = SWOT type at each period
    - Arrows connect same attribute across periods
    - Labels at first period point, adjustText avoids overlapping
    """
    if not period_swot_data:
        return

    # Collect per-attribute trajectories across periods
    trajectories = {}  # attr → [(delta_c, delta_f, swot, period), ...]
    for pd_item in period_swot_data:
        period = pd_item['period']
        swot_df = pd_item['swot_df']
        f_threshold = float(pd_item['focal_perf']['績效門檻'].iloc[0])
        c_threshold = float(pd_item['comp_perf']['績效門檻'].iloc[0])

        for _, row in swot_df.iterrows():
            attr = row['屬性']
            delta_f = float(row['焦點績效']) - f_threshold
            delta_c = float(row['競爭績效']) - c_threshold
            swot = row['SWOT']
            if attr not in trajectories:
                trajectories[attr] = []
            trajectories[attr].append((delta_c, delta_f, swot, period))

    if not trajectories:
        return

    fig, ax = plt.subplots(figsize=(11, 9))
    _quadrant_labels(ax)

    texts = []  # for adjustText
    label_xs, label_ys = [], []

    for attr, pts in trajectories.items():
        # Draw arrows connecting consecutive periods
        for i in range(len(pts) - 1):
            x0, y0 = pts[i][0], pts[i][1]
            x1, y1 = pts[i+1][0], pts[i+1][1]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#888',
                                        lw=1.2, shrinkA=4, shrinkB=4),
                        zorder=3)

        # Draw markers at each period with SWOT color
        for x, y, swot, period in pts:
            marker = SWOT_MARKERS.get(swot, 'o')
            color = SWOT_COLORS.get(swot, '#999')
            ax.scatter(x, y, marker=marker, s=70, c=color,
                       edgecolors='black', linewidths=0.5, zorder=5, alpha=0.85)

        # Label at FIRST point (start of trajectory)
        fx, fy = pts[0][0], pts[0][1]
        t = ax.text(fx, fy, attr, fontsize=8.5, fontweight='bold',
                    color='#222', zorder=6)
        texts.append(t)
        label_xs.append(fx)
        label_ys.append(fy)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)

    # Auto-adjust labels to avoid overlap
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=label_xs, y=label_ys, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.5),
                    force_points=(1.0, 1.0), force_text=(0.6, 0.6),
                    expand_points=(2.0, 2.0), expand_text=(1.5, 1.5))
    except ImportError:
        for t in texts:
            x0, y0 = t.get_position()
            t.set_position((x0 + 0.03, y0 + 0.03))

    ax.set_xlabel(f'ΔPerf ({comp_name})', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'ΔPerf ({focal_name})', fontsize=11, fontweight='bold')
    ax.set_title(f'Dynamic SWOT ({focal_name} vs. {comp_name})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.15)
    _swot_legend(ax)

    # Add padding to axes so labels have room
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad_x = (xmax - xmin) * 0.08
    pad_y = (ymax - ymin) * 0.08
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Dynamic SWOT: {output_path}")
    plt.close()


# ============================================================================
# Static PNG Charts
# ============================================================================

def plot_swot_matrix(swot_summary: pd.DataFrame, focal_hotel: str,
                     output_path: str = None):
    """2x2 SWOT matrix chart with attributes placed in quadrants."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{focal_hotel} - SWOT 分析矩陣', fontsize=16, fontweight='bold', y=0.98)

    quadrants = [
        ('S', axes[0, 0], '優勢 (Strengths)', '#27ae60'),
        ('W', axes[0, 1], '劣勢 (Weaknesses)', '#e74c3c'),
        ('O', axes[1, 0], '機會 (Opportunities)', '#3498db'),
        ('T', axes[1, 1], '威脅 (Threats)', '#f39c12'),
    ]

    for swot_type, ax, title, color in quadrants:
        attrs = swot_summary[swot_summary['主要SWOT'] == swot_type]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=10)
        ax.set_facecolor(color + '08')
        ax.set_xticks([])
        ax.set_yticks([])

        if len(attrs) == 0:
            ax.text(0.5, 0.5, '（無）', ha='center', va='center',
                    fontsize=14, color='#999', style='italic')
        else:
            for i, (_, row) in enumerate(attrs.iterrows()):
                y = 0.85 - i * (0.7 / max(len(attrs), 1))
                ax.text(0.08, y, f"● {row['屬性']}", fontsize=13,
                        fontweight='bold', color=color, va='center')

        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ SWOT 矩陣圖: {output_path}")
    plt.close()


def plot_swot_comparison(swot_results: dict, focal_hotel: str,
                         output_path: str = None):
    """Bar chart showing SWOT classification per competitor."""
    competitors = list(swot_results.keys())
    if not competitors:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(competitors))
    w = 0.2

    for i, (swot_type, color) in enumerate(SWOT_COLORS.items()):
        counts = []
        for comp in competitors:
            df = swot_results[comp]
            counts.append((df['SWOT'] == swot_type).sum())
        ax.bar(x + i * w, counts, w, label=SWOT_LABELS[swot_type],
               color=color, alpha=0.8)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f'vs {c}' for c in competitors], rotation=30, ha='right')
    ax.set_ylabel('屬性數量', fontsize=12, fontweight='bold')
    ax.set_title(f'{focal_hotel} - SWOT 競爭對比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ SWOT 競爭對比圖: {output_path}")
    plt.close()


def plot_swot_trend(trend_df: pd.DataFrame, focal_hotel: str,
                    output_path: str = None):
    """Line chart showing SWOT changes across time periods."""
    if trend_df.empty:
        return

    periods = trend_df['時期'].unique()
    attrs = [a for a in STANDARD_ATTRIBUTES if a in trend_df['屬性'].values]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Count S/W/O/T per period
    period_counts = []
    for period in periods:
        p_df = trend_df[trend_df['時期'] == period]
        counts = p_df['主要SWOT'].value_counts()
        period_counts.append({
            '時期': period,
            'S': counts.get('S', 0), 'W': counts.get('W', 0),
            'O': counts.get('O', 0), 'T': counts.get('T', 0),
        })
    pc_df = pd.DataFrame(period_counts)

    x = np.arange(len(periods))
    for swot_type, color in SWOT_COLORS.items():
        if swot_type in pc_df.columns:
            ax.plot(x, pc_df[swot_type], 'o-', color=color, linewidth=2.5,
                    markersize=10, label=SWOT_LABELS[swot_type], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=11)
    ax.set_ylabel('屬性數量', fontsize=12, fontweight='bold')
    ax.set_title(f'{focal_hotel} - SWOT 動態變化趨勢', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ SWOT 趨勢圖: {output_path}")
    plt.close()


def plot_attribute_swot_heatmap(trend_df: pd.DataFrame, focal_hotel: str,
                                output_path: str = None):
    """Heatmap showing each attribute's SWOT across periods."""
    if trend_df.empty:
        return

    periods = trend_df['時期'].unique()
    attrs = [a for a in STANDARD_ATTRIBUTES if a in trend_df['屬性'].values]

    swot_to_num = {'S': 3, 'O': 2, 'T': 1, 'W': 0, '?': -1}
    matrix = []
    for attr in attrs:
        row = []
        for period in periods:
            cell = trend_df[(trend_df['屬性'] == attr) & (trend_df['時期'] == period)]
            if len(cell) > 0:
                row.append(swot_to_num.get(cell.iloc[0]['主要SWOT'], -1))
            else:
                row.append(-1)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#cccccc', '#f39c12', '#3498db', '#27ae60'])

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, fontsize=11)
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels(attrs, fontsize=11)

    # Add text annotations
    for i, attr in enumerate(attrs):
        for j, period in enumerate(periods):
            cell = trend_df[(trend_df['屬性'] == attr) & (trend_df['時期'] == period)]
            if len(cell) > 0:
                swot = cell.iloc[0]['主要SWOT']
                ax.text(j, i, swot, ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white')

    ax.set_title(f'{focal_hotel} - 屬性 SWOT 動態變化', fontsize=14, fontweight='bold')

    legend_elements = [
        mpatches.Patch(color='#27ae60', label='S 優勢'),
        mpatches.Patch(color='#3498db', label='O 機會'),
        mpatches.Patch(color='#f39c12', label='T 威脅'),
        mpatches.Patch(color='#cccccc', label='W 劣勢'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ SWOT 熱力圖: {output_path}")
    plt.close()


# ============================================================================
# Interactive HTML Dashboard (Plotly.js)
# ============================================================================

def generate_swot_html(analysis: dict, output_path: str):
    """Generate fully interactive SWOT HTML dashboard with Plotly.js."""
    import json

    focal = analysis.get('focal_hotel', '未知')
    summary = analysis.get('swot_summary', pd.DataFrame())
    trend_df = analysis.get('trend', pd.DataFrame())
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- Prepare data for JS ----
    # Static summary
    summary_data = []
    if not summary.empty:
        for _, row in summary.iterrows():
            summary_data.append({
                'attr': row['屬性'],
                'swot': row['主要SWOT'],
                'S': int(row['S']), 'W': int(row['W']),
                'O': int(row['O']), 'T': int(row['T']),
            })

    # Trend data
    trend_data = []
    periods = []
    if not trend_df.empty:
        periods = trend_df['時期'].unique().tolist()
        for _, row in trend_df.iterrows():
            trend_data.append({
                'period': row['時期'],
                'attr': row['屬性'],
                'swot': row['主要SWOT'],
                'S': int(row['S']), 'W': int(row['W']),
                'O': int(row['O']), 'T': int(row['T']),
            })

    # Competitor detail data
    comp_data = {}
    for comp_name, swot_df in analysis.get('swot_results', {}).items():
        comp_data[comp_name] = []
        for _, row in swot_df.iterrows():
            comp_data[comp_name].append({
                'attr': row['屬性'],
                'focal_perf': round(float(row['焦點績效']), 2),
                'comp_perf': round(float(row['競爭績效']), 2),
                'diff': round(float(row['績效差異']), 2),
                'swot': row['SWOT'],
                'rule': row['判定規則'],
            })

    attrs = [a for a in STANDARD_ATTRIBUTES]

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{focal} - SWOT Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Noto Sans TC','Microsoft YaHei','Helvetica Neue',sans-serif;
       background:#fff; color:#333; padding:32px 24px; line-height:1.6; }}
.container {{ max-width:1200px; margin:0 auto; }}

header {{ border-bottom:2px solid #333; padding-bottom:16px; margin-bottom:28px; }}
h1 {{ font-size:1.6em; font-weight:700; color:#111; }}
.subtitle {{ color:#666; font-size:0.95em; margin-top:4px; }}
.timestamp {{ color:#999; font-size:0.8em; margin-top:4px; }}

.section {{ margin-bottom:32px; }}
.section-title {{ font-size:1.1em; font-weight:600; color:#333;
                  border-bottom:1px solid #ddd; padding-bottom:8px; margin-bottom:14px; }}
.section-note {{ color:#888; font-size:0.85em; margin-bottom:12px; }}

/* SWOT grid - clean, monochrome with subtle left border */
.swot-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.swot-box {{ border:1px solid #e0e0e0; padding:18px; min-height:120px;
            border-left:4px solid #999; }}
.swot-box h3 {{ font-size:1em; font-weight:600; margin-bottom:10px; color:#333; }}
.swot-attr {{ display:inline-block; padding:4px 10px; margin:3px;
             background:#f5f5f5; border:1px solid #ddd; font-size:0.9em; }}
.swot-empty {{ color:#aaa; font-style:italic; font-size:0.9em; }}
.sb {{ border-left-color:#555; }} .sb h3::before {{ content:'S '; }}
.wb {{ border-left-color:#999; }} .wb h3::before {{ content:'W '; }}
.ob {{ border-left-color:#777; }} .ob h3::before {{ content:'O '; }}
.tb {{ border-left-color:#bbb; }} .tb h3::before {{ content:'T '; }}

/* Tabs */
.tab-row {{ display:flex; gap:4px; margin-bottom:14px; flex-wrap:wrap; }}
.tab-btn {{ padding:6px 16px; border:1px solid #ccc; border-bottom:none;
           background:#fafafa; color:#555; cursor:pointer; font-size:0.85em;
           transition:all 0.15s; user-select:none; }}
.tab-btn.active {{ background:#333; color:#fff; border-color:#333; }}
.tab-btn:hover {{ background:#eee; }}

.chart-wrap {{ border:1px solid #e0e0e0; padding:12px; margin-bottom:16px; }}

/* Detail table */
.detail-table {{ width:100%; border-collapse:collapse; font-size:0.9em; }}
.detail-table th {{ background:#f5f5f5; color:#333; padding:10px 8px;
                   text-align:left; border-bottom:2px solid #333; font-weight:600; }}
.detail-table td {{ padding:8px; border-bottom:1px solid #eee; }}
.detail-table tr:hover {{ background:#fafafa; }}
.swot-label {{ display:inline-block; padding:2px 8px; font-weight:600;
              font-size:0.85em; border:1px solid #999; min-width:24px;
              text-align:center; }}
.diff-pos {{ font-weight:600; }}
.diff-neg {{ font-weight:600; }}

/* Info grid */
.info-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.info-box {{ border:1px solid #e0e0e0; padding:14px; }}
.info-box h4 {{ font-size:0.9em; font-weight:600; color:#333; margin-bottom:8px; }}
.info-box ul {{ padding-left:18px; color:#555; font-size:0.85em; line-height:1.8; }}

footer {{ border-top:1px solid #ddd; padding-top:12px; margin-top:24px;
          text-align:center; color:#999; font-size:0.8em; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>SWOT Analysis: {focal}</h1>
        <div class="subtitle">Based on Wu et al. (2024) R1-R8 Rules</div>
        <div class="timestamp">{now}</div>
    </header>

    <div class="section">
        <div class="section-title">Static SWOT (vs All Competitors)</div>
        <div class="swot-grid" id="static-grid"></div>
    </div>

    <div class="section">
        <div class="section-title">Dynamic SWOT Heatmap</div>
        <div class="section-note">Hover for details. Rows = attributes, columns = periods.</div>
        <div class="chart-wrap">
            <div id="heatmap-chart" style="width:100%; height:550px;"></div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">SWOT by Period</div>
        <div class="section-note">Select a period to view its SWOT classification.</div>
        <div class="tab-row" id="period-tabs"></div>
        <div class="swot-grid" id="period-grid"></div>
    </div>

    <div class="section">
        <div class="section-title">SWOT Category Trend</div>
        <div class="chart-wrap">
            <div id="trend-chart" style="width:100%; height:400px;"></div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Competitor Comparison</div>
        <div class="section-note">Select a competitor to view attribute-level detail.</div>
        <div class="tab-row" id="comp-tabs"></div>
        <div class="chart-wrap">
            <div id="comp-chart" style="width:100%; height:480px;"></div>
        </div>
        <table class="detail-table" style="margin-top:14px;">
            <thead><tr>
                <th>Attribute</th><th>Focal Perf.</th><th>Comp. Perf.</th>
                <th>Diff.</th><th>SWOT</th><th>Rule</th>
            </tr></thead>
            <tbody id="comp-tbody"></tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-title">R1-R8 Classification Rules</div>
        <div class="info-grid">
            <div class="info-box">
                <h4>S (Strength) / O (Opportunity)</h4>
                <ul>
                    <li><strong>R1 (S)</strong> Both strong, focal &ge; competitor</li>
                    <li><strong>R3 (O)</strong> Focal strong + comp. weak, focal &ge; comp.</li>
                    <li><strong>R6 (O)</strong> Both weak, focal &ge; competitor</li>
                    <li><strong>R8 (O)</strong> Focal weak + comp. strong, focal &ge; comp.</li>
                </ul>
            </div>
            <div class="info-box">
                <h4>W (Weakness) / T (Threat)</h4>
                <ul>
                    <li><strong>R2 (T)</strong> Both strong, focal &lt; competitor</li>
                    <li><strong>R4 (T)</strong> Focal strong + comp. weak, focal &lt; comp.</li>
                    <li><strong>R5 (W)</strong> Both weak, focal &lt; competitor</li>
                    <li><strong>R7 (T)</strong> Focal weak + comp. strong, focal &lt; comp.</li>
                </ul>
            </div>
        </div>
    </div>

    <footer>SWOT Analysis | {focal} | {now}</footer>
</div>

<script>
const FOCAL = {json.dumps(focal, ensure_ascii=False)};
const ATTRS = {json.dumps(attrs, ensure_ascii=False)};
const PERIODS = {json.dumps(periods, ensure_ascii=False)};
const SUMMARY = {json.dumps(summary_data, ensure_ascii=False)};
const TREND = {json.dumps(trend_data, ensure_ascii=False)};
const COMP = {json.dumps(comp_data, ensure_ascii=False)};

const SWOT_NAMES = {{'S':'Strength','W':'Weakness','O':'Opportunity','T':'Threat','?':'N/A'}};
const SWOT_NUM = {{'S':3,'O':2,'T':1,'W':0,'?':-0.5}};
const FONT = {{ family:'Noto Sans TC, Helvetica Neue, sans-serif' }};

// ===== 1. Static SWOT grid =====
function renderGrid(id, data) {{
    const g = document.getElementById(id); g.innerHTML = '';
    [['S','Strengths (優勢)','sb'],['W','Weaknesses (劣勢)','wb'],
     ['O','Opportunities (機會)','ob'],['T','Threats (威脅)','tb']
    ].forEach(([t,title,cls]) => {{
        const items = data.filter(d => d.swot === t);
        const box = document.createElement('div');
        box.className = 'swot-box ' + cls;
        box.innerHTML = `<h3>${{title}}</h3>` +
            (items.length ? items.map(d => `<span class="swot-attr">${{d.attr}}</span>`).join('')
                          : '<span class="swot-empty">None</span>');
        g.appendChild(box);
    }});
}}
renderGrid('static-grid', SUMMARY);

// ===== 2. Heatmap (grayscale with text) =====
if (TREND.length > 0) {{
    const z=[], text=[], hover=[];
    ATTRS.forEach(attr => {{
        const r=[], t=[], h=[];
        PERIODS.forEach(period => {{
            const d = TREND.find(x => x.period===period && x.attr===attr);
            if (d) {{
                r.push(SWOT_NUM[d.swot]??-0.5); t.push(d.swot);
                h.push(`${{attr}} | ${{period}}<br>SWOT: ${{d.swot}} (${{SWOT_NAMES[d.swot]}})<br>S:${{d.S}} W:${{d.W}} O:${{d.O}} T:${{d.T}}`);
            }} else {{ r.push(-0.5); t.push('?'); h.push(`${{attr}} | ${{period}}<br>No data`); }}
        }});
        z.push(r); text.push(t); hover.push(h);
    }});
    Plotly.newPlot('heatmap-chart', [{{
        z:z, x:PERIODS, y:ATTRS, text:text, hovertext:hover, type:'heatmap',
        colorscale: [[0,'#d4d4d4'],[0.33,'#aaa'],[0.67,'#777'],[1,'#333']],
        zmin:0, zmax:3, showscale:false,
        hovertemplate:'%{{hovertext}}<extra></extra>',
        texttemplate:'<b>%{{text}}</b>', textfont:{{ size:15, color:'white' }},
    }}], {{
        margin:{{ l:100, r:50, t:20, b:50 }},
        yaxis:{{ autorange:'reversed', tickfont:{{ size:13 }}, dtick:1 }},
        xaxis:{{ tickfont:{{ size:13 }}, side:'top' }}, plot_bgcolor:'#fff', font:FONT,
        annotations:[
            {{ x:1.04,y:1,xref:'paper',yref:'paper',text:'S',font:{{ size:12,color:'#333' }},showarrow:false,xanchor:'left' }},
            {{ x:1.04,y:0.7,xref:'paper',yref:'paper',text:'O',font:{{ size:12,color:'#666' }},showarrow:false,xanchor:'left' }},
            {{ x:1.04,y:0.4,xref:'paper',yref:'paper',text:'T',font:{{ size:12,color:'#999' }},showarrow:false,xanchor:'left' }},
            {{ x:1.04,y:0.1,xref:'paper',yref:'paper',text:'W',font:{{ size:12,color:'#bbb' }},showarrow:false,xanchor:'left' }},
        ],
    }}, {{responsive:true}});
}}

// ===== 3. Period tabs =====
let activePeriod = PERIODS[0] || null;
function renderPeriodTabs() {{
    const c = document.getElementById('period-tabs'); c.innerHTML = '';
    PERIODS.forEach(p => {{
        const t = document.createElement('span');
        t.className = 'tab-btn' + (p===activePeriod?' active':'');
        t.textContent = p;
        t.onclick = () => {{ activePeriod=p; renderPeriodTabs(); renderGrid('period-grid', TREND.filter(d=>d.period===activePeriod)); }};
        c.appendChild(t);
    }});
}}
renderPeriodTabs();
if (activePeriod) renderGrid('period-grid', TREND.filter(d=>d.period===activePeriod));

// ===== 4. Trend chart (monochrome line styles) =====
if (TREND.length > 0) {{
    const styles = [
        {{color:'#333', dash:'solid', sym:'circle'}},
        {{color:'#333', dash:'dash', sym:'square'}},
        {{color:'#888', dash:'solid', sym:'diamond'}},
        {{color:'#888', dash:'dot', sym:'triangle-up'}},
    ];
    const traces = ['S','O','T','W'].map((t,i) => {{
        const s = styles[i];
        return {{
            x:PERIODS,
            y:PERIODS.map(p=>TREND.filter(d=>d.period===p&&d.swot===t).length),
            name:`${{t}} (${{SWOT_NAMES[t]}})`, mode:'lines+markers',
            line:{{ width:2, color:s.color, dash:s.dash }},
            marker:{{ size:9, symbol:s.sym, color:s.color }},
        }};
    }});
    Plotly.newPlot('trend-chart', traces, {{
        xaxis:{{ tickfont:{{ size:12 }}, linecolor:'#ccc', linewidth:1, mirror:true }},
        yaxis:{{ title:'Count', tickfont:{{ size:12 }}, dtick:1, linecolor:'#ccc', linewidth:1, mirror:true }},
        legend:{{ orientation:'h', y:-0.18, x:0.5, xanchor:'center', font:{{ size:11 }} }},
        margin:{{ l:45, r:20, t:15, b:65 }}, plot_bgcolor:'#fff', font:FONT, hovermode:'x unified',
    }}, {{responsive:true}});
}}

// ===== 5. Competitor comparison =====
const compNames = Object.keys(COMP);
let activeComp = compNames[0] || null;
function renderCompTabs() {{
    const c = document.getElementById('comp-tabs'); c.innerHTML = '';
    compNames.forEach(n => {{
        const t = document.createElement('span');
        t.className = 'tab-btn' + (n===activeComp?' active':'');
        t.textContent = 'vs ' + n;
        t.onclick = () => {{ activeComp=n; renderCompTabs(); renderComp(); }};
        c.appendChild(t);
    }});
}}
function renderComp() {{
    if (!activeComp) return;
    const data = COMP[activeComp];
    Plotly.react('comp-chart', [
        {{ y:data.map(d=>d.attr), x:data.map(d=>d.focal_perf), name:FOCAL,
          type:'bar', orientation:'h', marker:{{color:'#333'}},
          text:data.map(d=>d.focal_perf.toFixed(2)), textposition:'outside', textfont:{{size:11}},
          hovertemplate:'%{{y}}<br>Focal: %{{x:.2f}}<extra></extra>' }},
        {{ y:data.map(d=>d.attr), x:data.map(d=>d.comp_perf), name:activeComp,
          type:'bar', orientation:'h', marker:{{color:'#aaa'}},
          text:data.map(d=>d.comp_perf.toFixed(2)), textposition:'outside', textfont:{{size:11}},
          hovertemplate:'%{{y}}<br>Competitor: %{{x:.2f}}<extra></extra>' }}
    ], {{
        barmode:'group', bargap:0.25, bargroupgap:0.1,
        xaxis:{{ title:'Performance', range:[0,5.5], tickfont:{{size:12}}, linecolor:'#ccc', linewidth:1, mirror:true }},
        yaxis:{{ autorange:'reversed', tickfont:{{size:12}}, dtick:1 }},
        legend:{{ orientation:'h', y:-0.12, x:0.5, xanchor:'center', font:{{size:11}} }},
        margin:{{ l:100, r:50, t:10, b:55 }}, plot_bgcolor:'#fff', font:FONT,
    }}, {{responsive:true}});

    document.getElementById('comp-tbody').innerHTML = data.map(d => `<tr>
        <td>${{d.attr}}</td><td>${{d.focal_perf.toFixed(2)}}</td><td>${{d.comp_perf.toFixed(2)}}</td>
        <td class="${{d.diff>=0?'diff-pos':'diff-neg'}}">${{d.diff>=0?'+':''}}${{d.diff.toFixed(2)}}</td>
        <td><span class="swot-label">${{d.swot}}</span></td><td>${{d.rule}}</td>
    </tr>`).join('');
}}
renderCompTabs(); renderComp();
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✓ SWOT 互動式儀表板: {output_path}")


# ============================================================================
# Main: Generate all visualizations
# ============================================================================

def generate_swot_visualizations(analysis: dict, output_dir: str):
    """Generate all SWOT visualizations (PNG + HTML)."""
    os.makedirs(output_dir, exist_ok=True)
    focal = analysis.get('focal_hotel', '未知')

    # Subdirectory for static charts
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    print(f"\n📊 生成 SWOT 視覺化...")

    # Static charts
    if 'swot_summary' in analysis:
        plot_swot_matrix(analysis['swot_summary'], focal,
                         os.path.join(charts_dir, 'swot_matrix.png'))

    if 'swot_results' in analysis:
        plot_swot_comparison(analysis['swot_results'], focal,
                             os.path.join(charts_dir, 'swot_comparison.png'))

    trend_df = analysis.get('trend', pd.DataFrame())
    if not trend_df.empty:
        plot_swot_trend(trend_df, focal,
                        os.path.join(charts_dir, 'swot_trend.png'))
        plot_attribute_swot_heatmap(trend_df, focal,
                                    os.path.join(charts_dir, 'swot_heatmap.png'))

    # HTML interactive dashboard
    generate_swot_html(analysis, os.path.join(output_dir, 'swot_dashboard.html'))

    print(f"✅ SWOT 視覺化完成")
    print(f"   儀表板: {output_dir}/swot_dashboard.html")
    print(f"   圖表:   {charts_dir}/")
