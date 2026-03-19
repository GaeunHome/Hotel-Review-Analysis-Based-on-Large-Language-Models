"""
Static matplotlib chart generation for IPA analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

from hotel_ipa.constants import (
    STANDARD_ATTRIBUTES, HOTEL_STYLES,
    POSITIVE_RATE_HIGH, POSITIVE_RATE_MID,
)

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================================
# Helpers
# ============================================================================

def _draw_ipa_quadrants(ax, mean_imp, mean_perf, x_lo, x_hi, y_lo, y_hi):
    """Draw IPA quadrant backgrounds, dividers, and labels."""
    ax.fill_between([mean_imp, x_hi], mean_perf, y_hi, alpha=0.06, color='#27ae60')
    ax.fill_between([x_lo, mean_imp], mean_perf, y_hi, alpha=0.06, color='#3498db')
    ax.fill_between([x_lo, mean_imp], y_lo, mean_perf, alpha=0.06, color='#95a5a6')
    ax.fill_between([mean_imp, x_hi], y_lo, mean_perf, alpha=0.06, color='#e74c3c')
    ax.axvline(x=mean_imp, color='#888', ls='--', lw=1, alpha=0.6)
    ax.axhline(y=mean_perf, color='#888', ls='--', lw=1, alpha=0.6)
    lbl_kw = dict(fontsize=10, fontweight='bold', alpha=0.4)
    cx_r = mean_imp + (x_hi - mean_imp) / 2
    cx_l = x_lo + (mean_imp - x_lo) / 2
    ax.text(cx_r, y_hi - 0.05, '繼續保持\nKeep Up', ha='center', va='top',
            color='#27ae60', **lbl_kw)
    ax.text(cx_l, y_hi - 0.05, '過度投入\nOverkill', ha='center', va='top',
            color='#3498db', **lbl_kw)
    ax.text(cx_l, y_lo + 0.05, '低優先\nLow Priority', ha='center', va='bottom',
            color='#95a5a6', **lbl_kw)
    ax.text(cx_r, y_lo + 0.05, '集中改善\nConcentrate', ha='center', va='bottom',
            color='#e74c3c', **lbl_kw)


def _draw_attr_index_table(ax_tbl, attr_num: dict):
    """Draw attribute index legend table on a subplot."""
    ax_tbl.axis('off')
    attrs_list = list(attr_num.keys())
    cols = 4
    lines = [f"{attr_num[a]:>2}. {a}" for a in attrs_list]
    n_rows = (len(lines) + cols - 1) // cols
    grid_rows = []
    for r in range(n_rows):
        parts = []
        for c in range(cols):
            idx = r + c * n_rows
            parts.append(f'{lines[idx]:<16}' if idx < len(lines) else '')
        grid_rows.append('  '.join(parts))
    ax_tbl.text(0.5, 0.85, '屬性編號對照 (Attribute Index)',
                ha='center', va='top', fontsize=11, fontweight='bold',
                transform=ax_tbl.transAxes)
    ax_tbl.text(0.5, 0.55, '\n'.join(grid_rows),
                ha='center', va='top', fontsize=10,
                transform=ax_tbl.transAxes, linespacing=1.7)


# ============================================================================
# Bar Charts
# ============================================================================

def plot_priority_ranking(priority_df, hotel_name="全部酒店", output_path=None):
    """Horizontal bar chart of improvement priority."""
    sorted_df = priority_df.sort_values('改進優先度', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#4CAF50' if r >= POSITIVE_RATE_HIGH else '#FFC107' if r >= POSITIVE_RATE_MID else '#F44336'
              for r in sorted_df['正向率%']]

    ax.barh(sorted_df['屬性'], sorted_df['改進優先度'], color=colors, alpha=0.7)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row['改進優先度'] + 0.1, i,
                f"{row['改進優先度']:.1f}", va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('改進優先度 (重要度 x 績效缺口)', fontsize=12, fontweight='bold')
    ax.set_title(f'{hotel_name} - 改進優先度排序', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', alpha=0.7, label='正向率 >= 80%'),
        mpatches.Patch(facecolor='#FFC107', alpha=0.7, label='正向率 60-80%'),
        mpatches.Patch(facecolor='#F44336', alpha=0.7, label='正向率 < 60%')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 優先度排序圖: {output_path}")
    plt.close()


def plot_importance_ranking(priority_df, hotel_name="全部酒店", output_path=None):
    """Horizontal bar chart of importance ranking."""
    sorted_df = priority_df.sort_values('平均重要度', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(sorted_df['屬性'], sorted_df['平均重要度'], color='#2196F3', alpha=0.7)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row['平均重要度'] + 0.05, i,
                f"{row['平均重要度']:.2f} ({row['提及次數']}次)",
                va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('平均重要度', fontsize=12, fontweight='bold')
    ax.set_title(f'{hotel_name} - 顧客重視程度排序', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 5.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 重要度排序圖: {output_path}")
    plt.close()


def plot_performance_ranking(priority_df, hotel_name="全部酒店", output_path=None):
    """Horizontal bar chart of performance ranking."""
    sorted_df = priority_df.sort_values('平均績效', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#4CAF50' if s >= 4 else '#FFC107' if s >= 3 else '#F44336'
              for s in sorted_df['平均績效']]

    ax.barh(sorted_df['屬性'], sorted_df['平均績效'], color=colors, alpha=0.7)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row['平均績效'] + 0.05, i,
                f"{row['平均績效']:.2f} ({row['正向率%']}%)",
                va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('平均績效', fontsize=12, fontweight='bold')
    ax.set_title(f'{hotel_name} - 實際表現排序', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 5.5)
    ax.axvline(x=3, color='gray', ls='--', lw=1, alpha=0.5, label='及格線 (3)')
    ax.axvline(x=4, color='green', ls='--', lw=1, alpha=0.5, label='優良線 (4)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 績效排序圖: {output_path}")
    plt.close()


def plot_comprehensive_view(priority_df, hotel_name="全部酒店", output_path=None):
    """Dual-bar chart comparing importance and performance."""
    sorted_df = priority_df.sort_values('改進優先度', ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(sorted_df))
    w = 0.35

    bars1 = ax.bar(x - w / 2, sorted_df['平均重要度'], w,
                   label='重要度', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + w / 2, sorted_df['平均績效'], w,
                   label='績效', color='#FF9800', alpha=0.8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('分數 (1-5)', fontsize=12, fontweight='bold')
    ax.set_title(f'{hotel_name} - 重要度與績效對比 (按改進優先度排序)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['屬性'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 6)
    ax.axhline(y=3, color='gray', ls='--', lw=1, alpha=0.5)
    ax.axhline(y=4, color='green', ls='--', lw=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 綜合對比圖: {output_path}")
    plt.close()


# ============================================================================
# IPA Scatter Plots
# ============================================================================

def plot_hotel_ipa_scatter(priority_df, hotel_name="全部酒店", output_path=None,
                           importance_label="重要度"):
    """
    IPA scatter plot for a single hotel.
    Uniform dark dots with white numbers inside; attribute index table below.
    """
    attr_num = {a: str(i + 1) for i, a in enumerate(STANDARD_ATTRIBUTES)}

    imp = priority_df['平均重要度']
    perf = priority_df['平均績效']
    mean_imp = imp.mean()
    mean_perf = perf.mean()

    pad = 0.4
    x_lo, x_hi = imp.min() - pad, imp.max() + pad
    y_lo, y_hi = perf.min() - pad, perf.max() + pad

    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    _draw_ipa_quadrants(ax, mean_imp, mean_perf, x_lo, x_hi, y_lo, y_hi)

    dot_color = '#2c3e50'
    dot_size = 700
    ax.scatter(imp, perf, s=dot_size, c=dot_color, alpha=0.85,
               edgecolors='white', lw=2, zorder=5)

    for _, row in priority_df.iterrows():
        num = attr_num.get(row['屬性'], '?')
        ax.annotate(num, (row['平均重要度'], row['平均績效']),
                    fontsize=12, fontweight='bold', color='white',
                    ha='center', va='center', zorder=6)

    ax.set_xlabel(f'Importance ({importance_label})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (績效)', fontsize=12, fontweight='bold')
    ax.set_title(f'{hotel_name} - IPA 四象限分析', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.15, ls=':')

    ax.text(mean_imp + 0.02, y_lo + 0.02,
            f'{importance_label}平均 = {mean_imp:.2f}',
            fontsize=8, color='#888', style='italic')
    ax.text(x_lo + 0.02, mean_perf + 0.02,
            f'績效平均 = {mean_perf:.2f}',
            fontsize=8, color='#888', style='italic')

    _draw_attr_index_table(ax_tbl, attr_num)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ IPA scatter: {output_path}")
    plt.close()


def plot_multi_hotel_ipa(all_priority_dfs: dict, output_path: str = None,
                         importance_label="重要度"):
    """Academic-quality scatter plot comparing multiple hotels."""
    attr_num = {a: str(i + 1) for i, a in enumerate(STANDARD_ATTRIBUTES)}
    hotels = list(all_priority_dfs.keys())

    all_imp, all_perf = [], []
    for pdf in all_priority_dfs.values():
        all_imp.extend(pdf['平均重要度'].tolist())
        all_perf.extend(pdf['平均績效'].tolist())
    mean_imp = np.mean(all_imp)
    mean_perf = np.mean(all_perf)

    pad = 0.3
    x_lo = min(all_imp) - pad
    x_hi = max(all_imp) + pad
    y_lo = min(all_perf) - pad
    y_hi = max(all_perf) + pad

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    _draw_ipa_quadrants(ax, mean_imp, mean_perf, x_lo, x_hi, y_lo, y_hi)

    for h_idx, (hotel, pdf) in enumerate(all_priority_dfs.items()):
        s = HOTEL_STYLES[h_idx % len(HOTEL_STYLES)]
        x_vals = pdf['平均重要度'].tolist()
        y_vals = pdf['平均績效'].tolist()
        ax.scatter(x_vals, y_vals, color=s['color'], marker=s['marker'],
                   s=320, alpha=0.85, edgecolors='white', lw=1.5, zorder=5)

    handles = [
        ax.scatter([], [], color=HOTEL_STYLES[i % len(HOTEL_STYLES)]['color'],
                   marker=HOTEL_STYLES[i % len(HOTEL_STYLES)]['marker'],
                   s=200, label=h, edgecolors='white', lw=1.5)
        for i, h in enumerate(hotels)
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#ccc')

    ax.set_xlabel(f'Importance ({importance_label})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance (績效)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Hotel IPA Comparison\n'
                 '五家酒店 Importance-Performance 比較分析',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.15, ls=':')

    ax.text(mean_imp + 0.02, y_lo + 0.02,
            f'{importance_label}平均 = {mean_imp:.2f}',
            fontsize=8, color='#888', style='italic')
    ax.text(x_lo + 0.02, mean_perf + 0.02,
            f'績效平均 = {mean_perf:.2f}',
            fontsize=8, color='#888', style='italic')

    _draw_attr_index_table(ax_tbl, attr_num)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"   ✓ 多酒店 IPA 比較圖: {output_path}")
    plt.close()


def plot_attribute_hotel_comparison(all_priority_dfs: dict, output_path: str = None):
    """Dot plot showing per-attribute performance/importance across all hotels."""
    hotels = list(all_priority_dfs.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10),
                                    gridspec_kw={'width_ratios': [1, 1]})

    for ax, metric, title, xlabel in [
        (ax1, '平均績效', 'Performance Comparison\n各屬性績效比較', 'Performance (績效)'),
        (ax2, '平均重要度', 'Importance Comparison\n各屬性重要度比較', 'Importance (重要度)')
    ]:
        attrs_shown = []
        for attr in STANDARD_ATTRIBUTES:
            vals = []
            for hotel, pdf in all_priority_dfs.items():
                row = pdf[pdf['屬性'] == attr]
                if len(row) > 0:
                    vals.append(row.iloc[0][metric])
            if vals:
                attrs_shown.append(attr)

        y_pos = np.arange(len(attrs_shown))

        for i, attr in enumerate(attrs_shown):
            vals = []
            for pdf in all_priority_dfs.values():
                row = pdf[pdf['屬性'] == attr]
                if len(row) > 0:
                    vals.append(row.iloc[0][metric])
            if len(vals) >= 2:
                ax.plot([min(vals), max(vals)], [i, i],
                        color='#ddd', lw=2, zorder=1)

        for h_idx, (hotel, pdf) in enumerate(all_priority_dfs.items()):
            s = HOTEL_STYLES[h_idx % len(HOTEL_STYLES)]
            x_vals, y_vals = [], []
            for i, attr in enumerate(attrs_shown):
                row = pdf[pdf['屬性'] == attr]
                if len(row) > 0:
                    x_vals.append(row.iloc[0][metric])
                    y_vals.append(i)

            ax.scatter(x_vals, y_vals, color=s['color'], marker=s['marker'],
                       s=100, alpha=0.85, edgecolors='white', lw=0.6,
                       zorder=5, label=hotel if ax == ax1 else None)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(attrs_shown, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlim(1, 5.3)
        ax.grid(axis='x', alpha=0.2, ls=':')
        ax.invert_yaxis()

        if metric == '平均績效':
            ax.axvline(x=3, color='gray', ls='--', lw=0.8, alpha=0.4)
            ax.axvline(x=4, color='green', ls='--', lw=0.8, alpha=0.4)

    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"   ✓ 屬性比較圖: {output_path}")
    plt.close()


def plot_importance_comparison(review_imp: dict, posthoc_imp: dict,
                               output_path: str = None):
    """Bar chart comparing per-review average vs post-hoc AI importance."""
    attrs = [a for a in STANDARD_ATTRIBUTES if a in review_imp and a in posthoc_imp]
    if not attrs:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(attrs))
    w = 0.35

    vals_review = [review_imp[a] for a in attrs]
    vals_posthoc = [posthoc_imp[a] for a in attrs]

    bars1 = ax.bar(x - w / 2, vals_review, w, label='逐筆平均重要度',
                   color='#2c3e50', alpha=0.8)
    bars2 = ax.bar(x + w / 2, vals_posthoc, w, label='Post-hoc AI 重要度',
                   color='#e67e22', alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.03,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('重要度 (1-5)', fontsize=12, fontweight='bold')
    ax.set_title('重要度比較：逐筆平均 vs Post-hoc AI 整體判斷',
                 fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.2, ls=':')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 重要度比較圖: {output_path}")
    plt.close()
