"""
Dynamic SWOT Analysis Engine.

Implements Wu et al. (2024) R1-R8 SWOT classification rules:
  - Compares focal hotel vs competitor hotels
  - Classifies each attribute into S/W/O/T
  - Supports multi-period dynamic analysis

Reference:
  Wu, J., Zhao, N., & Yang, T. (2024). Wisdom of crowds: SWOT analysis
  based on hybrid text mining methods using online reviews.
  Journal of Business Research, 171, 114378.
"""

import pandas as pd
import os

from hotel_ipa.constants import STANDARD_ATTRIBUTES, SWOT_PERIODS
from hotel_ipa.swot.swot_detector import load_and_normalize


# ============================================================================
# Performance Calculation
# ============================================================================

def calculate_attribute_performance(df: pd.DataFrame, hotel_name: str = None) -> pd.DataFrame:
    """
    Calculate per-attribute performance metrics for a hotel.

    Returns DataFrame with columns:
    [屬性, 平均績效, 平均重要度, 提及次數, 正向率%]
    """
    if hotel_name:
        df = df[df['Hotel Name'] == hotel_name]

    stats = []
    for attr in STANDARD_ATTRIBUTES:
        cat = df[df['Category'] == attr]
        if len(cat) == 0:
            continue

        n = len(cat)
        sentiments = cat['Sentiment'].value_counts()
        pos = sentiments.get('正面', 0) + sentiments.get('正向', 0)

        stats.append({
            '屬性': attr,
            '平均績效': round(cat['Score'].mean(), 4),
            '平均重要度': round(cat['Importance'].mean(), 4) if 'Importance' in cat.columns else 3.0,
            '提及次數': n,
            '正向率%': round(pos / n * 100, 1),
        })

    return pd.DataFrame(stats)


# ============================================================================
# SWOT Classification Rules (R1-R8)
# ============================================================================

def classify_internal(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each attribute as internal strength or weakness.

    Rule: If attribute performance >= mean performance → Strength
          If attribute performance < mean performance → Weakness
    """
    threshold = perf_df['平均績效'].mean()
    perf_df = perf_df.copy()
    perf_df['內部分類'] = perf_df['平均績效'].apply(
        lambda p: '優勢' if p >= threshold else '劣勢'
    )
    perf_df['績效門檻'] = round(threshold, 4)
    return perf_df


def apply_swot_rules(focal_df: pd.DataFrame, competitor_df: pd.DataFrame,
                     focal_name: str, competitor_name: str) -> pd.DataFrame:
    """
    Apply R1-R8 SWOT classification rules.

    Args:
        focal_df: Focal hotel's performance with internal classification
        competitor_df: Competitor hotel's performance with internal classification
        focal_name: Focal hotel name
        competitor_name: Competitor hotel name

    Returns:
        DataFrame with SWOT classification per attribute
    """
    results = []

    for attr in STANDARD_ATTRIBUTES:
        f_row = focal_df[focal_df['屬性'] == attr]
        c_row = competitor_df[competitor_df['屬性'] == attr]

        if len(f_row) == 0 or len(c_row) == 0:
            continue

        f_perf = f_row.iloc[0]['平均績效']
        c_perf = c_row.iloc[0]['平均績效']
        f_class = f_row.iloc[0]['內部分類']
        c_class = c_row.iloc[0]['內部分類']
        perf_diff = round(f_perf - c_perf, 4)

        # R1-R8 rules
        if f_class == '優勢' and c_class == '優勢':
            if f_perf >= c_perf:
                swot = 'S'  # R1: Both strength, focal >= competitor
                rule = 'R1'
            else:
                swot = 'T'  # R2: Both strength, focal < competitor
                rule = 'R2'
        elif f_class == '優勢' and c_class == '劣勢':
            if f_perf >= c_perf:
                swot = 'O'  # R3: Focal strength + competitor weakness, focal >= comp
                rule = 'R3'
            else:
                swot = 'T'  # R4: Focal strength + competitor weakness, focal < comp
                rule = 'R4'
        elif f_class == '劣勢' and c_class == '劣勢':
            if f_perf < c_perf:
                swot = 'W'  # R5: Both weakness, focal < competitor
                rule = 'R5'
            else:
                swot = 'O'  # R6: Both weakness, focal >= competitor
                rule = 'R6'
        elif f_class == '劣勢' and c_class == '優勢':
            if f_perf < c_perf:
                swot = 'T'  # R7: Focal weakness + competitor strength, focal < comp
                rule = 'R7'
            else:
                swot = 'O'  # R8: Focal weakness + competitor strength, focal >= comp
                rule = 'R8'
        else:
            swot = '?'
            rule = '?'

        results.append({
            '屬性': attr,
            '焦點酒店': focal_name,
            '競爭酒店': competitor_name,
            '焦點績效': round(f_perf, 2),
            '競爭績效': round(c_perf, 2),
            '績效差異': round(perf_diff, 2),
            '焦點內部': f_class,
            '競爭內部': c_class,
            'SWOT': swot,
            '判定規則': rule,
        })

    return pd.DataFrame(results)


# ============================================================================
# Multi-Hotel SWOT Comparison
# ============================================================================

def run_swot_analysis(df: pd.DataFrame, focal_hotel: str = None) -> dict:
    """
    Run SWOT analysis comparing the focal hotel against all competitors.

    Args:
        df: Normalized review data (from swot_detector)
        focal_hotel: Focal hotel name. If None, uses the first hotel.

    Returns:
        dict with keys:
          'focal_hotel': str
          'hotel_performances': {hotel_name: perf_df}
          'swot_results': {competitor_name: swot_df}
          'swot_summary': DataFrame (attribute × SWOT counts)
    """
    hotels = df['Hotel Name'].unique().tolist()
    if focal_hotel is None:
        focal_hotel = hotels[0]
    if focal_hotel not in hotels:
        raise ValueError(f"焦點酒店 '{focal_hotel}' 不在資料中。可選: {hotels}")

    competitors = [h for h in hotels if h != focal_hotel]
    print(f"\n🎯 焦點酒店: {focal_hotel}")
    print(f"   競爭酒店: {', '.join(competitors)}")

    # Calculate performance for all hotels
    hotel_perfs = {}
    for hotel in hotels:
        perf = calculate_attribute_performance(df, hotel)
        perf = classify_internal(perf)
        hotel_perfs[hotel] = perf

    # Apply SWOT rules: focal vs each competitor
    swot_results = {}
    for comp in competitors:
        swot_df = apply_swot_rules(
            hotel_perfs[focal_hotel], hotel_perfs[comp],
            focal_hotel, comp
        )
        swot_results[comp] = swot_df
        s_count = (swot_df['SWOT'] == 'S').sum()
        w_count = (swot_df['SWOT'] == 'W').sum()
        o_count = (swot_df['SWOT'] == 'O').sum()
        t_count = (swot_df['SWOT'] == 'T').sum()
        print(f"   vs {comp}: S={s_count} W={w_count} O={o_count} T={t_count}")

    # Summary: for each attribute, count S/W/O/T across all comparisons
    summary_rows = []
    for attr in STANDARD_ATTRIBUTES:
        counts = {'屬性': attr, 'S': 0, 'W': 0, 'O': 0, 'T': 0}
        for swot_df in swot_results.values():
            row = swot_df[swot_df['屬性'] == attr]
            if len(row) > 0:
                swot = row.iloc[0]['SWOT']
                if swot in counts:
                    counts[swot] += 1
        # Dominant SWOT = most frequent
        swot_vals = {k: counts[k] for k in ['S', 'W', 'O', 'T']}
        counts['主要SWOT'] = max(swot_vals, key=swot_vals.get) if any(swot_vals.values()) else '?'
        summary_rows.append(counts)

    summary_df = pd.DataFrame(summary_rows)

    return {
        'focal_hotel': focal_hotel,
        'hotel_performances': hotel_perfs,
        'swot_results': swot_results,
        'swot_summary': summary_df,
    }


# ============================================================================
# Dynamic SWOT (Multi-Period)
# ============================================================================

def run_dynamic_swot(df: pd.DataFrame, focal_hotel: str = None,
                     periods: dict = None) -> dict:
    """
    Run SWOT analysis across multiple time periods.

    Args:
        df: Normalized review data with Date column
        focal_hotel: Focal hotel name
        periods: Period definitions {name: {start, end}}. Uses SWOT_PERIODS if None.

    Returns:
        dict with keys:
          'periods': list of period names
          'period_results': {period_name: swot_analysis_result}
          'trend': DataFrame showing SWOT classification per attribute per period
    """
    if periods is None:
        periods = SWOT_PERIODS

    # Ensure Date is datetime
    df = df.copy()
    if 'Date' not in df.columns:
        raise KeyError(
            "動態 SWOT 需要 'Date' 欄位。請重新執行 "
            "python -m hotel_ipa.visualization.ipa_dashboard 以生成含 Date 的資料。"
        )
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    print(f"\n📅 動態 SWOT 分析 ({len(periods)} 個時期)")

    period_results = {}
    trend_data = []

    for period_name, period_range in periods.items():
        start = pd.to_datetime(period_range['start'])
        end = pd.to_datetime(period_range['end'])
        period_df = df[(df['Date'] >= start) & (df['Date'] <= end)]

        if len(period_df) == 0:
            print(f"   ⚠️ {period_name}: 無資料")
            continue

        hotels_in_period = period_df['Hotel Name'].nunique()
        print(f"   {period_name}: {len(period_df)} 筆，{hotels_in_period} 家酒店")

        if hotels_in_period < 2:
            print(f"      ⚠️ 至少需要 2 家酒店，跳過")
            continue

        result = run_swot_analysis(period_df, focal_hotel)
        period_results[period_name] = result

        # Collect trend data
        for _, row in result['swot_summary'].iterrows():
            trend_data.append({
                '時期': period_name,
                '屬性': row['屬性'],
                '主要SWOT': row['主要SWOT'],
                'S': row['S'], 'W': row['W'],
                'O': row['O'], 'T': row['T'],
            })

    trend_df = pd.DataFrame(trend_data)

    return {
        'periods': list(period_results.keys()),
        'period_results': period_results,
        'trend': trend_df,
    }


# ============================================================================
# Export
# ============================================================================

def export_swot_results(analysis: dict, output_dir: str):
    """Export SWOT analysis results to CSV + Excel."""
    os.makedirs(output_dir, exist_ok=True)

    focal = analysis.get('focal_hotel', 'unknown')

    # If it's a static analysis (no periods)
    if 'swot_summary' in analysis:
        # CSV intermediate
        summary_csv = os.path.join(output_dir, "swot_summary.csv")
        analysis['swot_summary'].to_csv(summary_csv, index=False, encoding='utf-8-sig')

        # Excel final report
        excel_path = os.path.join(output_dir, "swot_report.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            analysis['swot_summary'].to_excel(writer, sheet_name='SWOT彙總', index=False)
            for comp_name, swot_df in analysis['swot_results'].items():
                sheet = f'vs_{comp_name[:25]}'
                swot_df.to_excel(writer, sheet_name=sheet, index=False)
            for hotel, perf_df in analysis['hotel_performances'].items():
                sheet = f'績效_{hotel[:25]}'
                perf_df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"✅ SWOT 報告: {excel_path}")

    # If it's a dynamic analysis (with periods)
    if 'trend' in analysis:
        trend_csv = os.path.join(output_dir, "swot_trend.csv")
        analysis['trend'].to_csv(trend_csv, index=False, encoding='utf-8-sig')

        excel_path = os.path.join(output_dir, "swot_dynamic_report.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            analysis['trend'].to_excel(writer, sheet_name='SWOT趨勢', index=False)
            for period_name, result in analysis.get('period_results', {}).items():
                result['swot_summary'].to_excel(
                    writer, sheet_name=f'{period_name[:25]}_彙總', index=False
                )
        print(f"✅ 動態 SWOT 報告: {excel_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    from hotel_ipa.config_loader import load_config
    cfg = load_config()
    c = cfg["swot"]

    df = load_and_normalize(c["input_file"])

    focal = c.get("focal_hotel")
    periods = c.get("periods")

    if periods:
        result = run_dynamic_swot(df, focal_hotel=focal, periods=periods)
    else:
        # Run both static and dynamic with default periods
        print("\n" + "=" * 70)
        print("靜態 SWOT 分析")
        print("=" * 70)
        result = run_swot_analysis(df, focal_hotel=focal)

        print("\n" + "=" * 70)
        print("動態 SWOT 分析")
        print("=" * 70)
        dynamic = run_dynamic_swot(df, focal_hotel=focal)
        result['trend'] = dynamic['trend']
        result['period_results'] = dynamic['period_results']
        result['periods'] = dynamic['periods']

    export_swot_results(result, c["output_dir"])

    from hotel_ipa.visualization.swot_visualization import generate_swot_visualizations
    generate_swot_visualizations(result, c["output_dir"])

    print("\n✅ SWOT 分析完成！")


if __name__ == "__main__":
    main()
