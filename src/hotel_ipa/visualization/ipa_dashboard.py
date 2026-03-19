"""
IPA Dashboard - main pipeline.
Orchestrates data loading, metrics calculation, chart generation, and HTML dashboards.
"""

import pandas as pd
import os

from hotel_ipa.constants import STANDARD_ATTRIBUTES
from hotel_ipa.utils import parse_json_safe, match_category
from hotel_ipa.visualization.charts import (
    plot_priority_ranking,
    plot_importance_ranking,
    plot_performance_ranking,
    plot_comprehensive_view,
    plot_hotel_ipa_scatter,
    plot_multi_hotel_ipa,
    plot_attribute_hotel_comparison,
    plot_importance_comparison,
)
from hotel_ipa.visualization.html_dashboard import (
    generate_unified_dashboard,
)


# ============================================================================
# Data Loading
# ============================================================================

def load_classified_data(filepath: str) -> pd.DataFrame:
    """Load classified data from classify.py output."""
    print(f"📂 讀取分類結果: {filepath}")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    else:
        df = pd.read_excel(filepath, sheet_name="詳細數據")
    required = {"Hotel Name", "Target_Keyword", "Standardized_Category",
                "Sentiment", "Score", "Importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少欄位: {missing}")

    keep_cols = ["Review ID", "Hotel Name", "Target_Keyword", "Standardized_Category",
                 "Sentiment", "Score", "Importance"]
    if "AI_Raw_Tag" in df.columns:
        keep_cols.insert(3, "AI_Raw_Tag")
    if "Date" in df.columns:
        keep_cols.append("Date")

    result = df[keep_cols].copy()
    result = result.rename(columns={
        "Target_Keyword": "Keyword",
        "Standardized_Category": "Category"
    })

    result["Score"] = pd.to_numeric(result["Score"], errors="coerce").fillna(0).astype(int)
    result["Importance"] = pd.to_numeric(result["Importance"], errors="coerce").fillna(3).astype(int)
    print(f"✓ {len(result)} 條記錄")
    return result


def extract_from_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured data from raw analysis results."""
    records = []
    for _, row in df.iterrows():
        for item in parse_json_safe(row.get('分析結果', '')):
            if isinstance(item, dict):
                records.append({
                    'Hotel Name': row.get('Hotel Name', ''),
                    'Review ID': row.get('Review ID', ''),
                    'Date': row.get('Date', ''),
                    'Keyword': item.get('key', ''),
                    'Category': match_category(item.get('key', '')),
                    'Sentiment': item.get('sentiment', ''),
                    'Score': int(item.get('score', 0)),
                    'Importance': int(item.get('importance', 3))
                })
    return pd.DataFrame(records)


def _load_input(input_file: str) -> pd.DataFrame:
    """Auto-detect input format: CSV, classified Excel, or raw analysis Excel."""
    if input_file.endswith('.csv'):
        print("   偵測到 CSV 格式")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        if 'Standardized_Category' in df.columns:
            print("   偵測到分類結果格式")
            return load_classified_data(input_file)
        else:
            print("   偵測到原始分析格式，自動提取資料")
            return extract_from_analysis(df)
    else:
        xl = pd.ExcelFile(input_file)
        if "詳細數據" in xl.sheet_names:
            print("   偵測到分類結果格式")
            return load_classified_data(input_file)
        else:
            print("   偵測到原始分析格式，自動提取資料")
            df = pd.read_excel(input_file)
            return extract_from_analysis(df)


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_global_importance(df: pd.DataFrame) -> dict:
    """Calculate global importance for each attribute across ALL hotels."""
    global_imp = {}
    for attr in STANDARD_ATTRIBUTES:
        cat = df[df['Category'] == attr]
        if len(cat) > 0:
            global_imp[attr] = round(cat['Importance'].mean(), 2)
    return global_imp


def calculate_priority_metrics(df: pd.DataFrame,
                               hotel_name: str = None,
                               global_importance: dict = None) -> pd.DataFrame:
    """
    Calculate IPA priority metrics per attribute.
    Priority = Importance * (5 - Performance)
    """
    work_df = df[df['Hotel Name'] == hotel_name].copy() if hotel_name else df.copy()

    stats = []
    for attr in STANDARD_ATTRIBUTES:
        cat = work_df[work_df['Category'] == attr]
        if len(cat) == 0:
            continue

        if global_importance and attr in global_importance:
            avg_imp = global_importance[attr]
        else:
            avg_imp = round(cat['Importance'].mean(), 2)

        avg_perf = cat['Score'].mean()
        gap = 5 - avg_perf
        priority = avg_imp * gap
        n = len(cat)

        sentiments = cat['Sentiment'].value_counts()
        pos = sentiments.get('正面', 0) + sentiments.get('正向', 0)
        neu = sentiments.get('中立', 0)
        neg = sentiments.get('負面', 0) + sentiments.get('負向', 0)

        stats.append({
            '屬性': attr,
            '平均重要度': round(avg_imp, 2),
            '平均績效': round(avg_perf, 2),
            '績效缺口': round(gap, 2),
            '改進優先度': round(priority, 2),
            '提及次數': n,
            '正向數': pos, '中立數': neu, '負向數': neg,
            '正向率%': round(pos / n * 100, 1) if n else 0
        })

    return pd.DataFrame(stats)


# ============================================================================
# Post-hoc Importance
# ============================================================================

def _load_or_compute_posthoc_importance(stats_csv: str, output_dir: str,
                                         api_key: str = None) -> dict | None:
    """Load existing post-hoc importance or compute via GPT-4o."""
    posthoc_path = os.path.join(output_dir, "importance_posthoc.csv")

    if os.path.exists(posthoc_path):
        print(f"   載入既有 post-hoc 重要度: {posthoc_path}")
        df = pd.read_csv(posthoc_path, encoding='utf-8-sig')
        return dict(zip(df['屬性'], df['AI重要度']))

    if not api_key:
        print("   ⚠️ 無 API key，跳過 post-hoc 重要度計算")
        return None

    try:
        from hotel_ipa.importance.post_hoc import calculate_posthoc_importance
        print("   計算 post-hoc 重要度 (GPT-4o)...")
        result_df = calculate_posthoc_importance(
            input_file=stats_csv,
            output_file=posthoc_path,
            model="gpt-4o",
            num_runs=3,
            api_key=api_key,
        )
        if result_df.empty:
            return None
        return dict(zip(result_df['屬性'], result_df['AI重要度']))
    except Exception as e:
        print(f"   ⚠️ Post-hoc 重要度計算失敗: {e}")
        return None


# ============================================================================
# Overview Statistics
# ============================================================================

def _compute_overview_stats(df: pd.DataFrame) -> dict:
    """Compute overview statistics for the dashboard: sentiment, categories, etc."""
    # Sentiment (normalize 负面 → 負面)
    sent = df['Sentiment'].replace({'负面': '負面', '正向': '正面', '負向': '負面'}).value_counts()
    total = len(df)
    pos = int(sent.get('正面', 0))
    neg = int(sent.get('負面', 0))
    neu = int(sent.get('中立', 0))

    # Per-hotel sentiment
    hotel_sent = []
    for hotel in df['Hotel Name'].unique():
        h = df[df['Hotel Name'] == hotel].copy()
        h['Sentiment'] = h['Sentiment'].replace({'负面': '負面', '正向': '正面', '負向': '負面'})
        hs = h['Sentiment'].value_counts()
        ht = len(h)
        hp = int(hs.get('正面', 0))
        hn = int(hs.get('負面', 0))
        hne = int(hs.get('中立', 0))
        hotel_sent.append({
            'hotel': hotel, 'total': ht,
            'pos': hp, 'neg': hn, 'neu': hne,
            'pos_rate': round(hp / ht * 100, 1) if ht else 0,
        })

    # Standardized category distribution (12 標準 + 其他)
    cat_counts = df['Category'].value_counts().to_dict()

    # AI raw tag distribution (classify.py Stage 1 原始標籤)
    # Shows the full picture including non-standard tags before consolidation
    raw_tag_col = 'AI_Raw_Tag' if 'AI_Raw_Tag' in df.columns else None
    raw_tag_counts = df[raw_tag_col].value_counts().to_dict() if raw_tag_col else {}

    # Non-standard tags (tags that got consolidated or kept as 其他)
    non_std_tags = {}
    if raw_tag_col:
        for tag, cnt in raw_tag_counts.items():
            if tag not in STANDARD_ATTRIBUTES:
                non_std_tags[tag] = cnt

    # Unique reviews
    n_reviews = df['Review ID'].nunique() if 'Review ID' in df.columns else 0

    # Period trend data (performance per hotel per period per attribute)
    # 3 periods: 爆發期, 恢復期, 後疫情 (skip COVID前)
    import pandas as _pd
    TREND_PERIODS = {
        '爆發期': {'start': '2020-01-01', 'end': '2020-12-31'},
        '恢復期': {'start': '2021-01-01', 'end': '2021-12-31'},
        '後疫情': {'start': '2022-01-01', 'end': '2022-12-31'},
    }
    trend_data = []
    if 'Date' in df.columns:
        df_d = df.copy()
        df_d['Date'] = _pd.to_datetime(df_d['Date'], format='mixed', errors='coerce')
        for period_name, period_range in TREND_PERIODS.items():
            mask = (df_d['Date'] >= period_range['start']) & (df_d['Date'] <= period_range['end'])
            period_df = df_d[mask]
            for hotel in df_d['Hotel Name'].unique():
                h = period_df[period_df['Hotel Name'] == hotel]
                for attr in STANDARD_ATTRIBUTES:
                    cat = h[h['Category'] == attr]
                    if len(cat) >= 3:  # minimum sample
                        s_counts = cat['Sentiment'].replace(
                            {'负面': '負面', '正向': '正面', '負向': '負面'}
                        ).value_counts()
                        p = int(s_counts.get('正面', 0))
                        trend_data.append({
                            'period': period_name,
                            'hotel': hotel,
                            'attr': attr,
                            'perf': round(cat['Score'].mean(), 2),
                            'imp': round(cat['Importance'].mean(), 2),
                            'count': len(cat),
                            'pos_rate': round(p / len(cat) * 100, 1),
                        })

    return {
        'total_mentions': total,
        'n_reviews': n_reviews,
        'pos': pos, 'neg': neg, 'neu': neu,
        'hotel_sentiment': hotel_sent,
        'category_counts': cat_counts,
        'raw_tag_counts': raw_tag_counts,
        'non_std_tags': non_std_tags,
        'trend_data': trend_data,
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def analyze_ipa_dashboard(input_file: str, output_dir: str = "data/output",
                          api_key: str = None):
    """Run the full IPA dashboard analysis pipeline."""
    print("\n📊 讀取資料...")
    extracted_df = _load_input(input_file)

    extracted_path = os.path.join(output_dir, "ipa_extracted_data.csv")
    extracted_df.to_csv(extracted_path, index=False, encoding='utf-8-sig')
    print(f"✓ 提取資料: {extracted_path}")

    ipa_dir = os.path.join(output_dir, "ipa_dashboard")
    os.makedirs(ipa_dir, exist_ok=True)

    # ---- Global importance: per-review average ----
    print("\n📈 IPA 分析...")
    global_importance = calculate_global_importance(extracted_df)
    print(f"   逐筆平均重要度: {global_importance}")

    # ---- Subdirectory structure ----
    charts_dir = os.path.join(ipa_dir, "charts")
    all_charts_dir = os.path.join(charts_dir, "全部酒店")
    compare_dir = os.path.join(charts_dir, "comparison")
    for d in [charts_dir, all_charts_dir, compare_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- All hotels (using global importance) ----
    print("   全部酒店")
    priority_all = calculate_priority_metrics(extracted_df,
                                              global_importance=global_importance)

    chart_all = {
        'priority': os.path.join(all_charts_dir, "改進優先度.png"),
        'importance': os.path.join(all_charts_dir, "重要度排序.png"),
        'performance': os.path.join(all_charts_dir, "績效排序.png"),
        'comprehensive': os.path.join(all_charts_dir, "綜合對比.png"),
        'ipa_scatter': os.path.join(all_charts_dir, "IPA四象限.png"),
    }

    plot_priority_ranking(priority_all, "全部酒店", chart_all['priority'])
    plot_importance_ranking(priority_all, "全部酒店", chart_all['importance'])
    plot_performance_ranking(priority_all, "全部酒店", chart_all['performance'])
    plot_comprehensive_view(priority_all, "全部酒店", chart_all['comprehensive'])
    plot_hotel_ipa_scatter(priority_all, "全部酒店", chart_all['ipa_scatter'])

    # ---- Per-hotel analysis ----
    hotels = extracted_df['Hotel Name'].unique()
    print(f"\n✓ {len(hotels)} 家酒店")

    hotel_priority_dfs = {}

    stats_csv_path = os.path.join(output_dir, "ipa_all_hotels_statistics.csv")
    priority_all.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')

    with pd.ExcelWriter(os.path.join(output_dir, "ipa_all_hotels_statistics.xlsx"),
                        engine='openpyxl') as writer:
        priority_all.to_excel(writer, sheet_name='全部酒店', index=False)

        for hotel in hotels:
            print(f"   {hotel}")
            p = calculate_priority_metrics(extracted_df, hotel_name=hotel,
                                           global_importance=global_importance)
            hotel_priority_dfs[hotel] = p
            p.to_excel(writer, sheet_name=hotel[:30], index=False)

            hotel_chart_dir = os.path.join(charts_dir, hotel)
            os.makedirs(hotel_chart_dir, exist_ok=True)

            charts = {
                'priority': os.path.join(hotel_chart_dir, "改進優先度.png"),
                'importance': os.path.join(hotel_chart_dir, "重要度排序.png"),
                'performance': os.path.join(hotel_chart_dir, "績效排序.png"),
                'comprehensive': os.path.join(hotel_chart_dir, "綜合對比.png"),
                'ipa_scatter': os.path.join(hotel_chart_dir, "IPA四象限.png"),
            }
            plot_priority_ranking(p, hotel, charts['priority'])
            plot_importance_ranking(p, hotel, charts['importance'])
            plot_performance_ranking(p, hotel, charts['performance'])
            plot_comprehensive_view(p, hotel, charts['comprehensive'])
            plot_hotel_ipa_scatter(p, hotel, charts['ipa_scatter'])

    # ---- Multi-hotel comparison charts ----
    if len(hotel_priority_dfs) > 1:
        print("\n📊 生成多酒店比較圖...")
        multi_ipa_path = os.path.join(compare_dir, "多酒店IPA比較.png")
        plot_multi_hotel_ipa(hotel_priority_dfs, output_path=multi_ipa_path)

        attr_compare_path = os.path.join(compare_dir, "各屬性酒店對比.png")
        plot_attribute_hotel_comparison(hotel_priority_dfs, output_path=attr_compare_path)

    # ---- Post-hoc importance ----
    print("\n📊 Post-hoc 重要度 (AI 整體判斷)...")
    posthoc_importance = _load_or_compute_posthoc_importance(
        stats_csv_path, output_dir, api_key=api_key)

    if posthoc_importance:
        print(f"   Post-hoc AI重要度: {posthoc_importance}")
        posthoc_dir = os.path.join(charts_dir, "posthoc")
        os.makedirs(posthoc_dir, exist_ok=True)

        priority_all_ph = calculate_priority_metrics(
            extracted_df, global_importance=posthoc_importance)
        plot_hotel_ipa_scatter(priority_all_ph, "全部酒店（Post-hoc 重要度）",
                               os.path.join(posthoc_dir, "IPA四象限_posthoc_全部.png"),
                               importance_label="Post-hoc 重要度")

        hotel_ph_dfs = {}
        for hotel in hotels:
            p_ph = calculate_priority_metrics(
                extracted_df, hotel_name=hotel,
                global_importance=posthoc_importance)
            hotel_ph_dfs[hotel] = p_ph
            plot_hotel_ipa_scatter(
                p_ph, f"{hotel}（Post-hoc 重要度）",
                os.path.join(posthoc_dir, f"IPA四象限_posthoc_{hotel}.png"),
                importance_label="Post-hoc 重要度")

        if len(hotel_ph_dfs) > 1:
            plot_multi_hotel_ipa(
                hotel_ph_dfs,
                output_path=os.path.join(posthoc_dir, "多酒店IPA比較_posthoc.png"),
                importance_label="Post-hoc 重要度")

        plot_importance_comparison(global_importance, posthoc_importance,
                                   os.path.join(compare_dir, "重要度比較_逐筆vs_posthoc.png"))

    # ---- Compute overview stats ----
    print("\n📊 計算總覽統計...")
    overview_stats = _compute_overview_stats(extracted_df)

    # Load stability validation results if available
    validation_path = os.path.join(output_dir, "validation", "stability_results.json")
    if os.path.exists(validation_path):
        import json as _json
        with open(validation_path, 'r', encoding='utf-8') as f:
            overview_stats['validation'] = _json.load(f)
        print(f"   載入穩定性驗證結果: {validation_path}")

    # ---- Unified dashboard ----
    print("\n📄 生成統一儀表板...")
    unified_path = os.path.join(ipa_dir, "IPA統一儀表板.html")
    generate_unified_dashboard(
        all_priority_dfs=hotel_priority_dfs,
        priority_all=priority_all,
        global_importance=global_importance,
        charts_dir=charts_dir,
        compare_dir=compare_dir,
        posthoc_importance=posthoc_importance,
        posthoc_dir=os.path.join(charts_dir, "posthoc") if posthoc_importance else None,
        overview_stats=overview_stats,
        output_path=unified_path,
    )

    # ---- Clean up PNGs (already embedded as base64) ----
    import shutil
    png_count = 0
    for root, dirs, files in os.walk(charts_dir):
        for f in files:
            if f.endswith('.png'):
                os.remove(os.path.join(root, f))
                png_count += 1
    # Remove empty chart subdirectories
    for root, dirs, files in os.walk(charts_dir, topdown=False):
        if not os.listdir(root):
            os.rmdir(root)
    print(f"   已刪除 {png_count} 張 PNG（已嵌入統一儀表板）")

    print(f"\n{'='*70}")
    print(f"✅ IPA 儀表板分析完成！")
    print(f"   ★ 統一儀表板: {unified_path}")
    print(f"   統計表:       {output_dir}/ipa_all_hotels_statistics.xlsx")
    print(f"{'='*70}")


if __name__ == "__main__":
    from hotel_ipa.config_loader import load_config
    cfg = load_config()
    c = cfg["dashboard"]

    analyze_ipa_dashboard(
        c["input_file"],
        c["output_dir"],
        api_key=cfg["openai"]["api_key"]
    )
