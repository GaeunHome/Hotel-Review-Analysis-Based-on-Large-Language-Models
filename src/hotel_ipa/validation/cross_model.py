"""
Cross-Model Validation Module.

Compares GPT-4o and Claude Sonnet against GPT-4o-mini ground truth.
For each review, compares:
  - Attribute recall: did the model find the same categories?
  - Sentiment agreement: same sentiment label?
  - Score correlation: Spearman ρ on matched (review, category) pairs

Ground truth: extracted from the classified pipeline output (GPT-4o-mini + classify.py).
Validation models: run with a category-aware prompt (no classify.py needed).
"""

import pandas as pd
import numpy as np
import json
import time
import os
from typing import List, Dict
from scipy import stats

import openai
import anthropic

from hotel_ipa.config_loader import load_config
from hotel_ipa.constants import STANDARD_ATTRIBUTES


# ============================================================================
# Validation Prompt (includes 12 standard categories)
# ============================================================================

VALIDATION_PROMPT = """
# Role
你是一名酒店評論分析專家，負責提取屬性、情感評分（1-5分）以及重要度評估。

# Purpose
從評論中提取與飯店服務相關的關鍵詞，並將每個關鍵詞歸入以下 12 個標準屬性之一，
同時評估情感與分數。

# 12 個標準屬性
地理位置、性價比、服務、房間、餐廳、停車、清潔度、公共設施、周邊環境、衛浴、交通、睡眠品質

# Condition
1. 關鍵詞標準
- 提取「名詞」或「名詞+形容詞」組合（如：'前台態度'、'房間隔音'、'早餐種類'）。
- 保持簡潔（2-6個字），去除贅字。
- 若評論提及多個面向，請拆分為不同維度。

2. 屬性歸類
- 每個關鍵詞必須歸入上述 12 個標準屬性之一。
- 若無法歸入任何標準屬性，歸為「其他」。

3. 評分標準 (score: 1-5)
- 5分 (正面)：明確的正面評價或強烈讚賞，如「很好」「不錯」
- 4分 (尚可)：有保留的正面，如「還行」「還可以」
- 3分 (普通)：中立、無功無過、尚可
- 2分 (失望)：有待改進、小抱怨
- 1分 (極差)：強烈批評、憤怒、絕不再來

4. 重要度標準 (importance: 1-5)
- 5分 (非常重要)：顧客反覆強調、使用強烈語氣、明確表示影響入住決策
- 4分 (重要)：明確提及並著重描述、佔評論篇幅較大
- 3分 (普通重要)：一般性提及、正常描述
- 2分 (次要)：順帶一提、輕描淡寫
- 1分 (不重要)：僅在完整性考慮下提及、一筆帶過

# Format
請直接輸出一個 JSON Array。格式如下：
[
    {
        "category": "標準屬性名稱",
        "key": "關鍵詞",
        "sentiment": "正面/中立/負面",
        "score": 分數(1-5),
        "importance": 重要度(1-5)
    }
]
"""


# ============================================================================
# API Calls
# ============================================================================

# Simplified → Traditional Chinese mapping for the 12 standard attributes
_SIMP_TO_TRAD = {
    '地理位置': '地理位置', '性价比': '性價比', '服务': '服務',
    '房间': '房間', '餐厅': '餐廳', '停车': '停車',
    '清洁度': '清潔度', '公共设施': '公共設施', '周边环境': '周邊環境',
    '卫浴': '衛浴', '交通': '交通', '睡眠品质': '睡眠品質',
    '睡眠质量': '睡眠品質',
}
# Also add Traditional → Traditional (identity) for completeness
for attr in STANDARD_ATTRIBUTES:
    _SIMP_TO_TRAD[attr] = attr


def _normalize_category(cat: str) -> str:
    """Normalize category name: simplified → traditional, fuzzy match."""
    if cat in STANDARD_ATTRIBUTES:
        return cat
    if cat in _SIMP_TO_TRAD:
        return _SIMP_TO_TRAD[cat]
    # Fallback: keyword matching
    from hotel_ipa.utils import match_category
    matched = match_category(cat)
    return matched if matched in STANDARD_ATTRIBUTES else ''


def _parse_response(text: str) -> List[Dict]:
    """Parse JSON response, normalize categories to standard attributes."""
    from hotel_ipa.utils import parse_json_safe
    items = parse_json_safe(text)
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cat = _normalize_category(item.get('category', ''))
        if not cat:
            continue
        result.append({
            'category': cat,
            'sentiment': item.get('sentiment', ''),
            'score': int(item.get('score', 0)),
        })
    return result


def _call_openai_raw(review: str, model: str, api_key: str) -> str:
    """Call OpenAI API, return raw response text."""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VALIDATION_PROMPT},
            {"role": "user", "content": f"請分析以下酒店評論：{review}"},
        ],
        temperature=0.01,
        max_tokens=800,
    )
    return response.choices[0].message.content


def _call_claude_raw(review: str, model: str, api_key: str) -> str:
    """Call Claude API, return raw response text."""
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=VALIDATION_PROMPT,
        messages=[
            {"role": "user", "content": f"請分析以下酒店評論：{review}"},
        ],
        temperature=0.01,
    )
    return response.content[0].text if response.content else ""


# ============================================================================
# Cache (resume support)
# ============================================================================

def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# ============================================================================
# Run Validation
# ============================================================================

def _run_model(sample_df: pd.DataFrame, model_name: str, model_id: str,
               api_key: str, provider: str, raw_csv_path: str,
               sleep_seconds: float = 0.2) -> pd.DataFrame:
    """
    Run one model on all sample reviews with resume support.
    Saves raw response text to CSV. Parsing is done separately.

    Returns: DataFrame with columns [Review ID, raw_response]
    """
    call_fn = _call_openai_raw if provider == "openai" else _call_claude_raw

    # Load existing raw CSV for resume
    if os.path.exists(raw_csv_path):
        existing = pd.read_csv(raw_csv_path, encoding='utf-8-sig')
        done_ids = set(existing['Review ID'].astype(str).tolist())
        print(f"\n   [{model_name}] 已有 {len(done_ids)} 筆，續跑...")
    else:
        existing = pd.DataFrame(columns=['Review ID', 'raw_response'])
        done_ids = set()

    print(f"   [{model_name}] 分析 {len(sample_df)} 筆評論...")
    new_rows = []
    errors = 0

    for i, (_, row) in enumerate(sample_df.iterrows()):
        rid = str(row['Review ID'])
        if rid in done_ids:
            continue

        review = str(row['Review Text']).strip()
        if not review:
            new_rows.append({'Review ID': int(rid), 'raw_response': ''})
            continue

        try:
            raw_text = call_fn(review, model_id, api_key)
            new_rows.append({'Review ID': int(rid), 'raw_response': raw_text})
        except Exception as e:
            new_rows.append({'Review ID': int(rid), 'raw_response': ''})
            errors += 1
            if errors <= 3:
                print(f"      ⚠️ Review {rid}: {e}")

        time.sleep(sleep_seconds)

        # Save after every review
        batch = pd.DataFrame(new_rows)
        result_so_far = pd.concat([existing, batch], ignore_index=True)
        result_so_far.to_csv(raw_csv_path, index=False, encoding='utf-8-sig')

        total_done = len(done_ids) + len(new_rows)
        if total_done % 20 == 0:
            print(f"      進度: {total_done}/{len(sample_df)}")

    # Final result
    if new_rows:
        result = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        result = existing

    skipped = len(done_ids)
    print(f"      完成 (新跑 {len(new_rows)}，跳過 {skipped}，{errors} 錯誤)")
    print(f"      ✓ 原始回傳: {raw_csv_path}")
    return result


def _parse_raw_results(raw_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Parse raw response CSV into structured results dict."""
    results = {}
    for _, row in raw_df.iterrows():
        rid = str(row['Review ID'])
        raw = str(row.get('raw_response', ''))
        results[rid] = _parse_response(raw)
    return results


# ============================================================================
# Ground Truth Extraction
# ============================================================================

def _extract_ground_truth(classified_file: str, sample_ids: list) -> pd.DataFrame:
    """
    Extract ground truth from the classified pipeline output.
    Returns DataFrame with columns: [Review ID, category, sentiment, score]
    """
    print(f"\n📂 載入 ground truth: {classified_file}")
    if classified_file.endswith('.csv'):
        df = pd.read_csv(classified_file, encoding='utf-8-sig')
    else:
        df = pd.read_excel(classified_file, sheet_name="詳細數據")

    # Filter to sample reviews and standard categories
    df = df[df['Review ID'].isin(sample_ids)].copy()
    cat_col = 'Standardized_Category' if 'Standardized_Category' in df.columns else 'Category'
    df = df[df[cat_col].isin(STANDARD_ATTRIBUTES)]

    # Normalize sentiment
    df['Sentiment'] = df['Sentiment'].replace({
        '负面': '負面', '正向': '正面', '負向': '負面'
    })

    # Aggregate: per (Review ID, category), take mean score and majority sentiment
    gt_rows = []
    for (rid, cat), group in df.groupby(['Review ID', cat_col]):
        score = round(group['Score'].mean(), 1)
        sentiment = group['Sentiment'].mode().iloc[0] if len(group) > 0 else '中立'
        gt_rows.append({
            'Review ID': rid,
            'category': cat,
            'score': score,
            'sentiment': sentiment,
            'count': len(group),
        })

    gt = pd.DataFrame(gt_rows)
    print(f"   ✓ {len(gt)} 組配對 ({gt['Review ID'].nunique()} 筆評論)")
    return gt


# ============================================================================
# Metrics Computation
# ============================================================================

def _compute_metrics(gt: pd.DataFrame, model_results: dict,
                     model_name: str) -> dict:
    """
    Compare model results against ground truth.
    Returns dict with attr_recall, sent_agreement, score_spearman_rho, etc.
    """
    matched = 0
    sent_agree = 0
    gt_scores = []
    model_scores = []

    total_gt = len(gt)

    for _, row in gt.iterrows():
        rid = str(row['Review ID'])
        cat = row['category']
        gt_sent = row['sentiment']
        gt_score = row['score']

        # Check if model found this (review, category) pair
        model_items = model_results.get(rid, [])
        found = [it for it in model_items if it.get('category') == cat]

        if found:
            matched += 1
            m_item = found[0]
            m_sent = m_item.get('sentiment', '')
            m_score = m_item.get('score', 0)

            # Sentiment agreement (normalize)
            m_sent_norm = m_sent.replace('正向', '正面').replace('負向', '負面').replace('负面', '負面')
            if m_sent_norm == gt_sent:
                sent_agree += 1

            gt_scores.append(gt_score)
            model_scores.append(m_score)

    # Compute metrics
    attr_recall = round(matched / total_gt * 100, 1) if total_gt else 0
    sent_agreement = round(sent_agree / matched * 100, 1) if matched else 0

    if len(gt_scores) >= 3:
        rho, p_val = stats.spearmanr(gt_scores, model_scores)
        rho = round(rho, 4)
        p_val = round(p_val, 6)
    else:
        rho, p_val = 0, 1

    # Exact match and within-1
    exact = sum(1 for a, b in zip(gt_scores, model_scores) if a == b)
    within1 = sum(1 for a, b in zip(gt_scores, model_scores) if abs(a - b) <= 1)
    exact_pct = round(exact / matched * 100, 1) if matched else 0
    within1_pct = round(within1 / matched * 100, 1) if matched else 0

    result = {
        'attr_recall': attr_recall,
        'sent_agreement': sent_agreement,
        'score_spearman_rho': rho,
        'score_spearman_p': p_val,
        'score_exact_match': exact_pct,
        'score_within1': within1_pct,
        'n_gt_pairs': total_gt,
        'n_matched': matched,
    }

    print(f"\n   [{model_name}]")
    print(f"      屬性召回率:   {attr_recall}% ({matched}/{total_gt})")
    print(f"      情感一致率:   {sent_agreement}%")
    print(f"      Spearman ρ:  {rho} (p={p_val})")
    print(f"      分數完全一致: {exact_pct}%")
    print(f"      分數差≤1:    {within1_pct}%")

    return result


# ============================================================================
# Build Comparison Table
# ============================================================================

def _build_comparison_table(gt: pd.DataFrame, model_results: dict) -> pd.DataFrame:
    """Build per-(review, category) comparison table for all models."""
    rows = []
    for _, row in gt.iterrows():
        rid = str(row['Review ID'])
        cat = row['category']
        entry = {
            'review_id': int(rid),
            'category': cat,
            'gt_sentiment': row['sentiment'],
            'gt_score': row['score'],
        }
        for model_key, results in model_results.items():
            items = results.get(rid, [])
            found = [it for it in items if it.get('category') == cat]
            if found:
                entry[f'{model_key}_found'] = 1
                entry[f'{model_key}_sentiment'] = found[0].get('sentiment', '')
                entry[f'{model_key}_score'] = found[0].get('score', 0)
            else:
                entry[f'{model_key}_found'] = 0
                entry[f'{model_key}_sentiment'] = None
                entry[f'{model_key}_score'] = None
        rows.append(entry)
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def _score_stratified_sample(
    classified_file: str,
    reviews_file: str,
    sample_size: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Score-stratified sampling: uniform distribution across score bins 1-5.

    1. Compute per-review mean score from classified data.
    2. Bin into 1-5.
    3. Sample equally from each bin.
    """
    # Load classified data to compute per-review mean scores
    if classified_file.endswith('.csv'):
        clf = pd.read_csv(classified_file, encoding='utf-8-sig')
    else:
        clf = pd.read_excel(classified_file, sheet_name="詳細數據")

    cat_col = 'Standardized_Category' if 'Standardized_Category' in clf.columns else 'Category'
    clf = clf[clf[cat_col].isin(STANDARD_ATTRIBUTES)]
    review_mean_score = clf.groupby('Review ID')['Score'].mean()

    # Bin into 1-5
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    labels = [1, 2, 3, 4, 5]
    review_bins = pd.cut(review_mean_score, bins=bins, labels=labels).dropna()
    review_bins = review_bins.astype(int)

    # Load review texts
    reviews = pd.read_csv(reviews_file, encoding='utf-8-sig')
    reviews = reviews[reviews['Review ID'].isin(review_bins.index)]

    # Attach score bin
    reviews = reviews.copy()
    reviews['score_bin'] = reviews['Review ID'].map(review_bins)
    reviews = reviews.dropna(subset=['score_bin'])
    reviews['score_bin'] = reviews['score_bin'].astype(int)

    # Sample equally from each bin
    per_bin = sample_size // 5
    remainder = sample_size % 5
    sampled = []
    for b in labels:
        pool = reviews[reviews['score_bin'] == b]
        n = per_bin + (1 if b <= remainder else 0)
        n = min(n, len(pool))
        sampled.append(pool.sample(n=n, random_state=random_state))

    result = pd.concat(sampled).reset_index(drop=True)

    print(f"\n📊 分數分層抽樣: {len(result)} 筆")
    for b in labels:
        cnt = len(result[result['score_bin'] == b])
        print(f"   Score bin {b}: {cnt} 筆")

    # Drop the helper column before returning
    result = result.drop(columns=['score_bin'])
    return result


def run_cross_model_validation(
    classified_file: str,
    sample_file: str,
    output_dir: str,
    openai_key: str = None,
    anthropic_key: str = None,
    gpt4o_model: str = "gpt-4o",
    claude_model: str = "claude-sonnet-4-20250514",
    reviews_file: str = None,
    sample_size: int = 200,
    use_score_stratification: bool = True,
):
    """Run cross-model validation."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("跨模型驗證 (Cross-Model Validation)")
    print("=" * 70)

    # Score-stratified sampling or load existing sample
    new_sample_path = os.path.join(output_dir, "sample_reviews_stratified.csv")
    if use_score_stratification and reviews_file:
        sample = _score_stratified_sample(
            classified_file, reviews_file, sample_size)
        sample.to_csv(new_sample_path, index=False, encoding='utf-8-sig')
        sample_file = new_sample_path
    else:
        sample = pd.read_csv(sample_file, encoding='utf-8-sig')

    sample_ids = sample['Review ID'].tolist()
    print(f"   樣本: {len(sample)} 筆評論")

    # Extract ground truth from classified data
    gt = _extract_ground_truth(classified_file, sample_ids)
    gt.to_csv(os.path.join(output_dir, "ground_truth.csv"),
              index=False, encoding='utf-8-sig')

    # Run validation models → save raw responses to CSV
    raw_dfs = {}

    suffix = "_stratified" if use_score_stratification else ""

    if openai_key:
        gpt4o_csv = os.path.join(output_dir, f"raw_gpt4o_responses{suffix}.csv")
        raw_dfs['gpt4o'] = _run_model(
            sample, "GPT-4o", gpt4o_model, openai_key,
            provider="openai", raw_csv_path=gpt4o_csv, sleep_seconds=0.3)

    if anthropic_key:
        claude_csv = os.path.join(output_dir, f"raw_claude_responses{suffix}.csv")
        raw_dfs['claude'] = _run_model(
            sample, "Claude Sonnet", claude_model, anthropic_key,
            provider="anthropic", raw_csv_path=claude_csv, sleep_seconds=0.3)

    # Parse raw responses → structured results
    model_results = {}
    for model_key, raw_df in raw_dfs.items():
        model_results[model_key] = _parse_raw_results(raw_df)

    # Compute metrics
    print(f"\n{'='*50}\n驗證結果\n{'='*50}")

    all_metrics = {}
    for model_key, results in model_results.items():
        label = {'gpt4o': 'GPT-4o', 'claude': 'Claude Sonnet'}.get(model_key, model_key)
        all_metrics[model_key] = _compute_metrics(gt, results, label)

    # Build comparison table
    comparison = _build_comparison_table(gt, model_results)
    comparison.to_csv(os.path.join(output_dir, "validation_comparison.csv"),
                      index=False, encoding='utf-8-sig')

    # Export metrics
    metrics_rows = []
    for model_key, m in all_metrics.items():
        metrics_rows.append({'model': model_key, **m})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(output_dir, "validation_metrics.csv"),
                      index=False, encoding='utf-8-sig')

    # Update stability_results.json
    results_path = os.path.join(output_dir, "stability_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            sr = json.load(f)
    else:
        sr = {}

    sr['sample_size'] = len(sample)
    sr['gt_pairs'] = len(gt)
    sr['claude_model'] = claude_model
    sr['sampling_method'] = "分數分層抽樣（1-5 分均勻）" if use_score_stratification else "隨機抽樣"
    sr['validation_metrics'] = all_metrics
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(sr, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("✅ 跨模型驗證完成！")
    print(f"   ground truth:    {output_dir}/ground_truth.csv")
    print(f"   comparison:      {output_dir}/validation_comparison.csv")
    print(f"   metrics:         {output_dir}/validation_metrics.csv")
    print(f"   results JSON:    {results_path}")
    print(f"{'='*70}")


def main():
    cfg = load_config()
    c = cfg["validation"]
    d = cfg["dashboard"]

    run_cross_model_validation(
        classified_file=d["input_file"],
        sample_file=os.path.join(c["output_dir"], "sample_reviews.csv"),
        output_dir=c["output_dir"],
        openai_key=cfg["openai"]["api_key"],
        anthropic_key=cfg.get("anthropic", {}).get("api_key"),
        claude_model=c.get("models", {}).get("claude_sonnet", "claude-sonnet-4-20250514"),
        reviews_file=c["input_file"],
        sample_size=c.get("sample_size", 200),
        use_score_stratification=True,
    )


if __name__ == "__main__":
    main()
