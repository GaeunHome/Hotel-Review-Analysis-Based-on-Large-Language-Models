"""
Stability Validation Module.

Tests the reproducibility of the sentiment analysis prompt by:
1. Stratified random sampling of N reviews
2. Running each review through GPT-4o-mini and Claude Sonnet independently K times
3. Computing Fleiss' Kappa (intra-model consistency)
4. Computing Cohen's Kappa (inter-model agreement)

Reference: Thesis Section 3.8.1
"""

import pandas as pd
import numpy as np
import json
import time
import os
from typing import List, Dict

import openai
import anthropic

from hotel_ipa.config_loader import load_config
from hotel_ipa.utils import load_data_file, match_category
from hotel_ipa.constants import STANDARD_ATTRIBUTES


# ============================================================================
# Prompt (same as gpt_4o_mini_ipa.py for consistency)
# ============================================================================

ANALYSIS_PROMPT = """
# Role
你是一名酒店評論分析專家，負責提取屬性、情感評分（1-5分）以及重要度評估。

# Purpose
從評論中提取與飯店服務相關的關鍵詞，並為每個關鍵詞評估：
1. 情感分數 (score: 1-5分)
2. 重要度 (importance: 1-5分) - 評估顧客對此屬性的重視程度

# Condition
1. 關鍵詞標準
- 提取「名詞」或「名詞+形容詞」組合（如：'前台態度'、'房間隔音'、'早餐種類'）。
- 保持簡潔（2-6個字），去除贅字（如'我覺得'、'看起來'）。
- 若評論提及多個面向（如"房間大但舊"），請拆分為兩個維度（"房間空間"、"設施新舊"）。

2. 若評論僅為"很棒"、"一般"，關鍵詞標記為"整體評價"。

3. 評分標準 (score: 1-5)
- 5分 (極佳)：強烈讚賞、驚喜、完美
- 4分 (滿意)：正面評價、推薦
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
        "key": "關鍵詞1",
        "sentiment": "正面/中立/負面",
        "score": 分數(1-5),
        "importance": 重要度(1-5)
    }
]
"""


# ============================================================================
# Sampling
# ============================================================================

def stratified_sample(df: pd.DataFrame, sample_size: int = 200,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Stratified random sampling across hotels.
    Ensures proportional representation of each hotel.
    """
    hotel_counts = df['Hotel Name'].value_counts()
    total = len(df)
    samples = []

    for hotel, count in hotel_counts.items():
        n = max(1, round(sample_size * count / total))
        hotel_df = df[df['Hotel Name'] == hotel]
        sampled = hotel_df.sample(n=min(n, len(hotel_df)), random_state=random_state)
        samples.append(sampled)

    result = pd.concat(samples).head(sample_size).reset_index(drop=True)
    print(f"📊 分層抽樣: {len(result)} 則 (共 {result['Hotel Name'].nunique()} 家酒店)")
    for hotel in result['Hotel Name'].unique():
        print(f"   {hotel}: {len(result[result['Hotel Name'] == hotel])} 則")
    return result


# ============================================================================
# API Calls
# ============================================================================

def _parse_response(text: str) -> List[Dict]:
    """Parse JSON response from AI, extract and classify categories."""
    from hotel_ipa.utils import parse_json_safe
    items = parse_json_safe(text)
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                item['category'] = match_category(item.get('key', ''))
        return [i for i in items if isinstance(i, dict)]
    return []


def analyze_with_openai(review: str, model: str, api_key: str) -> List[Dict]:
    """Call OpenAI API for analysis."""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": f"請分析以下酒店評論：{review}"},
        ],
        temperature=0.01,
        max_tokens=800,
    )
    return _parse_response(response.choices[0].message.content)


def analyze_with_claude(review: str, model: str, api_key: str) -> List[Dict]:
    """Call Anthropic Claude API for analysis."""
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=ANALYSIS_PROMPT,
        messages=[
            {"role": "user", "content": f"請分析以下酒店評論：{review}"},
        ],
        temperature=0.01,
    )
    text = response.content[0].text if response.content else ""
    return _parse_response(text)


# ============================================================================
# Run Multiple Analyses
# ============================================================================

def run_multiple_analyses(
    sample_df: pd.DataFrame,
    num_runs: int,
    model_name: str,
    model_id: str,
    api_key: str,
    provider: str = "openai",
    sleep_seconds: float = 0.2,
) -> Dict[int, List[List[Dict]]]:
    """
    Run the same sample through a model multiple times.

    Returns:
        {run_index: [review_results_list]}
        where each review_results_list contains parsed JSON items
    """
    analyze_fn = analyze_with_openai if provider == "openai" else analyze_with_claude

    all_runs = {}
    for run in range(num_runs):
        print(f"\n   [{model_name}] 第 {run + 1}/{num_runs} 次測試...")
        run_results = []
        errors = 0

        for idx, row in sample_df.iterrows():
            review = str(row['Review Text']).strip()
            if not review:
                run_results.append([])
                continue

            try:
                items = analyze_fn(review, model_id, api_key)
                run_results.append(items)
            except Exception as e:
                run_results.append([])
                errors += 1
                if errors <= 3:
                    print(f"      ⚠️ 第 {idx} 則錯誤: {e}")

            time.sleep(sleep_seconds)

        all_runs[run] = run_results
        print(f"      完成 ({len(run_results)} 則，{errors} 錯誤)")

    return all_runs


# ============================================================================
# Kappa Calculations
# ============================================================================

def _extract_categories_per_review(run_results: List[List[Dict]]) -> List[set]:
    """Extract set of categories from each review's results."""
    cats_per_review = []
    for items in run_results:
        cats = set()
        for item in items:
            cat = item.get('category', '其他')
            if cat in STANDARD_ATTRIBUTES:
                cats.add(cat)
        cats_per_review.append(cats)
    return cats_per_review


def compute_fleiss_kappa(all_runs: Dict[int, List[List[Dict]]],
                         num_reviews: int) -> dict:
    """
    Compute Fleiss' Kappa for multi-rater agreement.

    Each "rater" is one run of the model.
    For each review, we check which categories were identified.
    """
    num_runs = len(all_runs)
    categories = STANDARD_ATTRIBUTES

    # Build rating matrix: (num_reviews * num_categories) × num_runs
    # For each review × category, count how many runs identified it
    n_items = num_reviews * len(categories)
    rating_matrix = np.zeros((n_items, 2))  # [not present, present]

    for run_idx, run_results in all_runs.items():
        cats_per_review = _extract_categories_per_review(run_results)
        for i in range(num_reviews):
            cats = cats_per_review[i] if i < len(cats_per_review) else set()
            for j, cat in enumerate(categories):
                item_idx = i * len(categories) + j
                if cat in cats:
                    rating_matrix[item_idx, 1] += 1
                else:
                    rating_matrix[item_idx, 0] += 1

    # Fleiss' Kappa calculation
    N = n_items  # number of items
    n = num_runs  # number of raters
    k = 2  # number of categories (present/not present)

    # P_i: proportion of agreement for item i
    P_i = np.sum(rating_matrix ** 2, axis=1) - n
    P_i = P_i / (n * (n - 1)) if n > 1 else P_i

    P_bar = np.mean(P_i)

    # P_j: proportion of all assignments to category j
    p_j = np.sum(rating_matrix, axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)

    if P_e == 1:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    # Interpretation
    if kappa >= 0.81:
        interpretation = "幾乎完全一致 (Almost Perfect)"
    elif kappa >= 0.61:
        interpretation = "高度一致 (Substantial)"
    elif kappa >= 0.41:
        interpretation = "中度一致 (Moderate)"
    elif kappa >= 0.21:
        interpretation = "一般一致 (Fair)"
    else:
        interpretation = "低度一致 (Slight/Poor)"

    return {
        'kappa': round(kappa, 4),
        'P_bar': round(P_bar, 4),
        'P_e': round(P_e, 4),
        'interpretation': interpretation,
        'num_runs': num_runs,
        'num_reviews': num_reviews,
    }


def compute_cohens_kappa(runs_a: Dict[int, List[List[Dict]]],
                          runs_b: Dict[int, List[List[Dict]]],
                          num_reviews: int) -> dict:
    """
    Compute Cohen's Kappa for inter-model agreement.

    Compares the most frequent classification from model A vs model B.
    """
    categories = STANDARD_ATTRIBUTES

    # Get majority vote per review per category for each model
    def get_majority(all_runs):
        votes = {}  # (review_idx, category) -> count of "present"
        for run_results in all_runs.values():
            cats_list = _extract_categories_per_review(run_results)
            for i in range(num_reviews):
                cats = cats_list[i] if i < len(cats_list) else set()
                for cat in categories:
                    key = (i, cat)
                    if key not in votes:
                        votes[key] = 0
                    if cat in cats:
                        votes[key] += 1

        threshold = len(all_runs) / 2
        majority = {}
        for key, count in votes.items():
            majority[key] = 1 if count > threshold else 0
        return majority

    maj_a = get_majority(runs_a)
    maj_b = get_majority(runs_b)

    # Build contingency
    agree = 0
    total = 0
    a_pos = 0
    b_pos = 0

    for i in range(num_reviews):
        for cat in categories:
            key = (i, cat)
            va = maj_a.get(key, 0)
            vb = maj_b.get(key, 0)
            if va == vb:
                agree += 1
            a_pos += va
            b_pos += vb
            total += 1

    if total == 0:
        return {'kappa': 0, 'interpretation': 'N/A'}

    po = agree / total
    pa = a_pos / total
    pb = b_pos / total
    pe = pa * pb + (1 - pa) * (1 - pb)

    if pe == 1:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    if kappa >= 0.81:
        interpretation = "幾乎完全一致 (Almost Perfect)"
    elif kappa >= 0.61:
        interpretation = "高度一致 (Substantial)"
    elif kappa >= 0.41:
        interpretation = "中度一致 (Moderate)"
    elif kappa >= 0.21:
        interpretation = "一般一致 (Fair)"
    else:
        interpretation = "低度一致 (Slight/Poor)"

    return {
        'kappa': round(kappa, 4),
        'po': round(po, 4),
        'pe': round(pe, 4),
        'interpretation': interpretation,
    }


# ============================================================================
# Main
# ============================================================================

def run_stability_validation(
    input_file: str,
    output_dir: str,
    sample_size: int = 200,
    num_runs: int = 5,
    openai_key: str = None,
    anthropic_key: str = None,
    openai_model: str = "gpt-4o-mini",
    claude_model: str = "claude-sonnet-4-6-20250514",
):
    """Run full stability validation."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("穩定性驗證")
    print("=" * 70)

    # Load and sample
    df = load_data_file(input_file)
    sample = stratified_sample(df, sample_size)
    sample.to_csv(os.path.join(output_dir, "validation_sample.csv"),
                  index=False, encoding='utf-8-sig')

    num_reviews = len(sample)
    results = {}

    # GPT-4o-mini runs
    if openai_key:
        print(f"\n{'='*50}\nGPT-4o-mini 穩定性測試 ({num_runs} 次)\n{'='*50}")
        gpt_runs = run_multiple_analyses(
            sample, num_runs, "GPT-4o-mini", openai_model, openai_key,
            provider="openai", sleep_seconds=0.15,
        )
        gpt_kappa = compute_fleiss_kappa(gpt_runs, num_reviews)
        results['gpt4o_mini'] = {
            'fleiss_kappa': gpt_kappa,
            'runs': gpt_runs,
        }
        print(f"\n   GPT-4o-mini Fleiss' Kappa: {gpt_kappa['kappa']} ({gpt_kappa['interpretation']})")
    else:
        print("⚠️ 未提供 OpenAI API Key，跳過 GPT-4o-mini 測試")
        gpt_runs = None

    # Claude Sonnet runs
    if anthropic_key:
        print(f"\n{'='*50}\nClaude Sonnet 穩定性測試 ({num_runs} 次)\n{'='*50}")
        claude_runs = run_multiple_analyses(
            sample, num_runs, "Claude Sonnet", claude_model, anthropic_key,
            provider="anthropic", sleep_seconds=0.3,
        )
        claude_kappa = compute_fleiss_kappa(claude_runs, num_reviews)
        results['claude_sonnet'] = {
            'fleiss_kappa': claude_kappa,
            'runs': claude_runs,
        }
        print(f"\n   Claude Sonnet Fleiss' Kappa: {claude_kappa['kappa']} ({claude_kappa['interpretation']})")
    else:
        print("⚠️ 未提供 Anthropic API Key，跳過 Claude 測試")
        claude_runs = None

    # Cross-model comparison
    if gpt_runs and claude_runs:
        print(f"\n{'='*50}\n跨模型一致性比較\n{'='*50}")
        cohens = compute_cohens_kappa(gpt_runs, claude_runs, num_reviews)
        results['cross_model'] = cohens
        print(f"   Cohen's Kappa: {cohens['kappa']} ({cohens['interpretation']})")

    # Export results
    export_stability_results(results, output_dir)

    print(f"\n{'='*70}")
    print("✅ 穩定性驗證完成！")
    print(f"   結果: {output_dir}")
    print(f"{'='*70}")

    return results


def export_stability_results(results: dict, output_dir: str):
    """Export stability validation results."""
    rows = []

    if 'gpt4o_mini' in results:
        k = results['gpt4o_mini']['fleiss_kappa']
        rows.append({
            '模型': 'GPT-4o-mini',
            '檢定方法': "Fleiss' Kappa",
            'Kappa': k['kappa'],
            '解釋': k['interpretation'],
            '測試次數': k['num_runs'],
            '樣本數': k['num_reviews'],
        })

    if 'claude_sonnet' in results:
        k = results['claude_sonnet']['fleiss_kappa']
        rows.append({
            '模型': 'Claude Sonnet',
            '檢定方法': "Fleiss' Kappa",
            'Kappa': k['kappa'],
            '解釋': k['interpretation'],
            '測試次數': k['num_runs'],
            '樣本數': k['num_reviews'],
        })

    if 'cross_model' in results:
        k = results['cross_model']
        rows.append({
            '模型': 'GPT-4o-mini vs Claude Sonnet',
            '檢定方法': "Cohen's Kappa",
            'Kappa': k['kappa'],
            '解釋': k['interpretation'],
            '測試次數': '-',
            '樣本數': '-',
        })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "stability_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        xlsx_path = os.path.join(output_dir, "stability_report.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"   ✓ 穩定性報告: {xlsx_path}")


def main():
    cfg = load_config()
    c = cfg["validation"]

    run_stability_validation(
        input_file=c["input_file"],
        output_dir=c["output_dir"],
        sample_size=c.get("sample_size", 200),
        num_runs=c.get("num_runs", 5),
        openai_key=cfg["openai"]["api_key"],
        anthropic_key=cfg.get("anthropic", {}).get("api_key"),
        openai_model=c.get("models", {}).get("gpt4o_mini", "gpt-4o-mini"),
        claude_model=c.get("models", {}).get("claude_sonnet", "claude-sonnet-4-6-20250514"),
    )


if __name__ == "__main__":
    main()
