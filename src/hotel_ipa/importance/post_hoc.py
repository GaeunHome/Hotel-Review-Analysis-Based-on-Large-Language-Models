"""
Post-hoc importance calculation via GPT-4o.
Aggregates attribute statistics, then asks GPT-4o to evaluate importance
based on domain knowledge.
"""

import pandas as pd
import json
import openai
from hotel_ipa.config_loader import load_config
from hotel_ipa.constants import STANDARD_ATTRIBUTES


SYSTEM_PROMPT = """
你是一名資深酒店管理與消費者行為研究專家。

# 任務
根據以下酒店評論的彙總統計數據，評估每個服務屬性對消費者的相對重要度（1-5分）。

# 評估依據
1. 提及頻率：被顧客提到越多次，代表該屬性越受關注
2. 情感強度：正向率或負向率極端的屬性，通常對顧客更重要
3. 領域知識：根據飯店管理文獻，哪些屬性是顧客選擇飯店的關鍵因素
4. 決策影響：哪些屬性最可能影響顧客的再訪意願與推薦行為

# 評分標準
- 5分：對消費者決策至關重要的核心屬性
- 4分：重要但非決定性的屬性
- 3分：中等重要的屬性
- 2分：次要關注的屬性
- 1分：較少影響消費決策的屬性

# 輸出格式
請輸出 JSON：
{
    "importance_scores": [
        {"attribute": "屬性名稱", "importance": 分數, "reasoning": "簡短理由"}
    ]
}
"""


def build_statistics_summary(df: pd.DataFrame, hotel_name: str = "全部酒店") -> str:
    """Build a text summary of attribute statistics for the AI prompt."""
    lines = [f"# {hotel_name} 服務屬性統計彙總\n"]

    if '屬性' in df.columns:
        cols = ['屬性', '平均績效', '提及次數', '正向率%']
        cols = [c for c in cols if c in df.columns]
        lines.append(df[cols].to_markdown(index=False))
    elif 'Category' in df.columns:
        stats = []
        for attr in STANDARD_ATTRIBUTES:
            cat = df[df['Category'] == attr]
            if len(cat) == 0:
                continue
            stats.append({
                '屬性': attr,
                '提及次數': len(cat),
                '平均分數': round(cat['Score'].mean(), 2),
                '正向率%': round(
                    (cat['Sentiment'].isin(['正面', '正向']).sum() / len(cat)) * 100, 1
                )
            })
        lines.append(pd.DataFrame(stats).to_markdown(index=False))

    lines.append(f"\n總提及次數: {df['提及次數'].sum() if '提及次數' in df.columns else len(df)}")
    return '\n'.join(lines)


def calculate_posthoc_importance(
    input_file: str,
    output_file: str,
    model: str = "gpt-4o",
    num_runs: int = 3,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Calculate post-hoc importance scores using GPT-4o.

    Args:
        input_file: Path to aggregated statistics (CSV or Excel)
        output_file: Path to save results
        model: OpenAI model to use
        num_runs: Number of independent queries for consistency verification
        api_key: OpenAI API key
    """
    if api_key:
        openai.api_key = api_key

    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, encoding='utf-8-sig')
    else:
        df = pd.read_excel(input_file)

    summary = build_statistics_summary(df)
    print(f"📊 Post-hoc 重要度計算 (模型: {model}, 查詢次數: {num_runs})")

    # Run multiple queries for consistency
    all_results = []
    for run in range(num_runs):
        print(f"   查詢 {run + 1}/{num_runs}...")
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"請分析以下統計數據：\n\n{summary}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1500,
            )
            result = json.loads(response.choices[0].message.content)
            scores = {
                item['attribute']: item['importance']
                for item in result.get('importance_scores', [])
            }
            all_results.append(scores)
        except Exception as e:
            print(f"   ⚠️ 查詢 {run + 1} 失敗: {e}")

    if not all_results:
        print("❌ 所有查詢都失敗")
        return pd.DataFrame()

    # Average across runs
    all_attrs = set()
    for r in all_results:
        all_attrs.update(r.keys())

    avg_importance = {}
    for attr in all_attrs:
        values = [r[attr] for r in all_results if attr in r]
        avg_importance[attr] = round(sum(values) / len(values), 2)

    # Build output
    output_rows = []
    for attr in STANDARD_ATTRIBUTES:
        if attr in avg_importance:
            individual = [r.get(attr, None) for r in all_results]
            output_rows.append({
                '屬性': attr,
                'AI重要度': avg_importance[attr],
                **{f'查詢{i+1}': v for i, v in enumerate(individual)},
                '一致性': 'Y' if len(set(v for v in individual if v)) == 1 else 'N',
            })

    result_df = pd.DataFrame(output_rows)

    # Save
    if output_file.endswith('.csv'):
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        result_df.to_excel(output_file, index=False)
    print(f"✅ Post-hoc 重要度: {output_file}")

    return result_df


def main():
    cfg = load_config()
    openai.api_key = cfg["openai"]["api_key"]
    c = cfg["importance_posthoc"]

    calculate_posthoc_importance(
        input_file=c["input_file"],
        output_file=c["output_file"],
        model=c.get("model", "gpt-4o"),
        api_key=cfg["openai"]["api_key"],
    )


if __name__ == "__main__":
    main()
