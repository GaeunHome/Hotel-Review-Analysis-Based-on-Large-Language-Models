"""
AI Advisor - Analyzes IPA data and provides improvement recommendations.
Uses GPT-4o-mini to generate professional hotel management advice.
"""

import pandas as pd
import json
import openai
from typing import Dict
from hotel_ipa.config_loader import load_config


SYSTEM_PROMPT = """
你是一名資深的酒店管理顧問，專精於數據分析和服務改善。你的任務是分析酒店的 IPA（重要度-績效分析）數據，並提供專業的改進建議。

# 分析框架
1. 問題識別：找出改進優先度最高的 3-5 個項目，識別關鍵問題和資源錯配
2. 根因分析：分析表現不佳的原因
3. 改進建議：短期（1-3月）、中期（3-6月）、長期（6月+）措施
4. 優先行動計劃：3-5 個最優先行動

# 輸出格式 (JSON)
{
    "executive_summary": "執行摘要（100-150字）",
    "key_findings": ["發現1", "發現2", "發現3"],
    "critical_issues": [
        {"attribute": "", "problem": "", "root_cause": "", "impact": ""}
    ],
    "improvement_recommendations": [
        {"attribute": "", "short_term": "", "medium_term": "", "long_term": "", "expected_outcome": ""}
    ],
    "priority_actions": [
        {"priority": 1, "action": "", "reason": "", "timeline": "", "resources_needed": ""}
    ],
    "strengths_to_maintain": [
        {"attribute": "", "why_important": "", "how_to_maintain": ""}
    ],
    "resource_optimization": [
        {"attribute": "", "current_situation": "", "suggestion": ""}
    ]
}

# 注意事項
- 建議應具體、可執行
- 考慮酒店行業實際運營情況
- 平衡成本效益
- 優先處理高重要度、低績效項目
"""


class AIAdvisor:
    """AI advisor that analyzes IPA data and provides recommendations."""

    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = load_config()["openai"]["api_key"]
        self.model = "gpt-4o-mini"

    def analyze_ipa_data(self, priority_df: pd.DataFrame,
                         hotel_name: str = "該酒店") -> Dict:
        """Analyze IPA data and return structured recommendations."""
        summary = self._build_summary(priority_df, hotel_name)
        return self._call_ai(summary)

    def _build_summary(self, df: pd.DataFrame, hotel_name: str) -> str:
        """Build a markdown data summary for the AI."""
        df_sorted = df.sort_values('改進優先度', ascending=False)
        top_improve = df_sorted.head(5)
        top_strength = df.nlargest(5, '平均績效')
        critical = df[(df['平均重要度'] > 4) & (df['平均績效'] < 3)]
        over_invest = df[(df['平均重要度'] < 2.5) & (df['平均績效'] > 4)]

        cols_improve = ['屬性', '平均重要度', '平均績效', '改進優先度', '正向率%', '提及次數']
        cols_basic = ['屬性', '平均重要度', '平均績效', '正向率%', '提及次數']

        return f"""# {hotel_name} - IPA 數據分析

## 整體概況
- 分析 {len(df)} 個屬性，總提及 {df['提及次數'].sum()} 次
- 平均重要度: {df['平均重要度'].mean():.2f}，平均績效: {df['平均績效'].mean():.2f}

## 前 5 大需要改進的項目
{top_improve[cols_improve].to_markdown(index=False)}

## 前 5 大優勢項目
{top_strength[cols_basic].to_markdown(index=False)}

## 關鍵問題（重要度 > 4 且績效 < 3）
{critical[cols_basic].to_markdown(index=False) if len(critical) else "無"}

## 過度投入（重要度 < 2.5 且績效 > 4）
{over_invest[cols_basic].to_markdown(index=False) if len(over_invest) else "無"}

## 完整數據
{df_sorted[cols_improve].to_markdown(index=False)}
"""

    def _call_ai(self, data_summary: str) -> Dict:
        """Call GPT for analysis."""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"請分析以下數據並提供建議：\n\n{data_summary}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ AI 分析失敗: {e}")
            return {
                "executive_summary": "AI 分析暫時無法使用，請查看統計數據。",
                "key_findings": ["數據已生成，請參考圖表分析。"],
                "critical_issues": [], "improvement_recommendations": [],
                "priority_actions": [], "strengths_to_maintain": [],
                "resource_optimization": []
            }

    def format_analysis_to_markdown(self, analysis: Dict, hotel_name: str) -> str:
        """Format analysis results as Markdown report."""
        sections = [f"# AI 顧問分析報告 - {hotel_name}\n"]
        sections.append(f"## 執行摘要\n\n{analysis.get('executive_summary', '')}\n")

        sections.append("## 關鍵發現\n")
        for f in analysis.get('key_findings', []):
            sections.append(f"- {f}")
        sections.append("")

        if analysis.get('critical_issues'):
            sections.append("## 關鍵問題\n")
            for issue in analysis['critical_issues']:
                sections.append(f"### {issue.get('attribute', '')}\n")
                sections.append(f"- **問題**: {issue.get('problem', '')}")
                sections.append(f"- **根因**: {issue.get('root_cause', '')}")
                sections.append(f"- **影響**: {issue.get('impact', '')}\n")

        if analysis.get('improvement_recommendations'):
            sections.append("## 改進建議\n")
            for rec in analysis['improvement_recommendations']:
                sections.append(f"### {rec.get('attribute', '')}\n")
                sections.append(f"**短期 (1-3月)**: {rec.get('short_term', '')}")
                sections.append(f"**中期 (3-6月)**: {rec.get('medium_term', '')}")
                sections.append(f"**預期效果**: {rec.get('expected_outcome', '')}\n---\n")

        if analysis.get('priority_actions'):
            sections.append("## 優先行動計劃\n")
            for a in analysis['priority_actions']:
                sections.append(f"### 優先級 {a.get('priority', 0)}: {a.get('action', '')}\n")
                sections.append(f"- **理由**: {a.get('reason', '')}")
                sections.append(f"- **時間表**: {a.get('timeline', '')}")
                sections.append(f"- **資源**: {a.get('resources_needed', '')}\n")

        if analysis.get('strengths_to_maintain'):
            sections.append("## 應保持的優勢\n")
            for s in analysis['strengths_to_maintain']:
                sections.append(f"### {s.get('attribute', '')}\n")
                sections.append(f"- **重要性**: {s.get('why_important', '')}")
                sections.append(f"- **維護**: {s.get('how_to_maintain', '')}\n")

        sections.append("\n---\n*本報告由 AI 顧問系統自動生成。*")
        return '\n'.join(sections)


if __name__ == "__main__":
    print("AI 顧問模組")
    print("  advisor = AIAdvisor(api_key='your_key')")
    print("  analysis = advisor.analyze_ipa_data(priority_df, hotel_name)")
