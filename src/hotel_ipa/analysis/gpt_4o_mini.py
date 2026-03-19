"""
GPT-4o-mini 基礎分析 - 提取關鍵詞與情感評分（不含重要度）
"""

import openai
from hotel_ipa.config_loader import load_config
from hotel_ipa.constants import API_TEMPERATURE, API_MAX_TOKENS, COST_PER_REVIEW_4O_MINI
from hotel_ipa.analysis.base import (
    load_reviews, filter_hotels, stringify_columns,
    ProgressManager, call_openai, run_analysis
)


SYSTEM_PROMPT = """
# Role
你是一名酒店評論分析專家，負責提取屬性並給予 1-5 分情感評分。
# Purpose
從以下評論中提取與飯店服務相關的關鍵詞，並為每個關鍵詞評估情感分數。
# Condition
1. 關鍵詞標準
- 提取「名詞」或「名詞+形容詞」組合（如：'前台態度'、'房間隔音'、'早餐種類'）。
- 保持簡潔（2-6個字），去除贅字（如'我覺得'、'看起來'）。
- 若評論提及多個面向（如"房間大但舊"），請拆分為兩個維度（"房間空間"、"設施新舊"）。
2.若評論僅為"很棒"、"一般"，關鍵詞標記為"整體評價"。
3.Scoring Criteria (評分標準)
- 5分 (極佳)：強烈讚賞、驚喜、完美 (e.g., "太棒了", "無可挑剔")
- 4分 (滿意)：正面評價、推薦 (e.g., "不錯", "很好", "舒適")
- 3分 (普通)：中立、無功無過、尚可 (e.g., "還可以", "一般", "中規中矩")
- 2分 (失望)：有待改進、小抱怨 (e.g., "不太好", "有點舊", "不值這個價")
- 1分 (極差)：強烈批評、憤怒、絕不再來 (e.g., "糟糕透頂", "髒亂", "態度惡劣")
# Format
請直接輸出一個 JSON Array。格式如下：
[
    {"key": "關鍵詞1", "sentiment": "正面/中立/負面", "score": 分數},
    {"key": "關鍵詞2", "sentiment": "正面/中立/負面", "score": 分數}
]"""


def main():
    cfg = load_config()
    openai.api_key = cfg["openai"]["api_key"]
    c = cfg["analysis_mini"]
    model = c["model"]

    df = load_reviews(c["input_file"])
    df = filter_hotels(df, c.get("target_hotels"))
    stringify_columns(df)

    progress = ProgressManager(c["output_file"])
    if c.get("resume", True):
        df, start_idx = progress.resume(df)
    else:
        df["分析結果"] = ""
        start_idx = 0

    remaining = len(df) - start_idx
    print(f"\n模型: {model} | 待處理: {remaining} | "
          f"預估: ${remaining * COST_PER_REVIEW_4O_MINI:.2f} USD")

    def analyze(text):
        return call_openai(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"請分析以下酒店評論：{text}"}],
            model=model, temperature=API_TEMPERATURE, max_tokens=API_MAX_TOKENS
        )

    run_analysis(df, analyze, progress, start_idx,
                 c.get("sleep_seconds", 0.1), c.get("save_interval", 10))


if __name__ == "__main__":
    main()
