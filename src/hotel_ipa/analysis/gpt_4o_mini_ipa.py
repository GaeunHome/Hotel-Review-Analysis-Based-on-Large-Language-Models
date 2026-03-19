"""
GPT-4o-mini IPA 分析 - 提取關鍵詞、情感評分和重要度
"""

import openai
from hotel_ipa.config_loader import load_config
from hotel_ipa.constants import API_TEMPERATURE, API_MAX_TOKENS, COST_PER_REVIEW_4O_MINI
from hotel_ipa.analysis.base import (
    load_reviews, filter_hotels, stringify_columns,
    ProgressManager, call_openai, run_analysis
)


# ============================================================================
# Prompt
# ============================================================================

SYSTEM_PROMPT = """
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
- 5分 (極佳)：強烈讚賞、驚喜、完美 (e.g., "太棒了", "無可挑剔")
- 4分 (滿意)：正面評價、推薦 (e.g., "不錯", "很好", "舒適")
- 3分 (普通)：中立、無功無過、尚可 (e.g., "還可以", "一般", "中規中矩")
- 2分 (失望)：有待改進、小抱怨 (e.g., "不太好", "有點舊", "不值這個價")
- 1分 (極差)：強烈批評、憤怒、絕不再來 (e.g., "糟糕透頂", "髒亂", "態度惡劣")

4. 重要度標準 (importance: 1-5)
- 5分 (非常重要)：顧客反覆強調、使用強烈語氣、明確表示影響入住決策
  例如："最重要的是"、"特別在意"、"主要原因"、"決定性因素"
- 4分 (重要)：明確提及並著重描述、佔評論篇幅較大
  例如：詳細描述該屬性、用多個形容詞修飾
- 3分 (普通重要)：一般性提及、正常描述
  例如：單句簡單描述、不特別強調
- 2分 (次要)：順帶一提、輕描淡寫
  例如："還有"、"另外"、"順便說一下"
- 1分 (不重要)：僅在完整性考慮下提及、一筆帶過
  例如：列舉中的一項、沒有任何描述

# 判斷重要度的關鍵指標
- 篇幅：描述該屬性的文字量
- 語氣：是否使用強調詞（"非常"、"特別"、"最"、"太"等）
- 情緒強度：正面或負面情緒的強烈程度
- 因果關係：是否說明該屬性對整體評價的影響
- 決策影響：是否影響推薦意願或再次入住意願

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

# 範例
評論："位置非常好，就在地鐵站旁邊，這是我選擇這家酒店的主要原因。房間還可以。"
輸出：
[
    {"key": "地理位置", "sentiment": "正面", "score": 5, "importance": 5},
    {"key": "交通便利", "sentiment": "正面", "score": 5, "importance": 5},
    {"key": "房間", "sentiment": "中立", "score": 3, "importance": 2}
]

分析：
- "地理位置"和"交通便利"：importance=5，因為用戶明確表示"非常好"、"主要原因"
- "房間"：importance=2，因為僅順帶提及"還可以"，沒有詳細描述
"""


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = load_config()
    c = cfg["analysis_ipa"]
    openai.api_key = cfg["openai"]["api_key"]
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
          f"預估: ${remaining * COST_PER_REVIEW_4O_MINI:.2f} USD | 含重要度評估")

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
