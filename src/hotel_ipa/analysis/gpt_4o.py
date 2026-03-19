"""
GPT-4o 分析 - 支援結構化文字輸出與 JSON 輸出，支援時期篩選
"""

import openai
from hotel_ipa.config_loader import load_config
from hotel_ipa.constants import API_TEMPERATURE, API_MAX_TOKENS, COST_PER_REVIEW_4O
from hotel_ipa.analysis.base import (
    load_reviews, stringify_columns,
    select_reviews_by_periods,
    ProgressManager, call_openai, run_analysis
)


# ============================================================================
# Prompts
# ============================================================================

SYSTEM_PROMPT_TEXT = """
# Role
你是一名酒店評論分析專家，負責提取屬性並給予 1-5 分情感評分（5=極佳，1=極差）。

# Purpose
請根據語意將評論歸類至以下 12 項屬性：
1.地理位置 (位置/距離/景點) 2.周邊環境 (氛圍/景觀) 3.交通 (地鐵/機場/出行)
4.性價比 (價格/划算) 5.清潔度 (衛生/打掃) 6.房間 (設施/床品/裝修/空調)
7.服務 (員工/態度) 8.餐廳 (早餐/飲食) 9.衛浴 (熱水/排水/備品)
10.公共設施 (泳池/大堂/電梯) 11.停車 (車位) 12.睡眠品質 (隔音/噪音)

# Condition
1. 敏感識別：即使評論極短（如「床好軟」），也須歸類至對應屬性（如「房間」）並評分。
2. 評分標準：1(非常負面)-5(非常正面) 整數。
3. 關鍵詞：每項屬性提取 3-5 個關鍵詞。
4. 排除：若評論完全無具體指向（僅說「很棒」、「還行」），輸出「無具體屬性評價」。

# Format
【屬性名稱】
關鍵詞：詞1、詞2
情感：正面/中立/負面
分數：X"""

SYSTEM_PROMPT_JSON = """
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
# Main
# ============================================================================

def main():
    cfg = load_config()
    openai.api_key = cfg["openai"]["api_key"]
    c = cfg["analysis_4o"]
    model = c["model"]
    output_format = c.get("output_format", "text")

    # 選擇 prompt
    system_prompt = SYSTEM_PROMPT_JSON if output_format == "json" else SYSTEM_PROMPT_TEXT
    print(f"📝 輸出格式: {output_format}")

    df = load_reviews(c["input_file"])

    # Filtering: periods > target_hotels > all
    target_hotels = c.get("target_hotels")
    periods = c.get("periods")

    if periods and target_hotels:
        df = select_reviews_by_periods(df, target_hotels, periods, limit=200)
        if len(df) == 0:
            raise ValueError("分期篩選後無資料")
    elif target_hotels:
        df = df[df["Hotel Name"].isin(target_hotels)]
        df = df.groupby("Hotel Name").head(600).reset_index(drop=True)
        for h in target_hotels:
            print(f"   {h}: {len(df[df['Hotel Name'] == h])} 條")
    else:
        print(f"📊 分析全部 {len(df)} 條")

    df = df.reset_index(drop=True)
    stringify_columns(df)

    progress = ProgressManager(c["output_file"])
    if c.get("resume", True):
        df, start_idx = progress.resume(df)
    else:
        df["分析結果"] = ""
        start_idx = 0

    remaining = len(df) - start_idx
    print(f"\n模型: {model} | 待處理: {remaining} | "
          f"預估: ${remaining * COST_PER_REVIEW_4O:.2f} USD")

    def analyze(text):
        return call_openai(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": f"請分析以下酒店評論：{text}"}],
            model=model, temperature=API_TEMPERATURE, max_tokens=API_MAX_TOKENS
        )

    run_analysis(df, analyze, progress, start_idx,
                 c.get("sleep_seconds", 0.3), c.get("save_interval", 10))


if __name__ == "__main__":
    main()
