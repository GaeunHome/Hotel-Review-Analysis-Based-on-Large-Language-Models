"""
Shared infrastructure for review analysis scripts.
Provides data loading, progress management, and the main analysis loop.
"""

import os
import time
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
import openai

from hotel_ipa.utils import load_data_file


# ============================================================================
# Data Loading & Filtering
# ============================================================================

REQUIRED_COLS = ["Review ID", "Rating", "Date", "Hotel Name", "Review Text"]


def load_reviews(input_path: str) -> pd.DataFrame:
    """Load and validate review data (CSV or Excel)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到輸入檔案: {input_path}")

    print(f"📂 載入資料: {input_path}")
    df = load_data_file(input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")

    return df


def filter_hotels(df: pd.DataFrame, target_hotels=None) -> pd.DataFrame:
    """Filter to target hotels, print summary."""
    all_hotels = df["Hotel Name"].unique()
    print(f"\n📋 檔案中共有 {len(all_hotels)} 家酒店:")
    for i, h in enumerate(all_hotels, 1):
        print(f"   {i}. {h} ({len(df[df['Hotel Name'] == h])} 條評論)")

    if target_hotels:
        df = df[df["Hotel Name"].isin(target_hotels)].reset_index(drop=True)
        print(f"\n🎯 篩選後共 {len(df)} 條評論")
    else:
        print(f"\n📊 將分析全部 {len(df)} 條評論")

    return df


def stringify_columns(df: pd.DataFrame, cols=None):
    """Convert columns to string, fill NaN."""
    if cols is None:
        cols = ["Rating", "Date", "Hotel Name", "Review Text"]
    for c in cols:
        if c in df.columns:
            if c == "Date" and pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime('%Y-%m-%d')
            df[c] = df[c].fillna("").astype(str)


def select_reviews_by_periods(df: pd.DataFrame, target_hotels: list,
                              periods_config: dict, limit: int = 200) -> pd.DataFrame:
    """Select reviews by time periods, each hotel/period up to `limit` rows."""
    print(f"\n📅 執行分期篩選...")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().any():
        dropped = df['Date'].isna().sum()
        print(f"⚠️ {dropped} 筆日期格式錯誤，已忽略")
        df = df.dropna(subset=['Date'])

    frames = []
    for hotel in target_hotels:
        hotel_df = df[df["Hotel Name"] == hotel]
        if len(hotel_df) == 0:
            print(f"   ⚠️ 找不到 {hotel}")
            continue
        for p_name, p_range in periods_config.items():
            start, end = pd.to_datetime(p_range['start']), pd.to_datetime(p_range['end'])
            period = hotel_df[(hotel_df['Date'] >= start) & (hotel_df['Date'] <= end)]
            period = period.sort_values('Date')
            selected = period.head(limit).copy()
            selected['Period_Tag'] = p_name
            frames.append(selected)
            print(f"   {hotel} | {p_name}: {len(period)}筆 -> 選取{len(selected)}筆")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================================
# Progress Management
# ============================================================================

class ProgressManager:
    """Handles checkpoint saving and resume for long-running analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        base = output_path.rsplit('.', 1)[0]
        self.checkpoint_path = base + '_checkpoint.json'
        self.csv_path = base + '_progress.csv'

    def resume(self, df: pd.DataFrame) -> tuple:
        """Try to resume from saved progress. Returns (df, start_idx)."""
        sources = [
            (self.csv_path, lambda f: pd.read_csv(f, encoding='utf-8-sig')),
            (self.output_path, load_data_file),
        ]
        for path, reader in sources:
            if not os.path.exists(path):
                continue
            try:
                existing = reader(path)
                if "分析結果" not in existing.columns:
                    continue
                start_idx = len(existing)
                for idx in range(len(existing)):
                    val = existing.loc[idx, "分析結果"]
                    text = str(existing.loc[idx, "Review Text"]).strip()
                    if (pd.isna(val) or str(val).strip() == "") and text:
                        start_idx = idx
                        break
                df["分析結果"] = existing["分析結果"]
                print(f"📍 從第 {start_idx + 1} 條繼續 ({start_idx} 條已完成) "
                      f"[{os.path.basename(path)}]")
                return df, start_idx
            except Exception as e:
                print(f"⚠️ 無法讀取 {path}: {e}")

        df["分析結果"] = ""
        return df, 0

    def save(self, df: pd.DataFrame, stats: dict, final: bool = False):
        """Save progress: CSV for intermediate, Excel for final."""
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if final:
            if self.output_path.endswith('.csv'):
                df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            else:
                df.to_excel(self.output_path, index=False)
        else:
            df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')

        checkpoint = {
            "last_saved": datetime.now().isoformat(),
            "total_rows": len(df),
            "completed_rows": int(df["分析結果"].notna().sum()),
            **stats
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)


# ============================================================================
# API Call & Main Loop
# ============================================================================

from hotel_ipa.constants import API_TEMPERATURE, API_MAX_TOKENS


@retry(wait=wait_exponential(multiplier=1, min=2, max=30),
       stop=stop_after_attempt(5))
def call_openai(messages, model="gpt-4o-mini", temperature=API_TEMPERATURE,
                max_tokens=API_MAX_TOKENS):
    """Retry-wrapped OpenAI chat completion."""
    response = openai.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def run_analysis(df: pd.DataFrame, analyze_fn, progress: ProgressManager,
                 start_idx: int, sleep_seconds=0.1, save_interval=10):
    """
    Main analysis loop: iterate reviews, call analyze_fn, track progress.

    Args:
        df: DataFrame with 'Review Text' column
        analyze_fn: callable(review_text) -> str
        progress: ProgressManager instance
        start_idx: row index to start from
        sleep_seconds: delay between API calls
        save_interval: save checkpoint every N rows
    """
    total = len(df)
    success, errors = start_idx, 0
    start_time = time.time()

    with tqdm(range(start_idx, total), initial=start_idx,
              total=total, desc="分析中") as pbar:
        for idx in pbar:
            review = df.loc[idx, "Review Text"].strip()
            if not review:
                df.loc[idx, "分析結果"] = ""
                continue

            try:
                df.loc[idx, "分析結果"] = analyze_fn(review)
                success += 1
            except Exception as e:
                df.loc[idx, "分析結果"] = f"[ERROR] {e}"
                errors += 1
                pbar.write(f"❌ 第 {idx + 1} 條: {e}")

            if (idx + 1) % save_interval == 0:
                progress.save(df, {'success': success, 'errors': errors})
                elapsed = time.time() - start_time
                done = idx + 1 - start_idx
                remaining = (total - idx - 1) * (elapsed / done) / 60
                pbar.write(f"💾 已保存 | 剩餘 {remaining:.1f} 分鐘")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print("📝 輸出最終 Excel...")
    progress.save(df, {'success': success, 'errors': errors}, final=True)

    elapsed = time.time() - start_time
    print(f"\n✅ 完成！成功 {success}/{total}，錯誤 {errors}，"
          f"耗時 {elapsed / 3600:.2f} 小時")
    return df
