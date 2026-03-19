"""
Auto-detect input format and normalize to a standard DataFrame
for SWOT analysis consumption.

Supports two input paths:
  Path A: classify.py output (has Standardized_Category column)
  Path B: raw analysis output (has 分析結果 column, needs keyword matching)
"""

import pandas as pd

from hotel_ipa.constants import STANDARD_ATTRIBUTES
from hotel_ipa.utils import parse_json_safe, match_category


def load_and_normalize(input_file: str) -> pd.DataFrame:
    """
    Load input data and return a normalized DataFrame with columns:
    [Hotel Name, Review ID, Keyword, Category, Sentiment, Score, Importance, Date]

    Auto-detects the input format:
    - CSV with Standardized_Category → Path A (classified)
    - Excel with 詳細數據 sheet → Path A (classified)
    - CSV/Excel with 分析結果 column → Path B (raw analysis, apply keyword matching)
    """
    print(f"📂 SWOT Detector: {input_file}")

    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, encoding='utf-8-sig')
    else:
        try:
            xl = pd.ExcelFile(input_file)
            if "詳細數據" in xl.sheet_names:
                df = pd.read_excel(input_file, sheet_name="詳細數據")
            else:
                df = pd.read_excel(input_file)
        except Exception:
            df = pd.read_excel(input_file)

    # Path A: Already classified
    if 'Standardized_Category' in df.columns:
        print("   偵測到分類結果格式 (Path A)")
        result = df.rename(columns={
            'Target_Keyword': 'Keyword',
            'Standardized_Category': 'Category',
        })
        cols = ['Review ID', 'Hotel Name', 'Keyword', 'Category',
                'Sentiment', 'Score', 'Importance', 'Date']
        result = result[[c for c in cols if c in result.columns]].copy()

    # Path A variant: already has Category column (e.g. ipa_extracted_data)
    elif 'Category' in df.columns and 'Keyword' in df.columns:
        print("   偵測到已提取格式 (Path A variant)")
        result = df.copy()

    # Path B: Raw analysis results
    elif '分析結果' in df.columns:
        print("   偵測到原始分析格式 (Path B)，套用關鍵詞匹配")
        records = []
        for _, row in df.iterrows():
            for item in parse_json_safe(row.get('分析結果', '')):
                if isinstance(item, dict):
                    records.append({
                        'Hotel Name': row.get('Hotel Name', ''),
                        'Review ID': row.get('Review ID', ''),
                        'Keyword': item.get('key', ''),
                        'Category': match_category(item.get('key', '')),
                        'Sentiment': item.get('sentiment', ''),
                        'Score': int(item.get('score', 0)),
                        'Importance': int(item.get('importance', 3)),
                        'Date': row.get('Date', ''),
                    })
        result = pd.DataFrame(records)
    else:
        raise ValueError(
            f"無法辨識輸入格式。需要 'Standardized_Category'、'Category' 或 '分析結果' 欄位。"
            f"\n現有欄位: {list(df.columns)}"
        )

    # Ensure numeric types
    for col in ['Score', 'Importance']:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(3).astype(int)

    # Filter to standard attributes only
    if 'Category' in result.columns:
        result = result[result['Category'].isin(STANDARD_ATTRIBUTES)].copy()

    print(f"   ✓ {len(result)} 筆記錄，{result['Hotel Name'].nunique()} 家酒店")
    return result
