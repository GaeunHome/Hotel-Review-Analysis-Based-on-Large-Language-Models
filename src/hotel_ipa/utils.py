"""
Shared utility functions for hotel IPA/SWOT analysis.
"""

import json
import ast
import pandas as pd

from hotel_ipa.constants import KEYWORD_PATTERNS


def parse_json_safe(text) -> list:
    """Safely parse JSON analysis results from AI output.

    Handles NaN, empty strings, malformed JSON, and Python literal formats.
    Returns an empty list on failure.
    """
    if pd.isna(text) or text == "":
        return []
    text = str(text).strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return []


def match_category(keyword: str) -> str:
    """Match a keyword to the closest standard hotel attribute category.

    Uses substring matching against KEYWORD_PATTERNS defined in constants.py.
    Returns '其他' if no match is found.
    """
    for category, patterns in KEYWORD_PATTERNS.items():
        if any(p in keyword for p in patterns):
            return category
    return "其他"


def load_data_file(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Load a data file, auto-detecting CSV or Excel format.

    Args:
        filepath: Path to CSV or Excel file.
        sheet_name: Optional Excel sheet name (ignored for CSV).
    """
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, encoding="utf-8-sig")
    elif sheet_name:
        return pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        return pd.read_excel(filepath)
