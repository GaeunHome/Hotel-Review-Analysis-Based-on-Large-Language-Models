"""
共用設定載入工具。
從 config/config.json 讀取設定，若找不到則提示使用者建立。
"""
import json
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    載入 config/config.json。

    Args:
        config_path: 設定檔路徑，預設自動尋找專案根目錄下的 config/config.json

    Returns:
        設定字典

    Raises:
        FileNotFoundError: 找不到設定檔時
    """
    if config_path is None:
        root = Path(__file__).parent.parent.parent
        config_path = root / "config" / "config.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"找不到設定檔: {config_path}\n"
            f"請執行以下步驟：\n"
            f"  1. 複製 config/config.example.json 為 config/config.json\n"
            f"  2. 在 config/config.json 填入你的 OpenAI API 金鑰"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_api_key(cfg: dict = None) -> str:
    """取得 OpenAI API 金鑰，config 未傳入時自動載入。"""
    if cfg is None:
        cfg = load_config()
    return cfg["openai"]["api_key"]


def get_anthropic_key(cfg: dict = None) -> str:
    """取得 Anthropic API 金鑰，config 未傳入時自動載入。"""
    if cfg is None:
        cfg = load_config()
    return cfg["anthropic"]["api_key"]
