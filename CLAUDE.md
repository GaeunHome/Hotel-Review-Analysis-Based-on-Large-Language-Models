# CLAUDE.md

此檔案為 Claude Code 在此儲存庫中工作時提供指引。

## 專案概述

中文酒店評論 IPA 分析系統。GPT-4o-mini 提取情感屬性 → 兩階段 AI 分類（12 標準屬性）→ IPA 四象限 → 動態 SWOT。

**資料**：北京 5 家酒店、11,580 筆評論（2019-2022）

## 執行流程

```bash
python -m hotel_ipa.preparation.sequence         # 排序
python -m hotel_ipa.analysis.gpt_4o_mini_ipa     # GPT 分析
python -m hotel_ipa.classification.classify       # 兩階段分類
python -m hotel_ipa.visualization.ipa_dashboard   # 儀表板（含 Post-hoc + 驗證載入）
```

## 專案結構

```
├── config/config.example.json
├── src/hotel_ipa/
│   ├── constants.py               12 屬性、分期、色彩
│   ├── config_loader.py           設定載入
│   ├── utils.py                   JSON 解析、關鍵詞比對
│   ├── preparation/sequence.py    資料排序
│   ├── analysis/
│   │   ├── base.py                共用基礎（進度管理、API、續傳）
│   │   ├── gpt_4o_mini_ipa.py    主要分析模組
│   │   ├── gpt_4o_mini.py        基礎分析
│   │   └── gpt_4o.py             GPT-4o 分析
│   ├── classification/classify.py 兩階段 AI 分類
│   ├── visualization/
│   │   ├── ipa_dashboard.py       主 pipeline（資料→指標→圖表→儀表板）
│   │   ├── charts.py              matplotlib 靜態圖表
│   │   ├── html_dashboard.py      統一儀表板 HTML 生成
│   │   ├── ai_advisor.py          AI 顧問
│   │   └── swot_visualization.py  SWOT 視覺化
│   ├── swot/swot_engine.py        動態 SWOT（R1-R8 規則）
│   ├── importance/post_hoc.py     Post-hoc 重要度（GPT-4o）
│   ├── validation/stability.py    穩定性驗證（Fleiss'/Cohen's Kappa）
│   └── stats/                     統計檢定
├── data/                           全部 gitignore
└── tests/
```

## 12 標準屬性

地理位置、性價比、服務、房間、餐廳、停車、清潔度、公共設施、周邊環境、衛浴、交通、睡眠品質

## 重要度計算

- **逐筆平均**：每條評論 GPT-4o-mini 給的 importance 取平均
- **Post-hoc AI**：GPT-4o 看彙總統計後一次性判斷

## API 金鑰

`config/config.json`（gitignore）需要 OpenAI + Anthropic key

## 模型

- **GPT-4o-mini**：核心分析 + 分類
- **GPT-4o**：Post-hoc 重要度、驗證
- **Claude Sonnet 4**：跨模型驗證
