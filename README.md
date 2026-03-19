# 酒店評論 IPA 分析系統

基於 AI 的中文酒店評論情感分析系統，整合 IPA（重要度-績效分析）與動態 SWOT 競爭力評估。

## 研究背景

- **資料**：北京 5 家酒店、11,580 筆評論（2019-2022）
- **方法**：GPT-4o-mini 逐條提取 → 兩階段 AI 分類（12 標準屬性）→ IPA 四象限 → 動態趨勢
- **產出**：自包含 HTML 互動式儀表板

## 快速開始

```bash
pip install -e .
cp config/config.example.json config/config.json  # 填入 API key

python -m hotel_ipa.preparation.sequence         # 1. 資料排序
python -m hotel_ipa.analysis.gpt_4o_mini_ipa     # 2. GPT 逐條分析
python -m hotel_ipa.classification.classify       # 3. 兩階段分類
python -m hotel_ipa.visualization.ipa_dashboard   # 4. 生成儀表板

open data/output/ipa_dashboard/IPA統一儀表板.html
```

## 流程與資料

```
data/raw/Raw_dataset_for_JBR_paper.xlsx           11,580 筆原始評論
  │  preparation/sequence.py
  ↓
data/processed/sorted_dataset.csv                  按酒店+日期排序
  │  analysis/gpt_4o_mini_ipa.py                   ← GPT-4o-mini
  ↓
data/output/reviews_analysis_with_importance.csv   +JSON 分析結果
  │  classification/classify.py                    ← 兩階段 AI 分類
  ↓
data/output/final_report_hybrid_10k.xlsx           35,500 筆（19→12 標準屬性）
  │  visualization/ipa_dashboard.py                ← Post-hoc + 圖表 + 儀表板
  ↓
data/output/ipa_dashboard/IPA統一儀表板.html        自包含互動式儀表板
```

## 儀表板內容

| 分頁 | 內容 |
|------|------|
| 總覽 | 情感統計、分類標籤分布、非標準標籤整合說明、穩定性與有效性驗證 |
| 互動式 IPA | Plotly 散佈圖（重要度 × 績效），可篩選屬性/酒店 |
| 動態趨勢 | 選擇屬性，查看 5 家酒店在 2020/2021/2022 的變化 |
| 數據圖表 | 各酒店 12 屬性績效/重要度橫向柱狀圖 |

## 驗證結果

| 驗證 | 方法 | 結果 |
|------|------|------|
| 穩定性 | Fleiss' Kappa（GPT-4o-mini ×5） | 0.994（幾乎完全一致）|
| 有效性-分類 | 屬性召回率 | 80-87% |
| 有效性-情感 | 情感一致率（vs Ground Truth） | 96-98% |
| 有效性-評分 | Spearman ρ（vs Ground Truth） | 0.785-0.807 |

## 專案結構

```
├── config/config.example.json          設定範本
├── src/hotel_ipa/
│   ├── constants.py                    12 屬性、分期、色彩
│   ├── config_loader.py                設定載入
│   ├── utils.py                        JSON 解析、關鍵詞比對
│   ├── preparation/sequence.py         資料排序
│   ├── analysis/
│   │   ├── base.py                     共用基礎（進度、API、續傳）
│   │   ├── gpt_4o_mini_ipa.py          GPT-4o-mini IPA 分析（主要）
│   │   ├── gpt_4o_mini.py              基礎分析
│   │   └── gpt_4o.py                   GPT-4o 分析
│   ├── classification/classify.py      兩階段 AI 分類
│   ├── visualization/
│   │   ├── ipa_dashboard.py            主 pipeline
│   │   ├── charts.py                   matplotlib 靜態圖表
│   │   ├── html_dashboard.py           統一儀表板 HTML
│   │   ├── ai_advisor.py              AI 顧問
│   │   └── swot_visualization.py      SWOT 視覺化
│   ├── swot/swot_engine.py            動態 SWOT（R1-R8）
│   ├── importance/post_hoc.py         Post-hoc 重要度（GPT-4o）
│   ├── validation/stability.py        穩定性驗證
│   └── stats/                         統計檢定
├── data/                               gitignore（可重新生成）
├── docs/使用說明.md
└── tests/
```

## 注意事項

- `config/config.json` 含 API key（gitignore），範本見 `config.example.json`
- `data/` 全部 gitignore，所有產出可重新生成
- 需要 OpenAI API key（主要）+ Anthropic API key（驗證用）
