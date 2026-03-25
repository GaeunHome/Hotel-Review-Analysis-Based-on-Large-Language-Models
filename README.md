# Hotel Review Analysis Based on Large Language Models

## 研究背景

- **資料**：北京 5 家酒店、11,580 筆評論（2019-2022）
- **方法**：GPT-4o-mini 逐條提取 → 兩階段 AI 分類（12 標準屬性）→ IPA 四象限 → 動態 SWOT
- **參考**：Wu, J., Zhao, N., & Yang, T. (2024). Wisdom of crowds: SWOT analysis based on hybrid text mining methods using online reviews. *Journal of Business Research*, 171, 114378.
- **產出**：自包含 HTML 互動式儀表板

## 快速開始

```bash
pip install -e .
cp config/config.example.json config/config.json  # 填入 API key

python -m hotel_ipa.preparation.sequence         # 1. 資料排序
python -m hotel_ipa.analysis.gpt_4o_mini_ipa     # 2. GPT 逐條分析
python -m hotel_ipa.classification.classify       # 3. 兩階段分類
python -m hotel_ipa.visualization.ipa_dashboard   # 4. 生成儀表板（含 SWOT）

open data/output/ipa_dashboard/酒店評論分析儀表板.html
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
data/output/final_report_hybrid_10k.xlsx           35,500 筆（→12 標準屬性）
  │  visualization/ipa_dashboard.py                ← IPA + SWOT + 圖表 + 儀表板
  ↓
data/output/ipa_dashboard/酒店評論分析儀表板.html    自包含互動式儀表板
```

## 儀表板內容

| 分頁 | 內容 |
|------|------|
| 總覽 | 情感統計、分類標籤分布、非標準標籤整合說明、穩定性與有效性驗證 |
| 互動式 IPA | Plotly 散佈圖（重要度 × 績效），可篩選屬性/酒店 |
| 動態趨勢 | 選擇屬性，查看 5 家酒店在 2020/2021/2022 的變化 |
| 數據圖表 | 各酒店 12 屬性績效/重要度橫向柱狀圖 |
| SWOT 分析 | 3 組配對比較（績效對比 + 動態 SWOT 散佈圖 + 識別表 + AI 解讀） |

### SWOT 分析比較組合

| 焦點酒店 | 競爭酒店 |
|----------|----------|
| 励骏酒店 | 北京王府井希尔顿酒店 |
| 北京天安门王府井漫心酒店 | 全季酒店(北京国贸东店) |
| 北京天安门王府井漫心酒店 | 麗枫酒店(北京国贸店) |

SWOT 因子判定基於 Wu et al. (2024) R1-R8 規則：先將各酒店屬性績效與自身平均績效比較（內部優勢/劣勢），再交叉比較兩家酒店的內部分類及績效差異，判定 S/W/O/T。動態 SWOT 圖以箭頭呈現各屬性在四個時期（COVID前、爆發期、恢復期、後疫情）的變化軌跡，顏色深淺反映 Post-hoc 重要度。

## 驗證結果

| 驗證 | 方法 | 結果 |
|------|------|------|
| 穩定性 | Fleiss' Kappa（GPT-4o-mini ×5） | 0.994（幾乎完全一致）|
| 有效性-評分 | Spearman ρ（vs GPT-4o / Claude Sonnet 4） | 0.785-0.807 |

## 專案結構

```
├── config/config.example.json              設定範本（需複製為 config.json）
├── src/hotel_ipa/
│   ├── constants.py                        12 屬性、分期、色彩、酒店順序
│   ├── config_loader.py                    設定載入
│   ├── utils.py                            JSON 解析、關鍵詞比對
│   ├── preparation/sequence.py             資料排序（步驟 1）
│   ├── analysis/
│   │   ├── base.py                         共用基礎（進度管理、API、斷點續傳）
│   │   └── gpt_4o_mini_ipa.py              GPT-4o-mini IPA 分析（步驟 2）
│   ├── classification/classify.py          兩階段 AI 分類（步驟 3）
│   ├── visualization/
│   │   ├── ipa_dashboard.py                主 pipeline：IPA + SWOT + 儀表板（步驟 4）
│   │   ├── charts.py                       matplotlib 靜態圖表（IPA 四象限等）
│   │   ├── html_dashboard.py               統一儀表板 HTML 生成
│   │   ├── ai_advisor.py                   AI 顧問（IPA 建議 + SWOT 解讀）
│   │   └── swot_visualization.py           SWOT 圖表（績效對比、動態散佈圖）
│   ├── swot/
│   │   ├── swot_engine.py                  SWOT 分析引擎（R1-R8 規則）
│   │   └── swot_detector.py                輸入格式偵測與正規化
│   ├── importance/post_hoc.py              Post-hoc 重要度（GPT-4o 整體判斷）
│   ├── validation/stability.py             穩定性驗證（Fleiss'/Spearman）
│   └── stats/
│       ├── statistical_tests.py            統計檢定
│       └── ipa_priority_tests.py           IPA 優先度檢定
├── data/                                   全部 gitignore（可重新生成）
├── docs/                                   參考文獻
└── pyproject.toml
```

### 模組說明

| 模組 | 說明 |
|------|------|
| `ipa_dashboard.py` | 主入口，串接資料載入→指標計算→圖表生成→SWOT 比較→AI 解讀→HTML 儀表板 |
| `html_dashboard.py` | 產生自包含 HTML，所有圖表以 base64 嵌入，Plotly.js 互動圖表 |
| `swot_engine.py` | 實作 Wu (2024) R1-R8 SWOT 判定規則，支援靜態與動態（多時期）分析 |
| `swot_visualization.py` | 績效對比長條圖（Fig. 6 風格）+ 動態 SWOT 散佈圖（Fig. 7-9 風格） |
| `ai_advisor.py` | 呼叫 GPT-4o-mini，針對 IPA 數據和 SWOT 結果產生分析解讀 |
| `post_hoc.py` | GPT-4o 看彙總統計後一次性判斷各屬性重要度（對比逐筆平均） |
| `charts.py` | matplotlib 靜態圖表：IPA 四象限、改進優先度、績效/重要度排序 |

## 12 標準酒店屬性

地理位置、性價比、服務、房間、餐廳、停車、清潔度、公共設施、周邊環境、衛浴、交通、睡眠品質

## 重要度計算

| 方法 | 說明 |
|------|------|
| 逐筆平均 | 每條評論 GPT-4o-mini 給的 importance 取平均 |
| Post-hoc AI | GPT-4o 看彙總統計後一次性判斷（SWOT 分析預設使用此方法） |

## 注意事項

- `config/config.json` 含 API key（gitignore），範本見 `config.example.json`
- `data/` 全部 gitignore，所有產出可重新生成
- 需要 OpenAI API key（主要）+ Anthropic API key（驗證用）
