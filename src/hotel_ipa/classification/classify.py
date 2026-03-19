"""
Two-stage keyword classification:
1. Raw classification: map keywords to standard or AI-generated labels
2. Consolidation: merge similar AI-generated labels back to standards
"""

import pandas as pd
import json
import time
import os
from openai import OpenAI
from tqdm import tqdm
from typing import Dict

from hotel_ipa.constants import STANDARD_ATTRIBUTES
from hotel_ipa.utils import parse_json_safe

CHECKPOINT_FILE = "data/output/classify_checkpoint.json"

STANDARD_LABELS = STANDARD_ATTRIBUTES


class TransparentClassifier:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.raw_mapping = {}
        self.final_mapping = {}
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.raw_mapping = data.get("raw_mapping", {})
            self.final_mapping = data.get("final_mapping", {})
            print(f"📌 載入 checkpoint: {len(self.raw_mapping)} 個分類結果")

    def _save_checkpoint(self):
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"raw_mapping": self.raw_mapping,
                       "final_mapping": self.final_mapping},
                      f, ensure_ascii=False, indent=2)

    # ---- Stage 0: Load data ----
    def load_data(self, filepath: str):
        """Load analysis results, extract keyword contexts and full records."""
        print(f"📂 讀取: {filepath}")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        else:
            df = pd.read_excel(filepath)
        col_name = next((c for c in df.columns if "分析" in c or "result" in c.lower()), None)
        if not col_name:
            raise ValueError("找不到分析欄位")

        text_col = next((c for c in df.columns if "review text" in c.lower() or "評論" in c), None)

        keyword_context = {}
        full_records = []

        for _, row in df.iterrows():
            items = parse_json_safe(row[col_name])
            review = str(row[text_col])[:150] if text_col else ""
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "key" in item:
                        kw = item["key"]
                        if kw not in keyword_context:
                            keyword_context[kw] = {
                                "review_text": review,
                                "sentiment": item.get("sentiment", ""),
                                "score": item.get("score", "")
                            }
                        full_records.append({
                            **row.to_dict(),
                            "Target_Keyword": kw,
                            "Sentiment": item.get("sentiment"),
                            "Score": item.get("score"),
                            "Importance": item.get("importance")
                        })
        return keyword_context, full_records

    # ---- Stage 1: Raw classification ----
    def classify_raw(self, keyword_context: Dict[str, dict], batch_size: int = 50):
        """Classify keywords using AI with review context."""
        print(f"\n{'='*50}\n階段一：關鍵詞分類\n{'='*50}")

        todo = [k for k in keyword_context if k not in self.raw_mapping]
        print(f"待分類: {len(todo)} (已跳過 {len(keyword_context) - len(todo)})")

        for batch in tqdm([todo[i:i+batch_size] for i in range(0, len(todo), batch_size)],
                          desc="AI 標註中"):
            items = [{
                "keyword": kw,
                "review_text": keyword_context[kw]["review_text"],
                "sentiment": keyword_context[kw]["sentiment"],
                "score": keyword_context[kw]["score"]
            } for kw in batch]

            prompt = f"""你是酒店顧客體驗研究員，將關鍵詞歸類為標準類別。

【標準標籤】{json.dumps(STANDARD_LABELS, ensure_ascii=False)}

【規則】
1. 優先對應標準標籤
2. 涵蓋多個標準標籤時，結合上下文選最接近的
3. 無法對應時，生成精準中文標籤（2-5字，禁用「其他」）

【待分類】{json.dumps(items, ensure_ascii=False)}

回傳 JSON: {{ "results": [{{"keyword": "...", "tag": "..."}}] }}"""

            for attempt in range(3):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.0
                    )
                    for item in json.loads(resp.choices[0].message.content).get("results", []):
                        self.raw_mapping[item["keyword"]] = item["tag"]
                    break
                except Exception as e:
                    print(f"⚠️ 錯誤 ({attempt+1}/3): {e}")
                    if attempt < 2:
                        time.sleep(5)
            self._save_checkpoint()

        print("✅ 階段一完成")
        return self.raw_mapping

    # ---- Stage 2: Consolidate tags ----
    def consolidate_tags(self):
        """Merge similar AI-generated tags back to standard labels."""
        print(f"\n{'='*50}\n階段二：合併標籤\n{'='*50}")

        new_tags = [t for t in set(self.raw_mapping.values()) if t not in STANDARD_LABELS]
        if not new_tags:
            print("所有關鍵詞已對應標準標籤")
            self.final_mapping = {t: t for t in set(self.raw_mapping.values())}
            self._save_checkpoint()
            return

        print(f"{len(new_tags)} 個自生成標籤待合併...")

        prompt = f"""你是酒店顧客體驗研究員。

【標準標籤】{json.dumps(STANDARD_LABELS, ensure_ascii=False)}
【自生成標籤】{json.dumps(new_tags, ensure_ascii=False)}

規則：
1. 語意吻合標準標籤 → 對應
2. 多個自生成標籤語意相近 → 合併
3. 確實獨特 → 保留
4. 禁用「其他」

回傳 JSON: {{ "mapping": {{ "標籤": "對應後標籤" }} }}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            mapping = json.loads(resp.choices[0].message.content).get("mapping", {})
            self.final_mapping = {t: t for t in STANDARD_LABELS}
            self.final_mapping.update(mapping)
            self._save_checkpoint()
            print(f"✅ 合併完成，最終 {len(set(self.final_mapping.values()))} 個標籤")
        except Exception as e:
            print(f"⚠️ 合併失敗: {e}")
            self.final_mapping = {t: t for t in set(self.raw_mapping.values())}

    # ---- Stage 3: Output report ----
    def save_full_report(self, full_records, output_file):
        """Apply classification and save to Excel."""
        print(f"\n💾 儲存: {output_file}")
        for record in full_records:
            kw = record["Target_Keyword"]
            raw_tag = self.raw_mapping.get(kw, "未分類")
            record["AI_Raw_Tag"] = raw_tag
            record["Standardized_Category"] = self.final_mapping.get(raw_tag, raw_tag)

        df = pd.DataFrame(full_records)
        cols = ["Review ID", "Hotel Name", "Target_Keyword", "AI_Raw_Tag",
                "Standardized_Category", "Sentiment", "Score", "Importance", "Date"]
        df = df[[c for c in cols if c in df.columns]]

        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, sheet_name="詳細數據", index=False)
            df["Standardized_Category"].value_counts().reset_index(
                ).rename(columns={"index": "標準類別", "count": "次數"}
                ).to_excel(writer, sheet_name="標準類別統計", index=False)
            df["AI_Raw_Tag"].value_counts().reset_index(
                ).rename(columns={"index": "原始標籤", "count": "次數"}
                ).to_excel(writer, sheet_name="原始標籤統計", index=False)
        print("✅ 儲存完畢")


if __name__ == "__main__":
    from hotel_ipa.config_loader import load_config
    cfg = load_config()
    c = cfg["classification"]

    classifier = TransparentClassifier(api_key=cfg["openai"]["api_key"], model=c["model"])
    keyword_context, records = classifier.load_data(c["input_file"])

    if keyword_context:
        classifier.classify_raw(keyword_context, batch_size=50)
        classifier.consolidate_tags()

    output = c["output_dir"] + "/final_report_hybrid_10k.xlsx"
    classifier.save_full_report(records, output)

    from collections import Counter
    cat_counts = Counter()
    for r in records:
        raw_tag = classifier.raw_mapping.get(r["Target_Keyword"], "未分類")
        cat_counts[classifier.final_mapping.get(raw_tag, raw_tag)] += 1
    print(f"\n{'='*50}\n📊 分類摘要\n{'='*50}")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:<12}: {count} 筆")
    print(f"\n總計: {len(records)} 筆 -> {output}")
