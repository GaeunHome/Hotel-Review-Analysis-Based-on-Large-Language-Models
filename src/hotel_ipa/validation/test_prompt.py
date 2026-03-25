"""
Quick prompt consistency test.
Pick a few reviews, run through GPT-4o-mini / GPT-4o / Claude,
compare scores side by side.

Usage:
    python -m hotel_ipa.validation.test_prompt          # default 5 reviews
    python -m hotel_ipa.validation.test_prompt --n 10   # 10 reviews
"""

import pandas as pd
import json
import argparse

import openai
import anthropic

from hotel_ipa.config_loader import load_config
from hotel_ipa.constants import STANDARD_ATTRIBUTES
from hotel_ipa.validation.cross_model import VALIDATION_PROMPT, _parse_response


def test_reviews(n: int = 5):
    cfg = load_config()
    openai_key = cfg["openai"]["api_key"]
    anthropic_key = cfg.get("anthropic", {}).get("api_key")
    claude_model = cfg["validation"].get("models", {}).get(
        "claude_sonnet", "claude-sonnet-4-20250514")

    # Pick reviews: prioritize GT=5 (the problematic boundary)
    gt = pd.read_csv("data/output/validation/ground_truth.csv", encoding="utf-8-sig")
    sample = pd.read_csv("data/output/validation/sample_reviews.csv", encoding="utf-8-sig")

    # Find reviews that have GT=5 attributes
    rids_with_5 = gt[gt["score"] == 5]["Review ID"].unique()
    test_reviews = sample[sample["Review ID"].isin(rids_with_5)].head(n)

    print(f"測試 {len(test_reviews)} 筆評論 (GT=5 為主)\n")

    for _, row in test_reviews.iterrows():
        rid = row["Review ID"]
        review = str(row["Review Text"]).strip()
        gt_items = gt[gt["Review ID"] == rid]

        print(f"{'='*70}")
        print(f"Review {rid}: {review[:60]}...")
        print(f"GT: {', '.join(f'{r.category}={int(r.score)}' for _, r in gt_items.iterrows())}")

        # GPT-4o-mini
        try:
            resp = openai.OpenAI(api_key=openai_key).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": VALIDATION_PROMPT},
                    {"role": "user", "content": f"請分析以下酒店評論：{review}"},
                ],
                temperature=0.01, max_tokens=800,
            )
            mini_items = _parse_response(resp.choices[0].message.content)
            print(f"GPT-4o-mini: {', '.join(f'{it['category']}={it['score']}' for it in mini_items)}")
        except Exception as e:
            print(f"GPT-4o-mini: ERROR {e}")

        # GPT-4o
        try:
            resp = openai.OpenAI(api_key=openai_key).chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": VALIDATION_PROMPT},
                    {"role": "user", "content": f"請分析以下酒店評論：{review}"},
                ],
                temperature=0.01, max_tokens=800,
            )
            gpt4o_items = _parse_response(resp.choices[0].message.content)
            print(f"GPT-4o:      {', '.join(f'{it['category']}={it['score']}' for it in gpt4o_items)}")
        except Exception as e:
            print(f"GPT-4o: ERROR {e}")

        # Claude
        if anthropic_key:
            try:
                resp = anthropic.Anthropic(api_key=anthropic_key).messages.create(
                    model=claude_model, max_tokens=800,
                    system=VALIDATION_PROMPT,
                    messages=[{"role": "user", "content": f"請分析以下酒店評論：{review}"}],
                    temperature=0.01,
                )
                claude_items = _parse_response(resp.content[0].text)
                print(f"Claude:      {', '.join(f'{it['category']}={it['score']}' for it in claude_items)}")
            except Exception as e:
                print(f"Claude: ERROR {e}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    test_reviews(args.n)
