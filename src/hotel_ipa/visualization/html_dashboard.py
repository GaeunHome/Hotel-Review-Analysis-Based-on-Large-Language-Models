"""
HTML dashboard generation for IPA analysis.
Generates a single self-contained HTML with all charts, stats, and interactive elements.
"""

import os
import json
import base64
import numpy as np
import pandas as pd
from datetime import datetime

from hotel_ipa.constants import STANDARD_ATTRIBUTES, HOTEL_STYLES


def _img_to_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


# Chart descriptions for hotel detail page
_CHART_DESC = {
    'ipa_scatter': '每個編號圓點代表一個屬性，位置由重要度（X）和績效（Y）決定。虛線為均值分隔線，右上為「繼續保持」，右下為「集中改善」，左上為「可調配資源」，左下為「低優先」。',
    'priority': '改進優先度 = 重要度 × 績效缺口。分數越高代表該屬性越需要優先投入資源改善。顏色反映正面率。',
    'comprehensive': '藍色為重要度，橘色為績效。兩者差距越大，代表顧客期望與實際表現落差越大。',
    'importance': '反映顧客對各屬性的重視程度。數值越高，代表顧客越在意該屬性。括號內為被提及次數。',
    'performance': '反映各屬性的實際表現。灰色虛線為及格線(3)，綠色虛線為優良線(4)。括號內為正面率。',
}


def generate_unified_dashboard(
    all_priority_dfs: dict,
    priority_all: pd.DataFrame,
    global_importance: dict,
    charts_dir: str,
    compare_dir: str,
    posthoc_importance: dict | None,
    posthoc_dir: str | None,
    overview_stats: dict | None = None,
    output_path: str = None,
):
    hotels = list(all_priority_dfs.keys())
    all_hotels_key = "全部酒店"
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- Collect chart images ----
    comp_images = {}
    for name, fname in [('imp_compare', '重要度比較_逐筆vs_posthoc.png')]:
        comp_images[name] = _img_to_base64(os.path.join(compare_dir, fname))

    hotel_keys = [all_hotels_key] + hotels
    chart_types = [
        ('ipa_scatter', 'IPA四象限.png', 'IPA 四象限分析'),
        ('priority', '改進優先度.png', '改進優先度排序'),
        ('comprehensive', '綜合對比.png', '重要度與績效對比'),
        ('importance', '重要度排序.png', '重要度排序'),
        ('performance', '績效排序.png', '績效排序'),
    ]
    hotel_charts = {}
    for h in hotel_keys:
        hotel_charts[h] = []
        h_dir = os.path.join(charts_dir, h)
        for ctype, fname, title in chart_types:
            b64 = _img_to_base64(os.path.join(h_dir, fname))
            if b64:
                hotel_charts[h].append((ctype, title, b64))

    posthoc_charts = {}
    if posthoc_dir and posthoc_importance:
        for h in hotel_keys:
            suffix = h if h != all_hotels_key else "全部"
            b64 = _img_to_base64(os.path.join(posthoc_dir, f"IPA四象限_posthoc_{suffix}.png"))
            if b64:
                posthoc_charts[h] = b64

    # ---- Stats tables ----
    all_dfs = {all_hotels_key: priority_all, **all_priority_dfs}
    tables_json = {}
    for h, pdf in all_dfs.items():
        tables_json[h] = [
            {k: round(row[k], 2) if isinstance(row[k], float) else int(row[k]) if k == '提及次數' else row[k]
             for k in ['屬性', '平均重要度', '平均績效', '績效缺口', '改進優先度', '提及次數', '正向率%']}
            for _, row in pdf.sort_values('改進優先度', ascending=False).iterrows()
        ]

    # ---- Plotly IPA data ----
    traces_data = []
    all_imp_vals, all_perf_vals = [], []
    hotel_colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
    for h_idx, (hotel, pdf) in enumerate(all_priority_dfs.items()):
        color = hotel_colors[h_idx % len(hotel_colors)]
        for _, row in pdf.iterrows():
            attr = row['屬性']
            imp = global_importance.get(attr, row['平均重要度'])
            perf = row['平均績效']
            all_imp_vals.append(imp)
            all_perf_vals.append(perf)
            traces_data.append({
                'hotel': hotel, 'attr': attr, 'imp': imp, 'perf': perf,
                'mentions': int(row['提及次數']), 'pos_rate': float(row['正向率%']),
                'priority': float(row['改進優先度']), 'color': color,
            })
    mean_imp = np.mean(all_imp_vals) if all_imp_vals else 3
    mean_perf = np.mean(all_perf_vals) if all_perf_vals else 3
    attrs_list = list(global_importance.keys())
    trend_json = json.dumps(overview_stats.get('trend_data', []) if overview_stats else [], ensure_ascii=False)

    # ---- Hotel detail sections (one chart per block with description) ----
    hotel_sections = {}
    for h in hotel_keys:
        html = ""
        for ctype, title, b64 in hotel_charts.get(h, []):
            desc = _CHART_DESC.get(ctype, '')
            html += f'''<div class="chart-block">
                <div class="chart-item"><img src="{b64}" alt="{title}"></div>
                <div class="chart-desc"><strong>{title}</strong><br>{desc}</div>
            </div>\n'''
        hotel_sections[h] = html

    # ---- Overview stats ----
    stats_html = ""
    if overview_stats:
        s = overview_stats
        total = s['total_mentions']
        pos, neg, neu = s['pos'], s['neg'], s['neu']
        pos_pct = round(pos / total * 100, 1) if total else 0
        neg_pct = round(neg / total * 100, 1) if total else 0
        neu_pct = round(neu / total * 100, 1) if total else 0

        hotel_sent_rows = ''.join(
            f'<tr><td>{h["hotel"]}</td>'
            f'<td>{h["total"]:,}</td>'
            f'<td class="g">{h["pos"]:,}</td>'
            f'<td class="rd">{h["neg"]:,}</td>'
            f'<td>{h["neu"]:,}</td>'
            f'<td><strong>{h["pos_rate"]}%</strong></td></tr>'
            for h in s['hotel_sentiment']
        )

        cat_rows = ""
        max_cat = max(s['category_counts'].values()) if s['category_counts'] else 1
        for cat, cnt in sorted(s['category_counts'].items(), key=lambda x: -x[1]):
            pct = round(cnt / total * 100, 1)
            is_other = cat not in STANDARD_ATTRIBUTES
            cls = ' class="other"' if is_other else ''
            bar_w = round(cnt / max_cat * 100)
            cat_rows += f'<tr{cls}><td>{cat}</td><td>{cnt:,}</td><td>{pct}%</td><td><div class="bar"><div class="fill{" alt" if is_other else ""}" style="width:{bar_w}%"></div></div></td></tr>'

        # Non-standard tag mapping detail
        non_std_detail = ""
        if s.get('non_std_tags'):
            mapping = {
                '安全': ('服務', '酒店安全管理屬於服務範疇'),
                '網絡品質': ('公共設施', 'WiFi/網絡屬於酒店公共設施'),
                '品牌信賴': ('服務', '品牌信任來自服務體驗累積'),
                '酒店形象': ('服務', '酒店形象由整體服務塑造'),
                '推薦': ('其他', '推薦意願為綜合評價，非單一屬性'),
                '家庭需求': ('其他', '家庭需求跨越多個屬性'),
                '其他': ('其他', '無法歸入標準屬性的雜項'),
            }
            rows = ""
            for tag, cnt in sorted(s['non_std_tags'].items(), key=lambda x: -x[1]):
                target, reason = mapping.get(tag, ('其他', ''))
                rows += f'<tr><td class="other-tag">{tag}</td><td>{cnt}</td><td>{target}</td><td style="color:#888">{reason}</td></tr>'
            non_std_detail = f'''
            <div class="section">
                <div class="section-title">非標準標籤整合說明</div>
                <p class="desc">AI 分類第一階段識別出 {len(s["non_std_tags"])} 個非標準標籤，第二階段依據語意關聯整合至標準屬性：</p>
                <table class="tbl"><thead><tr><th>AI 識別標籤</th><th>數量</th><th>歸入屬性</th><th>整合理由</th></tr></thead><tbody>{rows}</tbody></table>
            </div>'''

        # Validation results
        validation_html = ""
        if s.get('validation'):
            v = s['validation']
            fk = v.get('gpt4o_mini_fleiss_kappa', {})
            vm = v.get('validation_metrics', {})
            claude_model = v.get('claude_model', 'Claude Sonnet')
            gt_pairs = v.get('gt_pairs', 0)

            # Spearman results - only GPT-4o and Claude (not mini, since mini IS the ground truth)
            spearman_rows = ""
            compare_models = {
                'gpt4o': 'GPT-4o',
                'claude': 'Claude Sonnet 4',
            }
            for mkey, mlabel in compare_models.items():
                m = vm.get(mkey, {})
                if not m:
                    continue
                rho = m.get('score_spearman_rho', 0)
                rho_cls = 'g' if rho and rho >= 0.7 else ''
                spearman_rows += f'<tr><td>{mlabel}</td><td class="{rho_cls}"><strong>{rho}</strong></td></tr>'

            sample_n = v.get("sample_size", 50)

            validation_html = f'''
            <div class="section">
                <div class="section-title">穩定性與有效性驗證</div>
                <p class="desc">
                    本研究以 GPT-4o-mini 作為主要分析模型，從已分類完成的 {sample_n} 條評論中，
                    將相同評論分別送入 GPT-4o 和 Claude Sonnet 4 重新分析，
                    比較不同模型對同一條評論、同一屬性所給的績效分數是否一致。
                </p>

                <h4 style="margin:18px 0 8px;color:#1a1a2e;text-align:center">穩定性：Fleiss&apos; Kappa</h4>
                <p class="desc" style="text-align:center">
                    GPT-4o-mini 對相同 {sample_n} 條評論跑 {fk.get("num_runs",5)} 次，檢驗分類結果是否可重現。
                </p>
                <div class="cards" style="margin-bottom:20px;justify-content:center">
                    <div class="card cg" style="max-width:240px"><div class="card-n">{fk.get("kappa",0)}</div>
                    <div class="card-l">Fleiss&apos; Kappa<br><small>{fk.get("interpretation","")}</small></div></div>
                </div>

                <h4 style="margin:18px 0 8px;color:#1a1a2e;text-align:center">有效性：Spearman 等級相關係數</h4>
                <p class="desc" style="text-align:center">
                    以 GPT-4o-mini 的分類結果為基準，其他模型對同一條評論的同一屬性給出的績效分數，
                    與基準的等級相關。&rho; &gt; 0.7 表示高度相關，代表不同模型的評分趨勢一致。
                </p>
                <table class="tbl"><thead><tr>
                    <th>比較模型</th><th>Spearman &rho;</th>
                </tr></thead><tbody>{spearman_rows}</tbody></table>
                <p class="desc" style="text-align:center;margin-top:10px">
                    樣本：{sample_n} 條評論 | 比較方式：同一評論 &times; 同一屬性的績效分數配對
                </p>
            </div>'''

        # AI Advisor (per hotel)
        ai_html = ""
        ai_per_hotel = s.get('ai_per_hotel', {})
        if ai_per_hotel:
            ai_tabs = ''.join(
                f'<button class="tab{" active" if i==0 else ""}" onclick="sw(this,&apos;{h}&apos;,&apos;aip&apos;)" data-h="{h}">{h}</button>'
                for i, h in enumerate(ai_per_hotel.keys())
            )
            ai_panels = ""
            for i, (hotel, ai) in enumerate(ai_per_hotel.items()):
                show = " show" if i == 0 else ""
                findings = ''.join(f'<li>{f}</li>' for f in ai.get('key_findings', []))
                recs = ''
                for r in ai.get('improvement_recommendations', [])[:5]:
                    recs += f'<tr><td style="text-align:left;font-weight:600">{r.get("attribute","")}</td><td style="text-align:left">{r.get("short_term","")}</td><td style="text-align:left">{r.get("medium_term","")}</td><td style="text-align:left">{r.get("expected_outcome","")}</td></tr>'
                actions = ''
                for a in ai.get('priority_actions', [])[:5]:
                    actions += f'<tr><td>{a.get("priority","")}</td><td style="text-align:left">{a.get("action","")}</td><td style="text-align:left">{a.get("reason","")}</td><td>{a.get("timeline","")}</td></tr>'
                ai_panels += f'''<div class="panel aip{show}" data-h="{hotel}">
                    <div style="background:#f8f9fb;padding:16px 20px;border-radius:8px;margin-bottom:16px;font-size:.95em;line-height:1.8">{ai.get("executive_summary","")}</div>
                    <h4 style="margin:14px 0 8px;text-align:center;color:#1a1a2e">關鍵發現</h4>
                    <ul style="padding-left:20px;color:#555;line-height:2">{findings}</ul>
                    {"" if not recs else f'<h4 style="margin:18px 0 8px;text-align:center;color:#1a1a2e">改進建議</h4><table class="tbl"><thead><tr><th style="text-align:left">屬性</th><th style="text-align:left">短期</th><th style="text-align:left">中期</th><th style="text-align:left">預期效果</th></tr></thead><tbody>{recs}</tbody></table>'}
                    {"" if not actions else f'<h4 style="margin:18px 0 8px;text-align:center;color:#1a1a2e">優先行動</h4><table class="tbl"><thead><tr><th>優先級</th><th style="text-align:left">行動</th><th style="text-align:left">理由</th><th>時間</th></tr></thead><tbody>{actions}</tbody></table>'}
                </div>'''
            ai_html = f'''
            <div class="section" style="border-left:4px solid #e94560">
                <div class="section-title">AI 顧問分析（各酒店）</div>
                <div class="tabs">{ai_tabs}</div>
                {ai_panels}
            </div>'''

        stats_html = f'''
    <div class="section">
        <div class="section-title">資料總覽</div>
        <div class="cards">
            <div class="card"><div class="card-n">{s["n_reviews"]:,}</div><div class="card-l">評論數</div></div>
            <div class="card"><div class="card-n">{total:,}</div><div class="card-l">提及總數</div></div>
            <div class="card cg"><div class="card-n">{pos:,}</div><div class="card-l">正面 ({pos_pct}%)</div></div>
            <div class="card cr"><div class="card-n">{neg:,}</div><div class="card-l">負面 ({neg_pct}%)</div></div>
            <div class="card"><div class="card-n">{neu:,}</div><div class="card-l">中立 ({neu_pct}%)</div></div>
        </div>
    </div>
    <div class="section">
        <div class="section-title">各酒店情感統計</div>
        <table class="tbl"><thead><tr><th style="text-align:left">酒店</th><th>提及數</th><th style="color:#2ecc71">正面</th><th style="color:#e74c3c">負面</th><th>中立</th><th>正面率</th></tr></thead>
        <tbody>{hotel_sent_rows}</tbody></table>
    </div>
    <div class="section">
        <div class="section-title">分類標籤分布（12 標準屬性 + 其他）</div>
        <table class="tbl cat"><thead><tr><th style="text-align:left">類別</th><th>數量</th><th>佔比</th><th style="width:40%">分布</th></tr></thead><tbody>{cat_rows}</tbody></table>
    </div>
    {non_std_detail}
    {validation_html}
    {ai_html}'''

    # ---- Build HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>酒店評論分析儀表板</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Noto Sans TC','PingFang TC','Microsoft YaHei',-apple-system,sans-serif;
     background:#f4f5f7;color:#333;line-height:1.7;font-size:15px}}
.top{{background:#1a1a2e;color:#fff;padding:28px 32px;text-align:center}}
.top h1{{font-size:1.5em;font-weight:700;letter-spacing:1px}}
.top .sub{{color:rgba(255,255,255,.5);font-size:.82em;margin-top:6px}}
nav{{background:#16213e;padding:0 32px;display:flex;justify-content:center;overflow-x:auto;border-bottom:3px solid #0f3460}}
nav button{{background:0 0;border:0;color:#8899aa;padding:14px 22px;font-size:.88em;cursor:pointer;
           white-space:nowrap;border-bottom:3px solid transparent;margin-bottom:-3px;transition:.2s}}
nav button:hover{{color:#fff;background:rgba(255,255,255,.04)}}
nav button.active{{color:#fff;border-bottom-color:#e94560;font-weight:600}}
.wrap{{max-width:1300px;margin:0 auto;padding:24px}}
.section{{background:#fff;border-radius:10px;padding:28px;margin-bottom:24px;box-shadow:0 1px 6px rgba(0,0,0,.06)}}
.section-title{{font-size:1.1em;font-weight:700;color:#1a1a2e;border-left:4px solid #e94560;padding-left:12px;margin-bottom:18px;text-align:center}}
.desc{{color:#777;font-size:.85em;margin-bottom:14px;line-height:1.6}}
.cards{{display:flex;flex-wrap:wrap;gap:14px;justify-content:center}}
.card{{flex:1;min-width:130px;max-width:200px;background:#f8f9fb;border-radius:10px;padding:20px;text-align:center;border:1px solid #eee}}
.card-n{{font-size:1.9em;font-weight:800;color:#1a1a2e;line-height:1.2;font-variant-numeric:tabular-nums;text-align:center}}
.card-l{{font-size:.82em;color:#999;margin-top:4px;text-align:center}}
.cg .card-n{{color:#27ae60}}.cr .card-n{{color:#e74c3c}}
.tbl{{width:100%;border-collapse:collapse;font-size:.88em}}
.tbl th{{background:#1a1a2e;color:#fff;padding:11px 10px;font-weight:600;text-align:center}}
.tbl td{{padding:10px;border-bottom:1px solid #f0f0f0;text-align:center;font-variant-numeric:tabular-nums}}
.tbl tr:hover{{background:#fafbfc}}
.g{{color:#27ae60;font-weight:600}}.rd{{color:#e74c3c;font-weight:600}}
.cat td:first-child{{font-weight:600;text-align:left}}.other td:first-child,.other-tag{{color:#e94560;font-weight:600}}
.bar{{background:#eee;border-radius:3px;height:14px}}.fill{{background:#1a1a2e;border-radius:3px;height:100%}}.fill.alt{{background:#e94560}}
.tabs{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:20px;justify-content:center}}
.tab{{padding:8px 18px;border:1px solid #ddd;border-radius:6px;background:#fff;color:#555;cursor:pointer;font-size:.85em;transition:.15s}}
.tab.active{{background:#1a1a2e;color:#fff;border-color:#1a1a2e}}
.tab:hover{{border-color:#1a1a2e}}
.panel{{display:none}}.panel.show{{display:block}}
.chart-block{{margin-bottom:28px;border-bottom:1px solid #f0f0f0;padding-bottom:20px}}
.chart-block:last-child{{border-bottom:none}}
.chart-block img{{max-width:100%;height:auto;border-radius:6px}}
.chart-desc{{margin-top:10px;padding:12px 16px;background:#f8f9fb;border-radius:6px;font-size:.85em;color:#666;line-height:1.6}}
#plotly-chart{{width:100%;height:620px}}
#trend-chart{{width:100%;height:500px}}
footer{{text-align:center;color:#aaa;padding:20px;font-size:.78em}}
@media(max-width:768px){{.wrap{{padding:12px}}.cards{{gap:8px}}.card{{min-width:100px;padding:12px}}.card-n{{font-size:1.3em}}}}
</style></head>
<body>
<div class="top"><h1>酒店評論分析儀表板</h1>
<div class="sub">Importance-Performance Analysis | 五家北京酒店 | {now}</div></div>
<nav>
<button class="active" onclick="go('overview')">總覽</button>
<button onclick="go('interactive')">互動式分析</button>
<button onclick="go('trend')">動態趨勢</button>
<button onclick="go('data')">數據圖表</button>
</nav>
<div class="wrap">

<!-- ===== Overview ===== -->
<div id="p-overview" class="page">{stats_html}</div>

<!-- ===== Interactive IPA ===== -->
<div id="p-interactive" class="page" style="display:none">
<div class="section">
<div class="section-title">互動式分析 分析</div>
<p class="desc" style="text-align:center">X 軸為重要度，Y 軸為績效。每個點代表一家酒店的一個屬性。虛線為全體均值分隔線。可篩選屬性、hover 查看數值。</p>
<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;justify-content:center">
<button onclick="toggleA(1)" class="tab">全選</button><button onclick="toggleA(0)" class="tab">清除</button>
<div id="af" style="display:flex;flex-wrap:wrap;gap:6px"></div></div>
<div id="plotly-chart"></div>
</div></div>

<!-- ===== Trend ===== -->
<div id="p-trend" class="page" style="display:none">
<div class="section">
<div class="section-title">動態趨勢分析</div>
<p class="desc" style="text-align:center">選擇一個屬性，查看五家酒店在三個時期（2020 爆發期、2021 恢復期、2022 後疫情）的變化趨勢。</p>
<div style="display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap;justify-content:center">
<label style="font-weight:600">屬性：</label>
<select id="trend-attr" onchange="updTrend()" style="padding:8px 16px;border:1px solid #ddd;border-radius:6px;font-size:.9em;min-width:140px"></select>
<label style="font-weight:600;margin-left:12px">指標：</label>
<select id="trend-metric" onchange="updTrend()" style="padding:8px 16px;border:1px solid #ddd;border-radius:6px;font-size:.9em">
<option value="perf">績效</option><option value="pos_rate">正面率 (%)</option><option value="count">提及次數</option></select>
</div>
<div id="trend-chart"></div>
<div id="trend-tbl" style="margin-top:20px"></div>
</div></div>

<!-- ===== Data Charts ===== -->
<div id="p-data" class="page" style="display:none">
<div class="section">
<div class="section-title">數據圖表</div>
<p class="desc" style="text-align:center">選擇酒店查看各屬性的績效、重要度、正面率分布。</p>
<div class="tabs" id="dtabs">
{"".join(f'<button class="tab{" active" if i==0 else ""}" onclick="swData(&apos;{h}&apos;,this)" data-h="{h}">{h}</button>' for i,h in enumerate(hotel_keys))}
</div>
<div id="data-chart" style="width:100%;height:500px"></div>
</div></div>

</div>
<footer>酒店評論分析儀表板 | {now}</footer>

<script>
function go(id){{document.querySelectorAll('.page').forEach(p=>p.style.display='none');document.getElementById('p-'+id).style.display='block';document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));event.currentTarget.classList.add('active');if(id==='interactive'&&!window._pi){{initIPA();window._pi=1}};if(id==='trend'&&!window._ti){{initTrend();window._ti=1}};if(id==='data'&&!window._di){{initData();window._di=1}}}}
function sw(btn,h,cls){{btn.parentElement.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));btn.classList.add('active');document.querySelectorAll('.'+cls).forEach(p=>p.classList.toggle('show',p.dataset.h===h))}}

// IPA
const D={json.dumps(traces_data,ensure_ascii=False)};
const AT={json.dumps(attrs_list,ensure_ascii=False)};
const H={json.dumps(hotels,ensure_ascii=False)};
const MI={mean_imp:.4f},MP={mean_perf:.4f};
const MK=['circle','square','diamond','cross','triangle-up'];
let aa=new Set(AT);
function initIPA(){{const c=document.getElementById('af');AT.forEach(a=>{{const b=document.createElement('span');b.className='tab active';b.textContent=a;b.dataset.a=a;b.onclick=()=>{{aa.has(a)?aa.delete(a):aa.add(a);b.classList.toggle('active',aa.has(a));updIPA()}};c.appendChild(b)}});updIPA()}}
function toggleA(on){{aa=on?new Set(AT):new Set();document.querySelectorAll('#af .tab').forEach(b=>b.classList.toggle('active',on));updIPA()}}
function updIPA(){{const f=D.filter(d=>aa.has(d.attr));const jit=[0,-.02,.02,-.04,.04];const t=H.map((h,i)=>{{const p=f.filter(d=>d.hotel===h);return{{x:p.map(d=>d.imp+jit[i%5]),y:p.map(d=>d.perf+jit[(i+2)%5]),text:p.map(d=>`<b>${{d.attr}}</b><br>${{d.hotel}}<br>重要度:${{d.imp.toFixed(2)}}<br>績效:${{d.perf.toFixed(2)}}<br>提及:${{d.mentions}}|正面率:${{d.pos_rate.toFixed(1)}}%`),name:h,mode:'markers',marker:{{size:16,symbol:MK[i%5],color:p.map(d=>d.color),opacity:.85,line:{{width:2,color:'#fff'}}}},hovertemplate:'%{{text}}<extra></extra>',type:'scatter'}}}});Plotly.react('plotly-chart',t,{{xaxis:{{title:'重要度',gridcolor:'#f0f0f0',zeroline:false,linecolor:'#ddd',linewidth:1,mirror:true}},yaxis:{{title:'績效',gridcolor:'#f0f0f0',zeroline:false,linecolor:'#ddd',linewidth:1,mirror:true}},shapes:[{{type:'line',x0:MI,x1:MI,y0:0,y1:1,yref:'paper',line:{{color:'#bbb',width:1,dash:'dash'}}}},{{type:'line',y0:MP,y1:MP,x0:0,x1:1,xref:'paper',line:{{color:'#bbb',width:1,dash:'dash'}}}}],legend:{{orientation:'h',y:-.12,x:.5,xanchor:'center',font:{{size:12}}}},margin:{{l:60,r:20,t:20,b:70}},plot_bgcolor:'#fff',paper_bgcolor:'#fff',hovermode:'closest'}},{{responsive:true}})}}

// Trend
const TR={trend_json};
const PR=['2020','2021','2022'];
const PM={{'2020':'爆發期','2021':'恢復期','2022':'後疫情'}};
const TC={json.dumps({h:hotel_colors[i%len(hotel_colors)] for i,h in enumerate(hotels)},ensure_ascii=False)};
function initTrend(){{const s=document.getElementById('trend-attr');if(!s||s.options.length>0)return;[...new Set(TR.map(d=>d.attr))].forEach(a=>{{const o=document.createElement('option');o.value=a;o.text=a;s.appendChild(o)}});updTrend()}}
function updTrend(){{const attr=document.getElementById('trend-attr').value,metric=document.getElementById('trend-metric').value;const ml={{perf:'績效',pos_rate:'正面率(%)',count:'提及次數'}};const f=TR.filter(d=>d.attr===attr);const t=H.map((h,i)=>{{const pts=PR.map(p=>{{const d=f.find(x=>x.hotel===h&&x.period===PM[p]);return d?d[metric]:null}});return{{x:PR,y:pts,name:h,mode:'lines+markers',line:{{color:TC[h],width:3}},marker:{{size:10,symbol:MK[i%5],color:TC[h],line:{{width:1.5,color:'#fff'}}}},connectgaps:false,hovertemplate:`<b>${{h}}</b><br>%{{x}}<br>${{ml[metric]}}:%{{y}}<extra></extra>`,type:'scatter'}}}});Plotly.react('trend-chart',t,{{xaxis:{{title:'時期',type:'category',gridcolor:'#f0f0f0',linecolor:'#ddd',linewidth:1,mirror:true}},yaxis:{{title:ml[metric],gridcolor:'#f0f0f0',linecolor:'#ddd',linewidth:1,mirror:true}},legend:{{orientation:'h',y:-.18,x:.5,xanchor:'center',font:{{size:11}}}},margin:{{l:60,r:20,t:30,b:90}},plot_bgcolor:'#fff',paper_bgcolor:'#fff',hovermode:'x unified'}},{{responsive:true}});
let tb='<table class="tbl"><thead><tr><th style="text-align:left">酒店</th>';PR.forEach(p=>tb+=`<th>${{p}}</th>`);tb+='</tr></thead><tbody>';H.forEach(h=>{{tb+=`<tr><td style="text-align:left">${{h}}</td>`;PR.forEach(p=>{{const d=f.find(x=>x.hotel===h&&x.period===PM[p]);tb+=`<td>${{d?(metric==='count'?d[metric]:d[metric].toFixed(2)):'-'}}</td>`}});tb+='</tr>'}});tb+='</tbody></table>';document.getElementById('trend-tbl').innerHTML=tb}}

// Data charts
const TJ={json.dumps(tables_json,ensure_ascii=False)};
let curHotel='{all_hotels_key}';
function initData(){{updData(curHotel)}}
function swData(h,btn){{curHotel=h;btn.parentElement.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));btn.classList.add('active');updData(h)}}
function updData(h){{const d=TJ[h]||[];const attrs=d.map(r=>r['屬性']);const t=[{{x:d.map(r=>r['平均績效']),y:attrs,name:'績效',type:'bar',orientation:'h',marker:{{color:'#1a1a2e'}}}},{{x:d.map(r=>r['平均重要度']),y:attrs,name:'重要度',type:'bar',orientation:'h',marker:{{color:'#e94560'}}}},{{x:d.map(r=>r['正向率%']/100*5),y:attrs,name:'正面率(scaled)',type:'bar',orientation:'h',marker:{{color:'#2ecc71',opacity:.4}},visible:'legendonly'}}];Plotly.react('data-chart',t,{{barmode:'group',xaxis:{{title:'分數 (1-5)',range:[0,5.5],gridcolor:'#f0f0f0'}},yaxis:{{autorange:'reversed'}},legend:{{orientation:'h',y:1.08,x:.5,xanchor:'center'}},margin:{{l:80,r:20,t:40,b:50}},plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},{{responsive:true}})}}
</script></body></html>"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"   ✓ 統一儀表板: {output_path}")
