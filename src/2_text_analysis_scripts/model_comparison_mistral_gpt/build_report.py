"""Compile the results report. Emits two files:
  report.html           full standalone page (local; clickable relative links) - double-click
  report_artifact.html  body-only, theme-aware, de-linked - for publishing as a Claude Artifact
Both embed the four figures as base64 PNG (self-contained, CSP-safe). Diverging heatmap table.

Run: python src/2_text_analysis_scripts/model_comparison_mistral_gpt/build_report.py
"""
import os, base64, html, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "results_master_table.csv")
FIGS = ["fig1_distance_from_human", "fig2_family_effectsizes", "fig3_model_divergence", "fig4_sentiment_polarity"]
CELL_COLS = [f"{m} | {r} | {c}"
             for m in ["GPT-5.2", "Mistral"] for r in ["Informal", "Formal"]
             for c in ["Baseline", "P1 Human-like", "P2 Detector-aware", "P3 Detector-evasive"]]


def b64(stem):
    p = os.path.join(HERE, stem + ".png")
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""


def cell_bg(d):
    a = min(abs(d) / 1.5, 1.0) * 0.85
    r0, g0, b0 = (0, 114, 178) if d > 0 else (213, 85, 0)
    r = round(255 * (1 - a) + r0 * a); g = round(255 * (1 - a) + g0 * a); bb = round(255 * (1 - a) + b0 * a)
    return f"rgb({r},{g},{bb})", ("#fff" if a > 0.55 else "#1b1f27")


FINDINGS = [
    ("RQ1 · Detectability is register-dependent",
     "LLM Swedish diverges from human far more in <b>formal</b> abstracts (median |dz| 0.29–0.56, small–medium) "
     "than in <b>informal</b> comments (0.16–0.24, negligible–small), across both models. Short casual comments "
     "have little stylometric surface area; structured academic prose exposes the tells.",
     ["fig1_distance_from_human", "fig2_family_effectsizes"]),
    ("RQ2 · Adversarial prompting fails — and backfires",
     "Prompting the model to 'sound human / evade detection' does <b>not</b> reduce its structural distance from human; "
     "10 of 12 model×register×strategy cells are neutral or <b>more</b> detectable than baseline. The "
     "<b>detector-aware</b> strategy is the most counterproductive, via over-compliance (abnormally choppy prose).",
     []),
    ("RQ3 · Model identity matters most",
     "GPT-5.2 and Mistral differ at medium-or-larger effect on 183/532 features (99 large). <b>Mistral is the more "
     "human-like generator in formal Swedish</b> (closer to human on 61% of formal features). Under adversarial prompts "
     "Mistral over-compresses (shorter, more function words/periods) while GPT stays long.",
     ["fig3_model_divergence"]),
    ("RQ4 · Affect catches the informal tell stylometry misses",
     "Top-label sentiment is identical across sources (~77% neutral), but <b>continuous</b> signed polarity reveals a "
     "<b>GPT positivity bias</b> at baseline (dz +0.57 vs human; +0.74 vs Mistral — the largest single affective effect). "
     "Adversarial prompts move affect but <b>overshoot</b> (GPT swings from too-positive to more-negative-than-human); "
     "Mistral shows no positivity bias and tracks human throughout.",
     ["fig4_sentiment_polarity"]),
]
LINK_ITEMS = [("Master results table (CSV)", "results_master_table.csv"),
              ("Structural results (CSV)", "model_comparison_results.csv"),
              ("Affective results (CSV)", "affective_results.csv"),
              ("Comparison script", "model_comparison.py"),
              ("Affective script", "affective_sentiment.py"),
              ("Figures (vector PDF)", "fig1_distance_from_human.pdf")]

STYLE = """
:root{--bg:#f4f6f8;--card:#ffffff;--ink:#1b1f27;--muted:#667085;--line:#e3e7ec;--head:#eef1f5;
--accent:#0072B2;--na:#f2f4f6;--shadow:0 1px 2px rgba(16,24,40,.04)}
@media (prefers-color-scheme:dark){:root{--bg:#101318;--card:#191d24;--ink:#e6eaf0;--muted:#9aa4b2;
--line:#2a313b;--head:#20262f;--accent:#5aa9dd;--na:#20262f;--shadow:none}}
:root[data-theme="dark"]{--bg:#101318;--card:#191d24;--ink:#e6eaf0;--muted:#9aa4b2;--line:#2a313b;
--head:#20262f;--accent:#5aa9dd;--na:#20262f;--shadow:none}
:root[data-theme="light"]{--bg:#f4f6f8;--card:#ffffff;--ink:#1b1f27;--muted:#667085;--line:#e3e7ec;
--head:#eef1f5;--accent:#0072B2;--na:#f2f4f6;--shadow:0 1px 2px rgba(16,24,40,.04)}
*{box-sizing:border-box}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 64px;
font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink)}
.wrap h1{font-size:25px;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
.lead{color:var(--muted);margin:0 0 22px;max-width:70ch}
.wrap h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:34px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
.finding{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
margin:0 0 14px;box-shadow:var(--shadow)}
.finding h3{margin:0 0 6px;font-size:15.5px}.finding p{margin:0;max-width:78ch}
figure{margin:14px 0 0;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--muted);font-size:11.5px;margin-top:5px;font-variant:small-caps;letter-spacing:.03em}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card);box-shadow:var(--shadow)}
table{border-collapse:separate;border-spacing:0;font-size:12px;white-space:nowrap}
th,td{padding:5px 7px;text-align:center;border-bottom:1px solid var(--line)}
thead th{position:sticky;top:0;background:var(--head);color:var(--muted);font-weight:700;z-index:2}
thead th.grp{border-left:1px solid var(--line)}
td.stick,th.stick{position:sticky;left:0;background:var(--card);text-align:left;z-index:1;
max-width:220px;white-space:normal;border-right:1px solid var(--line)}
thead th.stick{z-index:3;background:var(--head)}
td.c{font-variant-numeric:tabular-nums;min-width:48px}
td.c.sig{font-weight:700}td.c.na{background:var(--na);color:var(--muted)}
td.fam{color:var(--muted);text-align:left;font-size:11px}td.n{color:var(--muted)}
tbody tr:hover td:not(.c){background:var(--head)}
.legend{color:var(--muted);font-size:12.5px;margin:12px 0 6px;max-width:100ch}.legend b{color:var(--ink)}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin:0 3px}
.links{color:var(--muted);font-size:13px}.links a{color:var(--accent);text-decoration:none}
.links a:hover{text-decoration:underline}
.foot{color:var(--muted);font-size:12px;margin-top:8px;max-width:90ch}
"""


def build_inner(links_html):
    df = pd.read_csv(TABLE, encoding="utf-8").fillna("")
    imgs = {f: b64(f) for f in FIGS}
    fsec = ""
    for title, body, figs in FINDINGS:
        pics = "".join(f'<figure><img src="{imgs[f]}" alt="{f}"><figcaption>{f}.pdf</figcaption></figure>'
                       for f in figs if imgs.get(f))
        fsec += f'<section class="finding"><h3>{title}</h3><p>{body}</p>{pics}</section>'

    thead = ('<tr><th class="stick">Test</th><th>Family</th>'
             '<th class="grp" colspan="4">GPT · Informal</th><th class="grp" colspan="4">GPT · Formal</th>'
             '<th class="grp" colspan="4">Mistral · Informal</th><th class="grp" colspan="4">Mistral · Formal</th>'
             '<th class="grp">N inf</th><th>N form</th></tr>'
             '<tr><th class="stick"></th><th></th>' +
             "".join('<th class="c grp">B</th><th class="c">P1</th><th class="c">P2</th><th class="c">P3</th>'
                     for _ in range(4)) + '<th class="grp"></th><th></th></tr>')
    rows = ""
    for _, r in df.iterrows():
        tip = html.escape(f"{r['Purpose of Test']}  |  e.g. {r['Example']}  ||  {r['Key Findings']}")
        rows += (f'<tr><td class="stick" title="{tip}"><b>{html.escape(str(r["Test"]))}</b></td>'
                 f'<td class="fam">{html.escape(str(r["Feature_family"]).replace("_"," "))}</td>')
        for i, col in enumerate(CELL_COLS):
            v = str(r[col]); grp = " grp" if i % 4 == 0 else ""
            m = re.match(r"([+-][\d.]+)", v)
            if m:
                bg, ink = cell_bg(float(m.group(1)))
                sig = " sig" if v.endswith("*") else ""
                rows += f'<td class="c{grp}{sig}" style="background:{bg};color:{ink}">{html.escape(v)}</td>'
            else:
                rows += f'<td class="c na{grp}">—</td>'
        rows += f'<td class="n grp">{r["N_informal"]}</td><td class="n">{r["N_formal"]}</td></tr>'

    return f"""<div class="wrap">
<h1>LLM-Generated Swedish — Results Report</h1>
<p class="lead">Human vs GPT-5.2 vs Mistral (mistral-small-2506), two registers × four prompt conditions.
Paired Wilcoxon + Cohen's dz + BH-FDR. Effect-size-first — at this n, p-values alone are uninformative.</p>
<h2>Key findings</h2>{fsec}
<h2>Full results — Cohen's dz vs human, by permutation</h2>
<div class="legend">Cell = dz vs human · <span class="swatch" style="background:#0072B2"></span><b>+ model &gt; human</b> ·
<span class="swatch" style="background:#D55E00"></span><b>− model &lt; human</b> · shade ∝ |dz| · <b>bold</b> = FDR&lt;0.05 ·
— = not measured in that register. B=Baseline, P1=Human-like, P2=Detector-aware, P3=Detector-evasive.
Hover a feature for its purpose, example, and finding.</div>
<div class="tablewrap"><table><thead>{thead}</thead><tbody>{rows}</tbody></table></div>
<h2>Artifacts</h2>{links_html}
<p class="foot">Figures embedded as PNG (vector PDFs alongside). GPT-SW3 (Swedish-native) is future work
(gated access). Emotion (zero-shot) deferred; sentiment via KBLab robust-swedish-sentiment-multiclass.</p>
</div>"""


def main():
    clickable = '<p class="links">' + " · ".join(
        f'<a href="{h}">{html.escape(t)}</a>' for t, h in LINK_ITEMS) + "</p>"
    plain = '<p class="links">In the repo folder <code>model_comparison_mistral_gpt/</code>: ' + \
            ", ".join(html.escape(h) for _, h in LINK_ITEMS) + ".</p>"

    full = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>CLUU — LLM-Generated Swedish: Results</title><style>{STYLE}</style></head>'
            f'<body style="margin:0;background:var(--bg)">{build_inner(clickable)}</body></html>')
    with open(os.path.join(HERE, "report.html"), "w", encoding="utf-8") as f:
        f.write(full)

    artifact = f"<style>{STYLE}\nbody{{margin:0;background:var(--bg)}}</style>{build_inner(plain)}"
    with open(os.path.join(HERE, "report_artifact.html"), "w", encoding="utf-8") as f:
        f.write(artifact)

    print("wrote report.html (local) + report_artifact.html (publishable, body-only, theme-aware)")


if __name__ == "__main__":
    main()
