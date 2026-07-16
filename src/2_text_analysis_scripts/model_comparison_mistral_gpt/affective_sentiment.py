"""Sentiment-only affective analysis of the INFORMAL corpus, three sources
(human, GPT-5.2, Mistral@1.0), for the human vs generator comparison.

Adapted from src/4_archive/stale_scripts/affective_analysis.py:
  - sentiment only (KBLab/robust-swedish-sentiment-multiclass); no zero-shot emotion.
  - fixed input paths (JUN26 moved to archive; Mistral from the corpus).
  - adds Mistral as a third source, batched + resumable.
Formal register is excluded by design (near-uniformly neutral).

Per text we record: sentiment_label, p_pos/p_neg/p_neu, signed_polarity (p_pos-p_neg),
affective_extremity (1-p_neu), is_subjective (top label != NEUTRAL).

Outputs (this folder):
  affective_sentiment_corpus.csv  per (source,doc_id,condition) scores  [resumable cache]
  affective_results.csv           paired Wilcoxon + dz + rank-biserial + BH-FDR
  affective_label_dist.csv        sentiment-label proportions per source x condition

Run: python src/2_text_analysis_scripts/model_comparison_mistral_gpt/affective_sentiment.py
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SCRIPTS = os.path.join(ROOT, "src", "2_text_analysis_scripts", "scripts")
sys.path.insert(0, SCRIPTS)
import significance_tests as sig

INFORMAL_ADV = os.path.join(ROOT, "src", "4_archive", "consolidated_informal_comments_adversarial.csv")
MISTRAL = os.path.join(ROOT, "src", "2_text_analysis_scripts", "mistral_temperature_ab", "generated_corpus_mistral_temps.csv")
CORPUS_OUT = os.path.join(HERE, "affective_sentiment_corpus.csv")
RESULTS_OUT = os.path.join(HERE, "affective_results.csv")
DIST_OUT = os.path.join(HERE, "affective_label_dist.csv")

MODEL = "KBLab/robust-swedish-sentiment-multiclass"
CONDITIONS = ["baseline", "human_like", "detector_aware", "detector_evasive"]
GPT_COL = {"baseline": "generated_comment", "human_like": "comment_human_like",
           "detector_aware": "comment_detector_aware", "detector_evasive": "comment_detector_evasive"}
NUM_FEATS = ["signed_polarity", "affective_extremity", "is_subjective"]
BATCH = 64


def _ok(x):
    return not (x is None or (isinstance(x, float) and pd.isna(x))) and bool(str(x).strip())


def build_tasks():
    """Return list of (source, doc_id, condition, text). Human once (condition '-')."""
    df = pd.read_csv(INFORMAL_ADV, encoding="utf-8")
    mist = pd.read_csv(MISTRAL, encoding="utf-8")
    mist = mist[(mist.register == "informal") & (mist.temp == 1.0)]
    mmap = {(int(r.doc_id), r.condition): r.text for _, r in mist.iterrows()}
    tasks = []
    for i in df.index:
        h = df.at[i, "human_comment"]
        if _ok(h):
            tasks.append(("human", int(i), "-", str(h)))
        for cond in CONDITIONS:
            g = df.at[i, GPT_COL[cond]]
            if _ok(g):
                tasks.append(("gpt", int(i), cond, str(g)))
            m = mmap.get((int(i), cond))
            if _ok(m):
                tasks.append(("mistral", int(i), cond, str(m)))
    return tasks


def score_all(tasks):
    done = set()
    if os.path.exists(CORPUS_OUT):
        d = pd.read_csv(CORPUS_OUT, encoding="utf-8")
        done = {(r.source, int(r.doc_id), str(r.condition)) for _, r in d.iterrows()}
    todo = [t for t in tasks if (t[0], t[1], t[2]) not in done]
    print(f"{len(todo)} texts to score ({len(done)} cached)", flush=True)
    if not todo:
        return
    from transformers import pipeline
    pipe = pipeline("text-classification", model=MODEL, top_k=None,
                    truncation=True, max_length=512, device=-1)
    header = not os.path.exists(CORPUS_OUT)
    n = 0
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        outs = pipe([t[3][:2000] for t in chunk])
        rows = []
        for (source, did, cond, _), sc in zip(chunk, outs):
            d = {x["label"].upper(): float(x["score"]) for x in sc}
            p_pos, p_neg, p_neu = d.get("POSITIVE", 0.0), d.get("NEGATIVE", 0.0), d.get("NEUTRAL", 0.0)
            top = max(sc, key=lambda x: x["score"])["label"].upper()
            rows.append({"source": source, "doc_id": did, "condition": cond,
                         "sentiment_label": top, "p_pos": round(p_pos, 6), "p_neg": round(p_neg, 6),
                         "p_neu": round(p_neu, 6), "signed_polarity": round(p_pos - p_neg, 6),
                         "affective_extremity": round(1.0 - p_neu, 6),
                         "is_subjective": int(top != "NEUTRAL")})
        pd.DataFrame(rows).to_csv(CORPUS_OUT, mode="a", header=header, index=False, encoding="utf-8")
        header = False
        n += len(rows)
        if n % 512 == 0 or s + BATCH >= len(todo):
            print(f"  scored {n}/{len(todo)}", flush=True)


def analyse():
    d = pd.read_csv(CORPUS_OUT, encoding="utf-8")
    human = d[d.source == "human"].set_index("doc_id")
    results, pv_by_comp = [], {}
    for comp, (a_src, b_src) in {"human_vs_gpt": ("human", "gpt"),
                                 "human_vs_mistral": ("human", "mistral"),
                                 "mistral_vs_gpt": ("mistral", "gpt")}.items():
        rows = []
        for cond in CONDITIONS:
            A = human if a_src == "human" else d[(d.source == a_src) & (d.condition == cond)].set_index("doc_id")
            B = d[(d.source == b_src) & (d.condition == cond)].set_index("doc_id")
            ids = A.index.intersection(B.index)
            for feat in NUM_FEATS:
                res = sig.test_feature(A.loc[ids, feat].reset_index(drop=True),
                                       B.loc[ids, feat].reset_index(drop=True))
                if res is None:
                    continue
                rows.append({"comparison": comp, "register": "informal", "condition": cond,
                             "feature": feat, "n_pairs": res["n"],
                             "median_A": res["median_human"], "median_B": res["median_llm"],
                             "cohens_dz": res["dz"], "rank_biserial": res["rbc"], "p_value": res["p"]})
        qs = sig.bh_fdr([r["p_value"] for r in rows]) if rows else []
        for r, q in zip(rows, qs):
            r["p_fdr_bh"] = q
            r["significant_fdr_0.05"] = bool(q < 0.05)
        results.extend(rows)
    pd.DataFrame(results).to_csv(RESULTS_OUT, index=False, encoding="utf-8")
    print(f"\nwrote {len(results)} paired tests -> {RESULTS_OUT}")

    # sentiment-label distribution per source (human overall; gpt/mistral per condition + overall)
    dist = []
    def props(sub, src, cond):
        vc = sub["sentiment_label"].value_counts(normalize=True)
        dist.append({"source": src, "condition": cond, "n": len(sub),
                     "POSITIVE": round(vc.get("POSITIVE", 0), 4), "NEUTRAL": round(vc.get("NEUTRAL", 0), 4),
                     "NEGATIVE": round(vc.get("NEGATIVE", 0), 4),
                     "median_signed_polarity": round(sub["signed_polarity"].median(), 4),
                     "mean_extremity": round(sub["affective_extremity"].mean(), 4)})
    props(human.reset_index(), "human", "-")
    for src in ["gpt", "mistral"]:
        props(d[d.source == src], src, "ALL")
        for cond in CONDITIONS:
            props(d[(d.source == src) & (d.condition == cond)], src, cond)
    pd.DataFrame(dist).to_csv(DIST_OUT, index=False, encoding="utf-8")
    print(f"wrote label distribution -> {DIST_OUT}\n")
    print("=== sentiment-label mix (%) + polarity by source ===")
    for r in dist:
        if r["condition"] in ("-", "ALL"):
            print(f"  {r['source']:8s} {r['condition']:4s}  POS {r['POSITIVE']*100:4.0f}  NEU {r['NEUTRAL']*100:4.0f}  "
                  f"NEG {r['NEGATIVE']*100:4.0f}  | med signed {r['median_signed_polarity']:+.3f}  extremity {r['mean_extremity']:.3f}")


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"tasks: {len(tasks)} (human {sum(t[0]=='human' for t in tasks)}, "
          f"gpt {sum(t[0]=='gpt' for t in tasks)}, mistral {sum(t[0]=='mistral' for t in tasks)})", flush=True)
    score_all(tasks)
    analyse()
