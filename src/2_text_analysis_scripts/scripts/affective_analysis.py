"""
affective_analysis.py  (INFORMAL register only)

Sentiment / affective analysis of the informal corpus on the CURRENT
post-exclusion, emoji-free corpus, so the numbers are consistent with the
fresh embedding + feature results.

Formal abstracts are near-uniformly neutral (register floor effect) and are
excluded by design; this runs on the informal register only.

Model: KBLab/robust-swedish-sentiment-multiclass (POSITIVE / NEUTRAL / NEGATIVE).
Per text we record:
    sentiment_label            top label (string; skipped by significance_tests)
    p_pos, p_neg, p_neu        class probabilities
    signed_polarity            p_pos - p_neg   (in [-1, 1])
    affective_extremity        1 - p_neu       (0 = neutral, 1 = fully polar)
    is_subjective              1 if top label != NEUTRAL   (binary)

Reads the canonical corpus (same file embedding_comparison.py / the feature
scripts use). Scores the human column once and each LLM condition column; caches
every (column, doc) score so the run is resumable.

Outputs to csv_files/:
    affective_analysis_informal_<condition>.csv   paired human_/llm_ per document
                                                  -> picked up by significance_tests.py
    inf_sentiment_distribution.csv                Human vs LLM (baseline) label mix
                                                  (drop-in replacement, same schema)
    inf_sentiment_distribution_by_condition.csv   label mix + polarity per condition
    affective_analysis_informal_cache.csv         resumable per-(column,doc) score cache

Run:
    python src/2_text_analysis_scripts/scripts/affective_analysis.py
    python src/2_text_analysis_scripts/scripts/affective_analysis.py --batch 32
"""
import argparse
import os

# model is cached locally -> stay offline so a missing network can't hang the run
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd

from data_utils import read_csv_robust, resolve_conditions, condition_tag, INFORMAL_CONDITIONS

HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(HERE)
_DATA = os.path.join(os.path.dirname(_2TAS), "1_data_collection")
CSV_DIR = os.path.join(_2TAS, "csv_files")

CORPUS = os.path.join(_DATA, "llm_comments", "consolidated_informal_comments_adversarial.csv")
CACHE = os.path.join(CSV_DIR, "affective_analysis_informal_cache.csv")

MODEL = "KBLab/robust-swedish-sentiment-multiclass"
HUMAN_COL = "human_comment"
MAX_LEN = 512
LABELS = ("POSITIVE", "NEUTRAL", "NEGATIVE")
# numeric per-text features written with human_/llm_ prefixes
NUM_FEATS = ("p_pos", "p_neg", "p_neu", "signed_polarity", "affective_extremity", "is_subjective")


def _ok(x):
    return not (x is None or (isinstance(x, float) and pd.isna(x))) and bool(str(x).strip())


def score_and_cache(pipe, df, col, batch):
    """Score every non-empty cell of `col` and APPEND each batch to the cache
    file immediately, so an interrupted run keeps its progress. Returns count."""
    todo = [(i, str(df.at[i, col])) for i in df.index if _ok(df.at[i, col])]
    n = 0
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        preds = pipe([t[1][:2000] for t in chunk])
        rows = []
        for (doc_id, _), sc in zip(chunk, preds):
            d = {x["label"].upper(): float(x["score"]) for x in sc}
            p_pos, p_neg, p_neu = d.get("POSITIVE", 0.0), d.get("NEGATIVE", 0.0), d.get("NEUTRAL", 0.0)
            top = max(sc, key=lambda x: x["score"])["label"].upper()
            rows.append(dict(
                column=col, doc_id=doc_id, sentiment_label=top,
                p_pos=round(p_pos, 6), p_neg=round(p_neg, 6), p_neu=round(p_neu, 6),
                signed_polarity=round(p_pos - p_neg, 6),
                affective_extremity=round(1.0 - p_neu, 6),
                is_subjective=int(top != "NEUTRAL"),
            ))
        header = not os.path.exists(CACHE)
        pd.DataFrame(rows).to_csv(CACHE, mode="a", header=header, index=False, encoding="utf-8")
        n += len(rows)
        print(f"    {col}: {min(s + batch, len(todo))}/{len(todo)}", end="\r", flush=True)
    print()
    return n


def build_cache(df, batch):
    """Score human + every LLM condition column, resumably (per-batch cache
    appends survive interruption). Returns a de-duplicated score DataFrame."""
    cols = [HUMAN_COL] + list(dict.fromkeys(INFORMAL_CONDITIONS.values()))
    done = set()
    if os.path.exists(CACHE):
        done = set(zip(*[pd.read_csv(CACHE)[c] for c in ("column", "doc_id")]))

    from transformers import pipeline
    pipe = None
    for col in cols:
        if col not in df.columns:
            print(f"  [skip] column '{col}' not in corpus")
            continue
        need = [i for i in df.index if _ok(df.at[i, col]) and (col, i) not in done]
        if not need:
            print(f"  [cache] {col}: all scored")
            continue
        if pipe is None:
            print(f"Loading {MODEL} ...", flush=True)
            pipe = pipeline("text-classification", model=MODEL, top_k=None,
                            truncation=True, max_length=MAX_LEN, device=-1)
        print(f"scoring {col} ({len(need)} texts)...", flush=True)
        score_and_cache(pipe, df.loc[need], col, batch)

    return pd.read_csv(CACHE, encoding="utf-8").drop_duplicates(["column", "doc_id"], keep="last")


def label_props(sub):
    """Counts + proportions for POSITIVE/NEUTRAL/NEGATIVE over a score frame."""
    vc = sub["sentiment_label"].value_counts()
    n = len(sub)
    out = {}
    for lb in LABELS:
        c = int(vc.get(lb, 0))
        out[f"{lb}_count"] = c
        out[f"{lb}_proportion"] = round(c / n, 6) if n else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="Affective/sentiment analysis (informal only)")
    ap.add_argument("--dataset", choices=["informal"], default="informal")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)
    df = read_csv_robust(CORPUS)
    print(f"corpus: {len(df)} rows ({CORPUS})")

    cache = build_cache(df, args.batch)
    by_col = {col: g.set_index("doc_id") for col, g in cache.groupby("column")}
    human = by_col.get(HUMAN_COL)
    if human is None or not len(human):
        print("no human scores — aborting")
        return

    # ---- per-document paired files (feed significance_tests.py) ----
    for cond, llm_col in resolve_conditions("informal", "all"):
        if llm_col not in by_col:
            print(f"  [skip] no scores for condition '{cond}' (column {llm_col})")
            continue
        llm = by_col[llm_col]
        ids = human.index.intersection(llm.index)
        rows = []
        for i in ids:
            r = {f"human_{f}": human.at[i, f] for f in ("sentiment_label",) + NUM_FEATS}
            r.update({f"llm_{f}": llm.at[i, f] for f in ("sentiment_label",) + NUM_FEATS})
            rows.append(r)
        tag = condition_tag("informal", cond)
        pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, f"affective_analysis_{tag}.csv"),
                                  index=False, encoding="utf-8")
        print(f"[{cond}] wrote affective_analysis_{tag}.csv ({len(rows)} pairs)")

    # ---- drop-in Human vs LLM (baseline) distribution ----
    base = by_col[INFORMAL_CONDITIONS["baseline"]]
    drop_in = pd.DataFrame([
        {"group": "Informal Human", **label_props(human)},
        {"group": "Informal LLM", **label_props(base)},
    ])
    # column order matches the previous file
    order = ["group"] + [f"{lb}_{k}" for lb in ("POSITIVE", "NEUTRAL", "NEGATIVE")
                         for k in ("count", "proportion")]
    drop_in[order].to_csv(os.path.join(CSV_DIR, "inf_sentiment_distribution.csv"),
                          index=False, encoding="utf-8")
    print("wrote inf_sentiment_distribution.csv")

    # ---- richer per-condition distribution (+ polarity / extremity) ----
    rows = []
    def summarise(name, sub):
        rows.append({"group": name, "n": len(sub), **label_props(sub),
                     "median_signed_polarity": round(sub["signed_polarity"].median(), 4),
                     "mean_affective_extremity": round(sub["affective_extremity"].mean(), 4),
                     "subjective_rate": round(sub["is_subjective"].mean(), 4)})
    summarise("Human", human)
    for cond, llm_col in resolve_conditions("informal", "all"):
        if llm_col in by_col:
            summarise(f"LLM {cond}", by_col[llm_col])
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "inf_sentiment_distribution_by_condition.csv"),
                              index=False, encoding="utf-8")
    print("wrote inf_sentiment_distribution_by_condition.csv\n")

    print("=== informal sentiment label mix (%) + polarity ===")
    for r in rows:
        print(f"  {r['group']:22s} n={r['n']:<5d} "
              f"POS {r['POSITIVE_proportion']*100:5.1f}  "
              f"NEU {r['NEUTRAL_proportion']*100:5.1f}  "
              f"NEG {r['NEGATIVE_proportion']*100:5.1f}  |  "
              f"med signed {r['median_signed_polarity']:+.3f}  "
              f"subj {r['subjective_rate']*100:4.1f}%")


if __name__ == "__main__":
    main()
