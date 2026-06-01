"""
Pragmatic and discourse marker analysis for Swedish text.

Four marker categories, each reported as a rate per 100 alpha tokens:

  hedges        — softeners expressing epistemic uncertainty or vagueness
                  (kanske, möjligen, typ, liksom, ungefär, …)
  boosters      — intensifiers amplifying force
                  (väldigt, verkligen, helt, jätte-, enormt, …)
  negation      — negation tokens (inte, ej, icke, aldrig, ingen, …)
  epistemic     — verbs/adverbs marking speaker's knowledge state
                  (verkar, tycks, troligen, förmodligen, tror, menar, …)
  discourse_ptcl— modal particles (ju, väl, nog, faktiskt, egentligen, …)

Also computes negation_sentence_rate = fraction of sentences containing ≥1 negation.

Matching is case-insensitive whole-word (spaCy alpha tokens only).

Usage:
  python src/2_text_analysis_scripts/pragmatic_markers.py --dataset formal
  python src/2_text_analysis_scripts/pragmatic_markers.py --dataset informal
"""
import argparse
import os

import numpy as np
import pandas as pd
import spacy

from data_utils import read_csv_robust

MODEL = "sv_core_news_sm"
_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")

# ---------------------------------------------------------------------------
# Swedish pragmatic marker lexicons
# ---------------------------------------------------------------------------

HEDGE_MARKERS = {
    "kanske", "möjligen", "eventuellt", "möjligtvis", "kanhända",
    "typ", "liksom", "ungefär", "lite", "ganska", "rätt", "nästan",
    "i viss mån", "på sätt och vis", "ibland", "ibland",
}

BOOSTER_MARKERS = {
    "väldigt", "verkligen", "helt", "javligt", "jättebra",
    "fullständigt", "absolut", "definitivt", "enormt", "otroligt",
    "fantastiskt", "extremt", "verklig", "genuint", "uppenbarligen",
    "riktigt", "super", "mega", "jätte", "alldeles", "precis",
    "exakt", "oerhört", "obetydligt", "avsevärt",
}

NEGATION_MARKERS = {
    "inte", "ej", "icke", "aldrig", "ingenting", "ingenstans",
    "ingen", "inget", "inga", "varken", "knappast", "sällan",
    "näppeligen", "omöjligen",
}

EPISTEMIC_MARKERS = {
    "verkar", "tycks", "troligen", "förmodligen", "antagligen",
    "tydligen", "troligtvis", "förmodar", "misstänker", "anser",
    "tror", "menar", "tycker", "uppfattar", "upplever",
    "verka", "tycka", "tro", "mena", "anse", "uppfatta",
    "förefaller", "påstår", "hävdar", "framhåller",
}

DISCOURSE_PARTICLES = {
    "ju", "väl", "nog", "visst", "faktiskt", "egentligen",
    "nämligen", "alltså", "dock", "ändå", "trots", "tillochmed",
    "dessutom", "däremot", "emellertid", "likväl", "visserligen",
}

# Map category name → lexicon set
CATEGORIES = {
    "hedge":              HEDGE_MARKERS,
    "booster":            BOOSTER_MARKERS,
    "negation":           NEGATION_MARKERS,
    "epistemic":          EPISTEMIC_MARKERS,
    "discourse_particle": DISCOURSE_PARTICLES,
}


def parse_args():
    p = argparse.ArgumentParser(description="Pragmatic marker rates")
    p.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    return p.parse_args()


def resolve_config(dataset):
    if dataset == "informal":
        return {
            "csv": os.path.join(_data, "llm_comments", "LLM_consolidated_reddit_comments_MAY26.csv"),
            "human_col": "human_comment",
            "llm_col": "generated_comment",
        }
    return {
        "csv": os.path.join(_data, "llm_abstracts", "abstracts", "sv_ai_generated_abstracts.csv"),
        "human_col": "Human_Abstract",
        "llm_col": "AI_Abstract",
    }


def load_nlp():
    try:
        return spacy.load(MODEL)
    except OSError:
        raise OSError(f"Run: python -m spacy download {MODEL}")


def compute_metrics(nlp, text):
    if pd.isna(text) or not str(text).strip():
        return None
    doc = nlp(str(text).strip())
    alpha_tokens = [t.text.lower() for t in doc if t.is_alpha]
    n = len(alpha_tokens)
    sents = list(doc.sents)
    n_sents = len(sents)

    result = {}
    for cat, lexicon in CATEGORIES.items():
        count = sum(1 for w in alpha_tokens if w in lexicon)
        result[f"{cat}_rate"] = count * 100.0 / n if n > 0 else 0.0
        result[f"{cat}_count"] = count

    # Sentence-level negation rate
    if n_sents > 0:
        neg_sents = sum(
            1 for s in sents
            if any(t.text.lower() in NEGATION_MARKERS for t in s if t.is_alpha)
        )
        result["negation_sentence_rate"] = neg_sents / n_sents
    else:
        result["negation_sentence_rate"] = 0.0

    result["n_alpha_tokens"] = n
    result["n_sentences"] = n_sents
    return result


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    nlp = load_nlp()
    df = read_csv_robust(cfg["csv"])
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    n = len(df)
    for i, row in df.iterrows():
        print(f"  {i + 1}/{n}", end="\r", flush=True)
        h = compute_metrics(nlp, row.get(cfg["human_col"]))
        l = compute_metrics(nlp, row.get(cfg["llm_col"]))
        if h is None or l is None:
            continue
        out_row = {f"human_{k}": v for k, v in h.items()}
        out_row.update({f"llm_{k}": v for k, v in l.items()})
        rows.append(out_row)

    print()
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"pragmatic_markers_{args.dataset}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} rows → {out_path}")

    rate_cols = [c for c in out_df.columns if c.startswith("human_") and "rate" in c]
    print("\nMean rates per 100 tokens (human | llm):")
    for hcol in sorted(rate_cols):
        lcol = "llm_" + hcol[len("human_"):]
        if lcol in out_df:
            print(f"  {hcol[6:]:35s}  {out_df[hcol].mean():.4f}  |  {out_df[lcol].mean():.4f}")


if __name__ == "__main__":
    main()
