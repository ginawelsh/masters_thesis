"""
Stylometric surface features:

  - N-gram repetition rate (bigram, trigram): fraction of n-grams that are repeated
    within the same document (1 = all repeated, 0 = all unique).
  - Punctuation patterns: per-100-token rates for . ! ? , ; : — and ...
  - Function-word rate: proportion of alpha tokens that are Swedish function words.
  - Vocabulary richness:
      TTR   — type-token ratio (unique lemmas / total tokens)
      MATTR — moving-average TTR over a 100-token window (McCarthy & Jarvis 2010)
      MTLD  — measure of textual lexical diversity (threshold = 0.72)

Usage:
  python src/2_text_analysis_scripts/stylometric_surface.py --dataset formal
  python src/2_text_analysis_scripts/stylometric_surface.py --dataset informal
"""
import argparse
import os

import numpy as np
import pandas as pd
import spacy

from data_utils import read_csv_robust

MODEL = "sv_core_news_sm"
MATTR_WINDOW = 100
MTLD_THRESHOLD = 0.72

_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")

# Swedish function words (lowercase surface forms)
SWEDISH_FUNCTION_WORDS = {
    # coordinating conjunctions
    "och", "eller", "men", "för", "så", "utan", "samt",
    # subordinating conjunctions / complementizers
    "att", "som", "när", "om", "än", "fast", "fastän", "även", "trots",
    "varken", "både", "antingen", "efter", "innan", "medan", "eftersom",
    "ifall", "huruvida", "tills",
    # prepositions
    "i", "på", "av", "med", "till", "om", "från", "under", "vid", "mot",
    "utom", "genom", "hos", "per", "sedan", "över", "inför", "bakom",
    "framför", "bredvid", "längs", "kring", "efter", "ut", "in", "upp",
    # determiners / articles
    "den", "det", "de", "en", "ett", "sin", "sitt", "sina",
    "min", "mitt", "mina", "din", "ditt", "dina", "hans", "hennes",
    "deras", "vår", "vårt", "våra", "er", "ert", "era",
    "varje", "alla", "varandra", "några", "något", "någon",
    "ingen", "inget", "inga", "denna", "detta", "dessa",
    # personal pronouns
    "jag", "du", "han", "hon", "vi", "ni", "de",
    "mig", "dig", "sig", "oss", "dem", "man",
    # interrogative / relative
    "vem", "vad", "vilket", "vilken", "vilka", "var", "hur",
    # auxiliaries
    "är", "var", "vara", "har", "hade", "ha", "ska", "skall", "skulle",
    "kan", "kunde", "vill", "ville", "bör", "måste", "får", "fick",
    "bli", "blir", "blev", "blivit", "varit",
    # negative / discourse particles
    "inte", "ej", "icke", "ju", "väl", "nog", "visst", "ja", "nej",
    "också", "redan", "bara", "ändå", "dock", "alltså", "därför",
    "nu", "då", "här", "där", "dit", "hit",
}


def parse_args():
    p = argparse.ArgumentParser(description="Stylometric surface features")
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


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def ngram_repetition_rate(tokens, n):
    """Fraction of n-grams that occur more than once in the document."""
    if len(tokens) < n:
        return 0.0
    from collections import Counter
    ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(ngrams)


def punctuation_rates(raw_text, n_tokens):
    """Per-100-token rates for common punctuation marks."""
    scale = 100.0 / n_tokens if n_tokens > 0 else 0.0
    return {
        "punct_period":    raw_text.count(".") * scale,
        "punct_exclaim":   raw_text.count("!") * scale,
        "punct_question":  raw_text.count("?") * scale,
        "punct_comma":     raw_text.count(",") * scale,
        "punct_semicolon": raw_text.count(";") * scale,
        "punct_colon":     raw_text.count(":") * scale,
        "punct_dash":      (raw_text.count("—") + raw_text.count("–")) * scale,
        "punct_ellipsis":  raw_text.count("...") * scale,
    }


def function_word_rate(alpha_tokens):
    if not alpha_tokens:
        return 0.0
    fw = sum(1 for w in alpha_tokens if w.lower() in SWEDISH_FUNCTION_WORDS)
    return fw / len(alpha_tokens)


def ttr(alpha_tokens):
    if not alpha_tokens:
        return 0.0
    return len({w.lower() for w in alpha_tokens}) / len(alpha_tokens)


def mattr(alpha_tokens, window=MATTR_WINDOW):
    n = len(alpha_tokens)
    if n < window:
        return ttr(alpha_tokens)
    ttrs = [
        ttr(alpha_tokens[i: i + window])
        for i in range(n - window + 1)
    ]
    return float(np.mean(ttrs))


def mtld(alpha_tokens, threshold=MTLD_THRESHOLD):
    """Bidirectional MTLD (McCarthy & Jarvis 2010)."""

    def _forward(tokens):
        if not tokens:
            return float("nan")
        factors, seen, start = 0, set(), 0
        for i, w in enumerate(tokens):
            seen.add(w.lower())
            if len(seen) / (i - start + 1) <= threshold:
                factors += 1
                seen, start = set(), i + 1
        remaining = len(tokens) - start
        if remaining > 0:
            partial_ttr = len(seen) / remaining
            if partial_ttr < 1.0:
                factors += (1.0 - partial_ttr) / (1.0 - threshold)
        return len(tokens) / factors if factors > 0 else float(len(tokens))

    if len(alpha_tokens) < 10:
        return float("nan")
    fwd = _forward(alpha_tokens)
    bwd = _forward(list(reversed(alpha_tokens)))
    return (fwd + bwd) / 2.0


# ---------------------------------------------------------------------------
# Per-document entry point
# ---------------------------------------------------------------------------

def compute_metrics(nlp, text):
    if pd.isna(text) or not str(text).strip():
        return None
    raw = str(text).strip()
    doc = nlp(raw)

    alpha_tokens = [t.text for t in doc if t.is_alpha]
    n_tokens = len([t for t in doc if not t.is_space])

    result = {
        "bigram_repetition_rate":  ngram_repetition_rate(alpha_tokens, n=2),
        "trigram_repetition_rate": ngram_repetition_rate(alpha_tokens, n=3),
        "function_word_rate":      function_word_rate(alpha_tokens),
        "ttr":                     ttr(alpha_tokens),
        "mattr":                   mattr(alpha_tokens),
        "mtld":                    mtld(alpha_tokens),
    }
    result.update(punctuation_rates(raw, n_tokens))
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
    out_path = os.path.join(OUT_DIR, f"stylometric_surface_{args.dataset}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} rows → {out_path}")

    print("\nMeans (human | llm):")
    feature_keys = [k for k in rows[0] if k.startswith("human_")]
    for hcol in sorted(feature_keys):
        lcol = "llm_" + hcol[len("human_"):]
        h_mean = out_df[hcol].mean()
        l_mean = out_df[lcol].mean()
        print(f"  {hcol[6:]:35s}  {h_mean:.4f}  |  {l_mean:.4f}")


if __name__ == "__main__":
    main()
