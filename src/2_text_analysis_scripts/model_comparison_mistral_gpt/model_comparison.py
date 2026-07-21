"""Model comparison on the paired feature pipeline. Generator-agnostic: compares any
available subset of {human, gpt (GPT-5.2), mistral (mistral-small-2506@1.0),
sw3 (gpt-sw3-*-instruct)} pairwise, per register x condition.

All generators answered the SAME prompts (same rows), so every comparison is paired by
row. Sources:
  - human + gpt   : the existing adversarial CSVs
  - mistral       : ../mistral_temperature_ab/generated_corpus_mistral_temps.csv (temp==1.0)
  - sw3 (optional): generated_corpus_gpt-sw3.csv in THIS folder (from the Colab run);
                    skipped automatically if the file is absent.
No new generation here. Features = the real pipeline extractors (raw text, no question
prepend). Stats per (register, condition, comparison): paired Wilcoxon + Cohen's dz +
matched-pairs rank-biserial + BH-FDR.

Run: python src/2_text_analysis_scripts/model_comparison_mistral_gpt/model_comparison.py
"""
import os, re, sys, argparse
from collections import Counter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SCRIPTS = os.path.join(ROOT, "src", "2_text_analysis_scripts", "scripts")
sys.path.insert(0, SCRIPTS)

import spacy
import textstat
import stylometric_surface as ss
import syntactic_complexity as sc
import pragmatic_markers as pm
import spacy_linguistic_analysis as sla
import ner_features as nf
import significance_tests as sig

CORPUS_MISTRAL = os.path.join(ROOT, "src", "2_text_analysis_scripts", "mistral_temperature_ab", "generated_corpus_mistral_temps.csv")
CORPUS_SW3 = os.path.join(HERE, "generated_corpus_gpt-sw3.csv")  # optional; from the Colab run
INFORMAL_ADV = os.path.join(ROOT, "src", "4_archive", "consolidated_informal_comments_adversarial.csv")
FORMAL_ADV = os.path.join(ROOT, "src", "1_data_collection", "llm_abstracts", "abstracts", "sv_abstracts_adversarial.csv")
OUT = os.path.join(HERE, "model_comparison_results.csv")

HUMAN_COL = {"informal": "human_comment", "formal": "Abstract"}
GPT_COL = {
    "informal": {"baseline": "generated_comment", "human_like": "comment_human_like",
                 "detector_aware": "comment_detector_aware", "detector_evasive": "comment_detector_evasive"},
    "formal": {"baseline": "Abstract_baseline", "human_like": "Abstract_human_like",
               "detector_aware": "Abstract_detector_aware", "detector_evasive": "Abstract_detector_evasive"},
}
CONDITIONS = ["baseline", "human_like", "detector_evasive"]  # detector_aware dropped (3-condition design)
GROUPS = ["stylometric_surface", "syntactic_complexity", "pragmatic_markers", "ner", "textstat", "pos_proportions"]
POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
            "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
# pairwise comparisons to run (only those whose BOTH generators have data are kept)
PAIRS = [("human", "gpt"), ("human", "mistral"), ("human", "sw3"),
         ("mistral", "gpt"), ("sw3", "gpt"), ("sw3", "mistral")]

nlp = spacy.load("sv_core_news_lg")


def _ok(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return bool(str(x).strip())


def textstat_feats(text):
    def rough_syllables_sv(t):
        t = re.sub(r"[^a-zåäöA-ZÅÄÖ]+", " ", t.lower())
        return len(re.findall(r"[aeiouyåäö]+", t)) if t.strip() else 0
    r = {"char_count": len(text), "word_count": max(len(text.split()), 1),
         "rough_syllable_count_sv": rough_syllables_sv(text)}
    for name in ("flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "smog_index"):
        try:
            r[f"textstat_{name}"] = getattr(textstat, name)(text)
        except Exception:
            r[f"textstat_{name}"] = np.nan
    return r


def pos_props(doc):
    c = Counter(t.pos_ for t in doc)
    tot = sum(c.values()) or 1
    return {f"pos_{tag}": c.get(tag, 0) / tot for tag in POS_TAGS}


def extract(text, register):
    drop_expr = register == "formal"
    min_tok = 30 if register == "formal" else 0
    matchers = pm.FORMAL_MATCHERS if register == "formal" else pm.MATCHERS
    g = {}
    g["stylometric_surface"] = ss.compute_metrics(nlp, text, drop_expressive=drop_expr) or {}
    g["syntactic_complexity"] = sc.compute_metrics(nlp, text, min_tokens=min_tok) or {}
    g["pragmatic_markers"] = pm.compute_metrics(nlp, text, matchers=matchers) or {}
    doc = nlp(text)
    ner = nf.doc_rates(sla.entity_counts_string(doc), ",".join(t.text for t in doc), "x")
    g["ner"] = {k[2:]: v for k, v in ner.items()}
    g["textstat"] = textstat_feats(text)
    g["pos_proportions"] = pos_props(doc)
    return g


def compare(rowsA, rowsB, label, register, condition):
    pvals, out = [], []
    for grp in GROUPS:
        feats = sorted({k for r in rowsA for k in r[grp]} | {k for r in rowsB for k in r[grp]})
        for feat in feats:
            a = pd.Series([r[grp].get(feat, np.nan) for r in rowsA], dtype="float64")
            b = pd.Series([r[grp].get(feat, np.nan) for r in rowsB], dtype="float64")
            res = sig.test_feature(a, b)
            if res is None:
                continue
            pvals.append(res["p"])
            out.append({"comparison": label, "register": register, "condition": condition,
                        "feature_group": grp, "feature": feat, "n_pairs": res["n"],
                        "median_A": res["median_human"], "median_B": res["median_llm"],
                        "cohens_dz": res["dz"], "rank_biserial": res["rbc"], "p_value": res["p"]})
    if pvals:
        for r, q in zip(out, sig.bh_fdr(pvals)):
            r["p_fdr_bh"] = q
            r["significant_fdr_0.05"] = bool(q < 0.05)
    return out


def load_corpus_map(path, temp=None):
    """Return {(register, doc_id, condition): text} from a generated corpus CSV, or {} if absent."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8")
    if temp is not None and "temp" in df.columns:
        df = df[df["temp"] == temp]
    return {(r["register"], int(r["doc_id"]), r["condition"]): r["text"] for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-cell", type=int, default=None)
    args = ap.parse_args()

    mistral_map = load_corpus_map(CORPUS_MISTRAL, temp=1.0)
    sw3_map = load_corpus_map(CORPUS_SW3, temp=1.0)
    adv = {"informal": pd.read_csv(INFORMAL_ADV, encoding="utf-8"),
           "formal": pd.read_csv(FORMAL_ADV, encoding="utf-8")}
    have = {"human", "gpt", "mistral"} | ({"sw3"} if sw3_map else set())
    active_pairs = [(a, b) for a, b in PAIRS if a in have and b in have]
    print(f"generators present: {sorted(have)} | comparisons: {['_vs_'.join(p) for p in active_pairs]}", flush=True)

    results = []
    human_cache = {}  # (register, doc_id) -> extracted human features (shared across conditions)
    for register in ["informal", "formal"]:
        df = adv[register]
        hcol = HUMAN_COL[register]
        for condition in CONDITIONS:
            gcol = GPT_COL[register][condition]
            feats_by_row = []  # list of {gen: feature_dict} for rows with >=1 valid generator pair
            count = 0
            for i in df.index:
                texts = {}
                if _ok(df.at[i, hcol]):
                    texts["human"] = str(df.at[i, hcol])
                if _ok(df.at[i, gcol]):
                    texts["gpt"] = str(df.at[i, gcol])
                m = mistral_map.get((register, int(i), condition))
                if _ok(m):
                    texts["mistral"] = str(m)
                s = sw3_map.get((register, int(i), condition))
                if _ok(s):
                    texts["sw3"] = str(s)
                if len(texts) < 2:
                    continue
                feats = {}
                for gen, txt in texts.items():
                    if gen == "human":
                        hf = human_cache.get((register, i))
                        if hf is None:
                            hf = extract(txt, register)
                            human_cache[(register, i)] = hf
                        feats["human"] = hf
                    else:
                        feats[gen] = extract(txt, register)
                feats_by_row.append(feats)
                count += 1
                if args.max_per_cell and count >= args.max_per_cell:
                    break
            for a, b in active_pairs:
                rowsA = [f[a] for f in feats_by_row if a in f and b in f]
                rowsB = [f[b] for f in feats_by_row if a in f and b in f]
                if len(rowsA) >= 3:
                    results += compare(rowsA, rowsB, f"{a}_vs_{b}", register, condition)
            print(f"  done {register}/{condition}: {len(feats_by_row)} rows", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {len(out)} rows -> {OUT}", flush=True)
    if len(out):
        out["absdz"] = out["cohens_dz"].abs()
        print("median |dz| by comparison (larger = more different):")
        print(out.groupby("comparison")["absdz"].median().round(3))


if __name__ == "__main__":
    main()
