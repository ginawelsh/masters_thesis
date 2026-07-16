"""Extract the GPT-pipeline feature families from the Mistral 0.15-vs-1.0 corpus and
run the SAME paired significance analysis (Wilcoxon signed-rank + Cohen's dz +
matched-pairs rank-biserial + BH-FDR per register x condition family) comparing the two
temperatures. Pairs are (register, doc_id, condition): 0.15 output vs 1.0 output of the
SAME prompt.

Reuses the real pipeline extractors from ../scripts/ (so numbers are computed exactly as
in the human-vs-generated analysis):
  stylometric_surface.compute_metrics, syntactic_complexity.compute_metrics,
  pragmatic_markers.compute_metrics, spacy_linguistic_analysis.entity_counts_string +
  ner_features.doc_rates, plus textstat (as in multi_layer_analysis) and UPOS proportions.

Input:  generated_corpus_mistral_temps.csv  (produced by generate_mistral_temps.py)
Output: significance_results.csv
Run:    python src/2_text_analysis_scripts/mistral_temperature_ab/extract_and_test.py
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

CACHE = os.path.join(HERE, "generated_corpus_mistral_temps.csv")
OUT = os.path.join(HERE, "significance_results.csv")
POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
            "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

nlp = spacy.load("sv_core_news_lg")


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
    g["ner"] = {k[2:]: v for k, v in ner.items()}  # strip "x_" prefix
    g["textstat"] = textstat_feats(text)
    g["pos_proportions"] = pos_props(doc)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-cell", type=int, default=None, help="cap docs per cell (quick validation)")
    args = ap.parse_args()

    df = pd.read_csv(CACHE, encoding="utf-8")
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    wide = df.pivot_table(index=["register", "doc_id", "condition"], columns="temp",
                          values="text", aggfunc="first").dropna(subset=[0.15, 1.0]).reset_index()
    print(f"paired docs available: {len(wide)}", flush=True)

    groups = ["stylometric_surface", "syntactic_complexity", "pragmatic_markers",
              "ner", "textstat", "pos_proportions"]
    results = []
    for register in ["informal", "formal"]:
        for condition in ["baseline", "human_like", "detector_aware", "detector_evasive"]:
            sub = wide[(wide["register"] == register) & (wide["condition"] == condition)]
            if args.max_per_cell:
                sub = sub.head(args.max_per_cell)
            if len(sub) < 3:
                continue
            rows015 = [extract(str(r[0.15]), register) for _, r in sub.iterrows()]
            rows100 = [extract(str(r[1.0]), register) for _, r in sub.iterrows()]
            fam_pvals, fam_rows = [], []
            for g in groups:
                feats = sorted({k for row in rows015 for k in row[g]} | {k for row in rows100 for k in row[g]})
                for feat in feats:
                    s015 = pd.Series([row[g].get(feat, np.nan) for row in rows015], dtype="float64")
                    s100 = pd.Series([row[g].get(feat, np.nan) for row in rows100], dtype="float64")
                    res = sig.test_feature(s015, s100)
                    if res is None:
                        continue
                    fam_pvals.append(res["p"])
                    fam_rows.append({
                        "register": register, "condition": condition, "feature_group": g, "feature": feat,
                        "n_pairs": res["n"], "median_t015": res["median_human"], "median_t100": res["median_llm"],
                        "direction": ("t1.0>t0.15" if res["median_llm"] > res["median_human"]
                                      else "t1.0<t0.15" if res["median_llm"] < res["median_human"] else "="),
                        "cohens_dz": res["dz"], "rank_biserial": res["rbc"], "p_value": res["p"],
                    })
            if fam_pvals:
                qs = sig.bh_fdr(fam_pvals)
                for row, q in zip(fam_rows, qs):
                    row["p_fdr_bh"] = q
                    row["significant_fdr_0.05"] = bool(q < 0.05)
                results.extend(fam_rows)
            print(f"  done {register}/{condition}: {len(sub)} pairs, {len(fam_rows)} features", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {len(out)} feature rows -> {OUT}", flush=True)
    if len(out):
        print("significant (FDR<0.05) by register/condition:")
        print(out[out["significant_fdr_0.05"]].groupby(["register", "condition"]).size())


if __name__ == "__main__":
    main()
