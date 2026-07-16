"""Build a single consolidated results table (results_master_table.csv) from the
structural (model_comparison_results.csv) and affective (affective_results.csv) results.

One row per feature ("Test"). One result cell per permutation
{GPT-5.2, Mistral} x {Informal, Formal} x {Baseline, P1 human-like, P2 detector-aware,
P3 detector-evasive}. Cell = Cohen's dz vs human (+ = model>human), '*' if FDR<0.05,
blank if not measured (e.g. affective in formal, expressive punctuation in formal).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(HERE, "model_comparison_results.csv")
AFFECT = os.path.join(HERE, "affective_results.csv")
OUT = os.path.join(HERE, "results_master_table.csv")

CONDS = [("baseline", "Baseline"), ("human_like", "P1 Human-like"),
         ("detector_aware", "P2 Detector-aware"), ("detector_evasive", "P3 Detector-evasive")]
MODELS = [("human_vs_gpt", "GPT-5.2"), ("human_vs_mistral", "Mistral")]
REGS = [("informal", "Informal"), ("formal", "Formal")]

REL_STRUCT = "src/2_text_analysis_scripts/model_comparison_mistral_gpt/model_comparison_results.csv"
REL_AFFECT = "src/2_text_analysis_scripts/model_comparison_mistral_gpt/affective_results.csv"
PY_STRUCT = "src/2_text_analysis_scripts/model_comparison_mistral_gpt/model_comparison.py"
PY_AFFECT = "src/2_text_analysis_scripts/model_comparison_mistral_gpt/affective_sentiment.py"

PRETTY = {
    "bigram_repetition_rate": "Bigram repetition", "trigram_repetition_rate": "Trigram repetition",
    "function_word_rate": "Function-word rate", "ttr": "Type-token ratio", "mattr": "MATTR",
    "mtld": "MTLD (lexical diversity)", "punct_period": "Periods /100tok", "punct_comma": "Commas /100tok",
    "punct_semicolon": "Semicolons /100tok", "punct_colon": "Colons /100tok", "punct_dash": "Dashes /100tok",
    "punct_exclaim": "Exclamation marks /100tok", "punct_question": "Question marks /100tok",
    "punct_ellipsis": "Ellipses /100tok", "dep_distance_mean": "Mean dependency distance",
    "tree_depth_mean": "Mean parse-tree depth", "tree_depth_max": "Max parse-tree depth",
    "subordinate_clause_rate": "Subordinate-clause rate", "n_sentences": "Sentence count",
    "hedge_rate": "Hedges /100tok", "hedge_count": "Hedges (count)",
    "hedge_formal_rate": "Hedges: formal /100tok", "hedge_formal_count": "Hedges: formal (count)",
    "hedge_neutral_rate": "Hedges: neutral /100tok", "hedge_neutral_count": "Hedges: neutral (count)",
    "hedge_informal_rate": "Hedges: informal /100tok", "hedge_informal_count": "Hedges: informal (count)",
    "negation_rate": "Negation /100tok", "negation_count": "Negation (count)",
    "negation_sentence_rate": "Negation-sentence rate", "epistemic_rate": "Epistemic /100tok",
    "epistemic_count": "Epistemic (count)", "booster_rate": "Boosters /100tok", "booster_count": "Boosters (count)",
    "discourse_particle_rate": "Discourse particles /100tok", "discourse_particle_count": "Discourse particles (count)",
    "n_alpha_tokens": "Length (alpha tokens)", "char_count": "Character count", "word_count": "Word count",
    "rough_syllable_count_sv": "Syllable count (approx.)", "textstat_flesch_reading_ease": "Flesch reading-ease",
    "textstat_flesch_kincaid_grade": "Flesch-Kincaid grade", "textstat_gunning_fog": "Gunning fog",
    "textstat_smog_index": "SMOG index", "signed_polarity": "Sentiment polarity (signed)",
    "affective_extremity": "Affective extremity", "is_subjective": "Subjectivity (non-neutral)",
}
PURPOSE = {
    "bigram_repetition_rate": "Repeated 2-grams; LLMs often repeat phrasing",
    "trigram_repetition_rate": "Repeated 3-grams; verbatim self-repetition",
    "function_word_rate": "Share of grammatical function words; stylometric fingerprint",
    "ttr": "Lexical diversity (unique/total words)", "mattr": "Moving-average TTR (length-robust diversity)",
    "mtld": "Lexical diversity robust to length", "punct_period": "Sentence-final density (short vs long sentences)",
    "punct_comma": "Clause segmentation / list style", "punct_semicolon": "Formal clause linking",
    "punct_colon": "Enumerative/formal punctuation", "punct_dash": "Dash usage (an LLM stylistic tell)",
    "punct_exclaim": "Expressive emphasis", "punct_question": "Rhetorical/interactive questions",
    "punct_ellipsis": "Trailing-off / informal tone", "dep_distance_mean": "Syntactic complexity (head-dependent distance)",
    "tree_depth_mean": "Sentence embedding depth", "tree_depth_max": "Deepest clause nesting",
    "subordinate_clause_rate": "Subordination / syntactic complexity", "n_sentences": "Number of sentences (length proxy)",
    "hedge_rate": "Epistemic caution / uncertainty", "hedge_count": "Absolute hedge use",
    "hedge_formal_rate": "Formal-register hedges", "hedge_neutral_rate": "Register-neutral hedges",
    "hedge_informal_rate": "Colloquial hedges", "negation_rate": "Use of negation",
    "negation_sentence_rate": "Share of negated sentences", "epistemic_rate": "Explicit epistemic stance markers",
    "booster_rate": "Intensifiers / certainty markers", "discourse_particle_rate": "Spoken-style discourse particles",
    "n_alpha_tokens": "Output length (alphabetic tokens)", "char_count": "Output length (characters)",
    "word_count": "Output length (words)", "rough_syllable_count_sv": "Phonological length proxy",
    "textstat_flesch_reading_ease": "Readability (higher = easier)", "textstat_flesch_kincaid_grade": "Reading grade level",
    "textstat_gunning_fog": "Readability via long-word density", "textstat_smog_index": "Readability via polysyllables",
    "signed_polarity": "P(positive)-P(negative); positivity/negativity lean",
    "affective_extremity": "Non-neutral sentiment mass (how polar)", "is_subjective": "Non-neutral (opinionated) share",
}
EXAMPLE = {
    "function_word_rate": "att, och, en, som, det", "mtld": "varied vocab vs 'bra bra bra'",
    "punct_period": "short: 'Ja. Precis. Instämmer.'", "punct_dash": "'det var - typ - konstigt'",
    "punct_exclaim": "'Grymt bra!!'", "punct_ellipsis": "'nja... vet inte...'",
    "subordinate_clause_rate": "'...eftersom att...', '...som jag tror...'",
    "hedge_rate": "kanske, möjligen, typ, väl", "hedge_informal_rate": "typ, liksom, ju",
    "booster_rate": "verkligen, absolut, väldigt", "negation_rate": "inte, aldrig, ingen",
    "discourse_particle_rate": "ju, väl, alltså, liksom", "signed_polarity": "'Toppen, tack!' (+) vs 'Skitdåligt.' (-)",
    "affective_extremity": "strong opinion (either sign) vs flat/neutral",
    "n_alpha_tokens": "longer vs terse replies",
}
POS_EX = {"ADJ": "stor", "ADP": "på", "ADV": "ofta", "AUX": "har", "CCONJ": "och", "DET": "den",
          "INTJ": "ja", "NOUN": "bil", "NUM": "tre", "PART": "att", "PRON": "jag", "PROPN": "Stockholm",
          "PUNCT": ".", "SCONJ": "eftersom", "SYM": "%", "VERB": "springa", "X": "xoxo"}
NER_EX = {"LOC": "Stockholm", "PRS": "Anna", "ORG": "SVT", "TME": "igår", "WRK": "Bibeln",
          "MSR": "5 kg", "OBJ": "bilen", "EVN": "valet"}


def meta(feat):
    if feat.startswith("pos_"):
        t = feat[4:]
        return f"POS: {t} proportion", f"Proportion of {t} tokens", f"e.g. '{POS_EX.get(t, t)}'"
    if feat.startswith("ner_"):
        t = feat.replace("ner_", "").replace("_rate", "")
        nm = {"LOC": "Location", "PRS": "Person", "ORG": "Organisation", "TME": "Time", "WRK": "Work/artefact",
              "MSR": "Measure", "OBJ": "Object", "EVN": "Event", "total": "All"}.get(t, t)
        return f"NER: {nm} /100tok", f"{nm} named-entity rate per 100 tokens", f"e.g. '{NER_EX.get(t, t)}'"
    return PRETTY.get(feat, feat), PURPOSE.get(feat, ""), EXAMPLE.get(feat, "")


def main():
    s = pd.read_csv(STRUCT); s["src"] = "struct"
    frames = [s]
    if os.path.exists(AFFECT):
        a = pd.read_csv(AFFECT); a["src"] = "affect"
        frames.append(a)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.comparison.isin([m[0] for m in MODELS])].copy()

    look = {(r.comparison, r.register, r.condition, r.feature): r for _, r in df.iterrows()}
    # order features by family then by max |dz|
    fam = df.groupby("feature")["feature_group"].first()
    maxdz = df.assign(a=df.cohens_dz.abs()).groupby("feature")["a"].max()
    fam_order = ["stylometric_surface", "syntactic_complexity", "pragmatic_markers", "ner",
                 "textstat", "pos_proportions", "affective_sentiment"]
    feats = sorted(df.feature.unique(),
                   key=lambda f: (fam_order.index(fam[f]) if fam[f] in fam_order else 99, -maxdz[f]))

    rows = []
    for feat in feats:
        pretty, purpose, example = meta(feat)
        is_aff = fam[feat] == "affective_sentiment"
        row = {"Test": pretty, "Feature_family": fam[feat], "Purpose of Test": purpose, "Example": example}
        cells = []  # (label, dz, sig) for the key-finding scan
        for comp, mlabel in MODELS:
            for reg, rlabel in REGS:
                for cond, clabel in CONDS:
                    col = f"{mlabel} | {rlabel} | {clabel}"
                    r = look.get((comp, reg, cond, feat))
                    if r is None:
                        row[col] = ""
                    else:
                        star = "*" if bool(r["significant_fdr_0.05"]) else ""
                        row[col] = f"{r.cohens_dz:+.2f}{star}"
                        cells.append((f"{mlabel}·{rlabel}·{clabel}", float(r.cohens_dz), bool(r["significant_fdr_0.05"])))
        # N per register (max n_pairs seen)
        sub = df[df.feature == feat]
        row["N_informal"] = int(sub[sub.register == "informal"]["n_pairs"].max()) if (sub.register == "informal").any() else 0
        row["N_formal"] = int(sub[sub.register == "formal"]["n_pairs"].max()) if (sub.register == "formal").any() else 0
        # key finding: largest-magnitude cell + #significant
        if cells:
            top = max(cells, key=lambda c: abs(c[1]))
            nsig = sum(1 for c in cells if c[2])
            direc = "model>human" if top[1] > 0 else "model<human"
            row["Key Findings"] = (f"Largest gap {top[0]} dz={top[1]:+.2f} ({direc}); "
                                   f"significant in {nsig}/{len(cells)} cells")
        else:
            row["Key Findings"] = "no data"
        row["Link to csv file"] = REL_AFFECT if is_aff else REL_STRUCT
        row["Link to python script"] = PY_AFFECT if is_aff else PY_STRUCT
        rows.append(row)

    # column order
    cols = ["Test", "Feature_family", "Purpose of Test", "Example"]
    cols += [f"{m} | {r} | {c}" for _, m in MODELS for _, r in REGS for _, c in CONDS]
    cols += ["Key Findings", "N_informal", "N_formal", "Link to csv file", "Link to python script"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {len(out)} rows x {len(cols)} cols -> {OUT}")
    print("families:", out.Feature_family.value_counts().to_dict())


if __name__ == "__main__":
    main()
