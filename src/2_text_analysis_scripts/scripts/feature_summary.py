"""Build a human-readable summary table of every feature-based analysis.

For each feature the table records: the feature group, a readable name, its
purpose, its equation/definition, and its results. Purpose + equation are
authored here (from the feature scripts); results are pulled live from
significance_tests_results.csv so they stay accurate when the tests are rerun.

Result convention: Cohen's dz, sign +ve = LLM > Human. A trailing '*' marks
BH-FDR<0.05 within that (dataset, condition) family. Conditions are shown in the
order base | h-like | det-aware. "not tested" = the feature is not computed for
that dataset (e.g. boosters / expressive punctuation are formal-floor effects).

Columns: results_formal / results_informal give the readable packed strings
(with significance stars); dz_formal_<cond> / dz_informal_<cond> give the same
effect sizes as plain numbers (blank = not tested) for sorting/filtering/plotting.

Output: csv_files/feature_analysis_summary.csv

Run:  python src/2_text_analysis_scripts/scripts/feature_summary.py
"""
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(os.path.dirname(_HERE), "csv_files")
SIG = os.path.join(CSV_DIR, "significance_tests_results.csv")
OUT = os.path.join(CSV_DIR, "feature_analysis_summary.csv")

COND_ORDER = ["baseline", "human_like", "detector_aware", "detector_evasive"]
COND_ABBR = {"baseline": "base", "human_like": "h-like",
             "detector_aware": "det-aware", "detector_evasive": "det-evasive"}

# (group, feature, analysis, purpose, equation) — order controls the output rows.
META = [
    # --- syntactic complexity ------------------------------------------------
    ("syntactic_complexity", "dep_distance_mean", "Mean dependency distance",
     "Syntactic complexity: how far dependents sit from their heads (larger = more embedded, non-local structure).",
     "mean over non-ROOT tokens of |token_index - head_index| per document."),
    ("syntactic_complexity", "tree_depth_mean", "Parse-tree depth (mean)",
     "Average depth of syntactic nesting per sentence.",
     "per sentence: longest ROOT-to-leaf path in the dependency tree; averaged over sentences."),
    ("syntactic_complexity", "tree_depth_max", "Parse-tree depth (max)",
     "Deepest syntactic nesting reached in the document.",
     "maximum over sentences of the longest ROOT-to-leaf path."),
    ("syntactic_complexity", "subordinate_clause_rate", "Subordinate clauses per sentence",
     "Density of subordination / clausal embedding.",
     "count of subordinating arcs (advcl, relcl, acl, acl:relcl, csubj, csubj:pass, ccomp) / number of sentences."),
    ("syntactic_complexity", "n_sentences", "Sentence count",
     "Document length in sentences (structural size).",
     "number of spaCy-segmented sentences in the document."),
    # --- stylometric surface -------------------------------------------------
    ("stylometric_surface", "mtld", "MTLD (lexical diversity)",
     "Length-robust lexical diversity (vocabulary variety independent of text length).",
     "bidirectional Measure of Textual Lexical Diversity: mean token-run length keeping sequential TTR > 0.72, averaged forward+backward (McCarthy & Jarvis 2010)."),
    ("stylometric_surface", "mattr", "MATTR (moving-average TTR)",
     "Length-robust vocabulary richness via a sliding window.",
     "mean of type-token ratio computed over every 100-token window."),
    ("stylometric_surface", "ttr", "Type-token ratio (TTR)",
     "Raw vocabulary richness (sensitive to text length).",
     "unique lowercased alpha tokens / total alpha tokens."),
    ("stylometric_surface", "function_word_rate", "Function-word rate",
     "Proportion of grammatical (closed-class) words vs content words.",
     "alpha tokens found in a Swedish function-word list / total alpha tokens."),
    ("stylometric_surface", "bigram_repetition_rate", "Bigram repetition rate",
     "Local repetitiveness / templated phrasing.",
     "repeated bigram occurrences / total bigrams over alpha tokens (repeated = sum of count-1 for bigrams occurring more than once)."),
    ("stylometric_surface", "trigram_repetition_rate", "Trigram repetition rate",
     "Repetition of 3-word sequences.",
     "repeated trigram occurrences / total trigrams over alpha tokens."),
    ("stylometric_surface", "punct_comma", "Commas per 100 tokens",
     "Comma usage; higher = more clause chaining / listing.",
     "count of ',' in raw text / non-space tokens * 100."),
    ("stylometric_surface", "punct_period", "Periods per 100 tokens",
     "Sentence-final punctuation density (inversely tracks sentence length).",
     "count of '.' / non-space tokens * 100."),
    ("stylometric_surface", "punct_semicolon", "Semicolons per 100 tokens",
     "Semicolon usage (formal clause linking).",
     "count of ';' / non-space tokens * 100."),
    ("stylometric_surface", "punct_colon", "Colons per 100 tokens",
     "Colon usage (introductions / lists).",
     "count of ':' / non-space tokens * 100."),
    ("stylometric_surface", "punct_dash", "Dashes per 100 tokens",
     "Em/en-dash usage (parenthetical / appositive style).",
     "(count of em-dash + count of en-dash) / non-space tokens * 100."),
    ("stylometric_surface", "punct_exclaim", "Exclamation marks per 100 tokens",
     "Expressive punctuation (informal register).",
     "count of '!' / non-space tokens * 100 (informal only)."),
    ("stylometric_surface", "punct_question", "Question marks per 100 tokens",
     "Interrogative / expressive punctuation (informal register).",
     "count of '?' / non-space tokens * 100 (informal only)."),
    ("stylometric_surface", "punct_ellipsis", "Ellipses per 100 tokens",
     "Trailing / expressive punctuation (informal register).",
     "count of '...' / non-space tokens * 100 (informal only)."),
    # --- pragmatic markers ---------------------------------------------------
    ("pragmatic_markers", "hedge_rate", "Hedges per 100 tokens",
     "Epistemic mitigation: markers reducing commitment (uncertainty, approximation, softening).",
     "matches of the 65-form hedge lexicon (case-insensitive; multi-word entries matched as non-overlapping phrases, longest first; polysemous items POS-gated) / alpha tokens * 100."),
    ("pragmatic_markers", "hedge_count", "Hedges (raw count)",
     "Absolute number of hedge markers in the document.",
     "number of hedge-lexicon matches."),
    ("pragmatic_markers", "hedge_formal_rate", "Formal-register hedges per 100 tokens",
     "Written-leaning hedging (e.g. sannolikt, torde, i viss man).",
     "hedge matches tagged register = formal / alpha tokens * 100."),
    ("pragmatic_markers", "hedge_formal_count", "Formal-register hedges (count)",
     "Absolute number of formal-register hedges.",
     "count of register = formal hedge matches."),
    ("pragmatic_markers", "hedge_neutral_rate", "Neutral-register hedges per 100 tokens",
     "Register-neutral hedging (e.g. kanske, kan, tror, ungefar).",
     "hedge matches tagged register = neutral / alpha tokens * 100."),
    ("pragmatic_markers", "hedge_neutral_count", "Neutral-register hedges (count)",
     "Absolute number of neutral-register hedges.",
     "count of register = neutral hedge matches."),
    ("pragmatic_markers", "hedge_informal_rate", "Informal-register hedges per 100 tokens",
     "Spoken-leaning hedging (e.g. typ, ba, liksom, gissar).",
     "hedge matches tagged register = informal / alpha tokens * 100."),
    ("pragmatic_markers", "hedge_informal_count", "Informal-register hedges (count)",
     "Absolute number of informal-register hedges.",
     "count of register = informal hedge matches."),
    ("pragmatic_markers", "booster_rate", "Boosters per 100 tokens",
     "Intensifiers that amplify force (e.g. valdigt, verkligen, helt).",
     "booster-lexicon matches (case-insensitive whole-word) / alpha tokens * 100 (informal only)."),
    ("pragmatic_markers", "booster_count", "Boosters (raw count)",
     "Absolute number of booster markers.",
     "number of booster-lexicon matches (informal only)."),
    ("pragmatic_markers", "negation_rate", "Negation per 100 tokens",
     "Frequency of negation markers (inte, ej, aldrig, ingen ...).",
     "negation-lexicon matches (case-insensitive whole-word) / alpha tokens * 100."),
    ("pragmatic_markers", "negation_count", "Negation (raw count)",
     "Absolute number of negation markers.",
     "number of negation-lexicon matches."),
    ("pragmatic_markers", "negation_sentence_rate", "Negation sentence rate",
     "How pervasive negation is across sentences (vs concentrated in a few).",
     "fraction of sentences containing at least one negation token."),
    ("pragmatic_markers", "epistemic_rate", "Epistemic markers per 100 tokens",
     "Explicit marking of the speaker's knowledge state (verkar, tror, troligen ...).",
     "epistemic-lexicon matches (case-insensitive whole-word) / alpha tokens * 100."),
    ("pragmatic_markers", "epistemic_count", "Epistemic markers (raw count)",
     "Absolute number of epistemic markers.",
     "number of epistemic-lexicon matches."),
    ("pragmatic_markers", "discourse_particle_rate", "Discourse particles per 100 tokens",
     "Interactional modal particles (ju, val, nog, faktiskt ...); spoken register.",
     "discourse-particle-lexicon matches (case-insensitive whole-word) / alpha tokens * 100 (informal only)."),
    ("pragmatic_markers", "discourse_particle_count", "Discourse particles (raw count)",
     "Absolute number of discourse particles.",
     "number of discourse-particle-lexicon matches (informal only)."),
    ("pragmatic_markers", "n_alpha_tokens", "Length (alpha tokens)",
     "Document length in alphabetic tokens.",
     "number of alphabetic spaCy tokens."),
    # --- named entity recognition -------------------------------------------
    ("ner", "ner_total_rate", "Named entities per 100 tokens (all types)",
     "Overall entity naming = concreteness / specificity of reference.",
     "per document: (total spaCy entities / token length) * 100."),
    ("ner", "ner_LOC_rate", "Location entities per 100 tokens",
     "Naming of geographic places (cities, countries, landmarks).",
     "per document: (count of LOC entities / token length) * 100."),
    ("ner", "ner_PRS_rate", "Person entities per 100 tokens",
     "Naming of people (historical / cultural figures).",
     "per document: (count of PRS entities / token length) * 100."),
    ("ner", "ner_ORG_rate", "Organisation entities per 100 tokens",
     "Naming of institutions, companies, teams.",
     "per document: (count of ORG entities / token length) * 100."),
    ("ner", "ner_TME_rate", "Time entities per 100 tokens",
     "Temporal expressions (dates, periods).",
     "per document: (count of TME entities / token length) * 100."),
    ("ner", "ner_WRK_rate", "Work/Artefact entities per 100 tokens",
     "Named works of art and artefacts.",
     "per document: (count of WRK entities / token length) * 100."),
    ("ner", "ner_MSR_rate", "Measure entities per 100 tokens",
     "Measures / numerical expressions.",
     "per document: (count of MSR entities / token length) * 100."),
    ("ner", "ner_OBJ_rate", "Object entities per 100 tokens",
     "Named objects / products.",
     "per document: (count of OBJ entities / token length) * 100."),
    ("ner", "ner_EVN_rate", "Event entities per 100 tokens",
     "Named events.",
     "per document: (count of EVN entities / token length) * 100."),
]


def results_str(df, feature, dataset):
    sub = df[(df["feature"] == feature) & (df["dataset"] == dataset)]
    if sub.empty:
        return "not tested"
    parts = []
    for c in COND_ORDER:
        r = sub[sub["condition"] == c]
        if r.empty:
            continue
        dz = r.iloc[0]["cohens_dz"]
        star = "*" if bool(r.iloc[0]["significant_fdr_0.05"]) else ""
        parts.append(f"{COND_ABBR[c]} {dz:+.2f}{star}")
    return " | ".join(parts) if parts else "not tested"


def dz_cell(df, feature, dataset, cond):
    """Numeric Cohen's dz for one (feature, dataset, condition); blank if not tested."""
    r = df[(df["feature"] == feature) & (df["dataset"] == dataset) & (df["condition"] == cond)]
    if r.empty:
        return ""
    return round(float(r.iloc[0]["cohens_dz"]), 3)


def summary_str(df, feature):
    sub = df[df["feature"] == feature]
    sig = sub[sub["significant_fdr_0.05"] == True]  # noqa: E712
    if sig.empty:
        return "no significant human/LLM difference"
    signs = {1 if v > 0 else -1 for v in sig["cohens_dz"] if v != 0}
    if signs == {1}:
        direction = "LLM > Human"
    elif signs == {-1}:
        direction = "Human > LLM"
    else:
        direction = "mixed (direction varies by condition)"
    mx = sig["cohens_dz"].abs().max()
    band = "large" if mx >= 0.8 else "medium" if mx >= 0.5 else "small"
    return f"{direction}; up to {band} effect (|dz|max {mx:.2f}); significant in {len(sig)} of {len(sub)} tests"


def main():
    df = pd.read_csv(SIG)
    rows = []
    for group, feature, analysis, purpose, equation in META:
        row = {
            "feature_group": group,
            "feature": feature,
            "analysis": analysis,
            "purpose": purpose,
            "equation": equation,
            "results_formal": results_str(df, feature, "formal"),
            "results_informal": results_str(df, feature, "informal"),
        }
        # numeric per-condition dz columns (sortable/plottable; blank = not tested)
        for ds, pfx in (("formal", "dz_formal"), ("informal", "dz_informal")):
            for cond in COND_ORDER:
                row[f"{pfx}_{cond}"] = dz_cell(df, feature, ds, cond)
        row["summary"] = summary_str(df, feature)
        rows.append(row)
    out = pd.DataFrame(rows)

    # sanity check: every tested feature is documented
    documented = {f for _, f, *_ in META}
    tested = set(df["feature"].unique())
    missing = tested - documented
    if missing:
        print(f"  [warn] tested but undocumented features: {sorted(missing)}")

    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(out)} features -> {OUT}")
    print("Legend: dz sign +ve = LLM > Human; '*' = BH-FDR<0.05; conditions = base | h-like | det-aware")


if __name__ == "__main__":
    main()
