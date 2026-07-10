"""
Pragmatic and discourse marker analysis for Swedish text.

Marker categories, each reported as a rate per 100 alpha tokens:

  hedges        — epistemic/mitigation markers that reduce commitment to a
                  proposition. A hand-curated Swedish lexicon spanning seven
                  functional subtypes: epistemic adverbs (kanske, troligen…),
                  modal particles (nog, väl, ju…), modal verbs (kan, borde,
                  skulle kunna…), mental-state verbs (tror, antar…),
                  approximators (ungefär, nästan, typ…), phrasal hedges
                  (i viss mån, mer eller mindre…) and evidential/attribution
                  markers (tyder på, så kallad, uppges…). Each entry is tagged
                  with a functional subtype and register — see HEDGE_LEXICON.
                  Also reported split by register as hedge_formal / hedge_neutral
                  / hedge_informal (these three sum to the overall hedge rate),
                  so formal- vs informal-leaning hedging can be compared.
  boosters      — intensifiers amplifying force (informal dataset only)
                  (väldigt, verkligen, helt, jätte-, enormt, …)
  negation      — negation tokens (inte, ej, icke, aldrig, ingen, …)
  epistemic     — verbs/adverbs marking speaker's knowledge state
                  (verkar, tycks, troligen, förmodligen, tror, menar, …)
                  NB: overlaps the expanded hedge lexicon (see notes below).
  discourse_ptcl— modal particles (ju, väl, nog, faktiskt, egentligen, …)
                  (informal dataset only)

Also computes negation_sentence_rate = fraction of sentences containing ≥1 negation.

Matching is case-insensitive over spaCy alpha tokens. Multi-word entries
(e.g. "skulle kunna", "i viss mån") are matched as non-overlapping phrases,
longest match first; single-token lexicons behave exactly as before. A small
POS gate (HEDGE_POS_GATE) restricts a few polysemous hedge tokens (rätt, runt,
kan…) to the part(s) of speech in which they actually hedge.

Usage:
  python src/2_text_analysis_scripts/pragmatic_markers.py --dataset formal
  python src/2_text_analysis_scripts/pragmatic_markers.py --dataset informal
"""
import argparse
import os

import numpy as np
import pandas as pd
import spacy

from data_utils import read_csv_robust, add_condition_arg, resolve_conditions, condition_tag

MODEL = "sv_core_news_lg"
_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")

# ---------------------------------------------------------------------------
# Swedish pragmatic marker lexicons
# ---------------------------------------------------------------------------

# Hedge lexicon: (surface form, functional subtype, register).
#   Register — F = formal / written-leaning, N = neutral / general,
#              I = informal / spoken-leaning.
# Multi-word entries are matched as phrases (see build_matcher / count_matches).
# NB: several entries also appear in EPISTEMIC_MARKERS / DISCOURSE_PARTICLES /
# NEGATION_MARKERS below, so the same token can be counted in more than one
# category. Categories are independent rates by design.
HEDGE_LEXICON = [
    # epistemic adverbs -----------------------------------------------------
    ("kanske",           "epistemic_adverb",  "N"),
    ("möjligen",         "epistemic_adverb",  "N"),
    ("möjligtvis",       "epistemic_adverb",  "N"),
    ("eventuellt",       "epistemic_adverb",  "N"),
    ("kanhända",         "epistemic_adverb",  "F"),
    ("troligen",         "epistemic_adverb",  "F"),
    ("troligtvis",       "epistemic_adverb",  "N"),
    ("sannolikt",        "epistemic_adverb",  "F"),
    ("förmodligen",      "epistemic_adverb",  "N"),
    ("antagligen",       "epistemic_adverb",  "N"),
    ("vanligen",         "epistemic_adverb",  "F"),
    ("vanligtvis",       "epistemic_adverb",  "N"),
    ("oftast",           "epistemic_adverb",  "N"),
    ("sällan",           "epistemic_adverb",  "N"),
    ("ibland",           "epistemic_adverb",  "N"),
    ("i allmänhet",      "epistemic_adverb",  "F"),
    # modal particles -------------------------------------------------------
    ("nog",              "modal_particle",    "N"),
    ("väl",              "modal_particle",    "I"),
    ("ju",               "modal_particle",    "N"),
    ("liksom",           "modal_particle",    "I"),
    ("bara",             "modal_particle",    "I"),
    ("ba",               "modal_particle",    "I"),   # colloquial clipping of "bara"
    # modal verbs -----------------------------------------------------------
    ("kan",              "modal_verb",        "N"),
    ("kunde",            "modal_verb",        "N"),
    ("skulle kunna",     "modal_verb",        "N"),
    ("borde",            "modal_verb",        "N"),
    ("torde",            "modal_verb",        "F"),
    ("lär",              "modal_verb",        "F"),
    ("verkar",           "modal_verb",        "N"),
    ("tycks",            "modal_verb",        "F"),
    # mental-state verbs ----------------------------------------------------
    ("tror",             "mental_state_verb", "N"),
    ("tycker",           "mental_state_verb", "N"),
    ("antar",            "mental_state_verb", "N"),
    ("gissar",           "mental_state_verb", "I"),
    ("misstänker",       "mental_state_verb", "N"),
    # approximators ---------------------------------------------------------
    ("ungefär",          "approximator",      "N"),
    ("cirka",            "approximator",      "F"),
    ("runt",             "approximator",      "I"),
    ("nästan",           "approximator",      "N"),
    ("lite",             "approximator",      "I"),
    ("ganska",           "approximator",      "I"),
    ("rätt",             "approximator",      "I"),
    ("typ",              "approximator",      "I"),
    # phrasal hedges --------------------------------------------------------
    ("i viss mån",       "phrasal_hedge",     "F"),
    ("i huvudsak",       "phrasal_hedge",     "F"),
    ("delvis",           "phrasal_hedge",     "F"),
    ("på sätt och vis",  "phrasal_hedge",     "N"),
    ("på något sätt",    "phrasal_hedge",     "N"),
    ("i stort sett",     "phrasal_hedge",     "N"),
    ("mer eller mindre", "phrasal_hedge",     "N"),
    ("så att säga",      "phrasal_hedge",     "N"),
    ("en slags",         "phrasal_hedge",     "N"),   # "en/ett slags"
    ("ett slags",        "phrasal_hedge",     "N"),
    ("enligt min mening","phrasal_hedge",     "F"),
    ("enligt mig",       "phrasal_hedge",     "I"),
    # evidential / attribution ---------------------------------------------
    ("antyder",             "evidential",     "F"),
    ("tyder på",            "evidential",     "N"),
    ("så kallad",           "evidential",     "F"),   # "så kallad(e)"
    ("så kallade",          "evidential",     "F"),
    ("påstådd",             "evidential",     "F"),   # "påstådd(a)"
    ("påstådda",            "evidential",     "F"),
    ("förment",             "evidential",     "F"),
    ("uppges",              "evidential",     "F"),
    ("enligt uppgift",      "evidential",     "F"),
    ("med kopplingar till", "evidential",     "F"),
]

HEDGE_MARKERS = {surface for surface, _subtype, _register in HEDGE_LEXICON}

# Split hedges by register so formal- vs informal-leaning hedging can be compared.
# Register (F/N/I) is a property of the hedge WORD, independent of which dataset
# (formal abstracts vs informal comments) the text comes from. The three buckets
# partition HEDGE_MARKERS, so hedge = hedge_formal + hedge_neutral + hedge_informal.
_REGISTER_NAME = {"F": "formal", "N": "neutral", "I": "informal"}
HEDGE_BY_REGISTER = {"formal": set(), "neutral": set(), "informal": set()}
for _surface, _subtype, _register in HEDGE_LEXICON:
    HEDGE_BY_REGISTER[_REGISTER_NAME[_register]].add(_surface)

# POS gate: restrict specific polysemous single-token hedges to the part(s) of
# speech in which they actually hedge, cutting false positives from unrelated
# senses. Keys are surface forms; values are allowed spaCy UPOS tags (token.pos_).
# Tokens not listed here are counted on surface form alone (as before).
# NB: this only fixes DIFFERENT-POS homographs (rätt "correct/court", runt
# "round"). It CANNOT separate same-POS senses — lite, bara, väl, nog, ju hedge
# and non-hedge uses are all ADV, so those stay surface-counted and may over-count.
HEDGE_POS_GATE = {
    "rätt":  {"ADV"},           # hedge "quite/pretty"; drops ADJ "correct", NOUN "court/right/dish"
    "runt":  {"ADV", "ADP"},    # approximator "around ~N"; drops ADJ "round"
    "kan":   {"AUX", "VERB"},   # modal "can/may"; drops stray noun homographs
    "kunde": {"AUX", "VERB"},
    "borde": {"AUX", "VERB"},
    "torde": {"AUX", "VERB"},
    "lär":   {"AUX", "VERB"},   # modal "is said to"; drops noun/name senses
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

# Map category name → lexicon set. hedge is the headline metric; the three
# hedge_<register> sub-categories partition it so formal vs informal hedging can
# be compared (hedge_rate == hedge_formal + hedge_neutral + hedge_informal).
CATEGORIES = {
    "hedge":              HEDGE_MARKERS,
    "hedge_formal":       HEDGE_BY_REGISTER["formal"],
    "hedge_neutral":      HEDGE_BY_REGISTER["neutral"],
    "hedge_informal":     HEDGE_BY_REGISTER["informal"],
    "booster":            BOOSTER_MARKERS,
    "negation":           NEGATION_MARKERS,
    "epistemic":          EPISTEMIC_MARKERS,
    "discourse_particle": DISCOURSE_PARTICLES,
}

# Formal abstracts: boosters and discourse particles are informal/spoken-register markers
# that barely occur in academic writing (floor effect), so they are excluded for --dataset
# formal. Hedges (with the register split), negation and epistemic markers are kept.
FORMAL_CATEGORIES = {k: CATEGORIES[k] for k in (
    "hedge", "hedge_formal", "hedge_neutral", "hedge_informal", "negation", "epistemic",
)}


# ---------------------------------------------------------------------------
# Phrase-aware matching
# ---------------------------------------------------------------------------
# A "matcher" indexes a lexicon's surface forms by token length so multi-word
# entries (e.g. "skulle kunna") can be found as phrases. A single-token lexicon
# reduces to the original whole-word membership test and yields identical counts.

def build_matcher(lexicon):
    """Index surface forms by token length: (max_len, {length: {token-tuple}})."""
    by_len = {}
    for surface in lexicon:
        toks = tuple(surface.lower().split())
        if toks:
            by_len.setdefault(len(toks), set()).add(toks)
    return (max(by_len) if by_len else 0), by_len


def count_matches(tokens, matcher, pos_gate=None):
    """Non-overlapping, longest-match-first count of lexicon hits.

    tokens: list of (lowercased_text, pos_tag). pos_gate (optional):
    {surface: {allowed UPOS}} restricts single-token matches for polysemous
    words to the POS in which they hedge; multi-word phrases are never gated.
    """
    max_len, by_len = matcher
    if max_len == 0:
        return 0
    words = [w for w, _pos in tokens]
    n = len(words)
    count = 0
    i = 0
    while i < n:
        for length in range(min(max_len, n - i), 0, -1):
            if length in by_len and tuple(words[i:i + length]) in by_len[length]:
                if (length == 1 and pos_gate and words[i] in pos_gate
                        and tokens[i][1] not in pos_gate[words[i]]):
                    continue  # right surface, wrong POS -> not a hedge here
                count += 1
                i += length
                break
        else:
            i += 1
    return count


# Prebuild matchers once at import so per-text scoring stays cheap.
MATCHERS = {name: build_matcher(lex) for name, lex in CATEGORIES.items()}
FORMAL_MATCHERS = {name: MATCHERS[name] for name in FORMAL_CATEGORIES}


def parse_args():
    p = argparse.ArgumentParser(description="Pragmatic marker rates")
    p.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    add_condition_arg(p)
    return p.parse_args()


def resolve_config(dataset):
    if dataset == "informal":
        return {
            "csv": os.path.join(_data, "llm_comments", "consolidated_informal_comments_adversarial.csv"),
            "human_col": "human_comment",
            "llm_col": "generated_comment",
        }
    return {
        "csv": os.path.join(_data, "llm_abstracts", "abstracts", "sv_abstracts_adversarial.csv"),
        "human_col": "Abstract",
        "llm_col": "Abstract_baseline",
    }


def load_nlp():
    try:
        return spacy.load(MODEL)
    except OSError:
        raise OSError(f"Run: python -m spacy download {MODEL}")


def compute_metrics(nlp, text, matchers=MATCHERS):
    if pd.isna(text) or not str(text).strip():
        return None
    doc = nlp(str(text).strip())
    alpha = [(t.text.lower(), t.pos_) for t in doc if t.is_alpha]
    n = len(alpha)
    sents = list(doc.sents)
    n_sents = len(sents)

    result = {}
    for cat, matcher in matchers.items():
        count = count_matches(alpha, matcher, HEDGE_POS_GATE)
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
    # NB: n_sentences is intentionally NOT emitted here — syntactic_complexity is
    # its owner. Emitting it in both files made it get tested twice (identical
    # values) and double-counted in each condition's BH-FDR family.
    return result


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    nlp = load_nlp()
    df = read_csv_robust(cfg["csv"])
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(df)

    # formal abstracts drop the informal-register marker categories (see FORMAL_CATEGORIES)
    matchers = MATCHERS if args.dataset == "informal" else FORMAL_MATCHERS

    # human side is identical across conditions -> parse it once and reuse
    print("parsing human texts…", flush=True)
    human_feats = [compute_metrics(nlp, row.get(cfg["human_col"]), matchers) for _, row in df.iterrows()]

    for cond, llm_col in resolve_conditions(args.dataset, args.condition):
        if llm_col not in df.columns:
            print(f"  skipping condition '{cond}': column '{llm_col}' not found")
            continue
        tag = condition_tag(args.dataset, cond)
        print(f"parsing LLM texts [{cond or 'generated'}]…", flush=True)
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"  {i + 1}/{n}", end="\r", flush=True)
            h = human_feats[i]
            l = compute_metrics(nlp, row.get(llm_col), matchers)
            if h is None or l is None:
                continue
            out_row = {f"human_{k}": v for k, v in h.items()}
            out_row.update({f"llm_{k}": v for k, v in l.items()})
            rows.append(out_row)
        print()

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"pragmatic_markers_{tag}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{cond or 'generated'}] Wrote {len(out_df)} rows → {out_path}")

        rate_cols = [c for c in out_df.columns if c.startswith("human_") and "rate" in c]
        print("Mean rates per 100 tokens (human | llm):")
        for hcol in sorted(rate_cols):
            lcol = "llm_" + hcol[len("human_"):]
            if lcol in out_df:
                print(f"  {hcol[6:]:35s}  {out_df[hcol].mean():.4f}  |  {out_df[lcol].mean():.4f}")


if __name__ == "__main__":
    main()
