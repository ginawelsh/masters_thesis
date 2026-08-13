"""Build the CLEANED formal detection corpus: all four sources, uniformly normalized.

Answers "how detectable is formal AI Swedish across generators, once the FORMATTING
confound is removed?" -- the markdown/heading/Nyckelord asymmetry (Mistral 86%,
GPT-5.6 90%, GPT-5.2 9-18%, human 2%) is neutralized by applying ONE normalizer
(text_normalizer.normalize_formal) to every AI source. Human is left untouched
(ground truth, already markdown-free).

Matched design: the 100 detection-quiz abstracts (the same set used by the GPT-5.2
balanced run). For each abstract:
    1 human  +  3 GPT-5.2  +  3 GPT-5.6  +  3 Mistral    (conditions baseline /
    human_like / detector_evasive)  =>  up to 10 items per abstract.

Backbone = the GPT-5.6 quiz file (it already carries quiz_source_row, quiz_pair_id,
the human Abstract, and the Title). GPT-5.2 is joined by source row into
sv_abstracts_adversarial; Mistral by doc_id == source row into the temps corpus.

GPT-5.6 detector_evasive has 2 hard refusals (empty cells) -> those items are
DROPPED here and the refusal rate is reported separately (a finding, not text to judge).

Output: formal_cleaned_quiz.csv
  item_id, register, generator, condition, is_human, context, text, source_row, pair_id
Run:
  python make_formal_cleaned_quiz.py
  python ai_detection_llm.py --provider all --corpus formal_cleaned_quiz.csv \
         --out-dir formal_cleaned_results
"""
import os
import pandas as pd

from text_normalizer import normalize_formal

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
_ABS = os.path.join(_src, "1_data_collection", "llm_abstracts", "abstracts")

G56 = os.path.join(_ABS, "generated_three_prompt_formal_14_07_26_gpt-5.6_quiz100.csv")
SV = os.path.join(_ABS, "sv_abstracts_adversarial.csv")
MIS = os.path.join(_src, "2_text_analysis_scripts", "mistral_temperature_ab",
                   "generated_corpus_mistral_temps.csv")
OUT = os.path.join(_here, "formal_cleaned_quiz.csv")

CONDITIONS = ["baseline", "human_like", "detector_evasive"]
GPT52_COL = {c: f"Abstract_{c}" for c in CONDITIONS}
GPT56_COL = {c: f"Abstract_{c}" for c in CONDITIONS}


def _nonempty(v):
    s = str(v).strip()
    return bool(s) and s.lower() != "nan"


def main():
    g56 = pd.read_csv(G56, encoding="utf-8")
    sv = pd.read_csv(SV, encoding="utf-8")
    mis = pd.read_csv(MIS, encoding="utf-8")
    misf = mis[(mis["register"] == "formal") & (mis["temp"] == 1.0)]
    mis_lookup = {(int(r.doc_id), r.condition): r.text for r in misf.itertuples()}

    recs = []
    dropped_refusal = 0
    for row in g56.itertuples():
        r = int(row.quiz_source_row)
        pid = row.quiz_pair_id
        title = str(row.Title)

        def add(generator, condition, text, is_human=False):
            if not _nonempty(text):
                return False
            recs.append({
                "register": "formal", "generator": generator, "condition": condition,
                "is_human": is_human, "context": title,
                "text": normalize_formal(text, is_human=is_human),
                "source_row": r, "pair_id": pid,
            })
            return True

        # 1 human (untouched by normalize)
        add("human", "human", row.Abstract, is_human=True)
        # GPT-5.2 (3 conditions), from sv_abstracts by source row
        svrow = sv.iloc[r]
        for c in CONDITIONS:
            add("gpt-5.2", c, svrow[GPT52_COL[c]])
        # GPT-5.6 (3 conditions), from the backbone file
        for c in CONDITIONS:
            ok = add("gpt-5.6", c, getattr(row, GPT56_COL[c]))
            if not ok and c == "detector_evasive":
                dropped_refusal += 1
        # Mistral (3 conditions), by doc_id == source row
        for c in CONDITIONS:
            add("mistral", c, mis_lookup.get((r, c)))

    out = pd.DataFrame(
        [{"item_id": i + 1, **rec} for i, rec in enumerate(recs)],
        columns=["item_id", "register", "generator", "condition", "is_human",
                 "context", "text", "source_row", "pair_id"],
    )
    out.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Wrote {len(out)} items to {os.path.basename(OUT)}  "
          f"({dropped_refusal} GPT-5.6 detector_evasive refusals dropped)")
    print("  by generator x condition:")
    tab = out.groupby(["generator", "condition"]).size().unstack(fill_value=0)
    print(tab.to_string())
    # residual formatting audit (should be ~human-level everywhere)
    md = out.assign(md=out["text"].str.contains(r"[*#`]|Nyckelord", regex=True)) \
            .groupby("generator")["md"].mean()
    print("\n  residual markdown/Nyckelord rate after normalization:")
    for gen, v in md.items():
        print(f"    {gen:10s} {v:.0%}")


if __name__ == "__main__":
    main()
