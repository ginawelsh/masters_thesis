"""
Affective and expressive features:

  - Affective extremity: confidence score of non-neutral sentiment predictions.
    Texts predicted POSITIVE or NEGATIVE get their model confidence as the
    extremity score; NEUTRAL texts score 0.  Higher = more polar.
  - Subjectivity rate (per batch): fraction of texts classified as non-neutral.
  - Emotion categories (zero-shot): joy, anger, sadness, fear, surprise, disgust,
    neutral — using joeddav/xlm-roberta-large-xnli which supports Swedish directly.
    Skip with --skip-emotion to save ~10 min of inference time.

Sentiment model: KBLab/robust-swedish-sentiment-multiclass
Emotion model:   joeddav/xlm-roberta-large-xnli  (multilingual zero-shot NLI)

Informal register only: formal abstracts are near-uniformly neutral, so sentiment/emotion
carries almost no variance there and is not analysed for abstracts.

Usage:
  python src/2_text_analysis_scripts/affective_analysis.py --dataset informal
  python src/2_text_analysis_scripts/affective_analysis.py --dataset informal --skip-emotion
"""
import argparse
import os

import numpy as np
import pandas as pd

from data_utils import read_csv_robust, resolve_conditions, condition_tag

_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")

SENTIMENT_MODEL = "KBLab/robust-swedish-sentiment-multiclass"
EMOTION_MODEL = "joeddav/xlm-roberta-large-xnli"
EMOTION_LABELS = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]
MAX_LEN = 512
NEUTRAL_LABELS = {"NEUTRAL", "neutral", "NEU"}


def parse_args():
    p = argparse.ArgumentParser(description="Affective and expressive analysis (informal register only)")
    # Formal abstracts are near-uniformly neutral, so sentiment/emotion has almost no
    # variance there; this analysis is restricted to the informal register.
    p.add_argument("--dataset", choices=["informal"], default="informal")
    p.add_argument("--skip-emotion", action="store_true",
                   help="Skip zero-shot emotion classification (saves ~10 min)")
    return p.parse_args()


def resolve_config(dataset):
    return {
        "csv": os.path.join(_data, "llm_comments", "consolidated_informal_comments_JUN26.csv"),
        "human_col": "human_comment",
        "llm_col": "generated_comment",
    }


def _prep(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    return str(text).strip()[:MAX_LEN]


def run_sentiment(pipe, text):
    """Return (top_label, top_score, affective_extremity, is_subjective).

    affective_extremity = combined score of POSITIVE + NEGATIVE labels,
    i.e. 1 - neutral_score.  This gives meaningful variation even when the
    model predicts NEUTRAL for most texts.
    """
    t = _prep(text)
    if not t:
        return "", 0.0, 0.0, 0
    try:
        all_scores = pipe(t, truncation=True, max_length=MAX_LEN, top_k=None)
        scores = {item["label"].upper(): float(item["score"]) for item in all_scores}
        neutral_score = max(scores.get(lbl.upper(), 0.0) for lbl in NEUTRAL_LABELS)
        extremity = round(1.0 - neutral_score, 6)
        top = max(all_scores, key=lambda x: x["score"])
        label = top.get("label", "")
        score = float(top.get("score", 0.0))
        is_subj = int(label.upper() not in {l.upper() for l in NEUTRAL_LABELS} and label != "")
        return label, score, extremity, is_subj
    except Exception:
        return "", 0.0, 0.0, 0


def run_emotion_zsc(pipe, text):
    t = _prep(text)
    empty = {lb: 0.0 for lb in EMOTION_LABELS}
    if not t:
        return empty
    try:
        out = pipe(t, candidate_labels=EMOTION_LABELS, multi_label=False)
        return {lb: float(sc) for lb, sc in zip(out["labels"], out["scores"])}
    except Exception:
        return empty


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)

    from transformers import pipeline as hf_pipeline

    print("Loading sentiment model…", flush=True)
    sent_pipe = hf_pipeline("text-classification", model=SENTIMENT_MODEL)

    emo_pipe = None
    if not args.skip_emotion:
        print("Loading emotion/ZSC model (may take a minute)…", flush=True)
        emo_pipe = hf_pipeline("zero-shot-classification", model=EMOTION_MODEL)

    df = read_csv_robust(cfg["csv"])
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(df)

    def feats(text):
        """Sentiment + (optional) emotion metrics for one text, without a human_/llm_ prefix."""
        out = {}
        label, score, extremity, is_subjective = run_sentiment(sent_pipe, text)
        out["sentiment_label"] = label
        out["sentiment_score"] = score
        out["affective_extremity"] = extremity
        out["is_subjective"] = is_subjective
        if emo_pipe is not None:
            for emo_lbl, emo_sc in run_emotion_zsc(emo_pipe, text).items():
                out[f"emotion_{emo_lbl}"] = emo_sc
        return out

    # the human side is identical across conditions -> score it once and reuse
    print("scoring human texts…", flush=True)
    human_feats = []
    for i, row in df.iterrows():
        print(f"  {i + 1}/{n}", end="\r", flush=True)
        human_feats.append(feats(row.get(cfg["human_col"])))
    print()

    for cond, llm_col in resolve_conditions(args.dataset):
        if llm_col not in df.columns:
            print(f"  skipping condition '{cond}': column '{llm_col}' not found")
            continue
        tag = condition_tag(args.dataset, cond)
        print(f"scoring LLM texts [{cond or 'generated'}]…", flush=True)
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"  {i + 1}/{n}", end="\r", flush=True)
            out_row = {f"human_{k}": v for k, v in human_feats[i].items()}
            out_row.update({f"llm_{k}": v for k, v in feats(row.get(llm_col)).items()})
            rows.append(out_row)
        print()

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"affective_analysis_{tag}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{cond or 'generated'}] Wrote {len(out_df)} rows → {out_path}")

        numeric_df = out_df.select_dtypes(include=[float, int])
        print("Means (human | llm):")
        for col in sorted(c for c in numeric_df.columns if c.startswith("human_")):
            lcol = "llm_" + col[len("human_"):]
            if lcol in numeric_df:
                print(f"  {col[6:]:40s}  {numeric_df[col].mean():.4f}  |  {numeric_df[lcol].mean():.4f}")


if __name__ == "__main__":
    main()
