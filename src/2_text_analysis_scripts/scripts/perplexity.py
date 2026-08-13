"""
Perplexity under a reference causal LM (a candidate AI-text-detection signal:
LLM-generated text often scores lower perplexity than human text under a
general-purpose LM, since it is drawn from that same distribution family).

Model: GPT-SW3 (AI-Sweden-Models/gpt-sw3-*) is the intended reference LM for a
Swedish-text thesis, but its base (non-instruct) checkpoints are currently
gated/pending for this account. MODEL_NAME defaults to facebook/xglm-564M --
ungated, multilingual, trained on CC100 (includes Swedish), and comparable in
size to gpt-sw3-356m. Swap MODEL_NAME (or pass --model) to a GPT-SW3 checkpoint
once access clears; nothing else here is model-specific.

Long documents are scored with the sliding-window method from HuggingFace's
perplexity guide (https://huggingface.co/docs/transformers/perplexity) rather
than truncated, so a long abstract isn't scored on only its first ~1024 tokens.

Also reports burstiness (GPTZero-style): the standard deviation of per-sentence
perplexity within a document. Human text tends to swing between easy and hard
sentences more than LLM text, so burstiness is a second signal orthogonal to
mean perplexity. Sentence splitting uses a bare rule-based spaCy sentencizer
(no Swedish model download needed -- only sentence boundaries are used here).

Usage:
  python src/2_text_analysis_scripts/scripts/perplexity.py --dataset formal
  python src/2_text_analysis_scripts/scripts/perplexity.py --dataset informal --limit 20
"""
import argparse
import math
import os
import statistics

import pandas as pd
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import read_csv_robust, add_condition_arg, resolve_conditions, condition_tag

MODEL_NAME = "facebook/xglm-564M"
MAX_LENGTH = 1024
STRIDE = 512

_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")


def parse_args():
    p = argparse.ArgumentParser(description="Perplexity under a reference causal LM")
    p.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    add_condition_arg(p)
    p.add_argument("--model", default=MODEL_NAME, help="HF causal-LM model id")
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto-detect)")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--limit", type=int, default=None,
                   help="Only score the first N rows (quick test run)")
    return p.parse_args()


def resolve_config(dataset):
    if dataset == "informal":
        return {
            "csv": os.path.join(_data, "llm_comments", "consolidated_informal_comments_adversarial.csv"),
            "human_col": "human_comment",
        }
    return {
        "csv": os.path.join(_data, "llm_abstracts", "abstracts", "sv_abstracts_adversarial.csv"),
        "human_col": "Abstract",
    }


def load_model(model_name, device):
    print(f"loading {model_name} on {device}…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model


def load_sentencizer():
    nlp = spacy.blank("sv")
    nlp.add_pipe("sentencizer")
    return nlp


@torch.no_grad()
def perplexity(text, tokenizer, model, device, max_length=MAX_LENGTH, stride=STRIDE):
    """Sliding-window perplexity: each window's target is masked to only the
    tokens not already scored by the previous window, so overlapping context
    doesn't get counted twice. Windows are averaged as in the HF recipe.
    """
    if pd.isna(text) or not str(text).strip():
        return None
    input_ids = tokenizer(str(text).strip(), return_tensors="pt")["input_ids"].to(device)
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return None

    nlls = []
    prev_end = 0
    for start in range(0, seq_len, stride):
        end = min(start + max_length, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, start:end]
        target = window.clone()
        target[:, :-trg_len] = -100
        out = model(window, labels=target)
        nlls.append(out.loss)
        prev_end = end
        if end == seq_len:
            break

    return float(torch.exp(torch.stack(nlls).mean())), seq_len


def sentence_burstiness(text, nlp, tokenizer, model, device, max_length, stride):
    """Std dev of per-sentence perplexity. None if fewer than 2 scoreable sentences."""
    sent_ppls = []
    for sent in nlp(str(text).strip()).sents:
        s = sent.text.strip()
        if not s:
            continue
        result = perplexity(s, tokenizer, model, device, max_length, stride)
        if result is not None and not math.isnan(result[0]) and not math.isinf(result[0]):
            sent_ppls.append(result[0])
    if len(sent_ppls) < 2:
        return None
    return statistics.stdev(sent_ppls)


def score(text, tokenizer, model, device, nlp, max_length, stride):
    if pd.isna(text) or not str(text).strip():
        return None
    result = perplexity(text, tokenizer, model, device, max_length, stride)
    if result is None:
        return None
    ppl, n_tokens = result
    if math.isnan(ppl) or math.isinf(ppl):
        return None
    burstiness = sentence_burstiness(text, nlp, tokenizer, model, device, max_length, stride)
    return {"perplexity": ppl, "n_tokens": n_tokens, "burstiness": burstiness}


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model, device)
    nlp = load_sentencizer()

    df = read_csv_robust(cfg["csv"])
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(df)

    # human side is identical across conditions -> score it once and reuse
    print("scoring human texts…", flush=True)
    human_feats = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  {i + 1}/{n}", end="\r", flush=True)
        human_feats.append(score(row.get(cfg["human_col"]), tokenizer, model, device, nlp,
                                  args.max_length, args.stride))
    print()

    for cond, llm_col in resolve_conditions(args.dataset, args.condition):
        if llm_col not in df.columns:
            print(f"  skipping condition '{cond}': column '{llm_col}' not found")
            continue
        tag = condition_tag(args.dataset, cond)
        print(f"scoring LLM texts [{cond or 'generated'}]…", flush=True)
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"  {i + 1}/{n}", end="\r", flush=True)
            h = human_feats[i]
            l = score(row.get(llm_col), tokenizer, model, device, nlp, args.max_length, args.stride)
            if h is None or l is None:
                continue
            rows.append({
                "human_perplexity": h["perplexity"], "human_n_tokens": h["n_tokens"],
                "human_burstiness": h["burstiness"],
                "llm_perplexity": l["perplexity"], "llm_n_tokens": l["n_tokens"],
                "llm_burstiness": l["burstiness"],
            })
        print()

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"perplexity_{tag}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{cond or 'generated'}] Wrote {len(out_df)} rows → {out_path}")
        if len(out_df):
            print(f"  mean perplexity   human={out_df['human_perplexity'].mean():.2f}  "
                  f"llm={out_df['llm_perplexity'].mean():.2f}")
            print(f"  mean burstiness   human={out_df['human_burstiness'].mean():.2f}  "
                  f"llm={out_df['llm_burstiness'].mean():.2f}")


if __name__ == "__main__":
    main()
