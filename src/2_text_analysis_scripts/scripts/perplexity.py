"""
Perplexity under a reference causal LM (a candidate AI-text-detection signal:
LLM-generated text often scores lower perplexity than human text under a
general-purpose LM, since it is drawn from that same distribution family).

Model: AI-Sweden-Models/gpt-sw3-356m -- the intended reference LM for a
Swedish-text thesis, now that access to the base (non-instruct) checkpoint has
cleared. Swap MODEL_NAME (or pass --model) to a different checkpoint if
needed; nothing else here is model-specific.

Long documents are scored with the sliding-window method from HuggingFace's
perplexity guide (https://huggingface.co/docs/transformers/perplexity) rather
than truncated, so a long abstract isn't scored on only its first ~1024 tokens.

Also reports burstiness (GPTZero-style): the standard deviation of per-sentence
perplexity within a document. Human text tends to swing between easy and hard
sentences more than LLM text, so burstiness is a second signal orthogonal to
mean perplexity. Sentence splitting uses a bare rule-based spaCy sentencizer
(no Swedish model download needed -- only sentence boundaries are used here).
Per-document sentences are scored in padded batches (not one-at-a-time), since
that loop is the dominant cost -- this is what actually benefits from a GPU.

Usage:
  python src/2_text_analysis_scripts/scripts/perplexity.py --dataset formal
  python src/2_text_analysis_scripts/scripts/perplexity.py --dataset informal --limit 20
  python src/2_text_analysis_scripts/scripts/perplexity.py --dataset informal --batch-size 32
"""
import argparse
import math
import os
import statistics
from contextlib import nullcontext

import pandas as pd
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import read_csv_robust, add_condition_arg, resolve_conditions, condition_tag

MODEL_NAME = "AI-Sweden-Models/gpt-sw3-356m"
MAX_LENGTH = 1024
STRIDE = 512
BATCH_SIZE = 16

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
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Sentences per batch for burstiness scoring")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    model.to(device)
    return tokenizer, model


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()


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
        with _autocast(device):
            out = model(window, labels=target)
        nlls.append(out.loss.float())
        prev_end = end
        if end == seq_len:
            break

    return float(torch.exp(torch.stack(nlls).mean())), seq_len


@torch.no_grad()
def _batch_sentence_ppls(sentences, tokenizer, model, device, max_length):
    """Perplexity for a list of sentences in one padded forward pass.

    Replaces one-sentence-per-forward-pass scoring (the dominant cost of
    burstiness, since it runs once per sentence per document): padding lets a
    whole batch share a single matmul, which is what actually makes a GPU
    (e.g. Colab) pay off here instead of just moving the same serial loop.
    """
    enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    if input_ids.size(1) < 2:
        return [None] * len(sentences)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    with _autocast(device):
        logits = model(input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(shift_labels.size())
    valid = (shift_labels != -100).float()
    n_valid = valid.sum(dim=1)
    per_seq_nll = (loss * valid).sum(dim=1) / n_valid.clamp(min=1)
    per_seq_ppl = torch.exp(per_seq_nll)

    out = []
    for i in range(len(sentences)):
        if n_valid[i].item() < 1:
            out.append(None)
            continue
        val = per_seq_ppl[i].item()
        out.append(val if not math.isnan(val) and not math.isinf(val) else None)
    return out


def sentence_burstiness(text, nlp, tokenizer, model, device, max_length, batch_size=BATCH_SIZE):
    """Std dev of per-sentence perplexity. None if fewer than 2 scoreable sentences."""
    sentences = [s.text.strip() for s in nlp(str(text).strip()).sents if s.text.strip()]
    if len(sentences) < 2:
        return None

    sent_ppls = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        sent_ppls.extend(p for p in _batch_sentence_ppls(chunk, tokenizer, model, device, max_length) if p is not None)
    if len(sent_ppls) < 2:
        return None
    return statistics.stdev(sent_ppls)


def score(text, tokenizer, model, device, nlp, max_length, stride, batch_size=BATCH_SIZE):
    if pd.isna(text) or not str(text).strip():
        return None
    result = perplexity(text, tokenizer, model, device, max_length, stride)
    if result is None:
        return None
    ppl, n_tokens = result
    if math.isnan(ppl) or math.isinf(ppl):
        return None
    burstiness = sentence_burstiness(text, nlp, tokenizer, model, device, max_length, batch_size)
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
                                  args.max_length, args.stride, args.batch_size))
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
            l = score(row.get(llm_col), tokenizer, model, device, nlp, args.max_length, args.stride, args.batch_size)
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
