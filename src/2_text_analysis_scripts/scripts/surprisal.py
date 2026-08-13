"""
Token-level surprisal under a reference causal LM (a candidate AI-text-detection
signal, complementary to perplexity.py): the same per-token negative log-likelihoods
that ppl is an average of are also examined directly, since their distribution --
tails, spread, spikes -- can separate human and LLM text even when the mean
(perplexity) does not.

Model: EuroLLM-1.7B (utter-project/EuroLLM-1.7B), a multilingual model with
Swedish coverage, comparable in role to the reference LM in perplexity.py.

Usage:
  python src/2_text_analysis_scripts/scripts/surprisal.py --dataset formal
  python src/2_text_analysis_scripts/scripts/surprisal.py --dataset informal --limit 20
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import read_csv_robust, add_condition_arg, resolve_conditions, condition_tag

MODEL_NAME = "utter-project/EuroLLM-1.7B"
MAX_LENGTH = 4096

_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")

FEATURE_KEYS = ["n_tokens", "mean_surprisal", "ppl", "std_surprisal",
                "max_surprisal", "p90_surprisal", "p95_surprisal"]


def parse_args():
    p = argparse.ArgumentParser(description="Token-level surprisal under a reference causal LM")
    p.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    add_condition_arg(p)
    p.add_argument("--model", default=MODEL_NAME, help="HF causal-LM model id")
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto-detect)")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--threshold", type=float, default=None,
                   help="Surprisal threshold in nats; adds frac_above_thr feature")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    )
    model.eval()
    model.to(device)
    return tokenizer, model


@torch.no_grad()
def surprisal_features(texts, tokenizer, model, device, max_length, threshold=None):
    """One feature dict per text. Surprisals are in nats (natural log), so
    ppl = exp(mean_surprisal).
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to(device)
    ids, mask = enc.input_ids, enc.attention_mask

    logits = model(ids, attention_mask=mask).logits
    sl = logits[:, :-1].contiguous()          # predict token t from <t
    lb = ids[:, 1:].contiguous()
    valid = mask[:, 1:].bool()                 # True where the target is real

    # per-token surprisal (NLL), shape (batch, seq-1)
    surp = F.cross_entropy(
        sl.transpose(1, 2), lb, reduction="none")
    surp = surp.masked_fill(~valid, float("nan"))  # blank out padding

    out = []
    for row, m in zip(surp, valid):
        s = row[m]                             # this text's real surprisals
        n = s.numel()
        feats = {
            "n_tokens": int(n),
            "mean_surprisal": s.mean().item(),
            "ppl": torch.exp(s.mean()).item(),
            "std_surprisal": s.std(unbiased=True).item() if n > 1 else 0.0,
            "max_surprisal": s.max().item(),
            "p90_surprisal": torch.quantile(s, 0.90).item(),
            "p95_surprisal": torch.quantile(s, 0.95).item(),
        }
        if threshold is not None:
            feats["frac_above_thr"] = (s > threshold).float().mean().item()
        out.append(feats)
    return out


def score(text, tokenizer, model, device, max_length, threshold):
    if pd.isna(text) or not str(text).strip():
        return None
    feats = surprisal_features([str(text).strip()], tokenizer, model, device, max_length, threshold)[0]
    if feats["n_tokens"] < 2:
        return None
    return feats


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model, device)

    df = read_csv_robust(cfg["csv"])
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(df)

    keys = list(FEATURE_KEYS) + (["frac_above_thr"] if args.threshold is not None else [])

    # human side is identical across conditions -> score it once and reuse
    print("scoring human texts…", flush=True)
    human_feats = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  {i + 1}/{n}", end="\r", flush=True)
        human_feats.append(score(row.get(cfg["human_col"]), tokenizer, model, device,
                                  args.max_length, args.threshold))
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
            l = score(row.get(llm_col), tokenizer, model, device, args.max_length, args.threshold)
            if h is None or l is None:
                continue
            r = {}
            for k in keys:
                r[f"human_{k}"] = h[k]
                r[f"llm_{k}"] = l[k]
            rows.append(r)
        print()

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"surprisal_{tag}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{cond or 'generated'}] Wrote {len(out_df)} rows → {out_path}")
        if len(out_df):
            print(f"  mean surprisal   human={out_df['human_mean_surprisal'].mean():.3f}  "
                  f"llm={out_df['llm_mean_surprisal'].mean():.3f}")
            print(f"  mean ppl         human={out_df['human_ppl'].mean():.2f}  "
                  f"llm={out_df['llm_ppl'].mean():.2f}")


if __name__ == "__main__":
    main()
