"""Estimate a model's REFUSAL and MARKDOWN rate per (register, condition) before
committing to a full generation run.

Why
---
The smoke test showed GPT-5.6 declining the formal detector_evasive prompt:
  "Jag kan inte hjälpa till att kringgå AI-detektering. Däremot kan jag skriva ..."
and then writing a NON-evasive abstract. Stored unchecked that is doubly wrong: the
cell is mislabelled, and the refusal sentence is a blatant AI tell for any judge.

GPT-5.2 does it too -- 5 of 170 abstracts in the shipped corpus open with a refusal or
a "Jag kan hjälpa dig att skriva ..." meta-preamble. So this is not a 5.6 quirk; it is
a rate that has to be measured per model x condition before you can trust a corpus.

One sample proves nothing at temperature 1.0. This script runs N real generations over
N different source documents and reports the rate, so you can decide between
"detect and retry" (low rate) and "this condition is not viable for this model"
(high rate -- itself a publishable finding).

It writes NOTHING into any corpus. Output is a report plus an evidence CSV.

Usage
  python refusal_probe.py                                  # gpt-5.6 formal evasive, n=20
  python refusal_probe.py --models gpt-5.2 gpt-5.6 --n 15
  python refusal_probe.py --registers formal informal --conditions baseline human_like detector_evasive
  python refusal_probe.py --n 30 --out probe_5_6.csv

Cost: n x len(models) x len(registers) x len(conditions) calls. The default is 20.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generation_pipeline as pipe  # noqa: E402

_dir = os.path.dirname(os.path.abspath(__file__))
FORMAL_CSV = os.path.join(_dir, "llm_abstracts", "abstracts", "sv_abstracts_openai_2.csv")
INFORMAL_CSV = os.path.join(_dir, "llm_comments",
                            "consolidated_informal_comments_adversarial.csv")


def load_items(register, n, seed):
    """n real source items from the corpus, so the estimate is representative."""
    if register == "formal":
        d = pd.read_csv(FORMAL_CSV, encoding="utf-8")
        d = d[d["Title"].notna()].sample(n=min(n, len(d)), random_state=seed)
        return [{"id": int(i), "title": r["Title"], "keywords": r.get("Keywords", "")}
                for i, r in d.iterrows()]
    d = pd.read_csv(INFORMAL_CSV, encoding="utf-8")
    d = d[d["question"].notna()].sample(n=min(n, len(d)), random_state=seed)
    return [{"id": int(i), "question": r["question"]} for i, r in d.iterrows()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gpt-5.6"])
    ap.add_argument("--registers", nargs="+", default=["formal"],
                    choices=["formal", "informal"])
    ap.add_argument("--conditions", nargs="+", default=["detector_evasive"],
                    choices=list(pipe.CONDITIONS))
    ap.add_argument("--n", type=int, default=20, help="documents per cell (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_dir, "refusal_probe_results.csv"))
    args = ap.parse_args()

    for m in args.models:
        pipe.get_client(m)  # fail fast on unknown model / missing key

    total = args.n * len(args.models) * len(args.registers) * len(args.conditions)
    print(f"Probing {total} generations "
          f"({args.n} docs x {len(args.models)} models x {len(args.registers)} registers "
          f"x {len(args.conditions)} conditions)\n")

    tasks = []
    for register in args.registers:
        items = load_items(register, args.n, args.seed)
        for model in args.models:
            for condition in args.conditions:
                for item in items:
                    tasks.append((model, register, condition, item))

    rows = []

    def _work(t):
        model, register, condition, item = t
        try:
            out = pipe.generate(model, condition, register, item).output
            err = None
        except Exception as e:
            out, err = "", f"{type(e).__name__}: {e}"
        return {
            "model": model, "register": register, "condition": condition,
            "doc_id": item["id"],
            "error": err,
            "refusal": pipe.looks_like_refusal(out) if out else None,
            "markdown_n": pipe.count_markdown(out) if out else 0,
            "chars": len(out),
            "text": out,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(_work, t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'model':10} {'register':9} {'condition':17} {'n':>4} {'refuse':>7} "
          f"{'md':>7} {'err':>4} {'chars':>7}")
    print("-" * 78)
    for (m, reg, cond), g in df.groupby(["model", "register", "condition"]):
        n = len(g)
        nref = int(g["refusal"].notna().sum())
        nmd = int((g["markdown_n"] > 0).sum())
        nerr = int(g["error"].notna().sum())
        print(f"{m:10} {reg:9} {cond:17} {n:4} "
              f"{nref:3}/{n:<3} {nmd:3}/{n:<3} {nerr:4} {g['chars'].mean():7.0f}")
    print("=" * 78)

    ref = df[df["refusal"].notna()]
    if len(ref):
        print(f"\n{len(ref)} refusal(s). Examples:")
        for _, r in ref.head(4).iterrows():
            print(f"\n  [{r['model']} / {r['register']} / {r['condition']} "
                  f"doc {r['doc_id']}]  {r['refusal']}")
            print("   " + str(r["text"])[:220].replace("\n", " | "))
        print("\nInterpretation:")
        print("  <5%   detect-and-retry is fine (the guard in the generation scripts")
        print("        already does this; refusals are never cached)")
        print("  5-20% viable but report the rate as a limitation")
        print("  >20%  the condition is not reliably obtainable from this model --")
        print("        treat that as a finding, not something to retry around")
    else:
        print("\nNo refusals detected in this sample.")

    md = df[df["markdown_n"] > 0]
    if len(md):
        print(f"\n{len(md)}/{len(df)} outputs contain markdown "
              f"(stripped automatically at write time by strip_markdown()).")
    print(f"\nEvidence CSV -> {args.out}")


if __name__ == "__main__":
    main()
