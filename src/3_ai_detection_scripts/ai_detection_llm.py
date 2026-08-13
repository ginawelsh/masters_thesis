"""LLM AI-text detection — reasoning models as detectors.

Runs a reasoning LLM as a zero-shot AI-vs-human classifier over a corpus CSV and
scores its accuracy overall and by register / prompt condition / generator.

Two corpora feed this (see the thesis's two detection types):
  - quiz_master.csv          (type 2, thesis-only) — the shared human-vs-LLM quiz;
                             humans and machines classify the SAME items.
  - a whole-corpus CSV       (type 1, paper+thesis) — larger, both generators;
                             built by make_detection_corpus.py.

Any CSV works as long as it has: register, condition, is_human, text
(optional: item_id, generator, context). Ground truth (is_human / condition /
generator) is used only for scoring — it is never sent to the model.

Providers:
  deepseek : deepseek-reasoner   reasoning LLM, OpenAI SDK (base_url api.deepseek.com)
  gemini   : gemini-3.6-flash    reasoning LLM, OpenAI SDK (Gemini OpenAI-compat endpoint)
  claude   : claude-opus-4-8     reasoning LLM, official Anthropic SDK, adaptive thinking
  gptzero  : gptzero-v2          commercial detector (REST) — independent baseline

Claude is called through the Anthropic SDK by design; DeepSeek/Gemini are separate
LLM providers reached through their own OpenAI-compatible access; GPTZero is a
commercial detector API (needs GPTZERO_API_KEY).

Robust to per-day quota caps: results are checkpointed and re-running RESUMES
(already-scored items are skipped, parse-fails are retried).

Usage:
  python ai_detection_llm.py --provider deepseek
  python ai_detection_llm.py --provider all --corpus quiz_master.csv
  python ai_detection_llm.py --provider gemini --corpus detection_corpus.csv --limit 50
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

# force UTF-8 stdout so Swedish text / ticks don't crash a cp1252 console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MAX_RETRIES = 5
RETRY_BASE_SEC = 10

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_script_dir))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# PROVIDERS — reasoning-model detectors
# ---------------------------------------------------------------------------
# Each entry: how to build a client and what model to call. DeepSeek and Gemini
# use the OpenAI SDK against their own endpoints; Claude uses the Anthropic SDK.
PROVIDERS = {
    "deepseek": {
        "kind": "openai",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "env": "DEEPSEEK_API_KEY",
        # deepseek-reasoner puts its (uncapped) chain-of-thought in a separate
        # field; max_tokens caps only the FINAL answer. 1024 was clipping the JSON
        # to empty/truncated -> UNKNOWN (counted wrong), so give it more headroom.
        # RAISED 4096 -> 8192 on 2026-08-01: 4096 still produced 8 UNKNOWNs in the
        # 300 Mistral items of quiz_followup.csv (0 in the original 800-item run).
        # This caps the ANSWER only, so it cannot change the judgement -- it just
        # stops a long reasoning trace truncating the JSON. Provenance note: the 698
        # already-scored items in that run were judged at 4096; only the 8 retried
        # UNKNOWNs use 8192, since the resume logic drops UNKNOWNs and re-asks them.
        "max_tokens": 8192,
    },
    "gemini": {
        # Pro flagship reasoning model — the fair peer to claude-opus-4-8 /
        # deepseek-reasoner. gemini-3.1-pro-preview is the newest gen-3.1 pro this key
        # can reach (gemini-2.5-pro / gemini-3-pro-preview 404 as unavailable to the
        # key). It occasionally 503s under load ("high demand") — the retry/resume
        # loop rides through that. Requires a paid (Tier 1+) key; on a free-tier key
        # the pro models 429, in which case fall back to "gemini-3.6-flash".
        "kind": "openai",
        "model": "gemini-3.1-pro-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env": "GEMINI_API_KEY",
        "reasoning_effort": "low",   # bound thinking budget so a full run isn't ~4h
    },
    "claude": {
        "kind": "anthropic",
        "model": "claude-opus-4-8",
        "env": "ANTHROPIC_API_KEY",
    },
    # Commercial detector (not a prompted LLM): returns a class directly.
    # Serves as the only detector NOT trained on / prompted about this corpus —
    # an independent, no-leakage baseline. English-trained; Swedish results are a
    # documented limitation. Short texts (many informal comments) may be rejected.
    "gptzero": {
        "kind": "gptzero",
        "model": "gptzero-v2",
        "url": "https://api.gptzero.me/v2/predict/text",
        "env": "GPTZERO_API_KEY",
    },
}


def build_caller(provider):
    """Return a `call(register, text) -> str` for the provider (str is a JSON blob
    in the shape parse_response expects), or raise with a clear reason.

    LLM providers build the Swedish detection prompt internally; the commercial
    detector (gptzero) sends the raw text and maps its class into the same shape.
    """
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["env"])
    if not key:
        raise RuntimeError(f"{cfg['env']} is not set in .env — cannot run {provider}.")

    if cfg["kind"] == "openai":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed (pip install openai)") from e
        client = OpenAI(api_key=key, base_url=cfg["base_url"])
        model = cfg["model"]
        # Optional thinking-budget cap. The gen-3.1 pro reasoning model otherwise
        # spends minutes/item; reasoning_effort="low" keeps the same model but bounds
        # its thinking so a full corpus finishes in a reasonable time. Only sent when
        # the provider sets it (DeepSeek-reasoner paces itself and takes no such flag).
        extra = {}
        if cfg.get("reasoning_effort"):
            extra["reasoning_effort"] = cfg["reasoning_effort"]
        max_tokens = cfg.get("max_tokens", 1024)   # per-provider answer budget

        def call(register, text):
            # no temperature: reasoning models pace their own sampling
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(register, text)}],
                max_tokens=max_tokens,
                **extra,
            )
            return resp.choices[0].message.content or ""
        return call

    if cfg["kind"] == "anthropic":
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic package not installed (pip install anthropic)") from e
        client = anthropic.Anthropic(api_key=key)
        model = cfg["model"]

        def call(register, text):
            # Adaptive thinking: the detector reasons about the text before deciding.
            resp = client.messages.create(
                model=model,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": build_prompt(register, text)}],
            )
            # final answer is in the text block(s); thinking blocks are separate
            return "".join(b.text for b in resp.content if b.type == "text")
        return call

    if cfg["kind"] == "gptzero":
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("requests package not installed (pip install requests)") from e
        url = cfg["url"]
        headers = {"x-api-key": key, "Content-Type": "application/json",
                   "Accept": "application/json"}

        def call(register, text):
            # NOTE: verify the exact request/response shape against current GPTZero
            # API docs — the `multilingual` flag and field names occasionally change.
            r = requests.post(url, headers=headers,
                              json={"document": text, "multilingual": True}, timeout=90)
            if r.status_code == 400:
                # per-item bad input (e.g. text too short) — record UNKNOWN, don't halt
                return json.dumps({"classification": "UNKNOWN", "confidence": 0,
                                   "reasoning": f"gptzero 400: {r.text[:150]}"})
            r.raise_for_status()  # 401/403/429/5xx -> raise -> retry, then halt+resume
            doc = r.json()["documents"][0]
            raw_cls = str(doc.get("predicted_class", "")).lower()
            prob = doc.get("completely_generated_prob",
                           doc.get("average_generated_prob", 0.0)) or 0.0
            if raw_cls == "human":
                cls = "MÄNNISKA"
            elif raw_cls in ("ai", "mixed"):        # "mixed" contains AI -> count as AI
                cls = "AI"
            else:
                cls = "UNKNOWN"
            return json.dumps({"classification": cls, "confidence": int(round(prob * 100)),
                               "reasoning": f"gptzero predicted_class={raw_cls}, "
                                            f"generated_prob={prob:.3f}"})
        return call

    raise RuntimeError(f"unknown provider kind {cfg['kind']!r}")


def call_with_retry(call, *args):
    for attempt in range(MAX_RETRIES):
        try:
            return call(*args)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f"    API error ({e}); retry in {wait}s...", flush=True)
            time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# PROMPTS (Swedish; ground truth is never included)
# ---------------------------------------------------------------------------
def build_prompt(register, text):
    if register == "formal":
        return (
            "Du deltar i en studie som handlar om att identifiera AI-genererad text på svenska.\n\n"
            "Analysera följande kandidatuppsatsabstrakt och avgör om det är skrivet av en AI "
            "eller av en människa. Titta på saker som språklig variation, formuleringsval, "
            "meningsbyggnad, och om texten känns autentiskt akademisk eller mallartad.\n\n"
            f"Abstrakt:\n{text}\n\n"
            "Svara med JSON i exakt detta format (ingen annan text utanför JSON-blocket):\n"
            '{"classification": "AI" or "MÄNNISKA", "confidence": <0-100>, '
            '"reasoning": "<kort förklaring på 50-100 ord>"}'
        )
    return (
        "Du deltar i en studie som handlar om att identifiera AI-genererad text på svenska.\n\n"
        "Analysera följande forumkommentar och avgör om den är skriven av en AI "
        "eller av en människa. Titta på saker som informellt/vardagligt språk, stavfel, "
        "personliga referenser, naturlig röst, och om texten känns genuin eller generisk.\n\n"
        f"Kommentar:\n{text}\n\n"
        "Svara med JSON i exakt detta format (ingen annan text utanför JSON-blocket):\n"
        '{"classification": "AI" or "MÄNNISKA", "confidence": <0-100>, '
        '"reasoning": "<kort förklaring på 50-100 ord>"}'
    )


def parse_response(raw):
    """Return (classification in {AI,MÄNNISKA,UNKNOWN}, confidence:int, reasoning:str)."""
    text = (raw or "").strip()
    if "```" in text:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    try:
        data = json.loads(text)
        cls = str(data.get("classification", "UNKNOWN")).upper()
        if "MÄNNISKA" in cls or "MANNISKA" in cls or "HUMAN" in cls:
            cls = "MÄNNISKA"
        elif "AI" in cls:
            cls = "AI"
        else:
            cls = "UNKNOWN"
        return cls, int(data.get("confidence", 50)), str(data.get("reasoning", ""))[:400]
    except (json.JSONDecodeError, ValueError, TypeError):
        upper = text.upper()
        if "MÄNNISKA" in upper or "MANNISKA" in upper or "HUMAN" in upper:
            return "MÄNNISKA", 50, text[:200]
        if "AI" in upper:
            return "AI", 50, text[:200]
        return "UNKNOWN", 0, text[:200]


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------
def _acc(sub):
    return (sum(r["correct"] for r in sub) / len(sub)) if sub else 0.0


def report(results):
    n = len(results)
    if not n:
        print("No results.")
        return
    print(f"\nOverall accuracy : {_acc(results):.1%}  ({sum(r['correct'] for r in results)}/{n})")

    # AI as the positive class
    tp = sum(1 for r in results if not r["is_human"] and r["predicted"] == "AI")
    fp = sum(1 for r in results if r["is_human"] and r["predicted"] == "AI")
    tn = sum(1 for r in results if r["is_human"] and r["predicted"] == "MÄNNISKA")
    fn = sum(1 for r in results if not r["is_human"] and r["predicted"] == "MÄNNISKA")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    print(f"AI precision/recall/F1 : {prec:.1%} / {rec:.1%} / {f1:.3f}   "
          f"(TP={tp} FP={fp} TN={tn} FN={fn})")

    unknown = sum(1 for r in results if r["predicted"] == "UNKNOWN")
    if unknown:
        print(f"Unparseable responses  : {unknown}")

    for dim in ("register", "generator", "condition"):
        vals = sorted({r.get(dim, "n/a") for r in results})
        if len(vals) <= 1 and vals == ["n/a"]:
            continue
        print(f"\nBy {dim}:")
        for v in vals:
            sub = [r for r in results if r.get(dim, "n/a") == v]
            tag = ("called human" if v == "human"
                   else "detected as AI" if dim in ("generator", "condition") and v != "n/a"
                   else "correct")
            print(f"  {str(v):16s} {_acc(sub):6.1%}  ({tag}, n={len(sub)})")


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def _save(results, out_path):
    df = pd.DataFrame(results)
    if "text" in df.columns:                       # keep the saved text short
        df["text"] = df["text"].astype(str).str.slice(0, 120) + "..."
    df.to_csv(out_path, index=False, encoding="utf-8")


def run_provider(provider, rows, out_dir):
    print("=" * 64)
    print(f"PROVIDER: {provider}  (model {PROVIDERS[provider]['model']})")
    print("=" * 64)
    try:
        call = build_caller(provider)
    except RuntimeError as e:
        print(f"[skip] {e}")
        return None

    out_path = os.path.join(out_dir, f"detection_results_{provider}.csv")

    # RESUME: keep already-scored items so a per-day quota cap (e.g. Gemini
    # free tier = 20 req/day) can be worked across sessions without redoing.
    results = []
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, encoding="utf-8")
        prev = prev[prev.get("predicted", "UNKNOWN") != "UNKNOWN"]  # re-try parse-fails
        results = prev.to_dict("records")
        done = {str(r.get("item_id")) for r in results}
        if done:
            print(f"[resume] {len(done)} items already scored in "
                  f"{os.path.basename(out_path)} — skipping those.")

    todo = [r for r in rows if str(r["item_id"]) not in done]
    print(f"[run] {len(todo)} of {len(rows)} items to classify.")

    interrupted = None
    for i, row in enumerate(todo, 1):
        try:
            raw = call_with_retry(call, row["register"], row["text"])
        except Exception as e:                      # quota/network exhaustion, etc.
            interrupted = e
            print(f"\n[stop] API failure after {i - 1} new items: "
                  f"{str(e)[:160]}\n  Partial results are saved; re-run to resume.")
            break
        cls, conf, reasoning = parse_response(raw)
        # UNKNOWN is a non-decision (unparseable) — never score it as a correct
        # detection; count it wrong so parse failures can't inflate accuracy.
        correct = False if cls == "UNKNOWN" else ((cls == "MÄNNISKA") == row["is_human"])
        results.append({**row, "predicted": cls, "confidence": conf,
                        "reasoning": reasoning, "correct": correct})
        tick = "OK " if correct else "XX "
        print(f"[{i:03d}/{len(todo)}] {tick} {row['register'][:4]}/{row.get('condition','?'):16s} "
              f"truth={'H' if row['is_human'] else 'AI'} pred={cls} ({conf})", flush=True)
        if i % 10 == 0:                             # checkpoint so a crash can't wipe progress
            _save(results, out_path)

    _save(results, out_path)
    print(f"\nSaved -> {out_path}  ({len(results)}/{len(rows)} items scored)")
    if len(results) == len(rows):
        report(results)
    else:
        print("[note] corpus not fully scored yet — re-run to finish before scoring.")
    return results


def load_corpus(path, limit=None):
    df = pd.read_csv(path, encoding="utf-8")
    required = {"register", "is_human", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"corpus {path} missing columns: {missing}")
    df["is_human"] = df["is_human"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "item_id": r.get("item_id", ""),
            "register": str(r["register"]),
            "generator": str(r.get("generator", "n/a")),
            "condition": str(r.get("condition", "n/a")),
            "is_human": bool(r["is_human"]),
            "text": str(r["text"]),
        })
    if limit:
        rows = rows[:limit]
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="LLM (reasoning-model) AI-text detector")
    p.add_argument("--provider", choices=list(PROVIDERS) + ["all"], default="deepseek")
    p.add_argument("--corpus", default=os.path.join(_script_dir, "quiz_master.csv"),
                   help="corpus CSV (default: quiz_master.csv)")
    p.add_argument("--limit", type=int, default=None, help="only classify the first N items (testing)")
    p.add_argument("--model", default=None,
                   help="override the provider's model id (e.g. gemini-3.6-flash when the "
                        "pro model is quota-blocked). Only valid with a single --provider, "
                        "since it applies to that provider's config.")
    p.add_argument("--out-dir", default=_script_dir,
                   help="directory for detection_results_<provider>.csv (default: script "
                        "dir). Use a separate dir per corpus so runs don't clobber each "
                        "other (e.g. gpt52_run_results/, formal_cleaned_results/).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model and args.provider == "all":
        raise SystemExit("--model can only be used with a single --provider, not 'all'.")
    if args.model:
        PROVIDERS[args.provider]["model"] = args.model
        print(f"[override] {args.provider} model -> {args.model}")
    rows = load_corpus(args.corpus, args.limit)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loaded {len(rows)} items from {os.path.basename(args.corpus)}  ->  {out_dir}")
    providers = list(PROVIDERS) if args.provider == "all" else [args.provider]
    for prov in providers:
        run_provider(prov, rows, out_dir)
