"""
Swedish generator smoke test.

Purpose: EuroEval tells you these models handle Swedish NLP tasks well, but it
does NOT test register naturalness (does the forum comment read like a real
forum poster? does the abstract sound like a real student?). This script closes
that gap: it generates a handful of items across every model x condition and
prints the Swedish side by side so you can READ it and judge fluency + register
+ evasive-condition compliance yourself.

It reuses your pipeline unchanged -- same build_prompt, same generate, same
byte-identical prompt contract. Adding Mistral / a real Qwen-Instruct is just
registry entries in generation_pipeline.py (Mistral is already registered there);
it does not touch prompt construction.

USAGE
-----
1. This file sits next to generation_pipeline.py (src/1_data_collection/).
2. Set env vars for whichever backends you're testing:
     OPENAI_API_KEY                      (gpt-5.2)
     MISTRAL_API_KEY                     (mistral; base_url + slug already default
                                          to OpenRouter in generation_pipeline.py --
                                          override with MISTRAL_BASE_URL / MISTRAL_MODEL)
     QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL   (optional; only if you test qwen-instruct)
   For OpenRouter, base_url = "https://openrouter.ai/api/v1" and the model
   strings are e.g. "mistralai/mistral-small-3.2-24b-instruct" and
   "qwen/qwen3-30b-a3b-instruct-2507" (confirm exact slugs with your provider).
3. SAMPLE_ITEMS below hold real items from the two corpora; swap in others freely.
4. Run:  python smoke_test.py
   Add --models / --conditions / --registers to narrow it (see bottom).

WHAT TO LOOK FOR WHILE READING
------------------------------
- Fluency parity: does any model's Swedish read visibly thinner/clumsier than
  the others? If so that's a confound, not a win.
- Register: formal = candidate-thesis abstract voice; informal = real Swedish
  forum-comment voice. Watch for a model that writes "essay" tone in a comment.
- Evasive compliance: on detector_aware / detector_evasive, did the model
  actually vary sentence length, drop the formulaic connectives, and avoid
  over-hedging? Smaller models sometimes ignore complex anti-detection prompts --
  that's exactly the behaviour you're measuring, so check it here.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Import the shared pipeline. It defines build_prompt, generate, MODELS,
# ModelSpec, OpenAICompatibleClient, GenParams, GenRecord.
# ---------------------------------------------------------------------------
PIPELINE_MODULE = "generation_pipeline"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find it regardless of cwd
pipe = importlib.import_module(PIPELINE_MODULE)

OpenAICompatibleClient = pipe.OpenAICompatibleClient
ModelSpec = pipe.ModelSpec
MODELS = pipe.MODELS


# ---------------------------------------------------------------------------
# Register the open generators for the test, if not already present. These are
# additive registry entries only -- no prompt logic changes. (Mistral is already
# registered in generation_pipeline.py, so that branch is a no-op here.)
# ---------------------------------------------------------------------------
def _register_open_models() -> None:
    if "mistral" not in MODELS:
        MODELS["mistral"] = ModelSpec(
            make_client=lambda: OpenAICompatibleClient(
                model=os.environ.get(
                    "MISTRAL_MODEL", "mistralai/mistral-small-3.2-24b-instruct"
                ),
                base_url=os.environ.get("MISTRAL_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key_env="MISTRAL_API_KEY",
            ),
        )
    # A real INSTRUCT Qwen -- NOT the -Base model. The -Base ranks well on
    # EuroEval but you'd never generate with it. Swap the default for whichever
    # instruct Qwen you actually deploy. (Optional; only used if you pass it.)
    if "qwen-instruct" not in MODELS:
        MODELS["qwen-instruct"] = ModelSpec(
            make_client=lambda: OpenAICompatibleClient(
                model=os.environ.get(
                    "QWEN_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"
                ),
                base_url=os.environ["QWEN_BASE_URL"],
                api_key_env="QWEN_API_KEY",
            ),
        )


# ---------------------------------------------------------------------------
# Sample items -- real content from the two corpora (informal question from the
# consolidated comments corpus; formal title/keywords from sv_abstracts). Keep it
# small (1-2 per register) so you can actually read every cell.
# ---------------------------------------------------------------------------
SAMPLE_ITEMS = {
    "formal": [
        {
            "id": "abs_001",
            "title": "Kraften av beröring: Beröring som komplementär metod för att minska stress på arbetsplatsen",
            "keywords": "fysisk beröring; massage; avslappning; arbetsrelaterad stress",
        },
    ],
    "informal": [
        {
            "id": "com_001",
            "question": (
                "Är det verkligen ingen kanal som visar Svensson Svensson och Wallace och "
                "Gromit ikväll? Jag tittar på TV.nu men finner inget."
            ),
        },
    ],
}

ALL_MODELS = ["gpt-5.2", "mistral"]
ALL_CONDITIONS = ["baseline", "human_like", "detector_aware", "detector_evasive"]
ALL_REGISTERS = ["formal", "informal"]


def _wrap(text: str, width: int = 88) -> str:
    out = []
    for para in text.splitlines() or [""]:
        out.append(textwrap.fill(para, width=width) if para.strip() else "")
    return "\n".join(out)


def run_smoke(models, conditions, registers, temps=None) -> None:
    """temps: optional list of temperatures. When given, every model is run once
    per temperature (a within-model A/B) so you can see how much sampling temp
    changes the Swedish -- otherwise the pipeline default (no override) is used,
    which is the byte-identical production contract."""
    _register_open_models()

    # (label, GenParams) pairs to run per model
    if temps:
        settings = [(f"temp={t}", pipe.GenParams(temperature=t)) for t in temps]
    else:
        settings = [("default", pipe.DEFAULT_PARAMS)]

    for register in registers:
        for item in SAMPLE_ITEMS.get(register, []):
            for condition in conditions:
                # Show the exact shared prompt once per (register, condition,
                # item) -- it's identical across models by contract.
                prompt = pipe.build_prompt(condition, register, item)
                header = f" {register} / {condition} / {item['id']} "
                print("\n" + "=" * 90)
                print(header.center(90, "="))
                print("=" * 90)
                print("PROMPT (identical for every model):")
                print(_wrap(prompt))
                print("-" * 90)

                for model in models:
                    for label, params in settings:
                        tag = model if label == "default" else f"{model} @ {label}"
                        print(f"\n>>> [{tag}]")
                        try:
                            rec = pipe.generate(model, condition, register, item, params)
                            print(_wrap(rec.output))
                        except KeyError:
                            print(f"    (model '{model}' not in registry -- skipped)")
                        except Exception as e:  # backend/env/auth errors
                            print(f"    (ERROR: {type(e).__name__}: {e})")
                print()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Swedish generator smoke test")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    help=f"subset of {ALL_MODELS}")
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                    help=f"subset of {ALL_CONDITIONS}")
    ap.add_argument("--registers", nargs="+", default=ALL_REGISTERS,
                    help=f"subset of {ALL_REGISTERS}")
    ap.add_argument("--temps", nargs="+", type=float, default=None,
                    help="optional temperatures to A/B per model, e.g. --temps 0.15 1.0 "
                         "(default: pipeline's no-override production contract)")
    args = ap.parse_args()

    run_smoke(args.models, args.conditions, args.registers, args.temps)
