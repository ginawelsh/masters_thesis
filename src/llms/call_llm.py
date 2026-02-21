"""
Base script for calling LLMs. Single entry point; supports local (e.g. GPT-Sw3)
and API backends. Use run_id to label runs (e.g. "gpt-5-2025-08-07 (zero-shot, val)").
"""

from __future__ import annotations

import os
from typing import Any


def call_llm(
    prompt: str,
    model: str = "gpt-sw3-6.7b-v2",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    run_id: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Call an LLM with the given prompt.

    Args:
        prompt: Input text for the model.
        model: Model identifier. Supported: "gpt-sw3-6.7b-v2" (local),
                "openai" or "gpt-4" / "gpt-5" (API, requires OPENAI_API_KEY).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0–2).
        run_id: Optional label for this run (e.g. "gpt-5-2025-08-07 (zero-shot, val)").
        **kwargs: Passed to the backend (e.g. do_sample for local).

    Returns:
        Generated text (may include prompt depending on backend).
    """
    if run_id:
        print(f"[{run_id}] calling {model}...")
    model_key = model.lower().strip()
    if "gpt-sw3" in model_key or model_key == "gpt-sw3-6.7b-v2":
        return _call_gpt_sw3(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
    if model_key in ("openai", "gpt-4", "gpt-5", "gpt-4o", "gpt-4o-mini"):
        return _call_openai(
            prompt,
            model=model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
    raise ValueError(
        f"Unknown model: {model}. Use 'gpt-sw3-6.7b-v2' or an OpenAI model (e.g. 'gpt-4', 'gpt-5')."
    )


def _call_gpt_sw3(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """Local GPT-Sw3 via transformers."""
    from run_gpt_sw3 import load_model, generate

    tokenizer, model, device = load_model()
    return generate(
        prompt,
        tokenizer,
        model,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    )


def _call_openai(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """OpenAI-compatible API (OpenAI, Azure, or other)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI API calls require: pip install openai\n"
            "Set OPENAI_API_KEY (or OPENAI_BASE_URL for other endpoints)."
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")  # e.g. for Azure or local
    if not api_key and not base_url:
        raise ValueError(
            "Set OPENAI_API_KEY in the environment for API-backed models."
        )
    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    # Use model as-is (e.g. "gpt-5", "gpt-5-2025-08-07"); default for generic "openai"
    model_id = model if model != "openai" else "gpt-4o"
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Call an LLM with a prompt.")
    p.add_argument("prompt", nargs="?", default="Träd är fina för att", help="Input prompt")
    p.add_argument("--model", "-m", default="gpt-sw3-6.7b-v2", help="Model: gpt-sw3-6.7b-v2 or openai name")
    p.add_argument("--max-tokens", "-n", type=int, default=100, help="Max new tokens")
    p.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--run-id", "-r", default=None, help='Run label e.g. "gpt-5-2025-08-07 (zero-shot, val)"')
    args = p.parse_args()

    out = call_llm(
        args.prompt,
        model=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        run_id=args.run_id,
    )
    print(out)


if __name__ == "__main__":
    main()
