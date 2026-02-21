"""
Base script for sending requests to AI-Sweden-Models/gpt-sw3-6.7b-v2.
Uses Hugging Face transformers; run on GPU if available.
"""

import torch
if not hasattr(torch, "Tensor") or getattr(torch, "__file__", None) is None:
    raise ImportError(
        "PyTorch is broken or not installed. Fix with:\n"
        "  python -m pip uninstall torch -y\n"
        "  python -m pip install torch"
    )
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "AI-Sweden-Models/gpt-sw3-6.7b-v2"


def load_model(device=None):
    """Load tokenizer and model. Uses CUDA if available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = model.to(device)
    return tokenizer, model, device


def generate(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 100,
    do_sample: bool = True,
    temperature: float = 0.7,
    **kwargs,
) -> str:
    """Send a prompt to the model and return the generated text (prompt + completion)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full


def main():
    tokenizer, model, device = load_model()
    prompt = "Träd är fina för att"
    print("Prompt:", prompt)
    response = generate(prompt, tokenizer, model, device, max_new_tokens=50)
    print("Response:", response)


if __name__ == "__main__":
    main()
