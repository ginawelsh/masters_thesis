"""
Informal: add LLM-generated comment column to reddit_comments.csv (SW3).
Reads 'question' per row, generates a short Swedish comment in response, writes to same CSV.
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

_data_dir = os.path.join(_root, "src", "data_collection")
csv_path = os.path.join(_data_dir, "reddit_comments.csv")
out_csv_path = os.path.join(_data_dir, "reddit_comments_sw3.csv")

MODEL_ID = "AI-Sweden-Models/gpt-sw3-356m"
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 256


def _get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return tokenizer, model


COMMENT_PROMPT = (
    "Svara på följande fråga med en kort, avslappnad svensk kommentar (som på ett forum). "
    "Skriv bara kommentaren.\n\nFråga: {question}\n\nKommentar:"
)


def generate_comment(question, tokenizer, model):
    prompt = COMMENT_PROMPT.format(question=question)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

    new_ids = output_ids[input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("Loading SW3 model and tokenizer...", flush=True)
    tokenizer, model = _get_model_and_tokenizer()
    model.eval()

    data = pd.read_csv(csv_path, encoding="utf-8")
    questions = list(data["question"])
    n = len(questions)

    data["Generated_SW3_Comment"] = pd.NA

    for i in range(n):
        q = questions[i]
        if pd.isna(q) or not str(q).strip():
            continue
        print(f"Processing {i + 1}/{n}...", end=" ", flush=True)
        try:
            data.loc[i, "Generated_SW3_Comment"] = generate_comment(str(q), tokenizer, model)
            print("done.")
        except Exception as e:
            print(f"Error: {e}")

    data.to_csv(out_csv_path, index=False, encoding="utf-8")
    print("Finished. Wrote", out_csv_path)
