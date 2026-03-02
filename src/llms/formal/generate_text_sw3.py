import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# project root and paths (same structure as OpenAI/Gemini scripts)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

csv_path = os.path.join(_root, "src", "data_collection", "sv_data_collection_csv.csv")
out_csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_sw3.csv")

# model config: leave room for generation (many SW3 models have 2048 context)
MODEL_ID = "AI-Sweden-Models/gpt-sw3-356m"
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 512


def _get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return tokenizer, model


# LLM-generated abstract creation (same prompt as OpenAI/Gemini)
def generate_abstract(abstract, tokenizer, model):
    prompt = f"Titta på det här abstraktet och generera ett nytt med egna ord, av samma längd: {abstract}"
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

    # decode only the newly generated part (strip prompt)
    new_ids = output_ids[input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("Loading SW3 model and tokenizer...", flush=True)
    tokenizer, model = _get_model_and_tokenizer()
    model.eval()

    data_csv = pd.read_csv(csv_path, encoding="utf-8")
    abstracts_csv = list(data_csv["Abstract"])
    n_test = len(abstracts_csv) if abstracts_csv else 0

    data_csv["Generated_SW3_Abstract"] = pd.NA

    if n_test == 0:
        test_abstract = "Detta är ett kort testabstrakt."
        print("Processing 1/1...", flush=True)
        result = generate_abstract(test_abstract, tokenizer, model)
        print("Generated abstract:", result)
        data_csv.loc[0, "Generated_SW3_Abstract"] = result
    else:
        for i in range(n_test):
            print(f"Processing {i + 1}/{n_test}...", end=" ", flush=True)
            result = generate_abstract(abstracts_csv[i], tokenizer, model)
            print("done.")
            print("Generated:", result[:80] + "..." if len(result) > 80 else result)
            print("-" * 40)
            data_csv.loc[i, "Generated_SW3_Abstract"] = result

    data_csv.to_csv(out_csv_path, index=False, encoding="utf-8")
    print("finished. Added column Generated_SW3_Abstract to", out_csv_path)
