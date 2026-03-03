import os
import time
import pandas as pd
from google import genai
from google.genai import errors as genai_errors

# load .env from project root (script is in src/llms/formal/)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# Google GenAI (Gemini) uses GOOGLE_API_KEY — get a key from https://aistudio.google.com/apikey
_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise ValueError(
        "GOOGLE_API_KEY not set. Get a key from https://aistudio.google.com/apikey then:\n"
        "  set GOOGLE_API_KEY=your-key   (Windows, current session)\n"
        "  export GOOGLE_API_KEY=your-key   (Linux/macOS)\n"
        "Or add GOOGLE_API_KEY=your-key to the .env file in the project root."
    )

client = genai.Client(api_key=_api_key)

# load abstracts data (CSV for reading/writing with new column)
_data_dir = os.path.join(_root, "src", "data_collection", "formal")
abstract_data_path = os.path.join(_data_dir, "sv_data_collection_kws.csv")

# read CSV, gather titles and keywords
csv_file = pd.read_csv(abstract_data_path, encoding="utf-8")
titles = list(csv_file["Title"])  # title1, title2, title3...
keywords = list(csv_file["Keywords"])  # [kw1, kw2, kw3]...[kw1, kw2, kw3]....

# set which csv file structure the new generated abstracts will appear in (added on)
csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_generated.csv")

# create new file with generated_abstracts col from Gemini
out_csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_gemini_2.csv")

# LLM-generated abstract creation - NEED TITLE + KEY WORDS cols for prompt (same as OpenAI)
# Retry on 503 (model overload) with backoff
MAX_RETRIES = 5
RETRY_BASE_SEC = 10


def generate_abstract(title, keywords):
    prompt = f"Titta på titeln och nyckelorden och skapa en sammanfattning i kandidatuppsatsstil med dina egna ord: {title, keywords}."
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            return response.text
        except genai_errors.ServerError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f" server busy (503), retry in {wait}s...", flush=True)
            time.sleep(wait)
    return None  # unreachable

# run over all rows, add Generated_Gemini_Abstract column, save to out CSV
if __name__ == "__main__":
    # load CSV so we can add a new column and save back
    data_csv = pd.read_csv(abstract_data_path, encoding="utf-8")

    titles = list(csv_file["Title"])  # title1, title2, title3...
    keywords = list(csv_file["Keywords"])  # [kw1, kw2, kw3]...[kw1, kw2, kw3]....

    n_test_titles = len(titles) if titles else 0

    # add new column (empty for rows we don't process)
    data_csv["Generated_Gemini_Abstract_keywords"] = pd.NA

    if n_test_titles == 0:
        test_abstract = "Detta är ett kort testabstrakt."
        print("Processing 1/1...", flush=True)
        result = generate_abstract(test_abstract, "")
        print("Generated abstract:", result)
        data_csv.loc[0, "Generated_Gemini_Abstract"] = result
    else:
        for i in range(n_test_titles):
            print(f"Processing {i + 1}/{n_test_titles}...", end=" ", flush=True)
            result = generate_abstract(titles[i], keywords[i])
            print("done.")
            print("Generated:", result[:80] + "..." if len(result) > 80 else result)
            print("-" * 40)
            data_csv.loc[i, "Generated_Gemini_Abstract"] = result

    data_csv.to_csv(out_csv_path, index=False, encoding="utf-8")
    print("finished. Added column Generated_Gemini_Abstract to", out_csv_path)
