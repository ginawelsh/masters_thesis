# LLM AI detection workflows
# LLM tool used for text generation: OpenAI
# LLM tool used for detection: Gemini

import os
import time
import pandas as pd
from google import genai
from google.genai import errors as genai_errors

# organise retry mechanism
MAX_RETRIES = 5
RETRY_BASE_SEC = 10

# load .env from project root (script is in src/llms/formal/)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# link to gemini API as client
client = genai.Client(api_key=_api_key)

abstracts_source = "list(abstracts file). needs to have both human and AI generated side by side. put in one col. there is a hidden col (h?)"
comments_source = "list(comments file). needs to concatenate human and AI into one col. hidden col: h? "

# NEED TO HAVE FILE OF ALL DATA - organised by: informal, formal. in the og document, there should be a column called human? with 1 and 0. 
# but this human? col should not be included in this script.

# set which csv file structure the new generated abstracts will appear in (added on)
csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_generated.csv")

# create new file with generated_abstracts col from Gemini
out_csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_gemini_2.csv")

# for bachelor's abstracts
def detect_llm_formal(abstract):
    prompt = f"Titta på det här abstraktet och avgör om det är skrivet av en AI eller en människa. Förklara din tankeprocess i din bedömning med 100–200 ord.: {abstract}."
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
    return None 

# for reddit comments
def detect_llm_informal(comment):
    prompt = f"Titta på det här kommentar och avgör om det är skrivet av en AI eller en människa. Förklara din tankeprocess i din bedömning med 100–200 ord.: {comment}."
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
    return None 


    def evaluate(results):
        pass


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