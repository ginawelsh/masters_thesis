import os
import pandas as pd
import openai
from openai import OpenAI

# load .env from project root (script is in src/llms/formal/)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# check openai api_key is set
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. In this terminal run:\n"
        "  set OPENAI_API_KEY=sk-your-key   (Windows, current session only)\n"
        "  export OPENAI_API_KEY=sk-your-key   (Linux/macOS)\n"
        "If you used setx, open a new terminal so it takes effect, or add OPENAI_API_KEY to a .env file in the project root."
    )
client = OpenAI(api_key=_api_key)

# load abstracts data (CSV for reading/writing with new column)
_data_dir = os.path.join(_root, "src", "data_collection", "formal")
abstract_data_path = os.path.join(_data_dir, "sv_data_collection_kws.csv")

# read CSV, gather titles and keywords
csv_file = pd.read_csv(abstract_data_path, encoding="utf-8")
titles = list(csv_file["Title"]) # title1, title2, title3...
keywords = list(csv_file["Keywords"]) # [kw1, kw2, kw3]...[kw1, kw2, kw3]....

# set which csv file structure the new generated abstracts will appear in (added on)
csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_generated.csv")

# create new file with generated_abstracts col from openai
out_csv_path = os.path.join(_root, "src", "data_collection", "sv_abstracts_openai_2.csv")

# LLM-generated abstract creation - NEED TITLE + KEY WORDS cols for prompt
def generate_abstract(title, keywords):
    response = client.chat.completions.create(model="gpt-5.2", messages=[{"role": "user", "content": \
    f"Titta på titeln och nyckelorden och skapa en sammanfattning i kandidatuppsatsstil med dina egna ord: {title, keywords}."}],)
    return response.choices[0].message.content

# simple test: call generate_abstract to verify API works (with progress)
if __name__ == "__main__":
    # load CSV so we can add a new column and save back
    data_csv = pd.read_csv(abstract_data_path, encoding="utf-8")

    abstracts_csv = list(data_csv["Abstract"])
    titles = list(csv_file["Title"]) # title1, title2, title3...
    keywords = list(csv_file["Keywords"]) # [kw1, kw2, kw3]...[kw1, kw2, kw3]....

    n_test_titles = len(titles) if titles else 0

    # add new column (empty for rows we don't process)
    data_csv["Generated_OpenAI_Abstract_keywords"] = pd.NA

    if n_test_titles == 0:
        test_abstract = "Detta är ett kort testabstrakt."
        print("Processing 1/1...", flush=True)
        result = generate_abstract(test_abstract)
        print("Generated abstract:", result)
        data_csv.loc[0, "Generated_OpenAI_Abstract"] = result
    else:
        for i in range(n_test_titles):
            print(f"Processing {i + 1}/{n_test_titles}...", end=" ", flush=True)
            result = generate_abstract(titles[i], keywords[i])
            print("done.")
            print("Generated:", result[:80] + "..." if len(result) > 80 else result)
            print("-" * 40)
            data_csv.loc[i, "Generated_OpenAI_Abstract"] = result

    data_csv.to_csv(out_csv_path, index=False, encoding="utf-8")
    print("finished. Added column Generated_OpenAI_Abstract to", out_csv_path)


# create a response
#response = client.chat.completions.create(
#    model="gpt-5.2",
#    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
#)

#print(response.choices[0].message.content)
