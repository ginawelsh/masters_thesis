"""Mean sentence length (tokens per sentence) for the three-generator appendix tables.

The appendix tables app:full-three-way-* compare human text with GPT-5.2, GPT-5.6 and
Mistral 3.2. Their source CSV (model_comparison_three_way_results.csv) and generating
script are not in the repository, so this script computes the one feature added after
submission, tokens_per_sentence, for all three generators with the same method as the
rest of those tables: syntactic_complexity.compute_metrics (sv_core_news_lg), paired
Wilcoxon signed-rank test (LLM vs. human, per document), Cohen's d_z on LLM - human.

GPT-5.2 pairs are read from syntactic_complexity_<register>_<condition>.csv (written by
syntactic_complexity.py), so its column matches the main results table. GPT-5.6 (the
100-document quiz subset) and Mistral (temperature 1.0) are parsed here.

Output: csv_files/tokens_per_sentence_three_way.csv
Run:    python src/2_text_analysis_scripts/scripts/tokens_per_sentence_three_way.py
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
sys.argv = sys.argv[:1]

import embedding_comparison as ec  # noqa: E402
import syntactic_complexity as sc  # noqa: E402
from data_utils import resolve_conditions  # noqa: E402
from significance_tests import test_feature  # noqa: E402

CONDS = ["baseline", "human_like", "detector_evasive"]
OUT = os.path.join(ec.CSV_DIR, "tokens_per_sentence_three_way.csv")
FEAT = "tokens_per_sentence"


def main():
    nlp = sc.load_nlp()
    cache = {}

    def tps(text, min_tokens):
        key = (str(text), min_tokens)
        if key not in cache:
            m = sc.compute_metrics(nlp, text, min_tokens)
            cache[key] = m[FEAT] if m else np.nan
        return cache[key]

    rows = []
    for dataset in ("formal", "informal"):
        reg = ec.REGISTER[dataset]
        min_tokens = sc.resolve_config(dataset)["min_tokens"]
        sources = {"GPT-5.6": ec.load_dataset("gpt56", dataset, 1.0),
                   "Mistral 3.2": ec.load_mistral(dataset, 1.0)}
        for cond, llm_col in resolve_conditions(dataset, "all"):
            if cond not in CONDS:
                continue
            pairs = {}
            g52 = pd.read_csv(os.path.join(ec.CSV_DIR, f"syntactic_complexity_{dataset}_{cond}.csv"))
            pairs["GPT-5.2"] = (g52[f"human_{FEAT}"], g52[f"llm_{FEAT}"])
            for gen, df in sources.items():
                if llm_col not in df.columns:
                    continue
                d = df[df[reg["human_col"]].notna() & df[llm_col].notna()]
                print(f"{dataset}/{cond}/{gen}: parsing {len(d)} pairs", flush=True)
                h = d[reg["human_col"]].map(lambda t: tps(t, min_tokens))
                l = d[llm_col].map(lambda t: tps(t, min_tokens))
                pairs[gen] = (h, l)
            for gen, (h, l) in pairs.items():
                res = test_feature(h, l)
                rows.append(dict(dataset=dataset, condition=cond, generator=gen, feature=FEAT,
                                 n_pairs=res["n"], median_human=res["median_human"],
                                 median_llm=res["median_llm"], cohens_dz=res["dz"],
                                 p_value=res["p"]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
