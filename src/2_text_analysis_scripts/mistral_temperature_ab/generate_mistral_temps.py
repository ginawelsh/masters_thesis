"""Generate Mistral (mistral-small-2506) outputs per ROW at temperature 0.15 and 1.0,
for both registers and all four prompt conditions. Paired design: the SAME prompt is
generated at both temperatures, so 0.15 and 1.0 can be compared pair-by-pair.

Requires MISTRAL_API_KEY (+ MISTRAL_BASE_URL / MISTRAL_MODEL) in the project-root .env,
exactly as generation_pipeline.py expects. Resumable: caches each
(register, doc_id, condition, temp) to generated_corpus_mistral_temps.csv and skips
completed cells on re-run.

Run:  python src/2_text_analysis_scripts/mistral_temperature_ab/generate_mistral_temps.py
"""
import os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # -> repo root
sys.path.insert(0, os.path.join(ROOT, "src", "1_data_collection"))
import generation_pipeline as pipe  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TEMPS = [0.15, 1.0]
CONDITIONS = ["baseline", "human_like", "detector_aware", "detector_evasive"]
CONCURRENCY = 8
CACHE = os.path.join(HERE, "generated_corpus_mistral_temps.csv")
_lock = threading.Lock()

# one generation target per row (paired human comment / abstract)
docs = []
comm = pd.read_csv(os.path.join(ROOT, "src", "4_archive", "consolidated_informal_comments_JUN26.csv"), encoding="utf-8")
for i, row in comm.iterrows():
    q = row.get("question")
    if pd.notna(q) and str(q).strip():
        docs.append(("informal", int(i), {"id": f"inf_{i}", "question": str(q)}))
absd = pd.read_csv(os.path.join(ROOT, "src", "1_data_collection", "llm_abstracts", "abstracts", "sv_abstracts_openai_2.csv"), encoding="utf-8")
for i, row in absd.iterrows():
    t, k = row.get("Title"), row.get("Keywords")
    if pd.notna(t) and str(t).strip():
        docs.append(("formal", int(i), {"id": f"for_{i}", "title": str(t), "keywords": "" if pd.isna(k) else str(k)}))
print(f"informal docs: {sum(d[0]=='informal' for d in docs)} | formal docs: {sum(d[0]=='formal' for d in docs)}", flush=True)


def load_cache():
    if not os.path.exists(CACHE):
        return set()
    df = pd.read_csv(CACHE, encoding="utf-8")
    return {(r["register"], int(r["doc_id"]), r["condition"], float(r["temp"])) for _, r in df.iterrows()}


def append(register, doc_id, condition, temp, text):
    line = pd.DataFrame([{"register": register, "doc_id": doc_id, "condition": condition, "temp": temp, "text": text}])
    with _lock:
        line.to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False, encoding="utf-8")


done = load_cache()
tasks = [(reg, did, item, c, t) for reg, did, item in docs for c in CONDITIONS for t in TEMPS
         if (reg, did, c, t) not in done]
print(f"{len(tasks)} cells to generate ({len(done)} cached)", flush=True)


def work(task):
    reg, did, item, c, t = task
    rec = pipe.generate("mistral", c, reg, item, pipe.GenParams(temperature=t))
    append(reg, did, c, t, rec.output)


n = 0
if tasks:
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            try:
                f.result(); n += 1
            except Exception as e:
                print("  [error]", str(e)[:100], flush=True)
            if n % 200 == 0:
                print(f"  {n}/{len(tasks)} generated", flush=True)
print(f"DONE. generated {n} new cells -> {CACHE}", flush=True)
