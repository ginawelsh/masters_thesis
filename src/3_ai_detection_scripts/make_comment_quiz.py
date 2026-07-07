"""Build a human-vs-LLM *comment* quiz that mixes the prompt conditions equally.

Companion to make_mixed_quiz.py (which does thesis abstracts). This one does the
informal forum comments (Reddit/Flashback), grouped into threads so the AI comments
stay ALIGNED with the thread question they were generated to answer.

Texts are copied VERBATIM from consolidated_informal_comments_adversarial.csv (read
straight from the CSV cells — no editing, no rephrasing). Human comments in particular
are never altered. Each comment shown in a thread comes from a DISTINCT source row of
that thread's question, so a human comment and its own AI paraphrase never co-occur.

Composition (edit the constants to resize):
  N_THREADS threads, COMMENTS_PER_THREAD comments each.
  Across the whole quiz the four conditions are balanced equally
  (human / baseline / human_like / detector_aware).
Default = 10 threads x 4 comments = 40 comments = 10 per condition.

The per-thread mix is randomised (NOT one-of-each), so a thread may be all-AI, all-human,
or any mix — this preserves the "don't assume it's always a mix" framing of the survey.

Outputs (into this folder):
  mixed_comment_quiz.csv   full data + answer key
    (thread_order, question, comment_order, text, is_human, condition, source_row)
"""
import os
import random

import pandas as pd

N_THREADS = 10
COMMENTS_PER_THREAD = 4
MIN_LEN = 40            # readability band for the comment text actually shown
MAX_LEN = 1000
MAX_QLEN = 300         # keep thread questions short enough to display cleanly
SEED = 42              # fixed for a reproducible quiz
MAX_ATTEMPTS = 500     # reshuffle budget to hit a balanced, valid assignment

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
IN_CSV = os.path.join(_src, "1_data_collection", "llm_comments",
                      "consolidated_informal_comments_adversarial.csv")
OUT_CSV = os.path.join(_here, "mixed_comment_quiz.csv")

# condition -> source column in the CSV (text copied exactly from these cells)
COND_COL = {
    "human": "human_comment",
    "baseline": "generated_comment",
    "human_like": "comment_human_like",
    "detector_aware": "comment_detector_aware",
}
CONDITIONS = ["human", "baseline", "human_like", "detector_aware"]


def _in_band(text):
    return MIN_LEN <= len(str(text)) <= MAX_LEN


def _candidate_questions(df):
    """Questions short enough, with >= COMMENTS_PER_THREAD rows and >=1 band-valid human row."""
    cands = []
    for q, g in df.groupby("question"):
        if len(str(q)) > MAX_QLEN:
            continue
        if len(g) < COMMENTS_PER_THREAD:
            continue
        if g["human_comment"].map(_in_band).sum() < 1:
            continue
        cands.append(q)
    return cands


def _assign_thread(question_rows, needed_conditions, rng):
    """Pick a distinct source row for each needed condition such that the text shown
    (that row's cell for that condition) is within the length band.

    Returns list of (row_index, condition) or None if it can't be satisfied.
    Hardest conditions (fewest band-valid rows) are placed first.
    """
    rows = list(question_rows.index)
    # order the requested conditions by scarcity within this question
    def valid_rows(cond):
        col = COND_COL[cond]
        return [r for r in rows if _in_band(question_rows.at[r, col])]

    order = sorted(range(len(needed_conditions)),
                   key=lambda i: len(valid_rows(needed_conditions[i])))

    used = set()
    result = [None] * len(needed_conditions)
    for i in order:
        cond = needed_conditions[i]
        options = [r for r in valid_rows(cond) if r not in used]
        if not options:
            return None
        pick = rng.choice(options)
        used.add(pick)
        result[i] = (pick, cond)
    return result


def _build(df):
    total = N_THREADS * COMMENTS_PER_THREAD
    if total % len(CONDITIONS) != 0:
        raise SystemExit("N_THREADS*COMMENTS_PER_THREAD must divide evenly by 4 conditions.")
    per_cond = total // len(CONDITIONS)

    candidates = _candidate_questions(df)
    if len(candidates) < N_THREADS:
        raise SystemExit(f"Only {len(candidates)} candidate questions; need {N_THREADS}.")

    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(SEED + attempt)

        questions = rng.sample(candidates, N_THREADS)

        # balanced condition pool, shuffled and split into one multiset per thread
        pool = CONDITIONS * per_cond
        rng.shuffle(pool)
        thread_conditions = [pool[i * COMMENTS_PER_THREAD:(i + 1) * COMMENTS_PER_THREAD]
                             for i in range(N_THREADS)]

        threads = []
        ok = True
        for q, needed in zip(questions, thread_conditions):
            g = df[df["question"] == q]
            assigned = _assign_thread(g, needed, rng)
            if assigned is None:
                ok = False
                break
            # shuffle display order within the thread so conditions aren't positional
            rng.shuffle(assigned)
            threads.append((q, assigned))
        if ok:
            return threads, attempt

    raise SystemExit("Could not find a balanced, valid assignment; loosen the constraints.")


def main():
    df = pd.read_csv(IN_CSV, encoding="utf-8").reset_index(drop=True)
    threads, attempt = _build(df)

    records = []
    order = 0
    for t_idx, (question, assigned) in enumerate(threads, 1):
        for c_idx, (row_idx, cond) in enumerate(assigned, 1):
            order += 1
            text = str(df.at[row_idx, COND_COL[cond]])  # verbatim
            records.append({
                "order": order,
                "thread_order": t_idx,
                "question": str(question),            # verbatim
                "comment_order": c_idx,
                "text": text,
                "is_human": cond == "human",
                "condition": cond,
                "source_row": int(row_idx),
            })

    out = pd.DataFrame(records, columns=[
        "order", "thread_order", "question", "comment_order",
        "text", "is_human", "condition", "source_row",
    ])
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    counts = out["condition"].value_counts().to_dict()
    print(f"Wrote {len(out)} comments across {N_THREADS} threads (attempt {attempt}).")
    print(f"  condition balance: {counts}")
    print(f"  distinct source rows: {out['source_row'].nunique()} (should equal {len(out)})")
    print(f"  {OUT_CSV}")


if __name__ == "__main__":
    main()
