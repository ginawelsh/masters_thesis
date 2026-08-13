# -*- coding: utf-8 -*-
"""Fill the 50 missing `Topic Category` values and merge near-duplicate labels.

Why: `Topic Category` as scraped from DiVA is unusable as a corpus statistic --
50 of 170 rows are blank, one row carries a DiVA metadata string instead of a
subject, and the vocabulary mixes SSIF (Standard for svensk indelning av
forskningsamnen 2011) levels, so "Medicine" and "Nursing" sit side by side as
if they were siblings.

Two columns are added; nothing existing is overwritten:

  Topic Category Canonical  fine-grained SSIF label, near-duplicates merged,
                            blanks filled from the abstract (the field the
                            feature analysis actually uses)
  SSIF Domain               SSIF level-1 rollup, so the sampling breadth claim
                            in the methodology can be stated at one consistent
                            level

  Topic Source              provenance: 'diva' (as scraped, unchanged),
                            'diva_merged' (near-duplicate collapsed),
                            'assigned' (was blank / metadata string)

FILLED assignments were read off the Swedish abstract + keywords, NOT the
Title -- see MISALIGNED below: several titles are author names or DiVA metadata
strings, so Title is not trustworthy for these rows.

Run:  python fill_topic_categories.py
Out:  sv_abstracts_topics_filled.csv   (corpus + 3 new columns)
      topic_category_map.csv           (audit trail: every raw -> canonical)
"""
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "sv_abstracts_adversarial.csv")
OUT_CSV = os.path.join(HERE, "sv_abstracts_topics_filled.csv")
MAP_CSV = os.path.join(HERE, "topic_category_map.csv")

# ---------------------------------------------------------------------------
# 1. Near-duplicate merges over the values DiVA actually returned.
#    Left = raw scraped value, right = canonical SSIF label.
# ---------------------------------------------------------------------------
MERGE = {
    # same subject, two DiVA spellings
    "Law (excluding Law and Society)": "Law",
    "Religious Studies Religious Studies Humanities": "Religious Studies",
    # domain-level labels for the same domain (old vs current DiVA naming)
    "Social and Behavioural Science": "Social Sciences",
    "Medicine": "Medical and Health Sciences",
}

# ---------------------------------------------------------------------------
# 2. SSIF level-1 rollup for every canonical label.
# ---------------------------------------------------------------------------
DOMAIN = {
    # 1 Natural Sciences (102 Computer and Information Sciences lives here)
    "Natural Sciences": "Natural Sciences",
    "Computer Sciences": "Natural Sciences",
    "Information Systems": "Natural Sciences",
    "Human Computer Interaction": "Natural Sciences",
    # 2 Engineering and Technology
    "Engineering and Technology": "Engineering and Technology",
    "Computer Engineering": "Engineering and Technology",
    "Production Engineering, Human Work Science and Ergonomics": "Engineering and Technology",
    # 3 Medical and Health Sciences
    "Medical and Health Sciences": "Medical and Health Sciences",
    "Nursing": "Medical and Health Sciences",
    "Immunology": "Medical and Health Sciences",
    # 4 Agricultural and Veterinary Sciences
    "Agricultural and Veterinary Sciences": "Agricultural and Veterinary Sciences",
    # 5 Social Sciences
    "Social Sciences": "Social Sciences",
    "Business Administration": "Social Sciences",
    "Economics": "Social Sciences",
    "Law": "Social Sciences",
    "Pedagogy": "Social Sciences",
    "Didactics": "Social Sciences",
    "Psychology": "Social Sciences",
    "Political Science": "Social Sciences",
    "Social Work": "Social Sciences",
    "Sociology": "Social Sciences",
    "Media and Communication Studies": "Social Sciences",
    "Other Social Sciences not elsewhere specified": "Social Sciences",
    # 6 Humanities and the Arts
    "Humanities and the Arts": "Humanities and the Arts",
    "Religious Studies": "Humanities and the Arts",
    "General Language Studies and Linguistics": "Humanities and the Arts",
}

# ---------------------------------------------------------------------------
# 3. Assignments for rows with no usable subject, keyed by row index.
#    Read from the Swedish abstract + keywords. Rows marked LOW in REVIEW below
#    are the judgement calls worth a second look.
# ---------------------------------------------------------------------------
FILLED = {
    50:  "Information Systems",        # affarssystem implementation, anpassning
    51:  "Business Administration",    # redovisningsval, periodiseringsfonder
    52:  "Sociology",                  # genus/sexualitet hos tonarstjejer
    54:  "Information Systems",        # informationssakerhet, intellektuellt kapital
    55:  "Humanities and the Arts",    # musikgenrerna raweh och rai
    56:  "Pedagogy",                   # laroplanens spar i pedagogiskt arbete
    57:  "Humanities and the Arts",    # operahusens design och arkitektur
    69:  "Sociology",                  # etableringsstrategier, unga akademiker
    70:  "Didactics",                  # individualisering i matematikundervisning
    71:  "Business Administration",    # varumarke och attityd till produkter
    72:  "Nursing",                    # transkulturell omvardnad
    73:  "Business Administration",    # beloningssystem, ekonomistyrning
    74:  "Social Work",                # vagen ut ur missbruk, sociala band
    75:  "Information Systems",        # kommunala hemsidor, koordination
    76:  "Religious Studies",          # Che Guevara/Jesus, diskursanalys      LOW
    77:  "Pedagogy",                   # larstilar pa omvardnadsprogrammet
    78:  "Nursing",                    # palliativ vard, sjukskoterskors upplevelser
    79:  "Business Administration",    # marknadsforing, aterkop av damtidningar
    80:  "Social Work",                # kuratorer, anmalningsskyldighet        LOW
    83:  "Media and Communication Studies",  # innehallsanalys av tidningstyper
    84:  "Production Engineering, Human Work Science and Ergonomics",  # makulatur, PDCA
    85:  "Business Administration",    # balanserat styrkort, ekonomistyrning
    86:  "Nursing",                    # ickeverbal kommunikation, demensvard
    87:  "Pedagogy",                   # dyskalkyli i skolan
    88:  "Business Administration",    # processorientering, lonsamhet
    89:  "Information Systems",        # "examensarbete i informatik", verksamhetsanalys
    90:  "Nursing",                    # etiskt ansvar i omvardnaden
    91:  "Business Administration",    # foretagsekonomi, utvardering av utbildningar
    92:  "Nursing",                    # hot och vald i somatisk vard
    93:  "Political Science",          # USA:s strategiska inflytande, Iran
    96:  "Information Systems",        # anvandarmedverkan i systemutveckling
    101: "Social Work",                # klient och socialsekreterare, makt
    104: "Pedagogy",                   # lek i grundskolan och traningsskolan
    107: "Law",                        # legalitetsprincipen, straff-/forvaltningsratt
    109: "Natural Sciences",           # ekologi/fiskvandring i Motala strom
    111: "Nursing",                    # omvardnad vid stickradsla
    113: "Business Administration",    # efterkalkyler, uppfoljning              LOW
    114: "Computer Engineering",       # GPRS/GSM/WLAN, systemarkitektur
    115: "Religious Studies",          # sloja i skolan
    116: "Nursing",                    # preoperativ information
    117: "Information Systems",        # e-post i IT-intensiv organisation
    118: "Nursing",                    # prostatacancer, livskvalitet, omvardnad
    119: "Information Systems",        # systemutveckling, designprocess
    122: "Psychology",                 # vuxenutvecklingspsykologi
    123: "Other Social Sciences not elsewhere specified",  # militar logistik      LOW
    124: "Media and Communication Studies",  # Islam i nyhetsprogram, orientalism  LOW
    125: "Social Work",                # familjehemsvard, socialtjanstlagen
    126: "Business Administration",    # utvecklingssamtal, samtalsmodell        LOW
    127: "Pedagogy",                   # daglig fysisk aktivitet, lararattityder
    128: "Pedagogy",                   # avbrott i distansstudier
    132: "Human Computer Interaction", # design for anvandbarhet, tyst kunnande
}

# Rows whose Title field is NOT the thesis title -- scrape misalignment. Topic
# was taken from the abstract; the Title itself still needs repair upstream.
MISALIGNED = {
    51:  "Title is a DiVA metadata string, not a title",
    54:  "Title is a DiVA metadata string, not a title",
    52:  "Title is about pedagogues/grief; Keywords+Abstract are about gender/sexuality",
    88:  "Title is an author name ('Andersson, Anna-Karin')",
    91:  "Title is an author name ('Andersson, Annika')",
    119: "Title is an author name ('Andersson, Helena')",
    125: "Title is an author name ('Andersson, Jennie')",
}

LOW_CONFIDENCE = {76, 80, 113, 123, 124, 126}

RAW_COL = "Topic Category"


def canonicalize(df):
    canon, source = [], []
    for idx, raw in df[RAW_COL].items():
        if idx in FILLED:
            canon.append(FILLED[idx])
            source.append("assigned")
            continue
        s = str(raw).strip() if pd.notna(raw) else ""
        if not s or s.lower() == "nan":
            raise SystemExit(f"row {idx} is blank but has no FILLED assignment")
        if s in MERGE:
            canon.append(MERGE[s])
            source.append("diva_merged")
        else:
            canon.append(s)
            source.append("diva")
    return canon, source


def main():
    df = pd.read_csv(IN_CSV, encoding="utf-8")
    print(f"read {len(df)} rows; {df[RAW_COL].isna().sum()} blank {RAW_COL}")

    canon, source = canonicalize(df)
    df["Topic Category Canonical"] = canon
    df["Topic Source"] = source

    unknown = sorted(set(canon) - set(DOMAIN))
    if unknown:
        raise SystemExit(f"no SSIF domain mapped for: {unknown}")
    df["SSIF Domain"] = [DOMAIN[c] for c in canon]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    rows = []
    for idx, r in df.iterrows():
        raw = r[RAW_COL]
        rows.append(dict(
            row_index=idx,
            raw_topic=("" if pd.isna(raw) else str(raw)),
            canonical_topic=r["Topic Category Canonical"],
            ssif_domain=r["SSIF Domain"],
            source=r["Topic Source"],
            low_confidence=idx in LOW_CONFIDENCE,
            title_misaligned=MISALIGNED.get(idx, ""),
            title=str(r["Title"])[:120],
        ))
    pd.DataFrame(rows).to_csv(MAP_CSV, index=False, encoding="utf-8")

    print(f"\nwrote {OUT_CSV}\nwrote {MAP_CSV}\n")
    print("--- provenance ---")
    print(df["Topic Source"].value_counts().to_string())
    print(f"\n--- canonical topics: {df['Topic Category Canonical'].nunique()} ---")
    print(df["Topic Category Canonical"].value_counts().to_string())
    print(f"\n--- SSIF domains: {df['SSIF Domain'].nunique()} ---")
    print(df["SSIF Domain"].value_counts().to_string())
    print(f"\nlow-confidence assignments: {len(LOW_CONFIDENCE)}")
    print(f"rows with a misaligned Title: {len(MISALIGNED)}")


if __name__ == "__main__":
    main()
