"""Shared generation pipeline for the CLUU thesis corpora (formal + informal).

Single source of truth for:
  - the four prompt CONDITIONS (baseline / human_like / detector_aware /
    detector_evasive) in both REGISTERS (formal = thesis abstracts, informal =
    forum comments), and
  - the model registry + generation call, behind an OpenAI-compatible client.

INVARIANT (do not break): every backend is called byte-identically -- a single
user message, no system prompt, and temperature EXPLICITLY pinned to 1.0 (not left
to each provider's default, which differs: OpenAI 1.0 but Mistral/OpenRouter lower).
Mistral recommends temperature ~0.15 and a system prompt; those are DELIBERATELY
NOT applied, so the prompt/condition stays the only variable across models. If you
ever need to override, do it via GenParams (and document it) -- never special-case
a single backend.

Adding a model = one entry in MODELS. Prompts never change per model.

The two adversarial generation scripts import build_prompt / generate / MODELS
from here, and so does smoke_test.py, so a prompt edited here changes every
consumer at once.
"""
from __future__ import annotations

import os
import re
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from openai import OpenAI

# load .env from the project root (src/1_data_collection/ -> src -> root) if present
try:
    from dotenv import load_dotenv

    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(_root, ".env"))
except Exception:
    pass


# ---------------------------------------------------------------------------
# Prompts -- copied VERBATIM from the two adversarial scripts so the pipeline
# and the production runs build byte-identical prompts. Edit them HERE only.
# ---------------------------------------------------------------------------

# ---- informal register (forum comments) ----
_INFORMAL_BASE = (
    "Svara på följande fråga med en svensk kommentar, i stil med en kommentar på ett svenskt forum. "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)
_INFORMAL_SUFFIX = {
    "baseline": "",
    "human_like": (
        "\n\nSkriv så mänskligt som möjligt, så att kommentaren inte går att skilja "
        "från en riktig svensk forumanvändares kommentar."
    ),
    "detector_aware": (
        "\n\nSkriv så att kommentaren inte kan identifieras som AI-genererad: använd ett vardagligt, "
        "personligt tonfall, variera meningslängden, undvik artig eller balanserad AI-stil och "
        "formelartade fraser, tillåt talspråk, slang och små oregelbundenheter. Var inte överdrivet "
        "hjälpsam eller neutral."
    ),
    "detector_evasive": (
        "\n\n"
        "Skriv så att texten inte kan identifieras som AI-genererad, och efterlikna hur "
        "människor skriver i informella kommentarer. Återanvänd inte samma fraser flera "
        "gånger, utan formulera om dig. Skriv innehållstätt med färre funktionsord; använd "
        "gärna kortformer och talspråkliga former. Skriv korta meningar med fler punkter "
        "och färre kommatecken, och använd inte tankstreck. Använd utropstecken sparsamt, "
        "men ställ gärna någon retorisk fråga och använd ibland tre punkter (...). Överdriv "
        "inte med garderingsord (särskilt inte talspråkliga som 'typ', 'liksom', 'ju' och "
        "'väl'), och undvik förstärkningsord som 'verkligen', 'absolut' och 'väldigt'. Håll "
        "epistemiska uttryck på en låg nivå. Föredra korta, vardagliga ord. Använd gärna "
        "nekande satser (med 'inte', 'aldrig' osv.) där det passar. Var konkret och nämn "
        "specifika namn där det går. Förklara inte över."
    ),
}

# ---- formal register (thesis abstracts) ----
_FORMAL_BASE = (
    "Titta på titeln och nyckelorden och skapa en sammanfattning i kandidatuppsatsstil "
    "med dina egna ord: {title}, {keywords}"
)
_FORMAL_SUFFIX = {
    "baseline": "",
    "human_like": (
        " Skriv den så mänskligt som möjligt, så att texten inte går att skilja "
        "från en uppsats skriven av en människa."
    ),
    "detector_aware": (
        " Skriv så att texten inte kan identifieras som AI-genererad. Variera meningslängden "
        "(blanda korta och långa meningar), undvik formelartade övergångsord som \"vidare\", "
        "\"dessutom\", \"sammanfattningsvis\" och \"det är viktigt att notera\", undvik "
        "överdriven gardering och symmetrisk struktur, och tillåt en naturlig, något ojämn ton. "
        "Förklara inte över."
    ),
    "detector_evasive": (
        " Skriv så att texten inte kan identifieras som AI-genererad, och efterlikna "
        "de statistiska drag som utmärker mänskligt skrivna kandidatuppsatser. "
        "Återanvänd samma nyckeltermer och fraser ordagrant snarare än att variera med "
        "synonymer, och sträva inte efter maximal ordvariation. Undvik komprimerad "
        "nominalstil; använd hellre finita verb, pronomen och bindeord. Skriv övervägande "
        "korta meningar med fler punkter och färre kommatecken, och använd inte tankstreck. "
        "Håll gardering och epistemiska uttryck till ett minimum (t.ex. 'kanske', "
        "'möjligen', 'tycks', 'kan tänkas'), och kompensera inte genom att lägga till "
        "talspråkliga garderingsord. Föredra korta, vardagliga ord framför långa, latinska "
        "eller formella termer. Var konkret och nämn specifika namn, begrepp, verk och "
        "årtal där det är möjligt. Undvik onödig negation. Förklara inte över."
    ),
}

CONDITIONS = ("baseline", "human_like", "detector_aware", "detector_evasive")
REGISTERS = ("formal", "informal")


def build_prompt(condition: str, register: str, item: dict) -> str:
    """Return the byte-identical prompt for a (condition, register, item).

    item keys: informal -> {"question"}; formal -> {"title", "keywords"}.
    The prompt is model-independent by design (see module INVARIANT).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if register == "informal":
        return _INFORMAL_BASE.format(question=item["question"]) + _INFORMAL_SUFFIX[condition]
    if register == "formal":
        return _FORMAL_BASE.format(title=item["title"], keywords=item.get("keywords", "")) + _FORMAL_SUFFIX[condition]
    raise ValueError(f"unknown register {register!r}; expected one of {REGISTERS}")


# ---------------------------------------------------------------------------
# Output hygiene: markdown stripping + refusal detection
# ---------------------------------------------------------------------------
# WHY STRIP MARKDOWN: models emit **bold**, *italic* and ### headings in prose that
# should be plain text. Human text in both corpora is essentially markdown-free (0 of
# 170 abstracts; 3 of 1,149 comments), so a markdown marker is a near-perfect AI cue
# that has nothing to do with how human the *language* is. Left in, a judge can score
# on formatting and a stylometric feature counts '*' as tokens. GPT-5.6 emits it far
# more heavily than GPT-5.2, so leaving it in would make a 5.2-vs-5.6 contrast a
# measurement of formatting rather than of Swedish.
#
# Deliberately conservative: only unambiguous inline markup and ATX headings. Bullet
# lists ('- x') and numbered lists ('1. x') are LEFT ALONE because they occur
# naturally in prose. Whitespace and newlines are preserved -- the informal register's
# multi-line structure is a genuine register signal, so this must not reflow text.
#
# NEVER APPLY THIS TO HUMAN TEXT. Human columns are read-only ground truth.
_MD_SUBS = [
    (re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+"), ""),            # ### Heading
    (re.compile(r"(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$"), ""),  # horizontal rule
    (re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S), r"\1"),  # **bold**
    (re.compile(r"__(?=\S)(.+?)(?<=\S)__", re.S), r"\1"),      # __bold__
    (re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)"), r"\1"),  # *italic*
    (re.compile(r"`(?=\S)([^`\n]+?)(?<=\S)`"), r"\1"),         # `code`
]


def strip_markdown(text) -> str:
    """Remove unambiguous markdown markup, preserving all whitespace and newlines.

    A no-op on text that contains no markdown. Not for human columns.
    """
    s = str(text)
    for rx, repl in _MD_SUBS:
        s = rx.sub(repl, s)
    return s


def count_markdown(text) -> int:
    """How many markdown constructs strip_markdown() would remove (for reporting)."""
    s = str(text)
    return sum(len(rx.findall(s)) for rx, _ in _MD_SUBS)


# Refusal / meta-commentary detection. A refusal is stored as if it were generated
# text unless something checks: GPT-5.6 answered the formal detector_evasive prompt
# with "Jag kan inte hjälpa till att kringgå AI-detektering. Däremot kan jag ..." and
# then wrote a NON-evasive abstract. That is doubly wrong -- the cell is mislabelled,
# and the refusal sentence is itself a giant AI tell for any judge reading it.
#
# Anchored to the OPENING of the text (refusals lead), and requires a first-person
# modal negation or an explicit detection-evasion reference, so an abstract that
# merely contains "kan inte" further in does not trip it.
_REFUSAL_HEAD_CHARS = 300
# A bare negation is NOT enough: "Jag kan inte hitta det på TV.nu" is a perfectly
# natural forum comment. A refusal is a first-person modal negation attached to a
# TASK verb (help / write / generate / comply), or an unambiguous meta-reference.
_SV_TASK_VERB = (r"hjälpa|bistå|assistera|skriva|generera|producera|skapa|framställa"
                 r"|uppfylla|tillmötesgå|utföra|göra\s+det|ställa\s+upp")
_REFUSAL_PATTERNS = [
    (re.compile(rf"\bjag\s+kan\s+(?:dessvärre\s+|tyvärr\s+)?inte\s+(?:\w+\s+){{0,3}}(?:{_SV_TASK_VERB})\b", re.I),
     "sv: 'jag kan inte <task verb>'"),
    (re.compile(rf"\bkan\s+jag\s+inte\s+(?:\w+\s+){{0,3}}(?:{_SV_TASK_VERB})\b", re.I),
     "sv: 'kan jag inte <task verb>'"),
    (re.compile(rf"\bjag\s+(?:får|kommer)\s+inte\s+(?:att\s+)?(?:\w+\s+){{0,2}}(?:{_SV_TASK_VERB})\b", re.I),
     "sv: 'jag får/kommer inte <task verb>'"),
    (re.compile(r"\bI\s+(?:can'?t|cannot|won'?t|am\s+not\s+able\s+to)\s+(?:\w+\s+){0,3}"
                r"(?:help|assist|write|generate|produce|create|comply|do\s+that)\b", re.I),
     "en: 'I can't <task verb>'"),
    (re.compile(r"\bkringgå\b[^.\n]{0,40}\b(?:detekt|AI|granskning)", re.I),
     "sv: 'kringgå ... detektering'"),
    (re.compile(r"\bAI[- ]?detekt\w*", re.I), "meta: mentions AI detection"),
    (re.compile(r"\bas an AI\b|\bsom en AI\b|\bAI-(?:modell|assistent)\b", re.I),
     "meta: refers to itself as an AI"),
]


def looks_like_refusal(text):
    """Return a short reason string if `text` opens like a refusal, else None."""
    head = str(text)[:_REFUSAL_HEAD_CHARS]
    for rx, why in _REFUSAL_PATTERNS:
        if rx.search(head):
            return why
    return None


# ---------------------------------------------------------------------------
# Run-scoping helpers (which conditions, which input file)
# ---------------------------------------------------------------------------
def conditions_from_env(default: list, env: str = "GEN_CONDITIONS"):
    """Return (conditions, tag). GEN_CONDITIONS="baseline" restricts a run to one
    condition; unset means `default`.

    The returned `tag` MUST be folded into the output/cache filenames. The generation
    cache is keyed on (row index, condition) with no record of which conditions a run
    covered, so a one-condition run and a three-condition run that shared a cache file
    would look identical to the resume logic and the narrower run would appear complete.
    """
    raw = os.environ.get(env, "").strip()
    if not raw:
        return list(default), ""
    want = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in want if c not in CONDITIONS]
    if unknown:
        raise SystemExit(
            f"{env}: unknown condition(s) {unknown}; valid: {list(CONDITIONS)}")
    outside = [c for c in want if c not in default]
    if outside:
        raise SystemExit(
            f"{env}: condition(s) {outside} are not part of this script's reported set "
            f"{list(default)}. Add them there deliberately if you really want them.")
    ordered = [c for c in default if c in want]           # keep canonical order
    return ordered, "_" + "-".join(ordered)


def input_from_env(default_path: str, env: str = "GEN_INPUT"):
    """Return (path, tag). GEN_INPUT overrides the corpus a script reads.

    The cache is keyed on ROW INDEX, so it is only valid for one input file and
    ordering; the returned tag keeps caches for different inputs apart.
    """
    raw = os.environ.get(env, "").strip()
    if not raw:
        return default_path, ""
    if not os.path.exists(raw):
        raise SystemExit(f"{env}: no such file: {raw}")
    import hashlib
    stem = os.path.splitext(os.path.basename(raw))[0]
    # short + hashed: readable enough to recognise, short enough to stay well inside
    # Windows' 260-char path limit given how deep these corpus folders sit
    h = hashlib.sha1(os.path.abspath(raw).encode("utf-8")).hexdigest()[:4]
    return raw, f"_in-{stem[:12]}-{h}"


# ---------------------------------------------------------------------------
# Quiz subsetting -- generate only the documents the detection quiz sampled.
# ---------------------------------------------------------------------------
# WHY THIS MATCHES ON TEXT, NOT ROW NUMBER: the quiz's `source_row` indexes the
# *_adversarial.csv files that make_balanced_quiz.py reads, which hold the same
# rows as the generation inputs but in a DIFFERENT ORDER (positional agreement on
# `question` between consolidated_informal_comments_JUN26.csv and
# consolidated_informal_comments_adversarial.csv is only ~37%). Using source_row
# as a positional index into a generation input would therefore silently select
# the wrong documents. The human text itself is stable across both files, so it is
# the only safe join key.
QUIZ_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "3_ai_detection_scripts", "quiz_master_balanced.csv",
)


def _norm_key(text) -> str:
    import re
    return re.sub(r"\s+", " ", str(text)).strip()


def quiz_subset(data, register: str, key_col: str, quiz_csv: Optional[str] = None,
                expected: Optional[int] = 100):
    """Return `data` restricted to the documents sampled into the detection quiz.

    Matches on the human text (`key_col`: "Abstract" for formal, "human_comment"
    for informal) against the quiz's human-condition rows. Keeps the first input row
    per matched key, so the result has exactly one row per quiz document, reindexed
    0..m-1. Raises if the match is incomplete -- a silent partial match would produce
    a corpus that looks fine and is not comparable to the GPT-5.2 run.

    Two `quiz_*` columns are attached so the generated CSV carries an EXPLICIT link
    back to the quiz document it corresponds to, rather than leaving a downstream
    consumer to re-derive this text join:
      quiz_pair_id    - the quiz's pair_id for this document
      quiz_source_row - the quiz's source_row (index into the *_adversarial.csv files)
    Pair on quiz_pair_id when building the GPT-5.6 quiz; do NOT rely on row order,
    which follows the generation input, not the quiz.
    """
    import pandas as pd

    quiz = pd.read_csv(quiz_csv or QUIZ_CSV, encoding="utf-8")
    hum = quiz[(quiz["register"] == register) & (quiz["condition"] == "human")]
    want = {_norm_key(t) for t in hum["text"]}
    if not want:
        raise ValueError(f"no human {register!r} rows found in {quiz_csv or QUIZ_CSV}")
    meta = {
        _norm_key(r["text"]): (r.get("pair_id"), r.get("source_row"))
        for _, r in hum.iterrows()
    }

    keys = data[key_col].map(_norm_key)
    hit = keys.isin(want)
    sub = data[hit].loc[~keys[hit].duplicated()].reset_index(drop=True)
    _k = sub[key_col].map(_norm_key)
    sub["quiz_pair_id"] = [meta[k][0] for k in _k]
    sub["quiz_source_row"] = [meta[k][1] for k in _k]

    found = len(sub)
    if found != len(want):
        missing = len(want) - found
        raise ValueError(
            f"quiz subset for {register!r} matched {found} of {len(want)} documents "
            f"({missing} unmatched) using key column {key_col!r}. Refusing to run: the "
            f"resulting corpus would not be comparable to the GPT-5.2 quiz."
        )
    if expected is not None and found != expected:
        raise ValueError(
            f"quiz subset for {register!r} produced {found} documents, expected {expected}."
        )
    return sub


# ---------------------------------------------------------------------------
# Generation params + client. Defaults preserve the cross-backend invariant.
# ---------------------------------------------------------------------------
@dataclass
class GenParams:
    """Generation params. Defaults keep the invariant: temperature pinned to 1.0
    (explicit, so every backend matches -- not left to differing provider defaults)
    and no system prompt. Set temperature=None to omit the field entirely; override
    only deliberately."""
    temperature: Optional[float] = 1.0
    system_prompt: Optional[str] = None
    max_retries: int = 5
    retry_base_sec: int = 4


DEFAULT_PARAMS = GenParams()


class OpenAICompatibleClient:
    """Thin wrapper over any OpenAI-compatible chat endpoint (OpenAI, OpenRouter,
    self-hosted vLLM / NIM). base_url=None -> the default OpenAI endpoint."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} not set (needed for model {model!r}). "
                f"Set it in the environment or in a .env file at the project root."
            )
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def complete(self, prompt: str, params: GenParams = DEFAULT_PARAMS) -> str:
        messages = []
        if params.system_prompt:
            messages.append({"role": "system", "content": params.system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model, "messages": messages}
        if params.temperature is not None:  # otherwise the provider default (1.0) is used
            kwargs["temperature"] = params.temperature
        resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


@dataclass
class ModelSpec:
    """A registry entry: a factory that builds the client for one model."""
    # instantiates 3 models: GPT-5.2, mistral-small-2506,  
    make_client: Callable[[], OpenAICompatibleClient]


# Adding a model = one entry here. Prompts and call contract never change.
MODELS: dict[str, ModelSpec] = {
    # OpenAI GPT-5.2 -- the thesis baseline model (default OpenAI endpoint).
    "gpt-5.2": ModelSpec(
        make_client=lambda: OpenAICompatibleClient(
            model="gpt-5.2", base_url=None, api_key_env="OPENAI_API_KEY",
        ),
    ),
    # OpenAI GPT-5.6 -- newer-generation comparison model, for the "has detectability
    # changed across model generations?" contrast. Called with the SAME contract as
    # gpt-5.2 (temp 1.0, no system prompt) so prompt/condition stays the only variable.
    # Override the upstream model string with GPT56_MODEL if OpenAI's id differs.
    "gpt-5.6": ModelSpec(
        make_client=lambda: OpenAICompatibleClient(
            model=os.environ.get("GPT56_MODEL", "gpt-5.6"),
            base_url=None, api_key_env="OPENAI_API_KEY",
        ),
    ),
    # Mistral Small (mistral-small-2506) via Mistral's own API / La Plateforme
    # (https://api.mistral.ai/v1, OpenAI-compatible). Called with the SAME contract
    # as every backend: temp 1.0, no system prompt (the model card recommends 0.15
    # + a system prompt; per the module INVARIANT those are NOT applied). Override
    # with MISTRAL_MODEL / MISTRAL_BASE_URL (e.g. OpenRouter slug
    # mistralai/mistral-small-3.2-24b-instruct at https://openrouter.ai/api/v1).
    "mistral": ModelSpec(
        make_client=lambda: OpenAICompatibleClient(
            model=os.environ.get("MISTRAL_MODEL", "mistral-small-2506"),
            base_url=os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            api_key_env="MISTRAL_API_KEY",
        ),
    ),
    # Same model, self-hosted behind a local vLLM OpenAI-compatible server:
    #   vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    #     --tokenizer_mode mistral --config_format mistral --load_format mistral
    # (~55 GB GPU RAM in bf16/fp16). The model card recommends temperature=0.15
    # and a system prompt; per the module INVARIANT those are NOT applied here so
    # the prompt/condition stays the only variable across models. vLLM ignores the
    # API key, so MISTRAL_LOCAL_API_KEY can be any non-empty value (e.g. "EMPTY").
    "mistral-local": ModelSpec(
        make_client=lambda: OpenAICompatibleClient(
            model=os.environ.get(
                "MISTRAL_LOCAL_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            ),
            base_url=os.environ.get("MISTRAL_LOCAL_BASE_URL", "http://localhost:8000/v1"),
            api_key_env="MISTRAL_LOCAL_API_KEY",
        ),
    ),
}


@dataclass
class GenRecord:
    model: str
    condition: str
    register: str
    item_id: Optional[str]
    prompt: str
    output: str


# clients are built once per model and reused (safe to call generate() from threads)
_clients: dict[str, OpenAICompatibleClient] = {}
_clients_lock = threading.Lock()


def get_client(model: str) -> OpenAICompatibleClient:
    """Return the (cached) client for a registered model. Raises early with a
    clear message if the model is unknown or its API key is missing."""
    if model not in MODELS:
        raise KeyError(f"model {model!r} not in registry {sorted(MODELS)}")
    with _clients_lock:
        if model not in _clients:
            _clients[model] = MODELS[model].make_client()
        return _clients[model]


def generate(model: str, condition: str, register: str, item: dict,
             params: GenParams = DEFAULT_PARAMS) -> GenRecord:
    """Build the shared prompt, call the model with retry/backoff, return a record.

    Raises the last error if all retries fail (callers decide how to handle it).
    """
    prompt = build_prompt(condition, register, item)
    client = get_client(model)
    last_err: Optional[Exception] = None
    for attempt in range(params.max_retries):
        try:
            text = client.complete(prompt, params)
            return GenRecord(model, condition, register, item.get("id"), prompt, text)
        except Exception as e:  # transient / rate-limit / backend errors
            last_err = e
            if attempt == params.max_retries - 1:
                break
            wait = params.retry_base_sec * (2 ** attempt)
            print(f"    [{model}] API error ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    assert last_err is not None
    raise last_err