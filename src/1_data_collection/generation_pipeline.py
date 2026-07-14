"""Shared generation pipeline for the CLUU thesis corpora (formal + informal).

Single source of truth for:
  - the four prompt CONDITIONS (baseline / human_like / detector_aware /
    detector_evasive) in both REGISTERS (formal = thesis abstracts, informal =
    forum comments), and
  - the model registry + generation call, behind an OpenAI-compatible client.

INVARIANT (do not break): every backend is called byte-identically -- a single
user message, no system prompt, and no temperature override (so each provider's
default of 1.0 is used). Mistral-Small-3.2 recommends temperature ~0.15 and a
system prompt; those are DELIBERATELY NOT applied, so the prompt/condition stays
the only variable across models. If you ever need to override, do it via
GenParams (and document it) -- never special-case a single backend.

Adding a model = one entry in MODELS. Prompts never change per model.

The two adversarial generation scripts import build_prompt / generate / MODELS
from here, and so does smoke_test.py, so a prompt edited here changes every
consumer at once.
"""
from __future__ import annotations

import os
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
# Generation params + client. Defaults preserve the cross-backend invariant.
# ---------------------------------------------------------------------------
@dataclass
class GenParams:
    """Generation params. Defaults keep the invariant: no temperature override
    (provider default 1.0) and no system prompt. Override only deliberately."""
    temperature: Optional[float] = None
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
    make_client: Callable[[], OpenAICompatibleClient]


# Adding a model = one entry here. Prompts and call contract never change.
MODELS: dict[str, ModelSpec] = {
    # OpenAI GPT-5.2 -- the thesis baseline model (default OpenAI endpoint).
    "gpt-5.2": ModelSpec(
        make_client=lambda: OpenAICompatibleClient(
            model="gpt-5.2", base_url=None, api_key_env="OPENAI_API_KEY",
        ),
    ),
    # Mistral-Small-3.2-24B-Instruct-2506 via OpenRouter (OpenAI-compatible).
    # Called with the SAME contract as every backend: temp 1.0, no system prompt.
    # Override the slug/endpoint with MISTRAL_MODEL / MISTRAL_BASE_URL if you
    # self-host (vLLM/NIM slug: mistralai/Mistral-Small-3.2-24B-Instruct-2506).
    "mistral": ModelSpec(
        make_client=lambda: OpenAICompatibleClient(
            model=os.environ.get("MISTRAL_MODEL", "mistralai/mistral-small-3.2-24b-instruct"),
            base_url=os.environ.get("MISTRAL_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key_env="MISTRAL_API_KEY",
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
