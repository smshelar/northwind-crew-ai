# """
# llm_factory.py
# --------------
# Returns the correct LLM config for CrewAI based on LLM_PROVIDER
# Compatible with Streamlit + CrewAI v0.11+
# """

# import os

# def get_llm(temperature: float = 0.2):
#     # Remove conflicting keys
#     os.environ.pop("OPENAI_API_KEY", None)

#     # Prefer Streamlit secrets if available
#     try:
#         import streamlit as st
#         provider = st.secrets.get("LLM_PROVIDER", "groq").lower().strip()
#         secrets = st.secrets
#     except:
#         provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
#         secrets = os.environ

#     # ── Google Gemini ─────────────────────────────────────────
#     if provider in ("google", "gemini"):
#         api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY not found")

#         return {
#             "model": "gemini/gemini-2.0-flash-lite",
#             "temperature": temperature,
#             "api_key": api_key,
#         }

#     # ── OpenAI ────────────────────────────────────────────────
#     elif provider == "openai":
#         api_key = secrets.get("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY not found")

#         return {
#             "model": "gpt-4o-mini",
#             "temperature": temperature,
#             "api_key": api_key,
#         }

#     # ── Groq ──────────────────────────────────────────────────
#     elif provider == "groq":
#         api_key = secrets.get("GROQ_API_KEY")
#         if not api_key:
#             raise ValueError("GROQ_API_KEY not found")

#         return {
#             "model": "groq/llama-3.1-8b-instant",
#             "temperature": temperature,
#             "api_key": api_key,
#         }

#     # ── Mistral ───────────────────────────────────────────────
#     elif provider == "mistral":
#         api_key = secrets.get("MISTRAL_API_KEY")
#         if not api_key:
#             raise ValueError("MISTRAL_API_KEY not found")

#         return {
#             "model": "mistral/mistral-small-latest",
#             "temperature": temperature,
#             "api_key": api_key,
#         }

#     # ── Cohere ────────────────────────────────────────────────
#     elif provider == "cohere":
#         api_key = secrets.get("COHERE_API_KEY")
#         if not api_key:
#             raise ValueError("COHERE_API_KEY not found")

#         return {
#             "model": "cohere/command-a-03-2025",
#             "temperature": temperature,
#             "api_key": api_key,
#         }

#     else:
#         raise ValueError(
#             f"Unknown LLM_PROVIDER: '{provider}'. "
#             f"Use: google | openai | groq | cohere | mistral"
#         )


# def get_current_model_name() -> str:
#     provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
#     models = {
#         "google":  "gemini-2.0-flash-lite",
#         "gemini":  "gemini-2.0-flash-lite",
#         "groq":    "groq/llama-3.1-8b-instant",
#         "openai":  "gpt-4o-mini",
#         "cohere":  "cohere/command-a-03-2025",
#         "mistral": "mistral/mistral-small-latest",
#     }
#     return models.get(provider, provider)


"""
llm_factory.py
--------------
Builds an LLM object compatible with both older CrewAI releases
(for example `crewai==0.11.2` on Streamlit Cloud) and newer versions.
"""

import os


def _load_provider_and_secrets():
    """Read provider + secrets from Streamlit when available, else environment."""
    try:
        import streamlit as st

        provider = st.secrets.get("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "groq"))
        return provider.lower().strip(), st.secrets
    except Exception:
        provider = os.getenv("LLM_PROVIDER", "groq")
        return provider.lower().strip(), os.environ


def _require_key(secrets, *names: str) -> str:
    for name in names:
        value = secrets.get(name)
        if value:
            return value
    raise ValueError(f"Required API key not found. Tried: {', '.join(names)}")


def _build_modern_crewai_llm(model: str, temperature: float, api_key: str, base_url: str | None = None):
    """Use CrewAI's native LLM wrapper when available."""
    try:
        from crewai import LLM

        kwargs = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return LLM(**kwargs)
    except Exception:
        return None


def _build_langchain_chat_openai(model: str, temperature: float, api_key: str, base_url: str | None = None):
    """Fallback for older CrewAI versions that expect a LangChain chat model."""
    try:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model_name": model,
            "temperature": temperature,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    except Exception:
        return None


def get_llm(temperature: float = 0.2):
    """Return an LLM object compatible with the installed CrewAI stack."""
    provider, secrets = _load_provider_and_secrets()

    if provider in ("google", "gemini"):
        api_key = _require_key(secrets, "GEMINI_API_KEY", "GOOGLE_API_KEY")
        os.environ["GEMINI_API_KEY"] = api_key

        llm = _build_modern_crewai_llm(
            model="gemini/gemini-2.0-flash-lite",
            temperature=temperature,
            api_key=api_key,
        )
        if llm is not None:
            return llm

        raise ValueError(
            "Gemini requires a newer CrewAI runtime in this project. "
            "Upgrade CrewAI or switch LLM_PROVIDER to openai/groq/mistral."
        )

    if provider == "openai":
        api_key = _require_key(secrets, "OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key

        llm = _build_modern_crewai_llm(
            model="openai/gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
        )
        if llm is not None:
            return llm

        llm = _build_langchain_chat_openai(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
        )
        if llm is not None:
            return llm

    if provider == "groq":
        api_key = _require_key(secrets, "GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = api_key

        llm = _build_modern_crewai_llm(
            model="groq/llama-3.1-8b-instant",
            temperature=temperature,
            api_key=api_key,
        )
        if llm is not None:
            return llm

        llm = _build_langchain_chat_openai(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        if llm is not None:
            return llm

    if provider == "mistral":
        api_key = _require_key(secrets, "MISTRAL_API_KEY")
        os.environ["MISTRAL_API_KEY"] = api_key

        llm = _build_modern_crewai_llm(
            model="mistral/mistral-small-latest",
            temperature=temperature,
            api_key=api_key,
        )
        if llm is not None:
            return llm

        llm = _build_langchain_chat_openai(
            model="mistral-small-latest",
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
        )
        if llm is not None:
            return llm

    if provider == "cohere":
        api_key = _require_key(secrets, "COHERE_API_KEY")
        os.environ["COHERE_API_KEY"] = api_key

        llm = _build_modern_crewai_llm(
            model="cohere/command-a-03-2025",
            temperature=temperature,
            api_key=api_key,
        )
        if llm is not None:
            return llm

        raise ValueError(
            "Cohere requires a newer CrewAI runtime in this project. "
            "Upgrade CrewAI or switch LLM_PROVIDER to openai/groq/mistral."
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. "
        "Use: google | gemini | openai | groq | cohere | mistral"
    )


def get_current_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    models = {
        "google": "gemini/gemini-2.0-flash-lite",
        "gemini": "gemini/gemini-2.0-flash-lite",
        "groq": "groq/llama-3.1-8b-instant",
        "openai": "gpt-4o-mini",
        "cohere": "cohere/command-a-03-2025",
        "mistral": "mistral/mistral-small-latest",
    }
    return models.get(provider, provider)
