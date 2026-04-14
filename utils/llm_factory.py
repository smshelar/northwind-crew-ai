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
Returns CrewAI-compatible LLM model strings based on LLM_PROVIDER.
Also hydrates provider API keys into environment variables so CrewAI can
initialize the selected provider across local/dev/Streamlit deployments.
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


def get_llm(temperature: float = 0.2) -> str:
    """
    Return a CrewAI-compatible model string for the configured provider.

    NOTE: `temperature` is kept in the signature for backward compatibility
    with existing call sites, even though CrewAI's Agent uses the model string.
    """
    _ = temperature  # intentionally unused, retained for compatibility
    provider, secrets = _load_provider_and_secrets()

    if provider in ("google", "gemini"):
        os.environ["GEMINI_API_KEY"] = _require_key(secrets, "GEMINI_API_KEY", "GOOGLE_API_KEY")
        return "gemini/gemini-2.0-flash-lite"

    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = _require_key(secrets, "OPENAI_API_KEY")
        return "gpt-4o-mini"

    if provider == "groq":
        os.environ["GROQ_API_KEY"] = _require_key(secrets, "GROQ_API_KEY")
        return "groq/llama-3.1-8b-instant"

    if provider == "mistral":
        os.environ["MISTRAL_API_KEY"] = _require_key(secrets, "MISTRAL_API_KEY")
        return "mistral/mistral-small-latest"

    if provider == "cohere":
        os.environ["COHERE_API_KEY"] = _require_key(secrets, "COHERE_API_KEY")
        return "cohere/command-a-03-2025"

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
