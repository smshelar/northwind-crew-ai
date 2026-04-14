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
