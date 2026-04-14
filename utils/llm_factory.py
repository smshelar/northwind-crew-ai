"""
llm_factory.py
--------------
Returns the correct LLM config for CrewAI based on LLM_PROVIDER
Compatible with Streamlit + CrewAI v0.11+
"""

import os

def get_llm(temperature: float = 0.2):
    # Remove conflicting keys
    os.environ.pop("OPENAI_API_KEY", None)

    # Prefer Streamlit secrets if available
    try:
        import streamlit as st
        provider = st.secrets.get("LLM_PROVIDER", "groq").lower().strip()
        secrets = st.secrets
    except:
        provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        secrets = os.environ

    # ── Google Gemini ─────────────────────────────────────────
    if provider in ("google", "gemini"):
        api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")

        return {
            "model": "gemini/gemini-2.0-flash-lite",
            "temperature": temperature,
            "api_key": api_key,
        }

    # ── OpenAI ────────────────────────────────────────────────
    elif provider == "openai":
        api_key = secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        return {
            "model": "gpt-4o-mini",
            "temperature": temperature,
            "api_key": api_key,
        }

    # ── Groq ──────────────────────────────────────────────────
    elif provider == "groq":
        api_key = secrets.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

        return {
            "model": "groq/llama-3.1-8b-instant",
            "temperature": temperature,
            "api_key": api_key,
        }

    # ── Mistral ───────────────────────────────────────────────
    elif provider == "mistral":
        api_key = secrets.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")

        return {
            "model": "mistral/mistral-small-latest",
            "temperature": temperature,
            "api_key": api_key,
        }

    # ── Cohere ────────────────────────────────────────────────
    elif provider == "cohere":
        api_key = secrets.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found")

        return {
            "model": "cohere/command-a-03-2025",
            "temperature": temperature,
            "api_key": api_key,
        }

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Use: google | openai | groq | cohere | mistral"
        )


def get_current_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    models = {
        "google":  "gemini-2.0-flash-lite",
        "gemini":  "gemini-2.0-flash-lite",
        "groq":    "groq/llama-3.1-8b-instant",
        "openai":  "gpt-4o-mini",
        "cohere":  "cohere/command-a-03-2025",
        "mistral": "mistral/mistral-small-latest",
    }
    return models.get(provider, provider)