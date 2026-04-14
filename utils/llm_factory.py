"""
llm_factory.py
--------------
Returns the correct LLM for CrewAI based on LLM_PROVIDER in .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_llm(temperature: float = 0.2):
    # Always remove bad OpenAI key first
    os.environ.pop("OPENAI_API_KEY", None)

    provider = os.getenv("LLM_PROVIDER", "google").lower().strip()

    # ── Google Gemini ─────────────────────────────────────────
    if provider in ("google", "gemini"):
        from crewai import LLM
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        os.environ["GOOGLE_API_KEY"] = api_key
        return LLM(
            model="gemini/gemini-2.0-flash-lite",
            api_key=api_key,
            temperature=temperature,
        )

    # ── OpenAI ────────────────────────────────────────────────
    elif provider == "openai":
        from crewai import LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        return LLM(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
        )

    # ── Groq ──────────────────────────────────────────────────
    elif provider == "groq":
        from crewai import LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["GROQ_API_KEY"] = api_key
        return LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=api_key,
            temperature=temperature,
        )

    # ── Mistral ───────────────────────────────────────────────
    elif provider == "mistral":
        from crewai import LLM
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file")
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["MISTRAL_API_KEY"] = api_key
        return LLM(
            model="mistral/mistral-small-latest",
            api_key=api_key,
            temperature=temperature,
        )

    # ── Cohere ────────────────────────────────────────────────
    elif provider == "cohere":
        from crewai import LLM
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in .env file")
        os.environ.pop("OPENAI_API_KEY", None)  # prevent LiteLLM fallback
        os.environ["COHERE_API_KEY"] = api_key
        return LLM(
            model="cohere/command-a-03-2025",  # cohere/ prefix is required
            api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Use: google | openai | groq | cohere | mistral"
        )


def get_current_model_name() -> str:
    """Returns the current model name for metrics tracking."""
    provider = os.getenv("LLM_PROVIDER", "google").lower().strip()
    models = {
        "google":  "gemini-2.0-flash-lite",
        "gemini":  "gemini-2.0-flash-lite",
        "groq":    "groq/llama-3.1-8b-instant",
        "openai":  "gpt-4o-mini",
        "cohere":  "cohere/command-r-plus",
        "mistral": "mistral/mistral-small-latest",
    }
    return models.get(provider, provider)