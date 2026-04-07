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
    provider = os.getenv("LLM_PROVIDER", "google").lower().strip()

    if provider in ("google", "gemini"):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        os.environ["GOOGLE_API_KEY"] = api_key

        # CrewAI uses LiteLLM format strings natively
        from crewai import LLM
        return LLM(
            model="gemini/gemini-2.0-flash",
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        from crewai import LLM
        return LLM(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        from crewai import LLM
        return LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use google | openai | groq")