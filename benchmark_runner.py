"""
benchmark_runner.py
-------------------
Automatically benchmarks multiple LLM models on the same
set of questions and prints a detailed comparison report.

Run with:
    python benchmark_runner.py

No Streamlit needed — runs purely in terminal.
"""

import os
import sys
import time
import sqlite3
import re
import json
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(override=True)

# Clear any stale OpenAI key that might interfere
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("Gx"):
    os.environ.pop("OPENAI_API_KEY", None)
    print("⚠️  Removed invalid OPENAI_API_KEY from environment")

# ── Test questions — same for both models ─────────────────
BENCHMARK_QUESTIONS = [
    "What were the top 5 products by total sales?",
    "Which customers have the highest order frequency?",
    # "What is the total revenue per year?",
    # "Which employee handled the most orders?",
    # "Which product category generates the most revenue?",
]

# ── Models to compare ─────────────────────────────────────
MODELS_TO_TEST = [
    {
        "provider": "gemini",
        "api_key_env": "GOOGLE_API_KEY",
        "display_name": "Gemini 2.0 Flash Lite",
        "model_string": "gemini/gemini-2.0-flash-lite",
    },
    {
        "provider": "groq",
        "api_key_env": "GROQ_API_KEY",
        "display_name": "Groq Llama 3.1 8b",
        "model_string": "groq/llama-3.1-8b-instant",
    },
    # {
    #     "provider":    "openai",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "display_name": "GPT-4o Mini",
    #     "model_string": "gpt-4o-mini",
    # },
    {
        "provider": "mistral",
        "api_key_env": "MISTRAL_API_KEY",
        "display_name": "Mistral 7b",
        "model_string": "mistral/mistral-tiny",
    },
    {
        "provider": "cohere",
        "api_key_env": "COHERE_API_KEY",
        "display_name": "Cohere Command A",
        "model_string": "cohere/command-a-03-2025",
    },
]

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "northwind.db")
SEP = "=" * 75


# ── Result dataclass ──────────────────────────────────────
@dataclass
class BenchmarkResult:
    model: str
    question: str
    sql_generated: str = ""
    sql_success: bool = False
    rows_returned: int = 0
    execution_time_ms: float = 0.0
    hallucinated_tables: list = field(default_factory=list)
    hallucinated_cols: list = field(default_factory=list)
    quality_score: int = 0
    error: str = ""


# ── Helpers ───────────────────────────────────────────────
def get_all_tables() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def run_sql(sql: str):
    import pandas as pd
    import re as _re

    sql = _re.sub(r"```(?:sql)?", "", sql).strip().rstrip("`").strip()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        raise e
    finally:
        conn.close()


# REPLACE WITH THIS
def extract_tables(sql: str, all_tables: list[str]) -> tuple:
    real = {t.lower(): t for t in all_tables}
    matches = re.findall(
        r'(?:FROM|JOIN)\s+[\[\`"]?([^\s\[\]`,;()]+)[\]\`"]?', sql, re.IGNORECASE
    )

    # Tables with spaces that get split by regex — never flag these
    KNOWN_VALID = ["order", "details"]  # fragments of [Order Details]

    used, hallucinated = [], []
    for t in matches:
        t_clean = t.strip("[]`\"' ")

        # Skip known false positives from spaced table names
        if t_clean.lower() in KNOWN_VALID:
            continue

        if t_clean.lower() in real:
            used.append(real[t_clean.lower()])
        else:
            hallucinated.append(t_clean)
    return used, hallucinated


def switch_model(provider: str, api_key_env: str):
    """Switch the active LLM provider by updating env vars."""
    os.environ["LLM_PROVIDER"] = provider
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"❌ {api_key_env} not found in .env file. " f"Add it to test {provider}."
        )
    if provider in ("gemini", "google"):
        os.environ["GOOGLE_API_KEY"] = api_key
    elif provider == "groq":
        os.environ["GROQ_API_KEY"] = api_key
    # elif provider == "openai":
    #     os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "cohere":
        os.environ["COHERE_API_KEY"] = api_key
    elif provider == "mistral":
        os.environ["MISTRAL_API_KEY"] = api_key


def run_single_benchmark(
    question: str,
    model_config: dict,
    all_tables: list[str],
) -> BenchmarkResult:
    """Run one question with one model and return metrics."""

    result = BenchmarkResult(
        model=model_config["display_name"],
        question=question,
    )

    try:
        # Switch model
        switch_model(model_config["provider"], model_config["api_key_env"])

        # Import here so env vars are picked up fresh
        from rag.schema_retriever import retrieve_relevant_schema

        # RAG retrieval
        try:
            schema_context, _ = retrieve_relevant_schema(question, top_k=5)
        except Exception:
            schema_context = ""

        # Build and run SQL writer agent
        from crewai import Crew, Process
        from tasks.sql_task import create_sql_task

        sql_task = create_sql_task(question, schema_context)
        sql_crew = Crew(
            agents=[sql_task.agent],
            tasks=[sql_task],
            process=Process.sequential,
            verbose=False,  # silent during benchmark
        )

        start = time.time()
        for attempt in range(3):
            try:
                sql_result = sql_crew.kickoff()
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 30 * (attempt + 1)
                    print(f"     ⏳ Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        raise
                else:
                    raise
        elapsed = (time.time() - start) * 1000

        sql_query = sql_result.tasks_output[0].raw.strip()
        sql_query = re.sub(r"```(?:sql)?", "", sql_query).strip().rstrip("`").strip()

        result.sql_generated = sql_query
        result.execution_time_ms = elapsed

        # Execute SQL
        df = run_sql(sql_query)
        result.sql_success = not df.empty
        result.rows_returned = len(df)

        # Check hallucinations
        used, hallucinated = extract_tables(sql_query, all_tables)
        result.hallucinated_tables = hallucinated

        # Quality score
        score = 100
        if not result.sql_success:
            score -= 50
        if result.rows_returned == 0:
            score -= 30
        score -= len(hallucinated) * 20
        result.quality_score = max(0, score)

    except Exception as e:
        result.error = str(e)[:100]
        result.sql_success = False
        result.quality_score = 0

    return result


def print_question_comparison(
    question: str,
    results: list[BenchmarkResult],
):
    """Print side by side comparison for one question."""
    print(f"\n{'─' * 75}")
    print(f"❓ {question}")
    print(f"{'─' * 75}")

    for r in results:
        status = "✅" if r.sql_success else "❌"
        halluc = "None ✅" if not r.hallucinated_tables else str(r.hallucinated_tables)
        print(f"\n  [{r.model}]")
        print(
            f"    Status       : {status} "
            f"{'Success' if r.sql_success else 'Failed'}"
        )
        print(f"    Rows         : {r.rows_returned}")
        print(f"    Time         : {r.execution_time_ms:.0f}ms")
        print(f"    Quality      : {r.quality_score}/100")
        print(f"    Hallucinated : {halluc}")
        if r.sql_generated:
            print(f"    SQL          : {r.sql_generated[:80]}...")
        if r.error:
            print(f"    Error        : {r.error}")


def print_final_report(all_results: list[BenchmarkResult]):
    """Print the final aggregated comparison table."""
    models = list(dict.fromkeys(r.model for r in all_results))

    print(f"\n\n{SEP}")
    print("📊  FINAL MODEL COMPARISON REPORT")
    print(f"    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Questions tested: {len(BENCHMARK_QUESTIONS)}")
    print(SEP)

    print(
        f"\n{'Model':<28} {'Avg Quality':>12} "
        f"{'Success%':>10} {'Halluc%':>10} "
        f"{'Avg Rows':>10} {'Avg Time':>12}"
    )
    print(f"{'─' * 28} {'─' * 12} {'─' * 10} " f"{'─' * 10} {'─' * 10} {'─' * 12}")

    scores = {}
    for model in models:
        model_results = [r for r in all_results if r.model == model]
        total = len(model_results)

        avg_quality = sum(r.quality_score for r in model_results) / total
        success_pct = sum(1 for r in model_results if r.sql_success) / total * 100
        halluc_pct = (
            sum(1 for r in model_results if r.hallucinated_tables) / total * 100
        )
        avg_rows = sum(r.rows_returned for r in model_results) / total
        avg_time = sum(r.execution_time_ms for r in model_results) / total

        scores[model] = avg_quality

        print(
            f"{model:<28} {avg_quality:>11.1f} "
            f"{success_pct:>9.0f}% {halluc_pct:>9.0f}% "
            f"{avg_rows:>10.1f} {avg_time:>10.0f}ms"
        )

    print(f"\n{SEP}")

    # Winner
    if len(scores) > 1:
        winner = max(scores, key=scores.get)
        loser = min(scores, key=scores.get)
        diff = scores[winner] - scores[loser]
        print(f"🏆  WINNER     : {winner}")
        print(f"📈  Score gap  : {diff:.1f} points")
        print("💡  Verdict    : ", end="")
        if diff > 20:
            print(f"{winner} is significantly better")
        elif diff > 10:
            print(f"{winner} is moderately better")
        else:
            print("Both models perform similarly — " "choose based on speed/cost")

    print(SEP)

    # Per-question winner
    print("\n📋  QUESTION-BY-QUESTION WINNERS")
    print(f"{'─' * 75}")
    questions = list(dict.fromkeys(r.question for r in all_results))
    for q in questions:
        q_results = [r for r in all_results if r.question == q]
        best = max(q_results, key=lambda r: r.quality_score)
        print(f"  {q[:55]:<55} → {best.model}")

    print(f"\n{SEP}\n")


def save_report_to_file(all_results: list[BenchmarkResult]):
    """Save full results to a JSON file for later analysis."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "questions": BENCHMARK_QUESTIONS,
        "models": [m["display_name"] for m in MODELS_TO_TEST],
        "results": [
            {
                "model": r.model,
                "question": r.question,
                "sql": r.sql_generated,
                "success": r.sql_success,
                "rows": r.rows_returned,
                "time_ms": r.execution_time_ms,
                "hallucinated_tables": r.hallucinated_tables,
                "quality_score": r.quality_score,
                "error": r.error,
            }
            for r in all_results
        ],
    }
    # Ensure folder exists
    os.makedirs("benchmark files", exist_ok=True)

    # Create filename inside folder
    filename = os.path.join(
        "benchmark files", f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📁 Full report saved to: {filename}")


# ── Main ──────────────────────────────────────────────────
def main():
    print(f"\n{SEP}")
    print("🔬  NORTHWIND AI — MODEL BENCHMARK")
    print(
        f"    Testing {len(MODELS_TO_TEST)} models "
        f"× {len(BENCHMARK_QUESTIONS)} questions"
    )
    print(SEP)

    # Check API keys first
    print("\n🔑 Checking API keys...")
    for model in MODELS_TO_TEST:
        key = os.getenv(model["api_key_env"])
        status = "✅ Found" if key else "❌ MISSING"
        print(f"   {model['display_name']:<25} " f"{model['api_key_env']}: {status}")
        if not key:
            print(
                f"   ⚠️  Add {model['api_key_env']} "
                f"to your .env file to test this model"
            )

    all_tables = get_all_tables()
    print(f"\n📊 Database: {len(all_tables)} tables found")

    all_results = []

    # Run each question on each model
    for q_num, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n{SEP}")
        print(f"Question {q_num}/{len(BENCHMARK_QUESTIONS)}: {question}")
        print(SEP)

        question_results = []
        for model_config in MODELS_TO_TEST:
            key = os.getenv(model_config["api_key_env"])
            if not key:
                print(f"  ⏭️  Skipping {model_config['display_name']} " f"— no API key")
                continue

            print(f"\n  🤖 Running with {model_config['display_name']}...")
            result = run_single_benchmark(question, model_config, all_tables)
            question_results.append(result)
            all_results.append(result)

            # Brief result per model
            status = "✅" if result.sql_success else "❌"
            print(
                f"     {status} Score: {result.quality_score}/100  "
                f"Rows: {result.rows_returned}  "
                f"Time: {result.execution_time_ms:.0f}ms"
            )

            # Rate limit pause between models
            print("     ⏳ Waiting 5s before next model...")
            time.sleep(20)

        # Side by side for this question
        if question_results:
            print_question_comparison(question, question_results)

        # Pause between questions to avoid rate limits
        if q_num < len(BENCHMARK_QUESTIONS):
            print("\n  ⏳ Waiting 10s before next question...")
            time.sleep(60)

    # Final report
    if all_results:
        print_final_report(all_results)
        save_report_to_file(all_results)
    else:
        print("\n❌ No results — check your API keys in .env")


if __name__ == "__main__":
    main()
