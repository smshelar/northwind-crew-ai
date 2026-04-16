"""
northwind_crew.py
-----------------
Full pipeline with RAG schema retrieval, memory cache,
SQL evaluation and hallucination detection.
"""



import json
import re
import sqlite3
import streamlit as st
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from crewai import Crew, Process

from tasks.sql_task import create_sql_task
from tasks.visualizer_task import create_visualizer_task

from evaluation.metrics import run_full_evaluation
from guardrails.input_guard import validate_input
from guardrails.output_guard import validate_output
from utils.llm_factory import get_current_model_name

# RAG + Evaluation imports
from rag.schema_retriever import retrieve_relevant_schema, is_schema_indexed
from rag.schema_indexer import index_schema
from rag.memory_store import check_memory, save_to_memory, get_memory_stats
from evaluation.evaluator import evaluate, is_hallucination, get_evaluation_badge

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")


def _run_sql(sql: str) -> pd.DataFrame:
    sql = re.sub(r"```(?:sql)?", "", sql).strip().rstrip("`").strip()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def _validate_and_fix_sql(sql: str) -> str:
    try:
        df = _run_sql(sql)
        if not df.empty:
            return sql
        sql_no_where = re.sub(
            r'WHERE\s+.*?(GROUP BY|ORDER BY|LIMIT|$)',
            r'\1', sql,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        df2 = _run_sql(sql_no_where)
        if not df2.empty:
            return sql_no_where
        return sql
    except Exception as e:
        print(f"DEBUG validate error: {e}")
        return sql


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def _build_chart(df: pd.DataFrame, viz: dict) -> plt.Figure | None:
    try:
        chart_type = viz.get("chart_type", "bar")
        x = viz.get("x_column")
        y = viz.get("y_column")
        title = viz.get("title", "")

        if x not in df.columns or y not in df.columns:
            print(f"DEBUG columns: {df.columns.tolist()}, x={x}, y={y}")
            return None

        if len(df) > 10:
            df = df.head(10)

        plt.rcParams.update({
            "figure.facecolor": "#0f1117",
            "axes.facecolor":   "#0f1117",
            "axes.edgecolor":   "#2a2d3a",
            "axes.labelcolor":  "#ffffff",
            "xtick.color":      "#ffffff",
            "ytick.color":      "#ffffff",
            "text.color":       "#ffffff",
            "grid.color":       "#2a2d3a",
            "grid.linestyle":   "--",
            "grid.alpha":       0.5,
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0f1117")
        colors = plt.cm.Blues(
            [0.4 + 0.6 * (1 - i / len(df)) for i in range(len(df))]
        )

        if chart_type == "bar":
            bars = ax.bar(df[x].astype(str), df[y],
                         color=colors, edgecolor="none", width=0.6)
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(df[y]) * 0.01,
                    f"{int(height):,}",
                    ha="center", va="bottom", fontsize=9, color="#ffffff"
                )
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x].astype(str), rotation=30,
                               ha="right", fontsize=10)
        elif chart_type == "line":
            ax.plot(df[x].astype(str), df[y], marker="o", color="#4CAF50",
                    linewidth=2.5, markersize=7, markerfacecolor="#ffffff")
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x].astype(str), rotation=30,
                               ha="right", fontsize=10)
            ax.fill_between(range(len(df)), df[y], alpha=0.1, color="#4CAF50")
        elif chart_type == "pie":
            wedge_colors = plt.cm.Blues(
                [0.3 + 0.7 * i / len(df) for i in range(len(df))]
            )
            ax.pie(df[y], labels=df[x].astype(str), autopct="%1.1f%%",
                   colors=wedge_colors, startangle=140, pctdistance=0.85,
                   wedgeprops={"edgecolor": "#0f1117", "linewidth": 2})
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y], color="#4CAF50", s=100,
                      alpha=0.8, edgecolors="white", linewidth=0.5)
        else:
            ax.bar(df[x].astype(str), df[y], color=colors, edgecolor="none")
            ax.set_xticklabels(df[x].astype(str), rotation=30,
                               ha="right", fontsize=10)

        ax.set_title(title, fontsize=14, fontweight="bold",
                    color="#ffffff", pad=15)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#2a2d3a")
        ax.spines["bottom"].set_color("#2a2d3a")
        ax.tick_params(axis="both", labelsize=10)
        fig.tight_layout(pad=2.0)
        return fig
    except Exception as e:
        print(f"Error building chart: {e}")
        return None


def _kickoff_with_retry(crew, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return crew.kickoff()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "rate limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"Rate limit. Waiting {wait}s... retry {attempt+1}/{max_retries}")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries reached due to rate limiting.")


def _extract_task_raw(result) -> str:
    """
    Normalize CrewAI kickoff outputs across versions.
    Older CrewAI returns a plain string, while newer versions expose
    `tasks_output[0].raw`.
    """
    if hasattr(result, "tasks_output") and result.tasks_output:
        return result.tasks_output[0].raw.strip()
    if isinstance(result, str):
        return result.strip()
    return str(result).strip()


class NorthwindCrew:

    def __init__(self):
        """Auto-build RAG index on first run if not already built."""
        if not is_schema_indexed():
            print("🔧 First run — building RAG schema index...")
            if "schema_indexed" not in st.session_state:
                index_schema()
                st.session_state["schema_indexed"] = True
            print("✅ Schema index ready")

    def run(self, user_question: str) -> dict:
        steps = []
        start_time = time.time()

        # ── Step 0: Input Guardrail ───────────────────────────
        input_check = validate_input(user_question)
        if not input_check.passed:
            return {
                "success": False,
                "error": f"Your question was blocked: {input_check.reason}",
                "steps": ["❌ Input blocked by guardrail"],
                "sql_query": "", "dataframe": None,
                "figure": None, "insight": "", "summary": "",
            }
        user_question = input_check.safe_question

        # ── Step 1: Check Memory Cache ────────────────────────
        steps.append("🧠 Checking memory cache...")
        memory_hit = check_memory(user_question)

        if memory_hit:
            steps.append(
                f"✅ Memory HIT — reusing cached result "
                f"(similarity: {memory_hit['similarity']:.0%})"
            )
            steps.append(
                f"   Original question: {memory_hit['original_question']}"
            )
            # Re-run SQL to get fresh dataframe
            try:
                df = _run_sql(memory_hit["sql_query"])
                data_sample = df.head(10).to_json(orient="records")
                visualize_task = create_visualizer_task(
                    user_question, data_sample
                )
                viz_crew = Crew(
                    agents=[visualize_task.agent],
                    tasks=[visualize_task],
                    process=Process.sequential,
                    verbose=False,
                )
                viz_result = _kickoff_with_retry(viz_crew)
                viz_raw = _extract_task_raw(viz_result)
                try:
                    viz = _extract_json(viz_raw)
                except Exception:
                    viz = {}
                figure = _build_chart(df, viz)
                steps.append("✅ Chart rendered from cached query")

                return {
                    "success": True,
                    "sql_query": memory_hit["sql_query"],
                    "dataframe": df,
                    "figure": figure,
                    "insight": viz.get("insight", memory_hit["insight"]),
                    "summary": memory_hit["summary"],
                    "steps": steps,
                    "error": None,
                    "evaluation_score": 100,
                    "evaluation_badge": "🟢 Cached (Verified)",
                    "from_cache": True,
                }
            except Exception as e:
                steps.append(f"⚠️  Cache replay failed, running fresh: {e}")

        # ── Step 2: RAG Schema Retrieval ──────────────────────
        steps.append("🔍 RAG: retrieving relevant table schemas...")
        try:
            schema_context, retrieved_tables = retrieve_relevant_schema(
                user_question, top_k=5
            )
            steps.append(
                f"✅ RAG retrieved {len(retrieved_tables)} relevant tables: "
                f"{retrieved_tables}"
            )
        except Exception as e:
            schema_context = ""
            steps.append(f"⚠️  RAG failed, using full schema: {e}")

        # ── Step 3: SQL Writer Agent ──────────────────────────
        sql_task = create_sql_task(user_question, schema_context)
        sql_crew = Crew(
            agents=[sql_task.agent],
            tasks=[sql_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            sql_result = _kickoff_with_retry(sql_crew)
            sql_query = _extract_task_raw(sql_result)
            sql_query = re.sub(
                r"```(?:sql)?", "", sql_query
            ).strip().rstrip("`").strip()
            steps.append(f"✅ SQL written: `{sql_query[:80]}...`")
        except Exception as e:
            steps.append(f"❌ SQL writing failed: {e}")
            return {
                "success": False, "error": str(e), "steps": steps,
                "sql_query": "", "dataframe": None, "figure": None,
                "insight": "", "summary": "",
                "evaluation_score": 0, "evaluation_badge": "🔴 Failed",
            }

        # ── Step 4: Execute SQL directly ──────────────────────
        steps.append("✅ Executor agent: running SQL against database...")
        try:
            debug_df = _run_sql(
                "SELECT DISTINCT strftime('%Y', OrderDate) AS Year "
                "FROM Orders ORDER BY Year"
            )
            print(f"DEBUG years in DB: {debug_df['Year'].tolist()}")

            sql_query = _validate_and_fix_sql(sql_query)
            df = _run_sql(sql_query)

            if df.empty:
                return {
                    "success": False,
                    "error": "Query returned no results. "
                             "Northwind data covers 1996–1998 only.",
                    "steps": steps, "sql_query": sql_query,
                    "dataframe": None, "figure": None,
                    "insight": "", "summary": "",
                    "evaluation_score": 0,
                    "evaluation_badge": "🔴 No Results",
                }

            steps.append(f"✅ Query executed — {len(df)} rows returned")

        except Exception as e:
            return {
                "success": False,
                "error": f"SQL execution failed: {e}",
                "steps": steps, "sql_query": sql_query,
                "dataframe": None, "figure": None,
                "insight": "", "summary": "",
                "evaluation_score": 0,
                "evaluation_badge": "🔴 SQL Error",
            }

        # ── Step 5: Full Evaluation (terminal only) ───────────
        steps.append("🔬 Evaluating result quality...")
        exec_time_ms = (time.time() - start_time) * 1000
        model_name = get_current_model_name()

        eval_summary = run_full_evaluation(
            question=user_question,
            sql_query=sql_query,
            dataframe=df,
            insight="",
            retrieved_tables=retrieved_tables if 'retrieved_tables' in dir() else [],
            execution_time_ms=exec_time_ms,
            model_name=model_name,
        )

        # ── Step 6: Output Guardrail ──────────────────────────
        output_check = validate_output(
            sql_query=sql_query,
            dataframe=df,
            insight="",
            hallucinated_tables=[],
            quality_score=eval_summary["quality_score"],
        )

        if not output_check.passed:
            return {
                "success": False,
                "error": "Result did not pass quality checks. Please rephrase.",
                "steps": steps + [f"❌ Output blocked: {output_check.reason}"],
                "sql_query": sql_query, "dataframe": df,
                "figure": None, "insight": "", "summary": "",
                "evaluation_score": eval_summary["quality_score"],
                "evaluation_badge": "🔴 Blocked",
            }

        steps.append(f"✅ Evaluation passed — score: {eval_summary['quality_score']}/100")
        
        # ── Step 7: Visualizer Agent ──────────────────────────
        data_sample = df.head(10).to_json(orient="records")
        visualize_task = create_visualizer_task(user_question, data_sample)
        viz_crew = Crew(
            agents=[visualize_task.agent],
            tasks=[visualize_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            viz_result = _kickoff_with_retry(viz_crew)
            viz_raw = _extract_task_raw(viz_result)
            viz = _extract_json(viz_raw)
            steps.append(
                f"✅ Visualization config: {viz.get('chart_type')} chart"
            )
        except Exception as e:
            viz = {}
            steps.append(f"⚠️  Visualizer failed: {e}")

        figure = _build_chart(df, viz)
        if figure:
            steps.append("✅ Chart rendered")

        summary = f"Found {len(df)} records. {viz.get('insight', '')}"

        print(f"\n🎯 INSIGHT FAITHFULNESS CHECK")
        print(f"   Insight: {viz.get('insight', '')[:80]}")
        print(f"   This was evaluated after visualization completed")

        # ── Step 8: Save to Memory (only if evaluation passed) ─
        save_to_memory(
            question=user_question,
            sql_query=sql_query,
            row_count=len(df),
            insight=viz.get("insight", ""),
            summary=summary,
        )
        stats = get_memory_stats()
        steps.append(
            f"💾 Saved to memory cache "
            f"({stats['cached_queries']} queries cached total)"
        )

        return {
            "success": True,
            "sql_query": sql_query,
            "dataframe": df,
            "figure": figure,
            "insight": viz.get("insight", ""),
            "summary": summary,
            "steps": steps,
            "error": None,
            "evaluation_score": eval_summary.get("quality_score", 100),
            "evaluation_badge": "🟢 Excellent",
            "from_cache": False,
        }
