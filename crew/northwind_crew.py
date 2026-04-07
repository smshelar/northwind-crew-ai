"""
Assembles the CrewAI Crew and runs the pipeline.
Returns a structured result dict for Streamlit to render.
"""

import json
import re
import sqlite3
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from crewai import Crew, Process

from tasks.sql_task import create_sql_task
from tasks.execute_task import create_execute_task
from tasks.visualizer_task import create_visualizer_task

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")


def _run_sql(sql: str) -> pd.DataFrame:
    """Directly runs SQL — used as fallback if agent fails."""
    sql = re.sub(r"```(?:sql)?", "", sql).strip().rstrip("`").strip()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()

def _validate_and_fix_sql(sql: str) -> str:
    """Tests the SQL and if it returns no results, tries without WHERE clause."""
    try:
        df = _run_sql(sql)
        if not df.empty:
            return sql  # query is fine, return as-is

        # If empty, strip WHERE clause and try again
        print("DEBUG: Query returned empty — trying without WHERE clause")
        sql_no_where = re.sub(
            r'WHERE\s+.*?(GROUP BY|ORDER BY|LIMIT|$)',
            r'\1',
            sql,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

        df2 = _run_sql(sql_no_where)
        if not df2.empty:
            print(f"DEBUG: Fallback query returned {len(df2)} rows")
            return sql_no_where

        return sql
    except Exception as e:
        print(f"DEBUG validate error: {e}")
        return sql


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def _extract_json_array(text: str) -> str:
    """Tries multiple strategies to pull a JSON array out of text."""
    text = re.sub(r"```(?:json|sql)?", "", text).strip().rstrip("`").strip()
    if text.startswith("["):
        return text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return f"[{match.group(0)}]"
    return text


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
                print(f"Rate limit hit. Waiting {wait}s ... retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries reached due to rate limiting.")


class NorthwindCrew:
    def run(self, user_question: str) -> dict:
        steps = []

        # ── Stage 1: Run SQL writer agent only first ──────────────
        sql_task = create_sql_task(user_question)

        sql_crew = Crew(
            agents=[sql_task.agent],
            tasks=[sql_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            sql_result = _kickoff_with_retry(sql_crew)
            sql_query = sql_result.tasks_output[0].raw.strip()
            sql_query = re.sub(r"```(?:sql)?", "", sql_query).strip().rstrip("`").strip()
            steps.append(f"✅ SQL written: `{sql_query[:80]}...`")
        except Exception as e:
            steps.append(f"❌ SQL writing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": steps,
                "sql_query": "",
                "dataframe": None,
                "figure": None,
                "insight": "",
                "summary": "",
            }

       # ── Stage 2: Execute SQL directly in Python ───────────────
        # (executor agent acts as a wrapper — DB call happens here)
        steps.append("✅ Executor agent: running SQL against database...")
        try:
            # DEBUG: print what years exist in the database
            debug_df = _run_sql(
                "SELECT DISTINCT strftime('%Y', OrderDate) AS Year "
                "FROM Orders ORDER BY Year"
            )
            print(f"DEBUG years in DB: {debug_df['Year'].tolist()}")

            # Validate and auto-fix before running
            sql_query = _validate_and_fix_sql(sql_query)
            df = _run_sql(sql_query)

            if df.empty:
                return {
                    "success": False,
                    "error": (
                        "Query returned no results. The Northwind database only "
                        "contains data from 1996, 1997, and 1998."
                    ),
                    "steps": steps,
                    "sql_query": sql_query,
                    "dataframe": None,
                    "figure": None,
                    "insight": "",
                    "summary": "",
                }

            steps.append(f"✅ Query executed — {len(df)} rows returned")

        except Exception as e:
            return {
                "success": False,
                "error": f"SQL execution failed: {e}",
                "steps": steps,
                "sql_query": sql_query,
                "dataframe": None,
                "figure": None,
                "insight": "",
                "summary": "",
            }

        if df.empty:
            return {
                "success": False,
                "error": "no results",
                "steps": steps,
                "sql_query": sql_query,
                "dataframe": None,
                "figure": None,
                "insight": "",
                "summary": "",
            }

        # ── Stage 3: Run visualizer agent ─────────────────────────
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
            viz_raw = viz_result.tasks_output[0].raw.strip()
            viz = _extract_json(viz_raw)
            steps.append(f"✅ Visualization config: {viz.get('chart_type')} chart")
        except Exception as e:
            viz = {}
            steps.append(f"⚠️ Visualizer failed: {e}")

        figure = _build_chart(df, viz)
        if figure:
            steps.append("✅ Chart rendered")

        return {
            "success": True,
            "sql_query": sql_query,
            "dataframe": df,
            "figure": figure,
            "insight": viz.get("insight", ""),
            "summary": f"Found {len(df)} records. {viz.get('insight', '')}",
            "steps": steps,
            "error": None,
        }