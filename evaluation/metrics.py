"""
metrics.py
----------
RAG and SQL evaluation metrics.
All output goes to TERMINAL ONLY
"""

import os
import sys
import re
import sqlite3
from dataclasses import dataclass, field
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")

SEPARATOR = "=" * 65


@dataclass
class RAGMetrics:
    """Metrics for the RAG schema retrieval step."""

    question: str
    retrieved_tables: list[str]
    all_tables: list[str]
    relevant_tables_ground_truth: list[str] = field(default_factory=list)
    k: int = 5

    @property
    def precision_at_k(self) -> float:
        """
        Precision@K = relevant retrieved / total retrieved
        How many of the K tables we retrieved were actually useful?
        """
        if not self.retrieved_tables:
            return 0.0
        if not self.relevant_tables_ground_truth:
            # No ground truth — estimate by checking if retrieved
            # tables appear in the generated SQL
            return 1.0  # assume all retrieved were used
        relevant_and_retrieved = set(self.retrieved_tables) & set(
            self.relevant_tables_ground_truth
        )
        return len(relevant_and_retrieved) / len(self.retrieved_tables)

    @property
    def recall_at_k(self) -> float:
        """
        Recall@K = relevant retrieved / total relevant
        Did we miss any important tables?
        """
        if not self.relevant_tables_ground_truth:
            return 1.0
        relevant_and_retrieved = set(self.retrieved_tables) & set(
            self.relevant_tables_ground_truth
        )
        return len(relevant_and_retrieved) / len(self.relevant_tables_ground_truth)

    @property
    def coverage_ratio(self) -> float:
        """What % of all tables did we narrow down to?"""
        if not self.all_tables:
            return 1.0
        return len(self.retrieved_tables) / len(self.all_tables)

    def print_metrics(self):
        print(f"\n{SEPARATOR}")
        print("📐 RAG RETRIEVAL METRICS")
        print(SEPARATOR)
        print(f"  Question      : {self.question[:60]}...")
        print(f"  All tables    : {len(self.all_tables)}")
        print(
            f"  Retrieved (K) : {len(self.retrieved_tables)} "
            f"→ {self.retrieved_tables}"
        )
        print(
            f"  Coverage ratio: {self.coverage_ratio:.1%} "
            f"(narrowed from {len(self.all_tables)} → "
            f"{len(self.retrieved_tables)} tables)"
        )
        if self.relevant_tables_ground_truth:
            print(f"  Ground truth  : {self.relevant_tables_ground_truth}")
            print(f"  Precision@{self.k}  : {self.precision_at_k:.3f}")
            print(f"  Recall@{self.k}    : {self.recall_at_k:.3f}")
        print(SEPARATOR)


@dataclass
class SQLMetrics:
    """Metrics for SQL generation and execution."""

    question: str
    sql_query: str
    execution_success: bool
    rows_returned: int
    execution_time_ms: float
    hallucinated_tables: list[str] = field(default_factory=list)
    hallucinated_columns: list[str] = field(default_factory=list)
    used_tables: list[str] = field(default_factory=list)
    model_name: str = "unknown"

    @property
    def hallucination_rate(self) -> float:
        """0.0 = no hallucination, 1.0 = fully hallucinated"""
        total_entities = len(self.used_tables) + len(self.hallucinated_tables)
        if total_entities == 0:
            return 0.0
        return len(self.hallucinated_tables) / total_entities

    @property
    def is_hallucinated(self) -> bool:
        return len(self.hallucinated_tables) > 0 or len(self.hallucinated_columns) > 0

    @property
    def quality_score(self) -> int:
        """0-100 composite score."""
        score = 100
        if not self.execution_success:
            score -= 50
        if self.rows_returned == 0:
            score -= 30
        score -= len(self.hallucinated_tables) * 20
        score -= len(self.hallucinated_columns) * 10
        return max(0, score)

    def print_metrics(self):
        print(f"\n{SEPARATOR}")
        print(f"🔬 SQL EVALUATION METRICS  [{self.model_name}]")
        print(SEPARATOR)
        print(f"  Question          : {self.question[:55]}...")
        print(f"  SQL               : {self.sql_query[:70]}...")
        print(
            f"  Execution success : {'✅ Yes' if self.execution_success else '❌ No'}"
        )
        print(f"  Rows returned     : {self.rows_returned}")
        print(f"  Execution time    : {self.execution_time_ms:.1f}ms")
        print(f"  Tables used       : {self.used_tables}")
        print(
            f"  Hallucinated tables: "
            f"{'None ✅' if not self.hallucinated_tables else self.hallucinated_tables}"
        )
        print(
            f"  Hallucinated cols : "
            f"{'None ✅' if not self.hallucinated_columns else self.hallucinated_columns}"
        )
        print(f"  Hallucination rate: {self.hallucination_rate:.1%}")
        print(f"  Quality score     : {self.quality_score}/100")
        badge = _score_badge(self.quality_score)
        print(f"  Badge             : {badge}")
        print(SEPARATOR)


@dataclass
class FaithfulnessMetrics:
    """
    Is the generated insight faithful to the actual data?
    Checks if numbers/names in the insight match the dataframe.
    """

    insight: str
    dataframe: pd.DataFrame
    question: str

    @property
    def faithfulness_score(self) -> float:
        """
        Check if key values mentioned in the insight
        actually appear in the dataframe.
        """
        if not self.insight or self.dataframe is None or self.dataframe.empty:
            return 0.0

        # Extract numbers from insight
        numbers_in_insight = set(re.findall(r"\b\d+\.?\d*\b", self.insight))

        # Get all values from dataframe as strings
        df_values = set()
        for col in self.dataframe.columns:
            for val in self.dataframe[col].astype(str).tolist():
                df_values.add(val)
                # Also add rounded versions
                try:
                    df_values.add(str(int(float(val))))
                    df_values.add(str(round(float(val), 1)))
                except Exception:
                    pass

        if not numbers_in_insight:
            return 0.8  # no numbers to verify, assume ok

        matched = numbers_in_insight & df_values
        return len(matched) / len(numbers_in_insight) if numbers_in_insight else 1.0

    @property
    def is_faithful(self) -> bool:
        return self.faithfulness_score >= 0.5

    def print_metrics(self):
        print(f"\n{SEPARATOR}")
        print("🎯 FAITHFULNESS METRICS")
        print(SEPARATOR)
        print(f"  Insight     : {self.insight[:80]}...")
        print(f"  Faithfulness: {self.faithfulness_score:.1%}")
        print(
            f"  Verdict     : "
            f"{'✅ Insight is grounded in data' if self.is_faithful else '⚠️  Insight may be hallucinated'}"
        )
        print(SEPARATOR)


@dataclass
class ModelComparisonEntry:
    """One row in the model comparison table."""

    model: str
    question: str
    sql_quality: int
    hallucination_rate: float
    rows_returned: int
    execution_time_ms: float
    faithfulness: float
    success: bool


class ModelComparator:
    """
    Tracks and compares performance across multiple LLM models
    for the same set of questions.
    """

    def __init__(self):
        self.entries: list[ModelComparisonEntry] = []

    def add(self, entry: ModelComparisonEntry):
        self.entries.append(entry)

    def print_comparison_table(self):
        if not self.entries:
            print("No comparison data yet.")
            return

        # Group by model
        models = list(set(e.model for e in self.entries))

        print(f"\n{'=' * 80}")
        print("📊 MODEL COMPARISON REPORT")
        print(f"{'=' * 80}")
        print(
            f"{'Model':<30} {'Avg Quality':>12} "
            f"{'Halluc Rate':>12} {'Avg Rows':>10} "
            f"{'Avg Time(ms)':>13} {'Success%':>10}"
        )
        print("-" * 80)

        for model in sorted(models):
            model_entries = [e for e in self.entries if e.model == model]
            avg_quality = sum(e.sql_quality for e in model_entries) / len(model_entries)
            avg_halluc = sum(e.hallucination_rate for e in model_entries) / len(
                model_entries
            )
            avg_rows = sum(e.rows_returned for e in model_entries) / len(model_entries)
            avg_time = sum(e.execution_time_ms for e in model_entries) / len(
                model_entries
            )
            success_rate = sum(1 for e in model_entries if e.success) / len(
                model_entries
            )

            print(
                f"{model:<30} {avg_quality:>11.1f} "
                f"{avg_halluc:>11.1%} {avg_rows:>10.1f} "
                f"{avg_time:>12.1f} {success_rate:>9.1%}"
            )

        print(f"{'=' * 80}")

        # Winner
        if len(models) > 1:
            best = max(
                models,
                key=lambda m: sum(e.sql_quality for e in self.entries if e.model == m)
                / len([e for e in self.entries if e.model == m]),
            )
            print(f"🏆 Best model: {best}")
        print(f"{'=' * 80}\n")


# Global comparator instance (shared across runs)
_comparator = ModelComparator()


def get_comparator() -> ModelComparator:
    return _comparator


def _score_badge(score: int) -> str:
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 70:
        return "🟡 Good"
    elif score >= 50:
        return "🟠 Questionable"
    else:
        return "🔴 Likely Hallucination"


def extract_used_tables(sql: str, all_tables: list[str]) -> tuple:
    """
    Extract which tables were used in the SQL and
    which ones don't exist (hallucinated).
    """
    real_tables_lower = {t.lower(): t for t in all_tables}
    table_matches = re.findall(
        r'(?:FROM|JOIN)\s+[\[\`"]?([^\s\[\]`,;()]+)[\]\`"]?', sql, re.IGNORECASE
    )
    used = []
    hallucinated = []
    for t in table_matches:
        t_clean = t.strip("[]`\"' ")
        if t_clean.lower() in real_tables_lower:
            used.append(real_tables_lower[t_clean.lower()])
        else:
            hallucinated.append(t_clean)
    return used, hallucinated


def check_column_hallucinations(sql: str, used_tables: list[str]) -> list[str]:
    """Check if SQL references columns that don't exist."""
    hallucinated_cols = []
    conn = sqlite3.connect(DB_PATH)

    real_columns = set()
    for table in used_tables:
        try:
            cursor = conn.execute(f'PRAGMA table_info("{table}")')
            for row in cursor.fetchall():
                real_columns.add(row[1].lower())
        except Exception:
            pass
    conn.close()

    # ── Extract SELECT aliases to exclude them ────────────────
    # e.g. COUNT(x) AS OrderFrequency → OrderFrequency is an alias
    aliases = set(re.findall(r"\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE))

    col_matches = re.findall(
        r"(?:SELECT|WHERE|GROUP BY|ORDER BY|ON)\s+" r"([a-zA-Z_][a-zA-Z0-9_\.]*)",
        sql,
        re.IGNORECASE,
    )

    keywords = {
        "distinct",
        "count",
        "sum",
        "avg",
        "max",
        "min",
        "as",
        "by",
        "from",
        "join",
        "on",
        "where",
        "group",
        "order",
        "limit",
        "having",
        "case",
        "when",
        "then",
        "else",
        "end",
        "and",
        "or",
        "not",
        "null",
        "asc",
        "desc",
        "*",
    }

    for col in col_matches:
        col_name = col.split(".")[-1].lower()
        if col_name in keywords:
            continue
        if col_name in {a.lower() for a in aliases}:  # ← skip aliases
            continue
        if len(col_name) <= 1:
            continue
        if real_columns and col_name not in real_columns:
            hallucinated_cols.append(col)

    return list(set(hallucinated_cols))


def run_full_evaluation(
    question: str,
    sql_query: str,
    dataframe,
    insight: str,
    retrieved_tables: list[str],
    execution_time_ms: float,
    model_name: str,
) -> dict:
    """
    Master evaluation function.
    Runs all metrics and prints to terminal.
    Returns summary dict for internal use.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    # ── RAG Metrics ───────────────────────────────────────────
    rag = RAGMetrics(
        question=question,
        retrieved_tables=retrieved_tables,
        all_tables=all_tables,
        k=len(retrieved_tables),
    )
    rag.print_metrics()

    # ── SQL Metrics ───────────────────────────────────────────
    used_tables, hallucinated_tables = extract_used_tables(sql_query, all_tables)
    hallucinated_cols = check_column_hallucinations(sql_query, used_tables)

    sql_metrics = SQLMetrics(
        question=question,
        sql_query=sql_query,
        execution_success=dataframe is not None and not dataframe.empty,
        rows_returned=len(dataframe) if dataframe is not None else 0,
        execution_time_ms=execution_time_ms,
        hallucinated_tables=hallucinated_tables,
        hallucinated_columns=hallucinated_cols,
        used_tables=used_tables,
        model_name=model_name,
    )
    sql_metrics.print_metrics()

    # ── Faithfulness Metrics ──────────────────────────────────
    faith = FaithfulnessMetrics(
        insight=insight,
        dataframe=dataframe,
        question=question,
    )
    faith.print_metrics()

    # ── Add to Model Comparator ───────────────────────────────
    comparator = get_comparator()
    comparator.add(
        ModelComparisonEntry(
            model=model_name,
            question=question,
            sql_quality=sql_metrics.quality_score,
            hallucination_rate=sql_metrics.hallucination_rate,
            rows_returned=sql_metrics.rows_returned,
            execution_time_ms=execution_time_ms,
            faithfulness=faith.faithfulness_score,
            success=sql_metrics.execution_success,
        )
    )

    # Print running comparison after every 2+ entries
    if len(comparator.entries) >= 2:
        comparator.print_comparison_table()

    return {
        "quality_score": sql_metrics.quality_score,
        "hallucination_rate": sql_metrics.hallucination_rate,
        "faithfulness": faith.faithfulness_score,
        "is_hallucinated": sql_metrics.is_hallucinated,
        "precision_at_k": rag.precision_at_k,
        "recall_at_k": rag.recall_at_k,
    }
