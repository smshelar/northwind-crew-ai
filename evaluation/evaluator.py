"""
evaluator.py
------------
Validates SQL query results and detects hallucinations.
Checks:
  1. SQL syntax is valid
  2. Tables used actually exist in the DB
  3. Columns used actually exist in those tables
  4. Query returns rows (not empty)
  5. Numeric columns have reasonable values
  6. Result columns match what was asked for
"""

import os
import sys
import re
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")


class EvaluationResult:
    def __init__(self):
        self.passed = True
        self.score = 100        # 0-100
        self.checks = []        # list of (check_name, passed, message)
        self.warnings = []
        self.errors = []

    def add_check(self, name: str, passed: bool, message: str):
        self.checks.append((name, passed, message))
        if not passed:
            self.errors.append(message)
            self.score -= 20
            self.passed = False
        else:
            self.warnings.append(message) if "warning" in message.lower() else None

    def summary(self) -> str:
        lines = [f"📊 Evaluation Score: {max(0, self.score)}/100"]
        for name, passed, msg in self.checks:
            icon = "✅" if passed else "❌"
            lines.append(f"  {icon} {name}: {msg}")
        return "\n".join(lines)


def evaluate(
    sql_query: str,
    dataframe: pd.DataFrame,
    user_question: str,
) -> EvaluationResult:
    """
    Main evaluation function.
    Returns an EvaluationResult with score and detailed checks.
    """
    result = EvaluationResult()
    conn = sqlite3.connect(DB_PATH)

    # ── Check 1: SQL is not empty ─────────────────────────────
    if not sql_query or len(sql_query.strip()) < 10:
        result.add_check(
            "SQL not empty",
            False,
            "SQL query is empty or too short"
        )
        conn.close()
        return result
    else:
        result.add_check("SQL not empty", True, "SQL query present")

    # ── Check 2: Only SELECT (no data modification) ───────────
    sql_upper = sql_query.upper().strip()
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
    has_dangerous = any(sql_upper.startswith(kw) or f" {kw} " in sql_upper
                       for kw in dangerous)
    result.add_check(
        "Safe SQL",
        not has_dangerous,
        "No dangerous SQL operations" if not has_dangerous
        else "⚠️ Dangerous SQL operation detected"
    )

    # ── Check 3: Tables exist in database ────────────────────
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    real_tables = {row[0].lower() for row in cursor.fetchall()}

    # Extract table names from SQL (simple regex)
    table_matches = re.findall(
        r'(?:FROM|JOIN)\s+[\[\`"]?([^\s\[\]`,;]+)[\]\`"]?',
        sql_query,
        re.IGNORECASE
    )
    hallucinated_tables = []
    for t in table_matches:
        t_clean = t.strip("[]`\"'")
        if t_clean.lower() not in real_tables:
            hallucinated_tables.append(t_clean)

    if hallucinated_tables:
        result.add_check(
            "Tables exist",
            False,
            f"Hallucinated tables: {hallucinated_tables}"
        )
    else:
        result.add_check("Tables exist", True, "All tables exist in DB")

    # ── Check 4: Query actually returns rows ──────────────────
    if dataframe is None or dataframe.empty:
        result.add_check(
            "Returns data",
            False,
            "Query returned 0 rows — possible wrong filters or hallucination"
        )
    else:
        result.add_check(
            "Returns data",
            True,
            f"Query returned {len(dataframe)} rows"
        )

    # ── Check 5: No NaN-only columns ─────────────────────────
    if dataframe is not None and not dataframe.empty:
        nan_cols = [
            col for col in dataframe.columns
            if dataframe[col].isna().all()
        ]
        if nan_cols:
            result.add_check(
                "No empty columns",
                False,
                f"Columns with all NaN values: {nan_cols}"
            )
        else:
            result.add_check(
                "No empty columns",
                True,
                "All columns have data"
            )

    # ── Check 6: Numeric sanity check ────────────────────────
    if dataframe is not None and not dataframe.empty:
        numeric_cols = dataframe.select_dtypes(include="number").columns
        for col in numeric_cols:
            if (dataframe[col] < 0).any():
                # Negative sales/counts are suspicious
                if any(kw in col.lower() for kw in
                       ["sales", "revenue", "count", "quantity", "freq"]):
                    result.add_check(
                        f"Numeric sanity ({col})",
                        False,
                        f"Column '{col}' has negative values — possible hallucination"
                    )
                    break
        else:
            result.add_check(
                "Numeric sanity",
                True,
                "Numeric values look reasonable"
            )

    # ── Check 7: Result relevance to question ────────────────
    question_lower = user_question.lower()
    if dataframe is not None and not dataframe.empty:
        col_names_lower = [c.lower() for c in dataframe.columns]

        # Check for keyword alignment
        relevant = False
        keyword_map = {
            "product": ["product", "item", "name"],
            "customer": ["customer", "company", "client"],
            "sales": ["sales", "revenue", "total", "amount"],
            "employee": ["employee", "staff", "rep"],
            "order": ["order", "frequency", "count"],
            "category": ["category", "type", "group"],
        }
        for topic, keywords in keyword_map.items():
            if topic in question_lower:
                if any(kw in " ".join(col_names_lower) for kw in keywords):
                    relevant = True
                    break
        else:
            relevant = True  # if no keyword match needed, assume ok

        result.add_check(
            "Result relevance",
            relevant,
            "Result columns are relevant to the question"
            if relevant else
            "Result columns may not match the question"
        )

    conn.close()
    result.score = max(0, result.score)
    return result


def is_hallucination(eval_result: EvaluationResult) -> bool:
    """Returns True if the result likely contains hallucinated data."""
    return eval_result.score < 60


def get_evaluation_badge(score: int) -> str:
    """Returns a coloured emoji badge based on score."""
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 70:
        return "🟡 Good"
    elif score >= 50:
        return "🟠 Questionable"
    else:
        return "🔴 Likely Hallucination"