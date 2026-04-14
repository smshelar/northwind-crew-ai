"""
output_guard.py
---------------
Validates agent output BEFORE showing to user.
Blocks hallucinated results, empty data, nonsensical insights.
"""

import re
from dataclasses import dataclass
import pandas as pd


@dataclass
class OutputGuardrailResult:
    passed: bool
    reason: str = ""
    safe_insight: str = ""


def validate_output(
    sql_query: str,
    dataframe: pd.DataFrame,
    insight: str,
    hallucinated_tables: list[str],
    quality_score: int,
) -> OutputGuardrailResult:
    """
    Validates pipeline output before it reaches the UI.
    Prints result to terminal.
    """
    print(f"\n{'=' * 65}")
    print("🛡️  OUTPUT GUARDRAIL CHECK")
    print(f"{'=' * 65}")

    # ── Check 1: No hallucinated tables ──────────────────────
    if hallucinated_tables:
        reason = (
            f"SQL references non-existent tables: {hallucinated_tables}. "
            "This is a hallucination."
        )
        print(f"  ❌ BLOCKED: {reason}")
        print(f"{'=' * 65}\n")
        return OutputGuardrailResult(passed=False, reason=reason)

    print("  ✅ No hallucinated tables")

    # ── Check 2: Data is not empty ────────────────────────────
    if dataframe is None or dataframe.empty:
        reason = "Query returned no data — cannot show empty results"
        print(f"  ❌ BLOCKED: {reason}")
        print(f"{'=' * 65}\n")
        return OutputGuardrailResult(passed=False, reason=reason)

    print(f"  ✅ Data has {len(dataframe)} rows")

    # ── Check 3: Quality score threshold ─────────────────────
    if quality_score < 50:
        reason = (
            f"Quality score too low ({quality_score}/100). "
            "Result reliability is insufficient."
        )
        print(f"  ❌ BLOCKED: {reason}")
        print(f"{'=' * 65}\n")
        return OutputGuardrailResult(passed=False, reason=reason)

    print(f"  ✅ Quality score: {quality_score}/100")

    # ── Check 4: Sanitise insight ─────────────────────────────
    safe_insight = insight or ""

    # Remove any SQL that might have leaked into insight
    safe_insight = re.sub(
        r'(SELECT|FROM|WHERE|JOIN|DROP|DELETE|INSERT)',
        '', safe_insight, flags=re.IGNORECASE
    ).strip()

    # Remove excessive punctuation/symbols
    safe_insight = re.sub(r'[^\w\s\.,!?%$£€\-\(\)]', '', safe_insight)

    print(f"  ✅ Insight sanitised")
    print(f"{'=' * 65}\n")

    return OutputGuardrailResult(
        passed=True,
        reason="All output checks passed",
        safe_insight=safe_insight,
    )