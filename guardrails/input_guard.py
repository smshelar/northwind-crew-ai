"""
input_guard.py
--------------
Validates user input BEFORE sending to any agent.
Blocks: prompt injections, harmful input,
        non-business questions, gibberish.
"""

import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    safe_question: str = ""


# Questions that are clearly not business database queries
NON_BUSINESS_PATTERNS = [
    r'\b(hack|exploit|inject|drop table|delete from|truncate)\b',
    r'\b(ignore previous|ignore all|disregard|forget your)\b',
    r'\b(you are now|act as|pretend to be|roleplay)\b',
    r'\b(password|credit card|ssn|social security)\b',
    r'\b(kill|murder|weapon|bomb|terrorist)\b',
]

# Must contain at least some business intent
BUSINESS_KEYWORDS = [
    "product", "customer", "order", "sale", "revenue", "employee",
    "supplier", "category", "ship", "freight", "profit", "stock",
    "inventory", "top", "best", "worst", "most", "least", "average",
    "total", "count", "how many", "which", "what", "show", "list",
    "compare", "trend", "monthly", "yearly", "quarterly", "country",
    "city", "company", "price", "quantity", "discount", "region",
]

MIN_QUESTION_LENGTH = 5
MAX_QUESTION_LENGTH = 500


def validate_input(question: str) -> GuardrailResult:
    """
    Validates user input before it reaches any agent.
    Prints result to terminal, returns GuardrailResult.
    """
    print(f"\n{'=' * 65}")
    print("🛡️  INPUT GUARDRAIL CHECK")
    print(f"{'=' * 65}")
    print(f"  Input: '{question[:80]}'")

    # ── Check 1: Length ───────────────────────────────────────
    if len(question.strip()) < MIN_QUESTION_LENGTH:
        reason = "Input too short — minimum 5 characters"
        print(f"  ❌ BLOCKED: {reason}")
        print(f"{'=' * 65}\n")
        return GuardrailResult(passed=False, reason=reason)

    if len(question) > MAX_QUESTION_LENGTH:
        reason = f"Input too long — maximum {MAX_QUESTION_LENGTH} characters"
        print(f"  ❌ BLOCKED: {reason}")
        print(f"{'=' * 65}\n")
        return GuardrailResult(passed=False, reason=reason)

    # ── Check 2: Prompt injection / harmful patterns ──────────
    question_lower = question.lower()
    for pattern in NON_BUSINESS_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            reason = f"Blocked pattern detected: '{pattern}'"
            print(f"  ❌ BLOCKED: Prompt injection or harmful input detected")
            print(f"{'=' * 65}\n")
            return GuardrailResult(passed=False, reason=reason)

    # ── Check 3: Must look like a business question ───────────
    has_business_intent = any(
        kw in question_lower for kw in BUSINESS_KEYWORDS
    )
    if not has_business_intent:
        reason = (
            "No business intent detected. "
            "Please ask about orders, products, customers, sales, etc."
        )
        print(f"  ⚠️  WARNING: {reason}")
        # Don't block — warn only, let it through
        # (user might phrase things differently)

    # ── Check 4: Not pure gibberish ───────────────────────────
    words = question.strip().split()
    if len(words) < 2:
        reason = "Input appears to be a single word — please ask a question"
        print(f"  ❌ BLOCKED: {reason}")
        print(f"{'=' * 65}\n")
        return GuardrailResult(passed=False, reason=reason)

    # ── Sanitise input ────────────────────────────────────────
    # Remove any SQL-like injections from the question text
    safe_question = re.sub(
        r'(;|--|\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b)',
        '', question, flags=re.IGNORECASE
    ).strip()

    print(f"  ✅ PASSED all guardrail checks")
    if safe_question != question:
        print(f"  🧹 Sanitised: '{safe_question[:80]}'")
    print(f"{'=' * 65}\n")

    return GuardrailResult(
        passed=True,
        reason="All checks passed",
        safe_question=safe_question,
    )