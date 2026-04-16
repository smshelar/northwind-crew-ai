import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Task
from agents.sql_writer import create_sql_writer


def create_sql_task(user_question: str, schema_context: str = "") -> Task:
    return Task(
        description=(
            f"The user asked: '{user_question}'\n\n"
            "Write a SQLite SQL query that answers this question exactly.\n"
            "Use ONLY the tables provided in your schema context.\n"
            "Return ONLY the raw SQL — no markdown, no code fences, no explanation."
        ),
        expected_output="A valid SQLite SELECT statement as plain text.",
        agent=create_sql_writer(schema_context=schema_context),
    )
