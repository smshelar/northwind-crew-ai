from crewai import Task
from agents.sql_writer import create_sql_writer

def create_sql_task(user_question: str) -> Task:
    return Task(
        description=(
            f"The user asked: '{user_question}'\n"
            "Write a SQLite SQL query that answers this question exactly. "
            "Return ONLY the raw SQL query — no markdown, no code fences."
        ),
        expected_output="A valid SQLite SELECT statement as plain text.",
        agent=create_sql_writer(),
    )