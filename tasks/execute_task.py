
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Task
from agents.executer import create_executer

def create_execute_task(sql_query: str = "") -> Task:
    return Task(
        description=(
            f"Execute this EXACT SQL query using the execute_sql tool:\n\n"
            f"```\n{sql_query}\n```\n\n"
            "Call execute_sql with this query and return ONLY the JSON array result.\n"
            "No text. No explanation. Just the JSON array."
        ),
        expected_output='A raw JSON array. Example: [{"col": "val"}]',
        agent=create_executer(),
    )