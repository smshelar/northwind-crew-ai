import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Task
from agents.visualizer import create_visualizer


def create_visualizer_task(user_question: str, data_sample: str = "") -> Task:
    return Task(
        description=(
            f"Original question: '{user_question}'\n\n"
            f"The query returned this data sample:\n{data_sample}\n\n"
            "Based on the data, decide the best chart type and return a JSON config.\n"
            "Return ONLY a JSON object — no markdown, no extra text.\n\n"
            "CORRECT output example:\n"
            '{"chart_type": "bar", "x_column": "CompanyName", '
            '"y_column": "OrderFrequency", "title": "Top Customers", '
            '"insight": "QUICK-Stop leads with 28 orders."}'
        ),
        expected_output=(
            "A JSON object only — no markdown, no extra text.\n"
            'Format: {"chart_type": "bar", "x_column": "...", '
            '"y_column": "...", "title": "...", "insight": "..."}'
        ),
        agent=create_visualizer(),
    )
