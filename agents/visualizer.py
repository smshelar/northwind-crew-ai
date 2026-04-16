"""
visualizer.py
-------------
CrewAI Agent: interprets data, produces chart config + business insight.
"""

from crewai import Agent
from utils.llm_factory import get_llm


def create_visualizer() -> Agent:
    return Agent(
        role="Data Visualizer & Analyst",
        goal=(
            "Analyze the query results and return a structured JSON block with: "
            "chart_type, x_column, y_column, title, and a one-paragraph insight."
        ),
        backstory=(
            "You are a senior business intelligence analyst. Given a JSON dataset "
            "and the original question, you decide the best chart type (bar, line, pie, scatter), "
            "If the visualization is complex, you can create multiple charts or an dashboard."
            "identify the correct columns, and write a clear business insight. "
            "Always respond with ONLY a JSON object — no markdown, no extra text.\n"
            "Format:\n"
            '{"chart_type": "bar", "x_column": "ProductName", "y_column": "TotalSales", '
            '"title": "Top Products by Sales", "insight": "..."}'
        ),
        # `get_llm()` returns a CrewAI-compatible LLM object for local and deploy.
        llm=get_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False,
    )
