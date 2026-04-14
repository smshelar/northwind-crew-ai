import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent
from tools.db_tool import ExecuteSQLTool
from utils.llm_factory import get_llm

def create_executer() -> Agent:
    return Agent(
        role="SQL Executor",
        goal="Call the execute_sql tool with the SQL query and return its exact output.",
        backstory=(
            "You are a silent database runner. You have ONE job only:\n"
            "1. Take the SQL query from the previous task\n"
            "2. Call the execute_sql tool with that query\n"
            "3. Copy and return the EXACT output of the tool\n\n"
            "You NEVER write explanations.\n"
            "You NEVER say 'the result is' or describe anything.\n"
            "You NEVER search the web.\n"
            "You ONLY call execute_sql and return its raw output.\n"
            "If you return anything other than a JSON array, you have failed."
        ),
        tools=[ExecuteSQLTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3,  # forces it to stop after 3 attempts
    )