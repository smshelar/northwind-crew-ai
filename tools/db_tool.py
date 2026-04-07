import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import sqlite3
import pandas as pd
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")

class SQLInput(BaseModel):
    query: str = Field(description="Valid SQLite SQL query to execute")

class ExecuteSQLTool(BaseTool):
    name: str = "execute_sql"
    description: str = (
        "USE THIS TOOL to execute a SQLite SQL query against the Northwind database. "
        "Pass the SQL query as the 'query' parameter. "
        "Returns a JSON array of records. "
        "You MUST use this tool — do not describe results without calling it."
    )
    args_schema: type[BaseModel] = SQLInput

    def _run(self, query: str) -> str:
        try:
            # Clean the query — strip markdown fences if agent wrapped it
            query = query.strip()
            query = query.replace("```sql", "").replace("```", "").strip()

            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                return json.dumps([])

            # Return compact JSON array
            result = df.to_json(orient="records")
            print(f"DEBUG tool returning: {result[:200]}")
            return result

        except Exception as e:
            return json.dumps({"error": str(e)})