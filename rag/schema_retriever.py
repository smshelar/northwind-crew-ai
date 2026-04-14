"""
schema_retriever.py
-------------------
Given a user question, retrieves the most relevant
table schemas from ChromaDB using vector similarity.
"""

import os
import sys
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()

def retrieve_relevant_schema(question: str, top_k: int = 5) -> str:
    """
    Given a natural language question, returns the schema
    of the top_k most relevant tables as a formatted string
    ready to inject into the SQL writer agent's prompt.
    """
    try:
        client = get_chroma_client()
        embed_fn = get_embedding_function()
        collection = client.get_collection(
            name="schema",
            embedding_function=embed_fn
        )

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),
        )

        if not results["documents"][0]:
            return _get_full_schema_fallback()

        # Build a clean schema string for the agent
        schema_parts = []
        retrieved_tables = []

        for i, (doc, meta) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0]
        )):
            table_name = meta["table_name"]
            columns = json.loads(meta["columns"])
            retrieved_tables.append(table_name)

            # Get full column details from DB
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
            col_details = [
                f"{row[1]} {row[2]}" for row in cursor.fetchall()
            ]
            conn.close()

            schema_parts.append(
                f"- {table_name}({', '.join(col_details)})"
            )

        print(f"🔍 RAG retrieved tables: {retrieved_tables}")

        schema_string = "\n".join(schema_parts)
        return schema_string, retrieved_tables

    except Exception as e:
        print(f"⚠️  RAG retrieval failed: {e} — using full schema")
        return _get_full_schema_fallback()


def _get_full_schema_fallback() -> tuple:
    """Fallback: return all tables if RAG fails."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor = conn.execute(f'PRAGMA table_info("{table}")')
        cols = [f"{row[1]} {row[2]}" for row in cursor.fetchall()]
        schema_parts.append(f"- {table}({', '.join(cols)})")

    conn.close()
    return "\n".join(schema_parts), tables


def is_schema_indexed() -> bool:
    """Check if the schema has been indexed into ChromaDB."""
    try:
        client = get_chroma_client()
        col = client.get_collection("schema")
        return col.count() > 0
    except Exception:
        return False