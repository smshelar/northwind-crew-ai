"""
schema_indexer.py
-----------------
Reads ALL tables from the database and indexes their
schema + sample data into ChromaDB for RAG retrieval.
Run this ONCE to build the index, or when schema changes.
"""

import os
import sys
import sqlite3
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions


from chromadb.config import Settings
import tempfile

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# def get_chroma_client():
#     """
#     Use in-memory/temp DB for Streamlit.
#     Use persistent DB locally.
#     """
    # try:
    #     import streamlit as st
    #     # If running in Streamlit Cloud
    #     if hasattr(st, "runtime"):
    #         return chromadb.Client(
    #             Settings(
    #                 persist_directory=tempfile.mkdtemp(),
    #                 anonymized_telemetry=False
    #             )
    #         )
    # except:
    #     pass

    # # Local fallback
    # return chromadb.PersistentClient(path=CHROMA_PATH)

def get_chroma_client():
    import chromadb
    from chromadb.config import Settings
    import tempfile

    return chromadb.Client(
        Settings(
            persist_directory=tempfile.mkdtemp(),  # ✅ valid path
            anonymized_telemetry=False
        )
    )


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()

def get_all_tables(conn) -> list[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def get_table_schema(conn, table_name: str) -> dict:
    """Get columns, types and sample rows for a table."""
    try:
        # Get columns
        cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
        columns = [
            {"name": row[1], "type": row[2], "nullable": not row[3]}
            for row in cursor.fetchall()
        ]

        # Get sample rows (3 rows)
        cursor = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        sample_rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        samples = [dict(zip(col_names, row)) for row in sample_rows]

        # Get row count
        cursor = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()[0]

        return {
            "table_name": table_name,
            "columns": columns,
            "sample_rows": samples,
            "row_count": row_count,
        }
    except Exception as e:
        print(f"  ⚠️  Could not read {table_name}: {e}")
        return None


def build_schema_document(schema: dict) -> str:
    """
    Convert schema dict into a rich text document for embedding.
    More detail = better retrieval accuracy.
    """
    table = schema["table_name"]
    cols = ", ".join(
        f"{c['name']} ({c['type']})" for c in schema["columns"]
    )
    col_names = [c["name"] for c in schema["columns"]]

    # Build sample values string
    sample_vals = ""
    if schema["sample_rows"]:
        for row in schema["sample_rows"][:2]:
            vals = ", ".join(str(v) for v in list(row.values())[:4])
            sample_vals += f"  Example: {vals}\n"

    doc = (
        f"Table: {table}\n"
        f"Columns: {cols}\n"
        f"Column names: {', '.join(col_names)}\n"
        f"Total rows: {schema['row_count']}\n"
        f"Sample data:\n{sample_vals}"
        f"Use this table for queries about: {table.lower().replace('_', ' ')}"
    )
    return doc


def index_schema():
    """Main function — reads DB and indexes all tables into ChromaDB."""
    print("🔍 Connecting to database...")
    conn = sqlite3.connect(DB_PATH)

    print("📚 Reading all tables...")
    tables = get_all_tables(conn)
    print(f"   Found {len(tables)} tables: {tables}")

    print("🔗 Connecting to ChromaDB...")
    client = get_chroma_client()
    embed_fn = get_embedding_function()

    # Delete existing collection if it exists (fresh rebuild)
    # try:
    #     client.delete_collection("schema")
    #     print("   Deleted existing schema collection")
    # except Exception:
    #     pass

    # # Create collection safely
    # try:
    #     collection = client.get_collection(name="schema")
    # except:
    #     collection = client.create_collection(
    #         name="schema",
    #         embedding_function=embed_fn,
    #         metadata={"description": "Database table schemas"}
    #     )

    collections = [c.name for c in client.list_collections()]

    if "schema" in collections:
        collection = client.get_collection(name="schema")
    else:
        collection = client.create_collection(
            name="schema",
            embedding_function=embed_fn,
            metadata={"description": "Database table schemas"}
        )

    print("📝 Indexing tables...")
    documents = []
    metadatas = []
    ids = []

    for table in tables:
        schema = get_table_schema(conn, table)
        if schema is None:
            continue

        doc = build_schema_document(schema)
        documents.append(doc)
        metadatas.append({
            "table_name": table,
            "columns": json.dumps([c["name"] for c in schema["columns"]]),
            "row_count": schema["row_count"],
        })
        ids.append(f"table_{table.replace(' ', '_')}")
        print(f"   ✅ Indexed: {table} ({schema['row_count']} rows, "
              f"{len(schema['columns'])} columns)")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    conn.close()
    print(f"\n✅ Schema index built — {len(documents)} tables indexed")
    print(f"   Stored at: {CHROMA_PATH}")
    return len(documents)


if __name__ == "__main__":
    index_schema()