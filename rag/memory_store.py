"""
memory_store.py
---------------
Caches verified question → SQL → result mappings in ChromaDB.
On new questions, checks for semantic similarity to avoid
re-running queries that have been answered correctly before.
"""

import os
import sys
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
SIMILARITY_THRESHOLD = 0.85


def get_chroma_path() -> str:
    if os.environ.get("CHROMA_PATH"):
        return os.environ["CHROMA_PATH"]

    if os.path.exists("/mount/src"):
        return "/tmp/chroma_db"

    return DEFAULT_CHROMA_PATH


def get_client():
    chroma_path = get_chroma_path()
    os.makedirs(chroma_path, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path)


def get_embed_fn():
    return embedding_functions.DefaultEmbeddingFunction()


def _get_or_create_collection():
    client = get_client()
    embed_fn = get_embed_fn()
    try:
        return client.get_collection(name="query_memory", embedding_function=embed_fn)
    except Exception:
        return client.create_collection(
            name="query_memory",
            embedding_function=embed_fn,
            metadata={"description": "Cached verified Q&A pairs"},
        )


def check_memory(question: str) -> dict | None:
    """
    Check if a similar question has been answered before.
    Returns cached result dict if found, None otherwise.
    """
    try:
        collection = _get_or_create_collection()

        if collection.count() == 0:
            return None

        results = collection.query(
            query_texts=[question],
            n_results=1,
        )

        if not results["documents"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance

        print(f"🧠 Memory check — best match similarity: {similarity:.3f}")

        if similarity >= SIMILARITY_THRESHOLD:
            meta = results["metadatas"][0][0]
            cached_question = results["documents"][0][0]

            print("✅ Memory HIT — reusing cached result")
            print(f"   Original question: {cached_question}")
            print(f"   Similarity: {similarity:.3f}")

            return {
                "hit": True,
                "similarity": similarity,
                "original_question": cached_question,
                "sql_query": meta["sql_query"],
                "insight": meta.get("insight", ""),
                "summary": meta.get("summary", ""),
                "row_count": meta.get("row_count", 0),
                "cached_at": meta.get("cached_at", ""),
            }

        return None

    except Exception as e:
        print(f"⚠️  Memory check failed: {e}")
        return None


def save_to_memory(
    question: str,
    sql_query: str,
    row_count: int,
    insight: str = "",
    summary: str = "",
):
    """
    Save a verified Q&A pair to memory.
    Only call this AFTER evaluation passes.
    """
    try:
        collection = _get_or_create_collection()

        q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()

        try:
            collection.delete(ids=[q_hash])
        except Exception:
            pass

        collection.add(
            documents=[question],
            metadatas=[
                {
                    "sql_query": sql_query,
                    "insight": insight,
                    "summary": summary,
                    "row_count": row_count,
                    "cached_at": datetime.now().isoformat(),
                }
            ],
            ids=[q_hash],
        )

        print(f"💾 Saved to memory: '{question[:60]}'")

    except Exception as e:
        print(f"⚠️  Memory save failed: {e}")


def get_memory_stats() -> dict:
    """Return stats about what is cached."""
    try:
        collection = _get_or_create_collection()
        count = collection.count()
        return {"cached_queries": count}
    except Exception:
        return {"cached_queries": 0}


def clear_memory():
    """Clear all cached queries."""
    try:
        client = get_client()
        client.delete_collection("query_memory")
        print("🗑️  Memory cleared")
    except Exception as e:
        print(f"⚠️  Clear failed: {e}")
