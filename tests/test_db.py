"""
test_db.py
----------
Test SQL queries and chart building without using any API.
Run with: python test_db.py
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")


def run_sql(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


# ── Test 1: Check DB connection ───────────────────────────
print("\n" + "=" * 50)
print("TEST 1: Database connection")
print("=" * 50)
try:
    df = run_sql("SELECT name FROM sqlite_master WHERE type='table'")
    print(f"✅ Connected! Tables found: {df['name'].tolist()}")
except Exception as e:
    print(f"❌ Connection failed: {e}")

# ── Test 2: Check years in database ──────────────────────
print("\n" + "=" * 50)
print("TEST 2: Years available in Orders")
print("=" * 50)
df = run_sql(
    "SELECT DISTINCT strftime('%Y', OrderDate) AS Year FROM Orders ORDER BY Year"
)
print(f"✅ Years in DB: {df['Year'].tolist()}")

# ── Test 3: Check table names exactly ────────────────────
print("\n" + "=" * 50)
print("TEST 3: Check [Order Details] table exists")
print("=" * 50)
try:
    df = run_sql("SELECT COUNT(*) AS total FROM [Order Details]")
    print(f"✅ [Order Details] has {df['total'][0]} rows")
except Exception as e:
    print(f"❌ Failed: {e}")

# ── Test 4: Top 5 products query ──────────────────────────
print("\n" + "=" * 50)
print("TEST 4: Top 5 products by sales")
print("=" * 50)
try:
    df = run_sql("""
        SELECT p.ProductName, 
               SUM(od.Quantity * od.UnitPrice * (1 - od.Discount)) AS TotalSales
        FROM Orders o
        JOIN [Order Details] od ON o.OrderID = od.OrderID
        JOIN Products p ON od.ProductID = p.ProductID
        GROUP BY p.ProductName
        ORDER BY TotalSales DESC
        LIMIT 5
    """)
    print(f"✅ Result:\n{df.to_string()}")
except Exception as e:
    print(f"❌ Failed: {e}")

# ── Test 5: Top customers by order frequency ─────────────
print("\n" + "=" * 50)
print("TEST 5: Top customers by order frequency")
print("=" * 50)
try:
    df = run_sql("""
        SELECT c.CompanyName, COUNT(o.OrderID) AS OrderFrequency
        FROM Customers c
        JOIN Orders o ON c.CustomerID = o.CustomerID
        GROUP BY c.CompanyName
        ORDER BY OrderFrequency DESC
        LIMIT 10
    """)
    print(f"✅ Result:\n{df.to_string()}")
except Exception as e:
    print(f"❌ Failed: {e}")

# ── Test 6: Monthly revenue ───────────────────────────────
print("\n" + "=" * 50)
print("TEST 6: Monthly revenue (1997)")
print("=" * 50)
try:
    df = run_sql("""
        SELECT strftime('%Y-%m', o.OrderDate) AS Month,
               SUM(od.Quantity * od.UnitPrice * (1 - od.Discount)) AS Revenue
        FROM Orders o
        JOIN [Order Details] od ON o.OrderID = od.OrderID
        WHERE strftime('%Y', o.OrderDate) = '1997'
        GROUP BY strftime('%Y-%m', o.OrderDate)
        ORDER BY Month
    """)
    print(f"✅ Result:\n{df.to_string()}")
except Exception as e:
    print(f"❌ Failed: {e}")

# ── Test 7: Chart rendering ───────────────────────────────
print("\n" + "=" * 50)
print("TEST 7: Chart rendering (no API needed)")
print("=" * 50)
try:
    df = run_sql("""
        SELECT p.ProductName, SUM(od.Quantity * od.UnitPrice) AS TotalSales
        FROM Orders o
        JOIN [Order Details] od ON o.OrderID = od.OrderID
        JOIN Products p ON od.ProductID = p.ProductID
        GROUP BY p.ProductName
        ORDER BY TotalSales DESC
        LIMIT 5
    """)

    plt.rcParams.update(
        {
            "figure.facecolor": "#0f1117",
            "axes.facecolor": "#0f1117",
            "axes.labelcolor": "#ffffff",
            "xtick.color": "#ffffff",
            "ytick.color": "#ffffff",
            "text.color": "#ffffff",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    colors = plt.cm.Blues([0.4 + 0.6 * (1 - i / len(df)) for i in range(len(df))])
    ax.bar(df["ProductName"], df["TotalSales"], color=colors, edgecolor="none")
    ax.set_title("Top 5 Products by Sales", color="#ffffff", fontsize=14)
    ax.set_xticklabels(df["ProductName"], rotation=30, ha="right")
    fig.tight_layout()
    plt.savefig("test_chart.png")
    print("✅ Chart saved as test_chart.png — open it to check!")
    plt.close()
except Exception as e:
    print(f"❌ Chart failed: {e}")

print("\n" + "=" * 50)
print("ALL TESTS DONE — no API used!")
print("=" * 50)
