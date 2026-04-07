import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent
from utils.llm_factory import get_llm

SCHEMA_SUMMARY = """
Northwind SQLite schema — use EXACTLY these table names (case-sensitive, with spaces where shown):

TABLES AND COLUMNS:
- Customers(CustomerID, CompanyName, ContactName, Country, City)
- Orders(OrderID, CustomerID, EmployeeID, OrderDate, ShippedDate, Freight)
- [Order Details](OrderID, ProductID, UnitPrice, Quantity, Discount)
- Products(ProductID, ProductName, SupplierID, CategoryID, UnitPrice, UnitsInStock)
- Categories(CategoryID, CategoryName, Description)
- Employees(EmployeeID, FirstName, LastName, Title)
- Suppliers(SupplierID, CompanyName, Country)
- Shippers(ShipperID, CompanyName)

CRITICAL RULES:
- This is SQLite — NOT PostgreSQL, NOT MySQL. Use ONLY SQLite syntax.
- The order details table MUST be written as [Order Details] with square brackets always.
- Dates are stored as text 'YYYY-MM-DD'. Use strftime() for date operations.
- The Northwind data covers years 1996, 1997, and 1998 ONLY.
- Never use EXTRACT() — it does not exist in SQLite.
- Never use DATE_PART() — it does not exist in SQLite.
- Never use DATE_TRUNC() — it does not exist in SQLite.
- Never use NOW() — use date('now') instead.

SQLITE DATE FUNCTIONS — use these instead:
- Year:    strftime('%Y', OrderDate)
- Month:   strftime('%m', OrderDate)
- Day:     strftime('%d', OrderDate)
- Quarter: CASE 
               WHEN strftime('%m', OrderDate) BETWEEN '01' AND '03' THEN 'Q1'
               WHEN strftime('%m', OrderDate) BETWEEN '04' AND '06' THEN 'Q2'
               WHEN strftime('%m', OrderDate) BETWEEN '07' AND '09' THEN 'Q3'
               ELSE 'Q4'
           END

REVENUE FORMULA:
SUM(od.Quantity * od.UnitPrice * (1 - od.Discount))

EXAMPLE CORRECT QUERIES:

-- Monthly revenue:
SELECT strftime('%Y-%m', o.OrderDate) AS Month,
       SUM(od.Quantity * od.UnitPrice * (1 - od.Discount)) AS Revenue
FROM Orders o
JOIN [Order Details] od ON o.OrderID = od.OrderID
GROUP BY strftime('%Y-%m', o.OrderDate)
ORDER BY Month;

-- Top 5 products:
SELECT p.ProductName, SUM(od.Quantity * od.UnitPrice) AS TotalSales
FROM Orders o
JOIN [Order Details] od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
GROUP BY p.ProductName
ORDER BY TotalSales DESC
LIMIT 5;
"""

def create_sql_writer() -> Agent:
    return Agent(
        role="SQL Query Writer",
        goal=(
            "Translate the user's natural language business question into "
            "a correct, efficient SQLite SQL query using the exact schema provided."
        ),
        backstory=(
            f"You are an expert data analyst who knows the Northwind database schema perfectly.\n"
            f"{SCHEMA_SUMMARY}\n"
            "Always write SELECT queries only. Never modify data. "
            "Return ONLY the raw SQL — no markdown, no explanation, no code fences."
        ),
        llm=get_llm(temperature=0.1),
        verbose=True,
        allow_delegation=False,
    )