import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent
from utils.llm_factory import get_llm

def create_sql_writer(schema_context: str = "") -> Agent:
    """
    schema_context: RAG-retrieved relevant table schemas.
    If empty, falls back to the default Northwind schema.
    """

    # Default fallback schema
    default_schema = """
- Customers(CustomerID TEXT, CompanyName TEXT, ContactName TEXT, Country TEXT, City TEXT)
- Orders(OrderID INTEGER, CustomerID TEXT, EmployeeID INTEGER, OrderDate TEXT, ShippedDate TEXT, Freight REAL)
- [Order Details](OrderID INTEGER, ProductID INTEGER, UnitPrice REAL, Quantity INTEGER, Discount REAL)
- Products(ProductID INTEGER, ProductName TEXT, SupplierID INTEGER, CategoryID INTEGER, UnitPrice REAL, UnitsInStock INTEGER)
- Categories(CategoryID INTEGER, CategoryName TEXT, Description TEXT)
- Employees(EmployeeID INTEGER, FirstName TEXT, LastName TEXT, Title TEXT)
- Suppliers(SupplierID INTEGER, CompanyName TEXT, Country TEXT)
- Shippers(ShipperID INTEGER, CompanyName TEXT)
"""
    schema_to_use = schema_context if schema_context else default_schema

    return Agent(
        role="SQL Query Writer",
        goal=(
            "Translate the user's natural language business question into "
            "a correct, efficient SQLite SQL query using ONLY the provided schema."
        ),
        backstory=(
            f"You are an expert SQLite data analyst.\n\n"
            f"RELEVANT DATABASE SCHEMA (use ONLY these tables):\n"
            f"{schema_to_use}\n\n"
            "CRITICAL RULES:\n"
            "- This is SQLite — NOT PostgreSQL. Use only SQLite syntax.\n"
            "- Use ONLY the tables listed above — do not invent table names.\n"
            "- Table '[Order Details]' MUST be written as [Order Details] with square brackets.\n"
            "- Use strftime() for dates, never EXTRACT() or DATE_PART().\n"
            "- Data covers 1996, 1997, 1998 ONLY — never use recent years.\n"
            "- Return ONLY raw SQL — no markdown, no explanation, no code fences.\n"
            "- Revenue formula: SUM(od.Quantity * od.UnitPrice * (1 - od.Discount))\n\n"
            "EXAMPLE:\n"
            "SELECT p.ProductName, SUM(od.Quantity * od.UnitPrice) AS TotalSales\n"
            "FROM Orders o\n"
            "JOIN [Order Details] od ON o.OrderID = od.OrderID\n"
            "JOIN Products p ON od.ProductID = p.ProductID\n"
            "GROUP BY p.ProductName ORDER BY TotalSales DESC LIMIT 5;"
        ),
        llm=get_llm(temperature=0.1),
        verbose=True,
        allow_delegation=False,
    )