# 🤖 Northwind Agentic AI

> A multi-agent AI system that answers business questions in plain English — powered by **CrewAI**, **Google Gemini**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?style=flat-square&logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Northwind%20DB-lightblue?style=flat-square&logo=sqlite)

---

## 📌 Overview

Northwind Agentic AI is a **multi-agent agentic AI system** that lets users query the Northwind business database using plain English — no SQL knowledge required.

The system uses **3 specialised AI agents** working in sequence:

1. **SQL Writer Agent** — Translates a natural language question into a SQLite query
2. **SQL Executor Agent** — Runs the query against the Northwind database
3. **Visualizer Agent** — Generates a chart and business insight from the results

All agents are orchestrated by **CrewAI** and presented through a **Streamlit** web interface with a custom dark theme.

---

## 🎯 Problem Statement

Business data locked inside databases is inaccessible to non-technical users. Analysts spend time writing repetitive queries instead of generating insights. This project solves that by:

- Allowing anyone to ask business questions in plain English
- Automatically generating accurate SQL from natural language
- Turning raw data into visualisations and insights instantly
- Demonstrating a real-world agentic AI pipeline architecture

---

## 🏗️ Architecture

```
User (Plain English Question)
        │
        ▼
┌─────────────────┐
│   Streamlit UI  │  ← app.py
│   (Dark Theme)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│              CrewAI Orchestrator            │
│           crew/northwind_crew.py            │
└──────┬──────────────┬───────────────┬───────┘
       │              │               │
       ▼              ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Agent 1 │   │  Agent 2 │   │   Agent 3    │
│SQL Writer│──▶│Executor  │──▶│  Visualizer  │
└──────────┘   └──────────┘   └──────────────┘
       │              │               │
       ▼              ▼               ▼
  SQL Query      SQLite DB       Chart + Insight
                Northwind.db    (matplotlib)
```

---

## 📁 Folder Structure

```
Northwind-Agentic-AI/
│
├── agents/                     # AI Agent definitions
│   ├── sql_writer.py           # Agent 1: writes SQL queries
│   ├── executer.py             # Agent 2: executes SQL
│   └── visualizer.py           # Agent 3: generates charts & insights
│
├── tasks/                      # Task instructions for each agent
│   ├── sql_task.py
│   ├── execute_task.py
│   └── visualizer_task.py
│
├── crew/                       # CrewAI pipeline orchestrator
│   └── northwind_crew.py
│
├── tools/                      # Custom CrewAI tools
│   └── db_tool.py              # ExecuteSQLTool — runs SQL on Northwind DB
│
├── utils/                      # Utilities
│   └── llm_factory.py          # LLM switcher: Gemini / OpenAI / Groq
│
├── northwind_db/               # Database manager
│   └── db_manager.py           # Auto-downloads northwind.db if missing
│
├── data/                       # Database file (auto-generated)
│   └── northwind.db
│
├── assets/                     # Frontend styling
│   └── styles.css              # Dark theme CSS for Streamlit
│
├── tests/                      # Offline tests (no API needed)
│   └── test_db.py
│
├── app.py                      # Streamlit web application
├── .env                        # API keys (not committed to Git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **AI Framework** | CrewAI |
| **LLM — Primary** | Google Gemini 2.0 Flash Lite |
| **LLM — Alternative** | Groq (Llama 3.1 8b Instant) |
| **LLM — Alternative** | OpenAI GPT-4o Mini |
| **LLM Routing** | LangChain + LiteLLM |
| **Database** | SQLite (Northwind dataset) |
| **Data Processing** | Pandas |
| **Visualisation** | Matplotlib |
| **Web UI** | Streamlit |
| **Language** | Python 3.11 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js (optional — only if regenerating the PowerPoint)
- A free API key from one of: [Google AI Studio](https://aistudio.google.com), [Groq](https://console.groq.com), or [OpenAI](https://platform.openai.com)

### 1. Clone the repository

```bash
git clone https://github.com/smshelar/northwind-agentic-ai.git
cd northwind-agentic-ai
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```env
# Choose ONE provider and add its key
GOOGLE_API_KEY=your_gemini_key_here
LLM_PROVIDER=google

# Or use Groq (free, fast)
# GROQ_API_KEY=your_groq_key_here
# LLM_PROVIDER=groq

# Or use OpenAI
# OPENAI_API_KEY=your_openai_key_here
# LLM_PROVIDER=openai
```

> 💡 Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) — no credit card required.

### 5. Run offline tests (no API needed)

```bash
python tests/test_db.py
```

All 7 tests should pass — this confirms your database and chart logic work before spending API quota.

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Example Questions

Try asking these in the app:

**Sales Analysis**
- What were the top 5 products by total sales revenue?
- Which product category generates the most revenue?
- What is the total revenue generated per year?

**Customer Analysis**
- Which customers have placed the most orders?
- Who are the top 10 customers by total amount spent?
- Which country has the most customers?

**Employee Performance**
- Which employee has handled the most orders?
- What is the total sales revenue generated by each employee?

**Inventory**
- Which products are running low on stock?
- Which products have never been ordered?

> ⚠️ The Northwind database contains data from **1996, 1997, and 1998 only**.

---

## 🔄 Switching LLM Providers

Change the `LLM_PROVIDER` in your `.env` file — no code changes needed:

| Provider | `.env` setting | Model used | Free? |
|---|---|---|---|
| Google Gemini | `LLM_PROVIDER=google` | gemini-2.0-flash-lite | ✅ Yes |
| Groq | `LLM_PROVIDER=groq` | llama-3.1-8b-instant | ✅ Yes |
| OpenAI | `LLM_PROVIDER=openai` | gpt-4o-mini | ❌ Paid |

---

## 🧪 Running Tests Without API

The `tests/test_db.py` file validates everything that doesn't need an LLM:

```bash
python tests/test_db.py
```

This tests:
- ✅ Database connection
- ✅ Available years in the data
- ✅ Table names (including `[Order Details]` with space)
- ✅ Top products query
- ✅ Top customers query
- ✅ Monthly revenue query
- ✅ Chart rendering (saves `test_chart.png`)

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: crewai` | Run `pip install -r requirements.txt` with Python 3.11 venv activated |
| `GOOGLE_API_KEY not found` | Check `.env` has no quotes or spaces around the `=` |
| `Rate limit exceeded` | Wait 1 minute and retry, or switch `LLM_PROVIDER` in `.env` |
| `No such table: OrderDetails` | Use `[Order Details]` with square brackets — the schema is fixed in `sql_writer.py` |
| `Executor returned non-JSON` | Known issue with small LLMs — SQL now executes directly in Python |
| `Python 3.9 venv` | Delete venv, recreate with `python3.11 -m venv venv` |

---

## 🗺️ Roadmap

### Phase 1 — Short Term
- [ ] Conversation memory — remember previous questions in the session
- [ ] Query history — save and replay past questions
- [ ] Export results to CSV / Excel
- [ ] Better error messages with suggested rephrasing

### Phase 2 — Medium Term
- [ ] Support for multiple databases beyond Northwind
- [ ] Voice input — ask questions by speaking
- [ ] Dashboard mode — pin favourite charts
- [ ] Scheduled email reports

### Phase 3 — Long Term
- [ ] Fine-tuned SQL model on company-specific schemas
- [ ] Role-based access control
- [ ] Predictive analytics and trend forecasting
- [ ] Integration with Tableau / Power BI

---

## 📊 Northwind Database Schema

```
Customers      ──┐
                 ├──▶ Orders ──▶ [Order Details] ──▶ Products ──▶ Categories
Employees      ──┘                                          └──▶ Suppliers
Shippers ──────────▶ Orders
```

Key tables:
- `Customers` — company info, country, city
- `Orders` — order dates, customer, employee, shipper
- `[Order Details]` — products, quantities, prices per order
- `Products` — product names, categories, stock levels
- `Employees` — sales rep info

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent AI framework
- [Northwind Dataset](https://github.com/jpwhite3/northwind-SQLite3) — sample business database
- [Google Gemini](https://aistudio.google.com) — LLM provider
- [Streamlit](https://streamlit.io) — web UI framework
- [Groq](https://console.groq.com) — fast free LLM inference

---

<div align="center">
  <strong>Built with 💚 using CrewAI + Gemini + Streamlit</strong><br/>
  <em>A practical demonstration of Agentic AI for business intelligence</em>
</div>