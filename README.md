# 🤖 Northwind Agentic AI

> A production-grade multi-agent AI system that answers business questions in plain English — powered by **CrewAI**, **ChromaDB RAG**, **Memory Caching**, **Evaluation Metrics**, **Guardrails**, and **4 LLM providers**.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat-square&logo=streamlit)

---

## 📌 Overview

Northwind Agentic AI lets anyone query a business database using plain English — no SQL knowledge required. It combines **Agentic AI** (3 specialised CrewAI agents) with **RAG** (ChromaDB vector search) to intelligently filter relevant tables before generating SQL, making it scalable to 100-200 table databases without performance loss.

**Ask:** *"Which customers have the highest order frequency?"*

**Get:** SQL query + bar chart + business insight — automatically.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **3 AI Agents** | SQL Writer → Executor → Visualizer, coordinated by CrewAI |
| **RAG Schema Retrieval** | ChromaDB finds top 5 relevant tables from 100+ using vector search |
| **Memory Cache** | Repeated questions answered in <200ms with no LLM call |
| **Evaluation Metrics** | SQL quality score, hallucination detection, faithfulness, Precision@K |
| **Guardrails** | Input sanitisation + output validation before any result is shown |
| **4 LLM Support** | Switch between Gemini, Groq, Mistral, Cohere via `.env` |
| **Model Benchmark** | Automated comparison of all models on the same questions |
| **Dark Theme UI** | Professional Streamlit interface with custom CSS |

---

## 🏗️ Architecture

```
User Question (plain English)
         │
         ▼
┌─────────────────┐
│  Input Guardrail│  ← Block injections, sanitise input
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Memory Check   │  ← ChromaDB: answered before? Return instantly
└────────┬────────┘
         │ (cache miss)
         ▼
┌─────────────────┐
│  RAG Retrieval  │  ← ChromaDB: find top 5 relevant tables
│  (ChromaDB)     │    Cosine similarity · ~50ms · 64-97% fewer tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent 1        │  ← SQL Writer: only sees retrieved table schemas
│  SQL Writer     │    temperature=0.1 · CrewAI + LiteLLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent 2        │  ← Executor: runs SQL directly via Python sqlite3
│  SQL Executor   │    reliable, fast, no LLM tool-call issues
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluator      │  ← Quality score 0-100, hallucination check
│  (Terminal)     │    Precision@K, faithfulness · terminal only
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Guard   │  ← Block hallucinated results before UI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent 3        │  ← Visualizer: chart config + business insight
│  Visualizer     │    temperature=0.3 · matplotlib dark theme
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Memory Save    │  ← Store verified result for next time
└────────┬────────┘
         │
         ▼
    Streamlit UI
  SQL + Chart + Insight
```

---

## 📁 Project Structure

```
northwind-agentic-ai/
│
├── agents/                      # AI Agent definitions
│   ├── sql_writer.py            # Agent 1: translates question to SQL
│   ├── executer.py              # Agent 2: runs SQL via Python
│   └── visualizer.py            # Agent 3: chart config + insight
│
├── tasks/                       # Task instructions per agent
│   ├── sql_task.py              # Passes RAG schema context to agent
│   ├── execute_task.py
│   └── visualizer_task.py       # Passes data sample for context
│
├── crew/
│   └── northwind_crew.py        # Full pipeline orchestrator
│
├── rag/                         # RAG layer (ChromaDB)
│   ├── schema_indexer.py        # Index all table schemas as vectors
│   ├── schema_retriever.py      # Find relevant tables per question
│   └── memory_store.py          # Cache verified Q&A pairs
│
├── evaluation/                  # Evaluation framework
│   ├── evaluator.py             # SQL quality + hallucination check
│   └── metrics.py               # Precision@K, faithfulness, model comparator
│
├── guardrails/                  # Safety checks
│   ├── input_guard.py           # Validate and sanitise user input
│   └── output_guard.py          # Validate agent output before display
│
├── tools/
│   └── db_tool.py               # CrewAI custom tool for SQL execution
│
├── utils/
│   └── llm_factory.py           # Switch LLMs via .env
│
├── northwind_db/
│   └── db_manager.py            # Auto-download Northwind DB
│
├── data/
│   └── northwind.db             # SQLite database (auto-generated)
│
├── chroma_db/                   # ChromaDB vector store (auto-created)
│
├── assets/
│   └── styles.css               # Dark theme CSS
│
├── tests/
│   └── test_db.py               # Offline tests — no API needed
│
├── workflows/
│   └── ci.yml                   # to see Running jobs, loggs failures/success
│
├── benchmark_runner.py          # Multi-model automated benchmark
├── app.py                       # Streamlit web application
├── .env                         # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **AI Framework** | CrewAI | Multi-agent orchestration |
| **Vector DB** | ChromaDB | RAG schema retrieval + memory cache |
| **Embedding** | DefaultEmbeddingFunction | No PyTorch dependency |
| **LLM — Primary** | Google Gemini 2.0 Flash Lite | Best accuracy |
| **LLM — Speed** | Groq Llama 3.1 8b Instant | Fastest (377ms avg) |
| **LLM — Alt 1** | Mistral Small | Free European model |
| **LLM — Alt 2** | Cohere Command A | Best business-friendly answers |
| **LLM Routing** | LiteLLM via CrewAI | Unified interface across providers |
| **Database** | SQLite (Northwind) | 14 tables, real business data |
| **Data Processing** | Pandas | Query result handling |
| **Visualisation** | Matplotlib | Dark-themed charts |
| **Web UI** | Streamlit | Frontend dashboard |
| **Language** | Python 3.11 | Core runtime |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- At least one free API key (Gemini or Groq recommended)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/northwind-agentic-ai.git
cd northwind-agentic-ai
```

### 2. Create virtual environment

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

### 4. Configure API keys

Create a `.env` file in the project root:

```env
# Choose your preferred provider
LLM_PROVIDER=groq               # google | groq | mistral | cohere

# Add keys for providers you want to use
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_key
COHERE_API_KEY=your_cohere_key
```

Get free API keys from:
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Groq: [console.groq.com](https://console.groq.com)
- Mistral: [console.mistral.ai](https://console.mistral.ai)
- Cohere: [dashboard.cohere.com](https://dashboard.cohere.com)

### 5. Run offline tests (no API needed)

```bash
python tests/test_db.py
```

All 7 tests should pass — confirms database, SQL and chart rendering work before spending API quota.

### 6. Build the RAG schema index (one time only)

```bash
python rag/schema_indexer.py
```

Reads all 14 tables and indexes them as vectors in ChromaDB. Only needed once — auto-rebuilds if the index is missing.

### 7. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🔍 How RAG Works

Without RAG, every query sends all table schemas to the LLM. At 100 tables this is ~20,000 tokens per call — slow, expensive and prone to hallucinations.

With RAG, ChromaDB finds the top 5 relevant tables using cosine similarity in ~50ms:

```
Question: "top 5 products by sales"
      │
      ▼
ChromaDB vector search across all indexed tables
      │
      ▼
Returns: Products (0.92), Order Details (0.88), Orders (0.81)...
      │
      ▼
Agent sees only 5 tables — ~1,000 tokens, regardless of total table count
```

| Total Tables | Without RAG | With RAG | Token Saving |
|---|---|---|---|
| 14 | ~2,800 tokens | ~1,000 tokens | 64% |
| 50 | ~10,000 tokens | ~1,000 tokens | 90% |
| 100 | ~20,000 tokens | ~1,000 tokens | 95% |
| 200 | ~40,000 tokens | ~1,000 tokens | 97.5% |

---

## 🧠 How Memory Cache Works

```
First time asking  → Full pipeline runs → Verified result saved to ChromaDB
Same question again → Similarity check (>85%) → Cache hit → Return in <200ms
```

Only results that pass the evaluation check are saved. No bad results are ever cached.

---

## 🛡️ Guardrails

**Input Guardrail** (`guardrails/input_guard.py`) blocks:
- Questions under 5 or over 500 characters
- Prompt injections: `ignore previous`, `act as`, `roleplay`
- SQL injections: `DROP`, `DELETE`, `INSERT`, `UPDATE`
- Single-word gibberish inputs

**Output Guardrail** (`guardrails/output_guard.py`) blocks:
- Results using hallucinated table names
- Empty query results (0 rows)
- Results scoring below 50/100 quality
- SQL keywords leaked into insight text

---

## 📊 Evaluation Metrics

All metrics print to terminal only — never shown to end users.

| Metric | Description | Good Value |
|---|---|---|
| SQL Quality Score | 0-100 composite, deducted for failures | 90-100 |
| Hallucination Rate | % of tables used that don't exist in DB | 0% |
| Precision@K | Relevant tables in top K retrieved | >80% |
| Faithfulness | Numbers in insight match actual data | >80% |
| Execution Time | LLM response time in milliseconds | <2000ms |

---

## 🔄 Switching LLM Providers

Change one line in `.env` — no code changes needed:

| Provider | Setting | Model | Free? | Avg Speed |
|---|---|---|---|---|
| Google Gemini | `LLM_PROVIDER=google` | gemini-2.0-flash-lite | ✅ | ~2766ms |
| Groq | `LLM_PROVIDER=groq` | llama-3.1-8b-instant | ✅ | ~377ms |
| Mistral | `LLM_PROVIDER=mistral` | mistral-small-latest | ✅ | ~1046ms |
| Cohere | `LLM_PROVIDER=cohere` | command-a-03-2025 | ✅ | ~9514ms |

---

## 🔬 Model Benchmark

Run the automated benchmark to compare all models on the same questions:

```bash
python benchmark_runner.py
```

**Results from our benchmark (2 questions × 4 models):**

| Model | Avg Quality | Success Rate | Avg Speed |
|---|---|---|---|
| Groq Llama 3.1 8b | 100/100 | 100% | 377ms |
| Mistral 7b | 100/100 | 100% | 1046ms |
| Cohere Command A | 100/100 | 100% | 9514ms |
| Gemini 2.0 Flash Lite | 50/100 | 50% | — |

> Gemini scored lower due to free tier quota limits during benchmark — not a model quality issue. In normal use Gemini performs at 100/100.

Full results are saved automatically to `benchmark_TIMESTAMP.json`.

---

## 💬 Example Questions

**Sales Analysis**
- What were the top 5 products by total sales revenue?
- Which product category generates the most revenue?
- What is the total revenue generated per year?

**Customer Analysis**
- Which customers have the highest order frequency?
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

## 🗄️ Northwind Database Schema

| Table | Key Columns |
|---|---|
| `Customers` | CustomerID, CompanyName, ContactName, Country, City |
| `Orders` | OrderID, CustomerID, EmployeeID, OrderDate, ShippedDate, Freight |
| `[Order Details]` | OrderID, ProductID, UnitPrice, Quantity, Discount |
| `Products` | ProductID, ProductName, CategoryID, UnitPrice, UnitsInStock |
| `Categories` | CategoryID, CategoryName, Description |
| `Employees` | EmployeeID, FirstName, LastName, Title, HireDate |
| `Suppliers` | SupplierID, CompanyName, Country |
| `Shippers` | ShipperID, CompanyName, Phone |
| `Territories` | TerritoryID, TerritoryDescription, RegionID |
| `Regions` | RegionID, RegionDescription |

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: crewai` | Recreate venv with Python 3.11, reinstall requirements |
| `GOOGLE_API_KEY not found` | Check `.env` — no quotes or spaces around `=` |
| `429 RESOURCE_EXHAUSTED` | Wait 1 minute or switch `LLM_PROVIDER` in `.env` |
| `command-r-plus was removed` | Use `cohere/command-a-03-2025` in `llm_factory.py` |
| `RAG coverage 100%` | Rebuild index: `rm -rf chroma_db/ && python rag/schema_indexer.py` |
| `OpenAI 401 on Groq/Cohere` | Remove `OPENAI_API_KEY` from `.env` — run `env | grep OPENAI` to check |
| `sentence-transformers error` | Use `DefaultEmbeddingFunction()` — no PyTorch needed |
| `eval_result not defined` | Replace with `eval_summary.get("quality_score", 100)` in final return |

---

## 🗺️ Roadmap

### Phase 1 — Short Term
- [ ] Conversation memory across sessions
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
- [ ] Integration with Tableau and Power BI

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄Link for app

- [my app](https://northwind-crew-ai-agentic-rag.streamlit.app/)

---

## 🙏 Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent AI framework
- [ChromaDB](https://www.trychroma.com) — vector database for RAG and memory
- [Northwind SQLite](https://github.com/jpwhite3/northwind-SQLite3) — sample business database
- [Google Gemini](https://aistudio.google.com) — LLM provider
- [Groq](https://console.groq.com) — fast free LLM inference
- [Mistral AI](https://mistral.ai) — open-weight European LLM
- [Cohere](https://cohere.com) — enterprise language models
- [Streamlit](https://streamlit.io) — web UI framework

---

<div align="center">
  <strong>Built with 💚 using CrewAI + ChromaDB + Streamlit</strong><br/>
  <em>Agentic AI + RAG for business intelligence — scalable to 100-200 tables</em>
</div>