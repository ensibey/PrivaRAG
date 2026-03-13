"""
app.py — Local Offline RAG System — Streamlit UI (v2)

Modes:
  📄 Document Q&A  — Conversational RAG with memory + source citations
  📊 Excel Analysis — PandasDataFrameAgent on uploaded spreadsheets
  🗄️  SQL Query     — Natural language → SQLite

UI features:
  • WhatsApp-style chat bubbles (user right / AI left)
  • Pulsing green "🔒 OFFLINE · DATA SECURE" badge
  • Live Ollama status indicator (🟢/🔴)
  • Drag-and-drop file upload with animated "Local Processing…" bar
  • Inline source citations: "Source: Finance.pdf, Page 12"
"""

import os
import tempfile
import time
import sqlite3
from pathlib import Path

import requests
import streamlit as st
import psutil
from PIL import Image
from config import LLM_MODEL, EMBEDDING_MODEL, DOCS_STORE_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local RAG — Offline AI",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Dark premium theme + WhatsApp-style bubbles
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg,#0d0d14 0%,#111827 60%,#0d1117 100%); color:#e2e8f0; }

/* ── OFFLINE badge ── */
.offline-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:linear-gradient(135deg,#064e3b,#065f46);
    border:1px solid #10b981; border-radius:50px;
    padding:7px 18px; font-size:.82rem; font-weight:600;
    color:#6ee7b7; letter-spacing:.05em;
    animation:pulse-glow 2.2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { box-shadow:0 0 12px rgba(16,185,129,.2); }
    50%      { box-shadow:0 0 28px rgba(16,185,129,.55); }
}

/* ── Ollama status dot ── */
.status-running { display:inline-flex; align-items:center; gap:6px; font-size:.78rem;
    color:#4ade80; font-weight:600; }
.status-offline { display:inline-flex; align-items:center; gap:6px; font-size:.78rem;
    color:#f87171; font-weight:600; }
.dot { width:9px; height:9px; border-radius:50%; }
.dot-green { background:#4ade80; animation:blink 1.5s infinite; }
.dot-red   { background:#f87171; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.3;} }

/* ── Header ── */
.header-bar {
    display:flex; justify-content:space-between; align-items:center;
    padding:14px 0 20px; border-bottom:1px solid #1e293b; margin-bottom:20px;
}
.header-title {
    font-size:1.5rem; font-weight:700;
    background:linear-gradient(135deg,#818cf8,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── WhatsApp chat bubbles ── */
.bubble-wrap-user { display:flex; justify-content:flex-end; margin:6px 0; }
.bubble-wrap-ai   { display:flex; justify-content:flex-start; margin:6px 0; }

.bubble-user {
    background:linear-gradient(135deg,#1d4ed8,#4f46e5);
    border-radius:18px 18px 4px 18px;
    padding:11px 16px; max-width:78%; color:#e0e7ff;
    font-size:.9rem; line-height:1.55;
    box-shadow:0 2px 8px rgba(79,70,229,.3);
}
.bubble-ai {
    background:linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid #334155;
    border-radius:18px 18px 18px 4px;
    padding:11px 16px; max-width:78%; color:#cbd5e1;
    font-size:.9rem; line-height:1.55;
}
.bubble-label {
    font-size:.68rem; color:#64748b; margin-bottom:4px; padding-left:4px;
}

/* ── Source citation card ── */
.src-card {
    background:#0f172a; border:1px solid #1e3a5f;
    border-left:3px solid #3b82f6; border-radius:7px;
    padding:8px 12px; margin:3px 0; font-size:.78rem;
    color:#94a3b8; font-family:monospace; line-height:1.5;
}

/* ── SQL result card ── */
.sql-card {
    background:#0f172a; border:1px solid #1e3a5f;
    border-left:3px solid #a78bfa; border-radius:7px;
    padding:8px 12px; margin:4px 0; font-size:.78rem;
    color:#c4b5fd; font-family:monospace;
}

/* ── Stat pill ── */
.stat-pill { background:#1e293b; border:1px solid #334155;
    border-radius:8px; padding:10px 14px; text-align:center; margin-bottom:6px; }
.stat-val { font-size:1.5rem; font-weight:700; color:#818cf8; }
.stat-lbl { font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e293b; }

/* ── Inputs & buttons ── */
.stTextInput>div>div>input { background:#1e293b; border:1px solid #334155;
    border-radius:10px; color:#e2e8f0; }
.stButton>button { background:linear-gradient(135deg,#4f46e5,#7c3aed); color:white;
    border:none; border-radius:10px; font-weight:600; transition:all .2s; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 18px rgba(99,102,241,.4); }
.stProgress>div>div>div>div { background:linear-gradient(90deg,#10b981,#3b82f6); }

hr { border-color:#1e293b; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Ollama status check
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ollama_badge(ok: bool) -> str:
    if ok:
        return ('<span class="status-running">'
                '<span class="dot dot-green"></span>Ollama: Running</span>')
    return ('<span class="status-offline">'
            '<span class="dot dot-red"></span>Ollama: Offline</span>')

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        # RAG mode
        "rag_messages": [],        # list of {"role","content","sources"}
        "rag_history": [],         # list of (human_str, ai_str) for ConversationalChain
        "doc_count": 0,
        "ingested_files": [],
        # Excel mode
        "excel_messages": [],
        "excel_file_path": None,
        "excel_file_name": None,
        "excel_sheets": [],
        # SQL mode
        "sql_messages": [],
        "sql_db_path": None,
        "sql_db_name": None,
        "sql_tables": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
ollama_ok = check_ollama()
st.markdown(f"""
<div class="header-bar">
  <div class="header-title">🧠 Local RAG — Offline AI Assistant</div>
  <div style="display:flex;align-items:center;gap:20px;">
    {ollama_badge(ollama_ok)}
    <span class="offline-badge">🔒 OFFLINE · DATA SECURE</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not ollama_ok:
    st.warning("⚠️ Ollama is not running. Start it with `ollama serve` before continuing.", icon="⚠️")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — mode selector + file upload
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    mode = st.radio(
        "**Mode**",
        ["📄 Document Q&A", "📊 Excel Analysis", "🗄️ SQL Query"],
        label_visibility="visible",
    )
    st.markdown("---")

    st.markdown("### 🖥️ System Monitor")
    st.caption(f"**LLM:** `{LLM_MODEL}` | **Embed:** `{EMBEDDING_MODEL}`")
    cpu_col, ram_col = st.columns(2)
    with cpu_col:
        st.metric("CPU", f"{psutil.cpu_percent(interval=None)}%")
    with ram_col:
        st.metric("RAM", f"{psutil.virtual_memory().percent}%")
    st.markdown("---")

    # ── MODE: Document Q&A ──────────────────────────────────────────────────
    if mode == "📄 Document Q&A":
        st.markdown("### 📂 Upload Documents")
        st.caption("PDF and Excel files — embedded locally.")

        uploaded = st.file_uploader(
            "Drag & drop files here",
            type=["pdf", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            if st.button("⚡ Ingest Files", use_container_width=True):
                from ingest import ingest_file, get_doc_count

                prog = st.progress(0, text="Starting…")
                status = st.empty()
                total = 0

                for idx, uf in enumerate(uploaded):
                    if uf.name in st.session_state.ingested_files:
                        st.info(f"⏭ Already indexed: {uf.name}")
                        continue

                    os.makedirs(DOCS_STORE_DIR, exist_ok=True)
                    doc_path = os.path.join(DOCS_STORE_DIR, uf.name)
                    with open(doc_path, "wb") as f:
                        f.write(uf.getbuffer())

                    def _cb(frac, msg, _idx=idx, _n=len(uploaded), _name=uf.name):
                        overall = (_idx + frac) / _n
                        prog.progress(overall, text=f"🔐 Local Processing · {_name} · {msg}")
                        status.caption("🖥 CPU only — zero network calls")

                    try:
                        n = ingest_file(doc_path, progress_callback=_cb)
                        total += n
                        st.session_state.ingested_files.append(uf.name)
                        st.success(f"✅ {uf.name} — {n} chunks")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")

                prog.progress(1.0, text="✅ Done!")
                time.sleep(0.4)
                prog.empty(); status.empty()

                try:
                    st.session_state.doc_count = get_doc_count()
                except Exception:
                    st.session_state.doc_count += total

        st.markdown("---")
        st.markdown("### 📊 Knowledge Base")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-pill"><div class="stat-val">{st.session_state.doc_count}</div>'
                        f'<div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-pill"><div class="stat-val">{len(st.session_state.ingested_files)}</div>'
                        f'<div class="stat-lbl">Files</div></div>', unsafe_allow_html=True)

        if st.session_state.ingested_files:
            for f in st.session_state.ingested_files:
                st.markdown(f"- 📄 `{f}`")

        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.rag_messages = []
            st.session_state.rag_history = []
            st.rerun()

    # ── MODE: Excel Analysis ────────────────────────────────────────────────
    elif mode == "📊 Excel Analysis":
        st.markdown("### 📊 Upload Spreadsheet")
        st.caption("Upload an Excel or CSV file to analyse it with AI.")

        uf = st.file_uploader(
            "Drag & drop here",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        if uf:
            suf = Path(uf.name).suffix
            saved_path = os.path.join(tempfile.gettempdir(), f"rag_excel_{uf.name}")
            with open(saved_path, "wb") as f:
                f.write(uf.getbuffer())

            if st.session_state.excel_file_name != uf.name:
                st.session_state.excel_file_path = saved_path
                st.session_state.excel_file_name = uf.name
                st.session_state.excel_messages = []

                from excel_agent import get_sheet_names
                st.session_state.excel_sheets = get_sheet_names(saved_path)

            st.success(f"✅ Loaded: `{uf.name}`")

            if len(st.session_state.excel_sheets) > 1:
                st.session_state.selected_sheet = st.selectbox(
                    "Sheet", st.session_state.excel_sheets
                )
            else:
                st.session_state.selected_sheet = (
                    st.session_state.excel_sheets[0]
                    if st.session_state.excel_sheets else None
                )

        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.excel_messages = []
            st.rerun()

    # ── MODE: SQL Query ─────────────────────────────────────────────────────
    else:
        st.markdown("### 🗄️ Database Source")
        st.caption("Upload a SQLite `.db` file, or upload Excel to auto-convert.")

        db_file = st.file_uploader(
            "SQLite .db file",
            type=["db", "sqlite", "sqlite3"],
            label_visibility="collapsed",
        )
        st.markdown("**— or convert Excel →**")
        xl_file = st.file_uploader(
            "Excel / CSV → SQLite",
            type=["xlsx", "xls", "csv"],
            label_visibility="collapsed",
            key="sql_xl",
        )

        if db_file:
            db_path = os.path.join(tempfile.gettempdir(), db_file.name)
            with open(db_path, "wb") as f:
                f.write(db_file.getbuffer())
            if st.session_state.sql_db_name != db_file.name:
                st.session_state.sql_db_path = db_path
                st.session_state.sql_db_name = db_file.name
                st.session_state.sql_messages = []
                from sql_agent import get_tables
                st.session_state.sql_tables = get_tables(db_path)
            st.success(f"✅ Loaded: `{db_file.name}`")

        elif xl_file:
            db_path = os.path.join(
                tempfile.gettempdir(), Path(xl_file.name).stem + "_rag.db"
            )
            xl_path = os.path.join(tempfile.gettempdir(), xl_file.name)
            with open(xl_path, "wb") as f:
                f.write(xl_file.getbuffer())

            if st.session_state.sql_db_name != xl_file.name:
                with st.spinner("Converting Excel → SQLite…"):
                    from sql_agent import excel_to_sqlite, get_tables
                    tables = excel_to_sqlite(xl_path, db_path)
                st.session_state.sql_db_path = db_path
                st.session_state.sql_db_name = xl_file.name
                st.session_state.sql_tables = tables
                st.session_state.sql_messages = []
                st.success(f"✅ Converted `{xl_file.name}` → {len(tables)} table(s)")

        if st.session_state.sql_tables:
            st.markdown("**Tables:**")
            for t in st.session_state.sql_tables:
                st.markdown(f"- 🗃 `{t}`")

        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.sql_messages = []
            st.rerun()

    # ── Privacy panel (all modes) ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
<div style="font-size:.76rem;color:#64748b;line-height:1.7;">
🔒 <b style="color:#6ee7b7">LLM:</b> Ollama (local CPU/GPU)<br>
🔒 <b style="color:#6ee7b7">Embeddings:</b> nomic-embed-text (local)<br>
🔒 <b style="color:#6ee7b7">Vector DB:</b> ChromaDB (on-disk)<br>
🔒 <b style="color:#6ee7b7">SQL DB:</b> SQLite (on-disk)<br>
🌐 <b style="color:#f87171">Network calls:</b> None
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: render a WhatsApp-style chat thread
# ──────────────────────────────────────────────────────────────────────────────

def render_chat(messages: list[dict]):
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f"""
<div class="bubble-wrap-user">
  <div>
    <div class="bubble-label" style="text-align:right;">You</div>
    <div class="bubble-user">{content}</div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="bubble-wrap-ai">
  <div>
    <div class="bubble-label">🤖 AI</div>
    <div class="bubble-ai">{content}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Source citations
            sources = msg.get("sources", [])
            if sources:
                label_parts = []
                for s in sources:
                    parts = [s["source"]]
                    if s.get("page"):
                        parts.append(f"Page {s['page']}")
                    elif s.get("sheet"):
                        parts.append(f"Sheet: {s['sheet']}")
                    label_parts.append(", ".join(parts))
                citation_label = " · ".join(dict.fromkeys(label_parts))  # deduplicate
                with st.expander(f"📎 Source: {citation_label}", expanded=False):
                    for s in sources:
                        loc = ""
                        if s.get("page"):
                            loc = f"Page {s['page']}"
                        elif s.get("sheet"):
                            loc = f"Sheet: {s['sheet']}"
                        st.markdown(
                            f'<div class="src-card">'
                            f'<b style="color:#60a5fa">📄 {s["source"]}</b>'
                            f'{" · " + loc if loc else ""}<br>'
                            f'{s["text"][:400]}{"…" if len(s["text"]) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        
                        # Add PDF preview
                        if s["source"].lower().endswith(".pdf") and s.get("page"):
                            pdf_path = os.path.join(DOCS_STORE_DIR, s["source"])
                            if os.path.exists(pdf_path):
                                try:
                                    import fitz
                                    doc = fitz.open(pdf_path)
                                    idx_page = s["page"] - 1 # 0-indexed
                                    if 0 <= idx_page < len(doc):
                                        page_obj = doc.load_page(idx_page)
                                        pix = page_obj.get_pixmap(dpi=100) # High enough for UI readability
                                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                        st.image(img, caption=f"{s['source']} - Page {s['page']}", use_container_width=True)
                                except Exception as e:
                                    st.caption(f"Failed to load PDF preview: {e}")

            # SQL-specific rendering
            if msg.get("sql_query"):
                with st.expander("🔍 Generated SQL", expanded=False):
                    st.code(msg["sql_query"], language="sql")
                if msg.get("raw_result"):
                    with st.expander("📋 Raw Query Result", expanded=False):
                        st.markdown(f'<div class="sql-card">{msg["raw_result"]}</div>',
                                    unsafe_allow_html=True)

            # Excel dataframe info
            if msg.get("df_info"):
                st.caption(f"📊 {msg['df_info']}")

# ──────────────────────────────────────────────────────────────────────────────
# Main area — per-mode chat
# ──────────────────────────────────────────────────────────────────────────────

if mode == "📄 Document Q&A":
    if not st.session_state.rag_messages:
        st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#475569;">
  <div style="font-size:3rem;margin-bottom:14px;">🔍</div>
  <div style="font-size:1.05rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">
    Upload documents, then ask anything
  </div>
  <div style="font-size:.83rem;">
    AI answers from your files only · Multi-turn memory enabled<br>
    Ask follow-ups — the AI remembers previous answers
  </div>
</div>""", unsafe_allow_html=True)
    else:
        render_chat(st.session_state.rag_messages)

    if st.session_state.doc_count == 0:
        st.info("📂 Upload and ingest at least one document via the sidebar.", icon="ℹ️")
    else:
        q = st.chat_input("Ask a question about your documents…")
        if q:
            st.session_state.rag_messages.append({"role": "user", "content": q})
            with st.spinner("🔮 Searching documents + thinking locally…"):
                try:
                    from rag_engine import query as rag_query
                    result = rag_query(q, st.session_state.rag_history)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    sources = []

            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            st.session_state.rag_history.append((q, answer))
            st.rerun()

elif mode == "📊 Excel Analysis":
    if not st.session_state.excel_file_path:
        st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#475569;">
  <div style="font-size:3rem;margin-bottom:14px;">📊</div>
  <div style="font-size:1.05rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">
    Upload a spreadsheet to start analysis
  </div>
  <div style="font-size:.83rem;">
    Ask in plain English · AI uses Pandas under the hood · 100% local
  </div>
</div>""", unsafe_allow_html=True)
    else:
        if not st.session_state.excel_messages:
            st.info(f"📊 `{st.session_state.excel_file_name}` loaded. Ask anything about the data!", icon="ℹ️")
        else:
            render_chat(st.session_state.excel_messages)

        q = st.chat_input("E.g. 'Which region had the highest sales?' or 'Show average per category'…")
        if q:
            st.session_state.excel_messages.append({"role": "user", "content": q})
            with st.spinner("📊 Analysing spreadsheet locally…"):
                try:
                    from excel_agent import query_excel
                    sheet = getattr(st.session_state, "selected_sheet", None)
                    result = query_excel(
                        st.session_state.excel_file_path, q,
                        sheet_name=sheet if sheet != "CSV" else None,
                    )
                    answer = result["answer"]
                    df_info = result["dataframe_info"]
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    df_info = ""

            st.session_state.excel_messages.append({
                "role": "assistant",
                "content": answer,
                "df_info": df_info,
            })
            st.rerun()

else:  # SQL Query
    if not st.session_state.sql_db_path:
        st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#475569;">
  <div style="font-size:3rem;margin-bottom:14px;">🗄️</div>
  <div style="font-size:1.05rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">
    Upload a SQLite database or convert an Excel file
  </div>
  <div style="font-size:.83rem;">
    AI translates your question to SQL, runs it, and explains the result
  </div>
</div>""", unsafe_allow_html=True)
    else:
        if not st.session_state.sql_messages:
            tables_str = ", ".join([f"`{t}`" for t in st.session_state.sql_tables])
            st.info(f"🗄️ Database ready. Tables: {tables_str}", icon="ℹ️")
        else:
            render_chat(st.session_state.sql_messages)

        q = st.chat_input("E.g. 'Total sales per month' or 'Top 5 customers by revenue'…")
        if q:
            st.session_state.sql_messages.append({"role": "user", "content": q})
            with st.spinner("🗄️ Generating SQL and running query locally…"):
                try:
                    from sql_agent import query_sql
                    result = query_sql(st.session_state.sql_db_path, q)
                    answer = result["answer"]
                    sql_q = result["sql_query"]
                    raw = result["raw_result"]
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    sql_q = ""
                    raw = ""

            st.session_state.sql_messages.append({
                "role": "assistant",
                "content": answer,
                "sql_query": sql_q,
                "raw_result": raw,
            })
            st.rerun()
