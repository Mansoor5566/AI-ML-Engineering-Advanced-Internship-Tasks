import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Global tokens ──────────────────────────────────────── */
:root {
    --bg:           #f0f2f6;   /* Page background – light grey */
    --surface:      #ffffff;   /* Sidebar & card surfaces      */
    --surface2:     #f8fafc;   /* Subtle secondary surface     */
    --border:       #e2e8f0;   /* Dividers / card borders      */

    /* Accent palette */
    --accent:       #2563eb;   /* Primary blue                 */
    --accent-light: #eff6ff;   /* Blue tint for bot bubble     */
    --accent-dark:  #1d4ed8;   /* Darker blue for hover        */
    --green:        #16a34a;   /* Success / online indicator   */
    --green-light:  #f0fdf4;   /* Green tint (unused currently)*/

    /* Text */
    --text-primary:   #0f172a; /* Headings & body text         */
    --text-secondary: #475569; /* Labels, captions             */
    --text-muted:     #94a3b8; /* Placeholder, disabled        */

    /* Bubble colours  (high contrast against --bg) */
    --user-bubble-bg:    #1e40af; /* Deep blue  → white text  */
    --user-bubble-text:  #ffffff;
    --bot-bubble-bg:     #ffffff; /* White card → dark text    */
    --bot-bubble-text:   #0f172a;
    --bot-bubble-border: #bfdbfe; /* Light blue border          */
}

/* ── Base reset ─────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: "Inter", "Segoe UI", sans-serif !important;
}

.stApp { background-color: var(--bg) !important; }

.block-container { padding-top: 2rem !important; }

/* ── Sidebar ────────────────────────────────────────────── */
div[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Page header ────────────────────────────────────────── */
.chat-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
}

.chat-accent { color: var(--accent); }

.chat-subtitle {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
}

/* ── Chat bubbles ───────────────────────────────────────── */
.user-msg {
    background: var(--user-bubble-bg);
    color:       var(--user-bubble-text);
    padding: 0.85rem 1.1rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.75rem 0 0.75rem 15%;   /* indent from left */
    font-size: 0.9rem;
    line-height: 1.6;
}

.bot-msg {
    background: var(--bot-bubble-bg);
    color:       var(--bot-bubble-text);
    padding: 0.85rem 1.1rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.75rem 15% 0.75rem 0;  /* indent from right */
    border: 1px solid var(--bot-bubble-border);
    font-size: 0.9rem;
    line-height: 1.7;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Role labels ────────────────────────────────────────── */
.label-user {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: right;
    margin: 0.5rem 0 0.2rem;
}

.label-bot {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0.5rem 0 0.2rem;
}

/* ── Source tags ────────────────────────────────────────── */
.source-tag {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 0.2rem 0.65rem;
    margin: 0.2rem 0.15rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* ── Sidebar stat cards ─────────────────────────────────── */
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}

.stat-number {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

/* ── Topic pills (sidebar) ──────────────────────────────── */
.topic-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    margin: 0.2rem;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Divider ────────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}

/* ── Chat input override ────────────────────────────────── */
.stChatInput > div {
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD RAG SYSTEM
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    return llm, retriever


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("### ⚙️ Control Panel")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.total_queries}</div>
                <div class="stat-label">Total Queries</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.messages) // 2}</div>
                <div class="stat-label">Exchanges</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    show_sources = st.toggle("📚 Show Sources", value=True)
    show_chunks  = st.toggle("📄 Show Retrieved Chunks", value=False)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("**📌 Topics covered**")

    topics = [
        "Artificial Intelligence", "Machine Learning",
        "Deep Learning", "NLP",
        "Transformers", "LLMs",
        "RAG", "ChatGPT"
    ]

    pills_html = "".join(f'<span class="topic-pill">{t}</span>' for t in topics)
    st.markdown(f'<div style="margin-top:0.5rem">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="chat-title">🤖 RAG <span class="chat-accent">Chatbot</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="chat-subtitle">Context-aware · Memory-enabled · Powered by Groq + LangChain</div>',
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────
# LOAD COMPONENTS
# ─────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading AI system…"):
    llm, retriever = load_rag_system()


# ─────────────────────────────────────────────────────────────
# STARTER QUESTIONS
# ─────────────────────────────────────────────────────────────
if not st.session_state.messages:

    st.markdown("#### 💡 Try asking…")

    starters = [
        "What is a large language model?",
        "Explain RAG in simple terms",
        "How do transformers work?",
        "Difference between ML and Deep Learning?"
    ]

    cols = st.columns(2)
    for i, question in enumerate(starters):
        with cols[i % 2]:
            if st.button(question, use_container_width=True):
                st.session_state["starter_prompt"] = question
                st.rerun()


# ─────────────────────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown('<div class="label-user">You</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="user-msg">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown('<div class="label-bot">Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="bot-msg">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

        if show_sources and msg.get("sources"):
            source_html = "".join(
                f'<span class="source-tag">📄 {src}</span>'
                for src in msg["sources"]
            )
            st.markdown(
                f'<div style="margin: 0.25rem 0 0.5rem;">{source_html}</div>',
                unsafe_allow_html=True
            )

        if show_chunks and msg.get("chunks"):
            with st.expander("📄 Retrieved context chunks"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk {i + 1} — {chunk['title']}**")
                    st.write(chunk["text"][:500] + "…")


# ─────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about AI, ML, NLP…")

if not user_input:
    user_input = st.session_state.pop("starter_prompt", None)


# ─────────────────────────────────────────────────────────────
# PROCESS USER INPUT
# ─────────────────────────────────────────────────────────────
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    st.markdown('<div class="label-user">You</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="user-msg">{user_input}</div>',
        unsafe_allow_html=True
    )

    with st.spinner("🔍 Retrieving context & generating answer…"):

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            verbose=False
        )

        result = chain.invoke({"question": user_input})

    answer = result["answer"]

    sources = list({
        doc.metadata.get("title", "Unknown")
        for doc in result["source_documents"]
    })

    chunks = [
        {
            "title": doc.metadata.get("title", "Unknown"),
            "text": doc.page_content
        }
        for doc in result["source_documents"]
    ]

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
        "chunks":  chunks
    })

    st.session_state.total_queries += 1

    st.rerun()