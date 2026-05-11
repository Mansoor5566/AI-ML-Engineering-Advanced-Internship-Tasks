🤖 Context-Aware RAG Chatbot
A simple Retrieval-Augmented Generation (RAG) chatbot built with LangChain, FAISS, HuggingFace embeddings, and Groq LLM. It answers questions based on a custom knowledge base with conversational memory.

🚀 Live Demo
https://contextawarechatboot.streamlit.app/

⚙️ Tech Stack
Streamlit
LangChain
FAISS
HuggingFace Embeddings
Groq (LLaMA 3)
PyTorch

📂 Features
Semantic search with FAISS
Context-aware responses (RAG)
Chat memory support
Fast AI responses via Groq
Streamlit UI

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

🔐 API Key

Add in .streamlit/secrets.toml:
GROQ_API_KEY="your_key"
