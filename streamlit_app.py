import streamlit as st
import os
import tempfile
import json
import logging
from typing import List, Dict
import streamlit.components.v1 as components

# Apply some basic configurations to be wide and title.
st.set_page_config(page_title="ReMindRAG Standalone UI", page_icon="🧠", layout="wide")

# =====================================================================
# RAG Initialization
# =====================================================================
@st.cache_resource(show_spinner="Loading ReMindRAG components...")
def initialize_rag(api_key: str, embedding_model="sentence-transformers/all-MiniLM-L6-v2", llm_model="gpt-4o-mini", hf_token: str = ""):
    """Initializes the ReMindRAG pipeline components."""
    from ReMindRag.llms import OpenaiAgent
    from ReMindRag.embeddings import HgEmbedding
    from ReMindRag.chunking import NaiveChunker
    from ReMindRag import ReMindRag
    from transformers import AutoTokenizer

    model_cache_dir = "./model_cache"
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Base URL for OpenAI. Could be exposed via UI if needed.
    base_url = "https://api.openai.com/v1"
    
    # Embeddings and Chunking
    st.sidebar.text("Loading Embeddings & Chunker...")
    embedding = HgEmbedding(embedding_model, model_cache_dir)
    chunker = NaiveChunker(embedding_model, model_cache_dir, max_token_length=200)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model, cache_dir=model_cache_dir)
    
    # Agents
    st.sidebar.text("Initializing LLM Agents...")
    chunk_agent = OpenaiAgent(base_url, api_key, llm_model)
    kg_agent = OpenaiAgent(base_url, api_key, llm_model)
    generate_agent = OpenaiAgent(base_url, api_key, llm_model)
    
    # Pipeline
    rag_instance = ReMindRag(
        logger_level=logging.WARNING, # Reduced noise
        log_path="logs/streamlit_rag.log",
        chunk_agent=chunk_agent,
        kg_agent=kg_agent,
        generate_agent=generate_agent,
        embedding=embedding,
        chunker=chunker,
        tokenizer=tokenizer,
        database_description="Standalone Streamlit Upload DB",
        save_dir="./Rag_Cache"
    )
    return rag_instance

# =====================================================================
# State Management
# =====================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_instance" not in st.session_state:
    st.session_state.rag_instance = None
if "api_key" not in st.session_state:
    # Try to load from default api_key.json if available
    default_key = ""
    try:
        with open("api_key.json", "r") as f:
            keys = json.load(f)
            if isinstance(keys, list) and len(keys) > 0:
                default_key = keys[0].get("api_key", "")
    except Exception:
        pass
    st.session_state.api_key = default_key

# =====================================================================
# Sidebar Layout
# =====================================================================
with st.sidebar:
    st.title("🧠 ReMindRAG UI")
    st.write("Standalone Streamlit Interface")
    st.markdown("---")
    
    st.header("Configuration")
    st.session_state.api_key = st.text_input("OpenAI API Key:", value=st.session_state.api_key, type="password")
    
    # Model Configurations
    llm_choice = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"], index=0)
    emb_choice = st.selectbox("Embedding Model", [
        "sentence-transformers/all-MiniLM-L6-v2", 
        "nomic-ai/nomic-embed-text-v2-moe"
    ], index=0)
    
    hf_token = ""
    if "nomic" in emb_choice:
        hf_token = st.text_input("HuggingFace Token (for Nomic):", type="password")
        st.caption("Required for gated HuggingFace models.")
        
    if st.button("Initialize ReMindRAG"):
        if not st.session_state.api_key:
            st.error("Please provide an OpenAI API Key!")
        else:
            try:
                st.session_state.rag_instance = initialize_rag(
                    api_key=st.session_state.api_key, 
                    embedding_model=emb_choice, 
                    llm_model=llm_choice,
                    hf_token=hf_token
                )
                st.success("Successfully initialized RAG pipeline!")
            except Exception as e:
                st.error(f"Error initializing: {str(e)}")
                
    st.markdown("---")
    
    # Document Upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a text document to index:", type=["txt", "md"])
    if uploaded_file is not None and st.session_state.rag_instance is not None:
        if st.button("Index Document"):
            with st.spinner("Processing document and extracting knowledge graph..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    # Index
                    st.session_state.rag_instance.load_file(tmp_path, language="en")
                    st.success(f"Indexed {uploaded_file.name} successfully!")
                    
                    # Cleanup
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error indexing: {str(e)}")

# =====================================================================
# Main Chat Layout
# =====================================================================
st.title("ReMindRAG Query Interface")
st.markdown("Ask questions against your indexed documents. ReMindRAG will traverse the Knowledge Graph to find the answer.")

if not st.session_state.rag_instance:
    st.info("👈 Please enter your API key and click 'Initialize ReMindRAG' in the sidebar to begin.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "metadata" in msg:
                with st.expander("Show Retrieval Details"):
                    display_meta = {k: v for k, v in msg["metadata"].items() if k != "graph_html"}
                    st.json(display_meta)
                    if "graph_html" in msg["metadata"]:
                        st.subheader("Knowledge Graph Traversal")
                        components.html(msg["metadata"]["graph_html"], height=500, scrolling=True)
                    
    # Chat Input
    query = st.chat_input("Ask a question...")
    if query:
        # Add user msg
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
            
        # Agent response
        with st.chat_message("assistant"):
            with st.spinner("Traversing Knowledge Graph..."):
                try:
                    response, chunks, edges = st.session_state.rag_instance.generate_response(query, force_do_rag=True)
                    
                    st.write(response)
                    
                    chunks_count = sum(len(c) for c in chunks) if isinstance(chunks, list) else (len(chunks) if chunks else 0)
                    
                    st.session_state.rag_instance.refresh_kg()
                    graph_html_str = ""
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                            tmp_html_path = tmp_html.name
                        st.session_state.rag_instance.kg.save_as_pyvis_for_quick_query(tmp_html_path, query)
                        with open(tmp_html_path, "r", encoding="utf-8") as f:
                            graph_html_str = f.read()
                        os.unlink(tmp_html_path)
                    except Exception as e:
                        logging.error(f"Failed to generate Knowledge Graph visual: {e}")

                    meta: Dict[str, str | int] = {
                        "Chunks Retrieved": chunks_count,
                        "Graph Edges Traversed": len(edges)
                    }
                    if graph_html_str:
                        meta["graph_html"] = graph_html_str

                    with st.expander("Show Retrieval Details"):
                        # Extract non-HTML metadata for display
                        display_meta = {k: v for k, v in meta.items() if k != "graph_html"}
                        st.json(display_meta)
                        if graph_html_str:
                            st.subheader("Knowledge Graph Traversal")
                            components.html(graph_html_str, height=500, scrolling=True)
                        
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "metadata": meta
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
