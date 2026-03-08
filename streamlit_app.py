import streamlit as st
import os
import tempfile
import json
import logging
from typing import List, Dict

# Apply some basic configurations to be wide and title.
st.set_page_config(page_title="ReMindRAG Standalone UI", page_icon="🧠", layout="wide")

# =====================================================================
# RAG Initialization
# =====================================================================
@st.cache_resource(show_spinner="Loading ReMindRAG components...")
def initialize_rag(api_key: str, llm_provider: str = "OpenAI", embedding_model="sentence-transformers/all-MiniLM-L6-v2", llm_model="gpt-4o-mini", hf_token: str = ""):
    """Initializes the ReMindRAG pipeline components."""
    from ReMindRag.llms import OpenaiAgent, AnthropicAgent, GeminiAgent
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
    st.sidebar.text(f"Initializing {llm_provider} Agents...")
    if llm_provider == "OpenAI":
        chunk_agent = OpenaiAgent(base_url, api_key, llm_model)
        kg_agent = OpenaiAgent(base_url, api_key, llm_model)
        generate_agent = OpenaiAgent(base_url, api_key, llm_model)
    elif llm_provider == "Anthropic":
        chunk_agent = AnthropicAgent(api_key, llm_model)
        kg_agent = AnthropicAgent(api_key, llm_model)
        generate_agent = AnthropicAgent(api_key, llm_model)
    elif llm_provider == "Gemini":
        chunk_agent = GeminiAgent(api_key, llm_model)
        kg_agent = GeminiAgent(api_key, llm_model)
        generate_agent = GeminiAgent(api_key, llm_model)
    else:
        raise ValueError(f"Unsupported LLM Provider: {llm_provider}")
    
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
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
    try:
        with open("api_key.json", "r") as f:
            st.session_state.api_keys = json.load(f)
    except Exception as e:
        st.warning(f"Could not load api_key.json: {e}")

# =====================================================================
# Sidebar Layout
# =====================================================================
with st.sidebar:
    st.title("🧠 ReMindRAG UI")
    st.write("Standalone Streamlit Interface")
    st.markdown("---")
    
    st.header("Configuration")
    
    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Gemini"], index=2) # Default to Gemini per user request
    
    # Retrieve the key securely from the loaded config rather than asking the user
    provider_config = st.session_state.api_keys.get(llm_provider, {})
    current_api_key = provider_config.get("api_key", "")
    
    # Model Configurations based on Provider
    if llm_provider == "OpenAI":
        default_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
    elif llm_provider == "Anthropic":
        default_models = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"]
    else: # Gemini
        default_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
        
    llm_choice = st.selectbox("LLM Model", default_models, index=0)
    
    emb_choice = st.selectbox("Embedding Model (Local HF)", [
        "sentence-transformers/all-MiniLM-L6-v2", 
        "nomic-ai/nomic-embed-text-v2-moe"
    ], index=0)
    
    hf_token = ""
    if "nomic" in emb_choice:
        hf_token = st.text_input("HuggingFace Token (for Nomic):", type="password")
        st.caption("Required for gated HuggingFace models.")
        
    if st.button("Initialize ReMindRAG"):
        if not current_api_key or current_api_key.startswith("YOUR_"):
            st.error(f"Please configure a valid {llm_provider} API Key in api_key.json!")
        else:
            try:
                st.session_state.rag_instance = initialize_rag(
                    api_key=current_api_key, 
                    llm_provider=llm_provider,
                    embedding_model=emb_choice, 
                    llm_model=llm_choice,
                    hf_token=hf_token
                )
                st.success(f"Successfully initialized RAG pipeline with {llm_provider}!")
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
    st.info("👈 Please select a provider and click 'Initialize ReMindRAG' in the sidebar to begin. (Ensure your keys are in `api_key.json`)")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "metadata" in msg:
                with st.expander("Show Retrieval Details"):
                    st.json(msg["metadata"])
                    
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
                    
                    meta = {
                        "Chunks Retrieved": chunks_count,
                        "Graph Edges Traversed": len(edges)
                    }
                    with st.expander("Show Retrieval Details"):
                        st.json(meta)
                        
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "metadata": meta
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
