import sys
import os
import json
from datetime import datetime
from typing import List, Dict

# Add the project root to sys.path
sys.path.append(os.getcwd())

from ReMindRag.llms import AgentBase
from ReMindRag.embeddings import HgEmbedding
from ReMindRag.chunking import NaiveChunker
from ReMindRag import ReMindRag
from transformers import AutoTokenizer

# Mock OpenaiAgent to avoid needing a real API key for testing the pipeline logic
class MockAgent(AgentBase):
    def __init__(self, name="MockAgent"):
        self.name = name

    def generate_response(self, system_prompt: str, chat_history: List[Dict[str, str]]) -> str:
        # Check the prompt to see what it's asking for
        if "extract" in system_prompt.lower() or "entity" in system_prompt.lower():
            # Mock entity extraction response in JSON format within triple backticks
            return '```json\n{"entities": [{"name": "Paladin", "type": "Class"}], "relations": []}\n```'
        
        if "decide whether we have sufficient information" in system_prompt.lower().replace("\n", " "):
             return '```cot-ans\nyes\n```'

        # Default QA response
        return "```cot-ans\nThis is a mocked response from ReMindRAG for testing purposes. Based on the document, a level 20 paladin gains a Sacred Oath feature.\n```"

# Step 1: Configuration
model_cache_dir = "./repro_model_cache"
if not os.path.exists(model_cache_dir):
    os.makedirs(model_cache_dir)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/repro_log_{timestamp}.log"
if not os.path.exists("logs"):
    os.makedirs("logs")

# Step 2: Load Base Components (using small models for reproduction)
print("Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding = HgEmbedding("sentence-transformers/all-MiniLM-L6-v2", model_cache_dir)

print("Loading Chunker and Tokenizer...")
chunker = NaiveChunker("sentence-transformers/all-MiniLM-L6-v2", model_cache_dir, max_token_length=200)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=model_cache_dir)

mock_agent = MockAgent()

# Step 3: Create ReMindRag Instance
print("Initializing ReMindRag instance...")
rag_instance = ReMindRag(
    logger_level=10,
    log_path=log_path,
    chunk_agent=mock_agent, 
    kg_agent=mock_agent,
    generate_agent=mock_agent, 
    embedding=embedding,
    chunker=chunker,
    tokenizer=tokenizer,
    database_description="DnD Player Handbook---Paladin Reproduction Test"
)

# Step 4: Load Content
print("Loading example data...")
example_data_path = os.path.join("example", "example_data.txt")
rag_instance.load_file(example_data_path, language="en")

# Step 5: Ask A Question
query = "What does a level 20 paladin gain?"
print(f"Querying: {query}")
response, chunks, edges = rag_instance.generate_response(query, force_do_rag=True)

print(f"\nResponse: {response}")
print(f"Retrieved {len(chunks)} chunks and {len(edges)} edges.")

# Validation
if len(chunks) > 0:
    print("\nSUCCESS: RAG pipeline successfully retrieved chunks from the database.")
else:
    print("\nFAILURE: RAG pipeline failed to retrieve chunks.")
