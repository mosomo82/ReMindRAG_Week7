# ============================================================
# Stage 1: builder — install all Python dependencies
# To pin to a specific digest for production:
#   docker pull python:3.13-slim
#   docker inspect python:3.13-slim --format '{{index .RepoDigests 0}}'
# Then replace the tag below with python:3.13-slim@sha256:<digest>
# ============================================================
FROM python:3.13-slim AS builder

# Install only what is needed to compile packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifest first (layer-cache friendly)
COPY requirements.txt .

# Filter out Windows-only / Conda-only packages that break Linux pip installs,
# then install everything. PyTorch CUDA wheels require the extra index URL.
RUN pip install --no-cache-dir --upgrade pip==25.0.1 && \
    grep -v -E "(pywin32|pywinpty|win_inet_pton|mkl|conda)" requirements.txt \
    > requirements_filtered.txt && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r requirements_filtered.txt

# ============================================================
# Stage 2: runtime — lean final image without build tools
# ============================================================
FROM python:3.13-slim AS runtime

# Runtime-only system libraries (no compiler toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-download NLTK data required for sentence tokenization (punkt_tab)
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

WORKDIR /app

# Copy project source
COPY . .

# Ensure critical runtime directories exist so first-run makedirs calls succeed
RUN mkdir -p /app/Rag_Cache/chroma_data \
    /app/Rag_Cache/model_cache \
    /app/eval/database \
    /app/eval/dataset_cache \
    /app/example/logs \
    /app/ReMindRag/webui/temp \
    /app/logs

# Environment: Hugging Face will read HF_TOKEN from runtime env
ENV HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Default entrypoint: run the Streamlit UI
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
