# setup_repro.ps1
# This script automates the environment setup for ReMindRAG reproduction.

Write-Host "Checking Python version..." -ForegroundColor Cyan
$pythonVersion = py -3.13 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($pythonVersion -ne "3.13.2") {
    Write-Host "Warning: Found Python $pythonVersion, but 3.13.2 is recommended." -ForegroundColor Yellow
} else {
    Write-Host "Python 3.13.2 detected." -ForegroundColor Green
}

if (-Not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -3.13 -m venv venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

Write-Host "Activating venv and installing dependencies..." -ForegroundColor Cyan
# Activation in script needs to handle the scope
. .\venv\Scripts\Activate.ps1

# Upgrade pip
.\venv\Scripts\python.exe -m pip install --upgrade pip

# Step 1: PyTorch with CUDA (large download, separate step)
Write-Host "Installing PyTorch (CUDA 12.6)..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install torch==2.6.0+cu126 torchaudio==2.6.0+cu126 torchvision==0.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Step 2: Core ML/NLP dependencies
Write-Host "Installing ML dependencies..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install transformers==4.49.0 sentence-transformers==3.4.1 chromadb==0.6.3 huggingface_hub==0.29.2 datasets==3.3.2 nltk==3.9.1

# Step 3: API and web dependencies
Write-Host "Installing API/web dependencies..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install openai==1.60.1 Flask==3.1.0 fastapi==0.115.11 streamlit>=1.40.0

# Step 4: Utilities and visualization
Write-Host "Installing utilities..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install networkx==3.4.2 pyvis==0.3.2 tqdm==4.67.1 pandas==2.2.3 numpy==2.2.2 matplotlib==3.10.1 rich==13.9.4 python-docx==1.1.2

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "To run the app: .\venv\Scripts\streamlit.exe run streamlit_app.py" -ForegroundColor Green
