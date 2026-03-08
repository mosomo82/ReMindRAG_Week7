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
python -m pip install --upgrade pip

# Install core dependencies with pinned exact versions
Write-Host "Installing core dependencies..." -ForegroundColor Cyan
pip install torch==2.6.0+cu126 transformers==4.49.0 sentence-transformers==3.4.1 chromadb==0.6.3 Flask==3.1.0 fastapi==0.115.11 openai==1.60.1 networkx==3.4.2 pyvis==0.3.2 datasets==3.3.2 huggingface_hub==0.29.2 tqdm==4.67.1 pandas==2.2.3 numpy==2.2.2 matplotlib==3.10.1 rich==13.9.4 --extra-index-url https://download.pytorch.org/whl/cu126

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1"
