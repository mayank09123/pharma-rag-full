#!/bin/bash
# One-command setup for Mac
# Usage: bash setup.sh

set -e

echo ""
echo "================================================"
echo "  Pharma RAG Pipeline — Mac Setup"
echo "================================================"
echo ""

# Check Python
if ! command -v python3.11 &>/dev/null; then
    echo "Python 3.11 not found. Installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install python@3.11
fi

echo "✓ Python: $(python3.11 --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3.11 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate venv
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies (this takes 2-3 minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Open .env and replace with your real OpenAI API key:"
    echo "     OPENAI_API_KEY=sk-your-actual-key-here"
    echo ""
    echo "Get your key at: https://platform.openai.com/api-keys"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your OpenAI API key"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python scripts/ingest.py"
echo "  4. Run: python scripts/query.py"
echo ""
