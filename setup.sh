#!/bin/bash
# CivicCam Setup Script
# Run this script to set up the project on a new machine

set -e

echo "🚔 CivicCam Setup"
echo "=================="
echo ""

# Check Python version
echo "📌 Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "   Found Python $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   Created venv/"
else
    echo "   venv/ already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "   Dependencies installed"

# Create .env if it doesn't exist
echo ""
echo "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || cat > .env << EOF
# Telegram Configuration
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
    echo "   Created .env file"
else
    echo "   .env already exists"
fi

# Create incidents directory
mkdir -p incidents/images
echo "   Created incidents/ directory"

# Check if model exists
echo ""
echo "🤖 Checking model..."
if [ -f "models/civiccam_best.pt" ]; then
    echo "   ✅ Model found: models/civiccam_best.pt"
else
    echo "   ⚠️  No model found. Please add your trained model to models/civiccam_best.pt"
fi

# Done
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Configure Telegram (optional):"
echo "     Edit .env with your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
echo ""
echo "  2. Run the app:"
echo "     source venv/bin/activate"
echo "     streamlit run app.py"
echo ""
echo "  3. Open in browser:"
echo "     http://localhost:8501"
echo ""
