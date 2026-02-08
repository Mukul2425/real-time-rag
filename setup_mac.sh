#!/bin/bash
# Mac Quick Setup Script for Multimodal EV RAG Assistant
# This script helps Mac users quickly set up the project

set -e  # Exit on error

echo "🍎 Multimodal EV RAG - Mac Quick Setup"
echo "======================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS"
    echo "It may not work correctly on other systems"
    echo ""
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Homebrew
echo "📦 Checking Homebrew..."
if ! command_exists brew; then
    echo "❌ Homebrew not found"
    echo "Please install Homebrew first:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
echo "✅ Homebrew installed"

# Check Python 3
echo ""
echo "🐍 Checking Python..."
if ! command_exists python3; then
    echo "❌ Python 3 not found"
    echo "Installing Python 3..."
    brew install python
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python $PYTHON_VERSION installed"
fi

# Check Docker
echo ""
echo "🐳 Checking Docker..."
if ! command_exists docker; then
    echo "❌ Docker not found"
    echo "Please install Docker Desktop for Mac:"
    echo "https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✅ Docker installed"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "⚠️  Docker is installed but not running"
    echo "Please start Docker Desktop and run this script again"
    exit 1
fi
echo "✅ Docker is running"

# Create virtual environment
echo ""
echo "🔧 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

# Check for .env file
echo ""
echo "🔑 Checking environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from .env.example"
        echo ""
        echo "⚠️  IMPORTANT: You need to add your API keys to .env file"
        echo "Edit .env and add:"
        echo "  - NEWS_API_KEY"
        echo "  - PINECONE_API_KEY"
        echo "  - OPENROUTER_API_KEY"
        echo ""
        echo "You can edit the file with:"
        echo "  nano .env"
    else
        echo "❌ .env.example not found"
        echo "Please create a .env file manually"
    fi
else
    echo "✅ .env file exists"
fi

# Validate setup
echo ""
echo "🔍 Validating setup..."
python validate_setup.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Make sure you've added your API keys to .env file"
echo "2. Setup Pinecone index:"
echo "   python setup_multimodal_index.py"
echo ""
echo "3. Start Kafka:"
echo "   docker compose up -d"
echo ""
echo "4. Run the data pipeline (in separate terminals):"
echo "   Terminal 1: python ingestion_scripts/producer.py"
echo "   Terminal 2: python data_processor/consumer_and_embedder.py"
echo ""
echo "5. Start the Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "💡 Tip: Use 'make help' to see available shortcuts"
echo ""
echo "📖 For more details, see SETUP_MAC.md"
echo ""
