# 🍎 Mac Setup Guide - Multimodal EV RAG Assistant

This guide is specifically for Mac users who are migrating from Windows or setting up this project for the first time on macOS.

> **🔄 Windows Users**: If you're migrating from Windows, also check out [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for a detailed comparison and migration steps.

## 📋 Prerequisites for Mac

### 1. Install Homebrew (if not already installed)
Homebrew is the package manager for macOS. Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, follow the instructions to add Homebrew to your PATH.

### 2. Install Python 3.8+
Check if Python 3 is installed:
```bash
python3 --version
```

If not installed or version is too old, install via Homebrew:
```bash
brew install python@3.11
```

### 3. Install Docker Desktop for Mac
1. Download from [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Install the `.dmg` file
3. Open Docker Desktop and ensure it's running
4. Docker Compose is included with Docker Desktop

Verify installation:
```bash
docker --version
docker compose version
```

### 4. Install Git (if not already installed)
```bash
brew install git
```

## 🚀 Quick Setup (Recommended for Mac)

### Option 1: Automated Setup Script (Easiest)

We've created a setup script that automates most of the setup process:

```bash
# Clone the repository
git clone https://github.com/Mukul2425/real-time-rag.git
cd real-time-rag

# Run the Mac setup script
chmod +x setup_mac.sh
./setup_mac.sh
```

The script will:
- Check all prerequisites
- Create a virtual environment
- Install all Python dependencies
- Create a `.env` file from the example
- Validate your setup

### Option 2: Manual Setup

Follow these steps if you prefer manual control:

### Step 1: Clone the Repository
```bash
git clone https://github.com/Mukul2425/real-time-rag.git
cd real-time-rag
```

### Step 2: Create Python Virtual Environment
On Mac, always use `python3` explicitly:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Mac/Linux)
source venv/bin/activate
```

**Important**: Your terminal prompt should now show `(venv)` to indicate the virtual environment is active.

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Mac-specific note**: If you encounter issues with PyTorch or other packages:
```bash
# For Apple Silicon (M1/M2/M3) Macs
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor (nano, vim, or VSCode)
nano .env
```

Add your API keys to the `.env` file:
```env
NEWS_API_KEY=your_news_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Save the file:
- In `nano`: Press `Ctrl+O` to save, then `Ctrl+X` to exit
- In `vim`: Press `ESC`, then type `:wq` and press Enter

### Step 5: Validate Your Setup
```bash
python validate_setup.py
```

This will check if all dependencies and configurations are correct.

### Step 6: Setup Pinecone Index
```bash
python setup_multimodal_index.py
```

### Step 7: Start Kafka
Ensure Docker Desktop is running, then:
```bash
docker compose up -d

# Verify Kafka is running
docker compose ps
```

### Step 8: Run the Data Pipeline
Open **three separate Terminal windows/tabs**:

**Terminal 1 - Producer** (fetches news):
```bash
cd real-time-rag
source venv/bin/activate
python ingestion_scripts/producer.py
```

**Terminal 2 - Consumer** (processes and embeds data):
```bash
cd real-time-rag
source venv/bin/activate
python data_processor/consumer_and_embedder.py
```

**Terminal 3 - Streamlit App**:
```bash
cd real-time-rag
source venv/bin/activate
streamlit run app.py
```

The app should automatically open in your default browser at `http://localhost:8501`

## 🛠️ Using Makefile on Mac

The project includes a Makefile that works great on Mac (since macOS is Unix-based):

```bash
# See all available commands
make help

# Install dependencies
make setup

# Validate environment
make validate

# Setup Pinecone
make setup-pinecone

# Start Kafka
make start-kafka

# In separate terminals:
make start-producer  # Terminal 1
make start-consumer  # Terminal 2
make start-app       # Terminal 3
```

## 🔧 Mac-Specific Troubleshooting

### Issue 1: Command not found - python
**Solution**: On Mac, use `python3` instead of `python`:
```bash
# Instead of: python script.py
python3 script.py

# Or create an alias (add to ~/.zshrc or ~/.bash_profile)
alias python=python3
```

### Issue 2: pip not found
**Solution**: Use `pip3`:
```bash
pip3 install -r requirements.txt
```

### Issue 3: Permission Denied when Installing Packages
**Solution**: Never use `sudo` with pip. Make sure you're in the virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 4: Docker Cannot Connect
**Solutions**:
1. Ensure Docker Desktop is running (check menu bar)
2. Restart Docker Desktop
3. Check Docker settings → Resources → ensure Docker has enough memory (4GB+)

### Issue 5: Port Already in Use
**For Kafka (port 9092)**:
```bash
# Find process using the port
lsof -i :9092

# Kill the process (replace PID with actual process ID)
kill -9 PID
```

**For Streamlit (port 8501)**:
```bash
# Find and kill
lsof -i :8501
kill -9 PID
```

### Issue 6: PyTorch/CUDA Issues on Apple Silicon
**Solution**: Apple Silicon (M1/M2/M3) doesn't support CUDA. The code will automatically fall back to CPU:
```bash
# The app detects this automatically
# For faster performance, consider using Metal Performance Shaders (MPS):
# This is handled automatically in newer PyTorch versions
```

### Issue 7: SSL Certificate Error
**Solution**: Update certificates:
```bash
# Install certificates
/Applications/Python\ 3.*/Install\ Certificates.command

# Or update certifi
pip install --upgrade certifi
```

### Issue 8: zsh: command not found: make
**Solution**: Install make (usually included with Xcode Command Line Tools):
```bash
xcode-select --install
```

### Issue 9: Virtual Environment Activation Issues
**For Zsh (default on modern macOS)**:
```bash
source venv/bin/activate
```

**If activation seems to fail**, check your shell:
```bash
echo $SHELL
# If it shows /bin/zsh, you're using zsh (correct)
# If it shows /bin/bash, follow bash instructions
```

## 🍎 Mac Performance Tips

### 1. Use Activity Monitor
Open Activity Monitor (Cmd+Space, type "Activity Monitor") to check:
- CPU usage
- Memory pressure
- Disk activity

### 2. Optimize for Apple Silicon (M1/M2/M3)
The project runs natively on Apple Silicon. PyTorch will use CPU optimizations automatically.

### 3. Increase Docker Resources
Docker Desktop → Settings → Resources:
- **CPUs**: Allocate 4+ cores
- **Memory**: Allocate 6-8 GB
- **Disk**: 20+ GB

### 4. Use iTerm2 or Terminal Multiplexer
Instead of multiple Terminal windows, consider:
- **iTerm2**: Better terminal with split panes
- **tmux**: Terminal multiplexer for managing multiple sessions

```bash
# Install iTerm2
brew install --cask iterm2

# Or install tmux
brew install tmux
```

## 📦 Mac-Specific Dependencies

Some Python packages might need additional tools on Mac:

### For Image Processing Issues:
```bash
brew install libjpeg libpng
```

### For Compilation Issues:
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

## 🔄 Migrating from Windows

### Key Differences from Windows:

| Aspect | Windows | Mac |
|--------|---------|-----|
| Python command | `python` | `python3` |
| Pip command | `pip` | `pip3` |
| Path separator | `\` | `/` |
| Venv activation | `venv\Scripts\activate` | `source venv/bin/activate` |
| Line endings | CRLF (`\r\n`) | LF (`\n`) |
| Make command | May need WSL or GnuWin32 | Built-in (with Xcode tools) |

### Converting Your Windows Setup:

1. **Environment Variables**: 
   - Windows: Set in System Properties or `.env`
   - Mac: Use `.env` file (same as before)

2. **Path Issues**:
   - The code uses Python's `os.path` which handles paths correctly on both systems
   - No changes needed in the code

3. **Docker**:
   - Windows: Docker Desktop for Windows (may use WSL2)
   - Mac: Docker Desktop for Mac (native)
   - Commands are the same

4. **Line Endings**:
   If you cloned on Windows and moved files:
   ```bash
   # Fix line endings for shell scripts
   find . -name "*.sh" -exec dos2unix {} \;
   
   # Or install dos2unix first
   brew install dos2unix
   ```

## 🎯 Quick Start Checklist

- [ ] Install Homebrew
- [ ] Install Python 3.8+ (`brew install python`)
- [ ] Install Docker Desktop and start it
- [ ] Clone the repository
- [ ] Create virtual environment (`python3 -m venv venv`)
- [ ] Activate virtual environment (`source venv/bin/activate`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Copy `.env.example` to `.env` and add API keys
- [ ] Run validation (`python validate_setup.py`)
- [ ] Setup Pinecone (`python setup_multimodal_index.py`)
- [ ] Start Kafka (`docker compose up -d`)
- [ ] Run producer, consumer, and app in separate terminals

## 📚 Additional Mac Resources

- [Homebrew Documentation](https://docs.brew.sh/)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
- [Python on Mac](https://docs.python.org/3/using/mac.html)
- [Terminal User Guide](https://support.apple.com/guide/terminal/welcome/mac)

## 🆘 Getting Help

If you encounter issues:

1. Check the main [README.md](README.md) for general troubleshooting
2. Review this Mac-specific guide
3. Run `python validate_setup.py` to check your configuration
4. Check [Docker Desktop logs](https://docs.docker.com/desktop/troubleshoot/)
5. Open an issue on GitHub with:
   - macOS version (`sw_vers`)
   - Python version (`python3 --version`)
   - Error message
   - Steps to reproduce

## ✅ Verification

After setup, verify everything works:

```bash
# 1. Check Python
python3 --version  # Should be 3.8+

# 2. Check virtual environment
source venv/bin/activate
which python  # Should point to venv/bin/python

# 3. Check Docker
docker --version
docker compose ps  # Should show Kafka running

# 4. Check imports
python -c "import streamlit, torch, transformers, pinecone; print('All imports OK')"

# 5. Run validation
python validate_setup.py
```

---

**Welcome to Mac! You should now be ready to run the Multimodal EV RAG Assistant on your Mac.** 🎉
