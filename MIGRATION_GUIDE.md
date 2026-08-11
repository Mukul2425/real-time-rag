# 🔄 Windows to Mac Migration Guide

This guide helps you migrate the Multimodal EV RAG Assistant from Windows to macOS.

## 📋 Pre-Migration Checklist

Before migrating, make sure you have:

- ✅ All your API keys saved (NEWS_API_KEY, PINECONE_API_KEY, OPENROUTER_API_KEY)
- ✅ Any custom configurations or modifications documented
- ✅ Backup of any local data (if applicable)

## 🎯 Quick Migration Steps

### 1. Export Your Configuration (Windows)

On your Windows machine, save your `.env` file:
```powershell
# On Windows, copy your .env file contents
type .env
```

Save this information securely (password manager, encrypted note, etc.)

### 2. Clean Checkout on Mac

On your Mac, start fresh:
```bash
# Clone the repository
git clone https://github.com/Mukul2425/real-time-rag.git
cd real-time-rag

# Run the Mac setup script
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Restore Configuration

Create your `.env` file with the saved API keys:
```bash
nano .env
```

Paste your API keys:
```env
NEWS_API_KEY=your_news_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## 🔧 Key Differences

### Command Differences

| Task | Windows | Mac |
|------|---------|-----|
| Python | `python` or `py` | `python3` |
| Pip | `pip` | `pip3` or `pip` (in venv) |
| Activate venv | `venv\Scripts\activate` | `source venv/bin/activate` |
| Path separator | `\` (backslash) | `/` (forward slash) |
| Clear screen | `cls` | `clear` |
| List files | `dir` | `ls` |
| Make utility | Requires installation | Built-in (via Xcode tools) |

### Terminal/Shell Differences

**Windows**:
- Command Prompt (cmd.exe)
- PowerShell
- Git Bash (if installed)

**Mac**:
- Terminal (default: zsh on modern macOS)
- iTerm2 (popular alternative)
- Built-in Unix shell

### File System Differences

**Windows**:
- Drive letters (C:\, D:\)
- Case-insensitive file system (usually)
- Backslashes in paths: `C:\Users\username\project`

**Mac**:
- Single root filesystem (/)
- Case-sensitive file system (APFS)
- Forward slashes in paths: `/Users/username/project`

## 🐳 Docker Differences

### Windows (with WSL2)
- Docker Desktop uses Windows Subsystem for Linux 2
- May have performance overhead
- Requires WSL2 features enabled
- File system access can be slower

### Mac
- Docker Desktop runs natively on macOS
- Better performance (especially on Apple Silicon)
- No WSL needed
- Direct file system access

### Commands (Same on Both)
```bash
# These work the same on Windows and Mac
docker --version
docker compose up -d
docker compose ps
docker compose down
```

## 🛠️ Common Migration Issues

### Issue 1: Scripts with Windows Line Endings

**Problem**: Shell scripts have CRLF line endings from Windows

**Solution**:
```bash
# Install dos2unix if needed
brew install dos2unix

# Fix line endings
dos2unix setup_mac.sh
```

Or configure Git to handle this:
```bash
git config --global core.autocrlf input
```

### Issue 2: Hardcoded Windows Paths

**Problem**: Code has paths like `C:\path\to\file`

**Solution**: This project uses `os.path` which is cross-platform. No changes needed.

### Issue 3: Python Command Not Found

**Problem**: Used to typing `python` on Windows

**Solution**: 
```bash
# Create an alias (add to ~/.zshrc)
alias python=python3
alias pip=pip3

# Reload shell configuration
source ~/.zshrc
```

### Issue 4: Virtual Environment Confusion

**Problem**: Different activation methods

**Windows**:
```powershell
venv\Scripts\activate
```

**Mac**:
```bash
source venv/bin/activate
```

### Issue 5: Package Installation Differences

Some packages may have different dependencies on Mac:

```bash
# If you encounter build errors, you may need:
brew install libjpeg libpng
xcode-select --install

# For PyTorch on Apple Silicon
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 🍎 Mac-Specific Advantages

### 1. Native Unix Environment
- Better command-line tools
- Unix utilities built-in
- No need for WSL

### 2. Apple Silicon (M1/M2/M3)
- Better power efficiency
- Fast CPU performance
- Good memory management
- Native ARM support

### 3. Better Docker Integration
- Native Docker Desktop
- Faster file I/O
- Lower overhead

## 📝 Recommended Mac Setup

### Essential Tools

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install useful tools
brew install git
brew install python
brew install wget
brew install tree
brew install htop
```

### Optional but Recommended

```bash
# iTerm2 - Better terminal
brew install --cask iterm2

# Visual Studio Code
brew install --cask visual-studio-code

# Rectangle - Window management
brew install --cask rectangle

# Docker Desktop
brew install --cask docker
```

### Development Environment

```bash
# Oh My Zsh - Better shell experience
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Syntax highlighting
brew install zsh-syntax-highlighting

# Auto-suggestions
brew install zsh-autosuggestions
```

## ✅ Migration Verification

After migration, verify everything works:

```bash
# 1. Check Python
python3 --version

# 2. Check virtual environment
source venv/bin/activate
which python  # Should show venv path

# 3. Verify packages
pip list

# 4. Check Docker
docker --version
docker compose version

# 5. Run validation
python validate_setup.py

# 6. Test import of key packages
python -c "import streamlit, torch, transformers, pinecone; print('✅ All imports successful')"
```

## 🎓 Learning Resources

### Mac Terminal Basics
- [Terminal User Guide](https://support.apple.com/guide/terminal/welcome/mac)
- [Learn the Command Line](https://www.codecademy.com/learn/learn-the-command-line)

### Python on Mac
- [Python Setup on Mac](https://docs.python-guide.org/starting/install3/osx/)
- [Real Python - Python on Mac](https://realpython.com/installing-python-on-mac/)

### Docker on Mac
- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Mac Specifics](https://docs.docker.com/desktop/mac/)

## 🚀 Post-Migration Tips

### 1. Update Your Workflow

Consider using:
- `make` commands instead of manual commands
- Shell aliases for frequently used commands
- iTerm2 with split panes for multiple terminals

### 2. Keyboard Shortcuts

Mac shortcuts are different:
- Copy: `Cmd+C` (not Ctrl+C)
- Paste: `Cmd+V` (not Ctrl+V)
- Terminal interrupt: `Ctrl+C` (same as Windows)

### 3. Package Management

Use Homebrew for system tools:
```bash
# Search for packages
brew search <package>

# Install packages
brew install <package>

# Update all packages
brew update && brew upgrade
```

### 4. Performance Monitoring

Use Activity Monitor (Mac's Task Manager):
```bash
# Open with command
open -a "Activity Monitor"

# Or press: Cmd+Space, type "Activity Monitor"
```

## 📞 Getting Help

If you encounter migration issues:

1. Check [SETUP_MAC.md](SETUP_MAC.md) for Mac-specific setup
2. Review this migration guide
3. Run `python validate_setup.py`
4. Check the main [README.md](README.md) troubleshooting section
5. Open a GitHub issue with:
   - "Migrated from Windows to Mac" in the title
   - macOS version: `sw_vers`
   - Error details
   - Steps already tried

## 🎉 Welcome to Mac!

Congratulations on your migration! Mac offers a great development environment for Python and Docker-based projects. Take some time to explore the native Unix tools and you'll find many workflows are actually simpler than on Windows.

### Quick Reference Card

Save this for quick access:

```bash
# Daily commands
source venv/bin/activate          # Activate Python environment
docker compose up -d              # Start Kafka
make start-producer               # Start news producer
make start-consumer               # Start data processor
make start-app                    # Start Streamlit app

# Troubleshooting
python validate_setup.py          # Check setup
docker compose ps                 # Check containers
docker compose logs kafka         # View Kafka logs
lsof -i :8501                    # Check port usage

# Maintenance
brew update && brew upgrade       # Update Homebrew packages
pip list --outdated              # Check Python packages
docker compose down              # Stop Kafka
make clean                       # Clean cache files
```

---

**Happy coding on your Mac!** 🍎✨
