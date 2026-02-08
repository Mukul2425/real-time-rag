# 📚 Documentation Index

Quick reference guide to all documentation in this repository.

## 🚀 Getting Started

### For All Users
- **[README.md](README.md)** - Main project documentation, features, and general setup
- **[.env.example](.env.example)** - Template for environment variables

### For Mac Users
- **[SETUP_MAC.md](SETUP_MAC.md)** - Comprehensive Mac setup guide ⭐ Start here if you're on Mac
- **[setup_mac.sh](setup_mac.sh)** - Automated setup script for Mac

### For Windows → Mac Migration
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Complete Windows to Mac migration guide

## 🛠️ Development

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributing to the project
- **[CODE_IMPROVEMENTS.md](CODE_IMPROVEMENTS.md)** - Suggested code improvements and enhancements

## 📖 Setup Guides by Operating System

### macOS (Mac)
1. Read [SETUP_MAC.md](SETUP_MAC.md)
2. Run `./setup_mac.sh` for automated setup
3. Or use the Makefile: `make help`

### Windows → Mac
1. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) first
2. Follow [SETUP_MAC.md](SETUP_MAC.md)
3. Reference the command comparison tables in both guides

### General (Windows/Linux/Mac)
1. Read [README.md](README.md)
2. Follow the Quick Start section
3. Use `python validate_setup.py` to verify your setup

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | Kafka infrastructure configuration |
| `Makefile` | Common commands and shortcuts |
| `.env` | Your API keys (you create this from `.env.example`) |
| `.gitignore` | Files to exclude from git |

## 🎯 Quick Command Reference

### Setup
```bash
# Mac - Automated
./setup_mac.sh

# Mac - Using Makefile
make setup
make validate
make setup-pinecone

# Manual
pip install -r requirements.txt
python validate_setup.py
python setup_multimodal_index.py
```

### Running
```bash
# Using Makefile (Mac/Linux)
make start-kafka
make start-producer    # Terminal 1
make start-consumer    # Terminal 2
make start-app         # Terminal 3

# Manual
docker compose up -d
python ingestion_scripts/producer.py
python data_processor/consumer_and_embedder.py
streamlit run app.py
```

## 🆘 Troubleshooting

### By Platform
- **Mac Issues**: See "Mac-Specific Troubleshooting" in [SETUP_MAC.md](SETUP_MAC.md)
- **Migration Issues**: See "Common Migration Issues" in [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **General Issues**: See "Troubleshooting" section in [README.md](README.md)

### By Component
- **Docker/Kafka**: [README.md](README.md#troubleshooting)
- **Python/Packages**: [SETUP_MAC.md](SETUP_MAC.md#mac-specific-troubleshooting) (Mac) or [README.md](README.md)
- **API Keys**: [README.md](README.md#configuration)

## 📋 Checklists

### First-Time Setup Checklist
- [ ] Install prerequisites (Python, Docker, Git)
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Set up `.env` file with API keys
- [ ] Run `python validate_setup.py`
- [ ] Set up Pinecone index
- [ ] Start Kafka
- [ ] Test the pipeline

### Mac-Specific Checklist
See detailed checklist in [SETUP_MAC.md](SETUP_MAC.md#quick-start-checklist)

### Migration Checklist
See detailed checklist in [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md#pre-migration-checklist)

## 🔗 External Resources

### Official Documentation
- [Python Documentation](https://docs.python.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pinecone Documentation](https://docs.pinecone.io/)

### Platform-Specific
- [Homebrew (Mac)](https://brew.sh/)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
- [Python on macOS](https://docs.python-guide.org/starting/install3/osx/)

## 📊 Documentation Structure

```
real-time-rag/
├── README.md                    # Main documentation
├── SETUP_MAC.md                 # Mac setup guide
├── MIGRATION_GUIDE.md           # Windows → Mac migration
├── CONTRIBUTING.md              # Contribution guidelines
├── CODE_IMPROVEMENTS.md         # Code suggestions
├── DOCUMENTATION_INDEX.md       # This file
├── .env.example                 # Environment template
├── setup_mac.sh                 # Mac setup script
└── Makefile                     # Command shortcuts
```

## 🎓 Learning Path

### Beginners
1. Start with [README.md](README.md) to understand the project
2. Follow the Quick Start guide for your OS
3. Run `python validate_setup.py` frequently
4. Ask for help if you get stuck

### Mac Users (New to Mac)
1. Read [SETUP_MAC.md](SETUP_MAC.md) "Mac-Specific Troubleshooting"
2. Learn basic Terminal commands
3. Understand the differences from Windows
4. Use the automated script: `./setup_mac.sh`

### Windows → Mac Migrants
1. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) completely
2. Note the command differences table
3. Set up Mac-specific tools (Homebrew, etc.)
4. Follow [SETUP_MAC.md](SETUP_MAC.md)

### Advanced Users
1. Review [CODE_IMPROVEMENTS.md](CODE_IMPROVEMENTS.md)
2. Read [CONTRIBUTING.md](CONTRIBUTING.md)
3. Use the Makefile for efficiency
4. Consider contributing improvements

## 🔍 Finding Information

### "How do I install on Mac?"
→ [SETUP_MAC.md](SETUP_MAC.md)

### "I'm switching from Windows to Mac"
→ [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

### "How does this project work?"
→ [README.md](README.md) - See "How It Works" section

### "I want to contribute"
→ [CONTRIBUTING.md](CONTRIBUTING.md)

### "Something isn't working"
→ Run `python validate_setup.py` first, then check:
- [SETUP_MAC.md](SETUP_MAC.md#mac-specific-troubleshooting) (Mac)
- [README.md](README.md#troubleshooting) (General)

### "What are all the available commands?"
→ Run `make help` or see [README.md](README.md#quick-commands-reference)

## 💡 Tips

- **Always activate your virtual environment** before running Python commands
- **Use `make help`** to see available shortcuts (Mac/Linux)
- **Run `python validate_setup.py`** to check your configuration
- **Check Docker Desktop is running** before starting Kafka
- **Read error messages carefully** - they often tell you what's wrong

## 📝 Keeping Documentation Updated

When contributing, please update relevant documentation:
- Code changes → Update [README.md](README.md) if user-facing
- Mac compatibility → Update [SETUP_MAC.md](SETUP_MAC.md)
- New features → Update all relevant docs
- Bug fixes → Update troubleshooting sections

---

**Need help?** Open an issue on GitHub with detailed information about your problem.
