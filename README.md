# 🚗🖼️ Multimodal EV RAG Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?logo=apache-kafka)](https://kafka.apache.org/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?logo=pinecone&logoColor=white)](https://www.pinecone.io/)

A Retrieval-Augmented Generation (RAG) system that processes both **text** and **images** from electric vehicle news articles using CLIP embeddings and multimodal LLMs.

> 🎯 **Real-time EV market intelligence powered by multimodal AI**

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Performance Tips](#-performance-tips)
- [Security Best Practices](#-security-best-practices)
- [Contributing](#-contributing)
- [Roadmap](#-roadmap)
- [FAQ](#-faq)
- [License](#-license)

## 🌟 Features

- **Multimodal RAG**: Search and analyze both text content and images from EV news
- **Real-time Data**: Kafka-based pipeline for continuous news ingestion
- **CLIP Embeddings**: Unified 512-dimensional embeddings for text and images
- **Visual Interface**: Streamlit app supporting image uploads and visual results
- **Smart Image Processing**: Automatic image extraction and validation from news articles
- **Multimodal LLM**: Google Gemini 1.5 Flash via OpenRouter for comprehensive analysis

## 🏗️ Architecture

```
News API → Kafka Producer → Kafka → Consumer → CLIP → Pinecone → Streamlit App
    ↓           ↓              ↓        ↓       ↓        ↓          ↓
  Articles   Enhanced      Queue    Process  Vector   Search   Multimodal
  + Images   Extraction            Text+Img  Store    Results      UI
```

## 🚀 Quick Start

### Prerequisites Checklist
Before starting, ensure you have:
- ✅ Python 3.8 or higher installed
- ✅ Docker and Docker Compose installed
- ✅ Git installed
- ✅ API keys for News API, Pinecone, and OpenRouter

### Quick Setup with Makefile (Recommended)

This project includes a Makefile for easy setup and management:

```bash
# See all available commands
make help

# Install dependencies
make setup

# Validate your environment
make validate

# Setup Pinecone index
make setup-pinecone

# Start Kafka
make start-kafka
```

Then start the pipeline in separate terminals:
```bash
# Terminal 1
make start-producer

# Terminal 2
make start-consumer

# Terminal 3
make start-app
```

### Manual Setup

If you prefer manual setup or don't have `make` installed:

### 1. Clone and Setup
```bash
git clone https://github.com/Mukul2425/real-time-rag.git
cd real-time-rag
pip install -r requirements.txt
```


### 2. Environment Configuration
Create `.env` file in the project root:
```bash
# Copy the example file
cp .env.example .env

# Edit with your actual API keys
nano .env  # or use your preferred editor
```

Your `.env` file should contain:
```env
NEWS_API_KEY=your_news_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here  
OPENROUTER_API_KEY=your_openrouter_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=auto
LLM_MODEL=
```

**Get your API keys:**
- **News API**: Sign up at [newsapi.org](https://newsapi.org/) (Free tier: 100 requests/day)
- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io/) (Free tier: 100k vectors)
- **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai/) (Pay-as-you-go)
- **Gemini API**: Sign up at [Google AI Studio](https://aistudio.google.com/) for direct Gemini access

**Validate your setup** (optional but recommended):
```bash
python validate_setup.py
```
This script checks all dependencies and configuration.

### 3. Setup Pinecone Index
```bash
python setup_multimodal_index.py
```
This creates a 512-dimensional index optimized for CLIP embeddings.

### 4. Start Infrastructure
```bash
docker-compose up -d
# Verify Kafka is running
docker-compose ps
```

### 5. Run Data Pipeline
Open two terminal windows:

**Terminal 1 - Producer** (Fetches news articles):
```bash
python ingestion_scripts/producer.py
```

**Terminal 2 - Consumer** (Processes and indexes data):
```bash
python data_processor/consumer_and_embedder.py
```

💡 **Tip**: Let the pipeline run for 5-10 minutes to collect initial data.

### 6. Launch Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### 7. Test Your Setup
1. Wait for some articles to be processed (check consumer logs)
2. Try a query like: "What are the latest Tesla charging innovations?"
3. Upload an image of an EV for multimodal search
4. Check the sources to verify data is being retrieved

## 📋 Prerequisites

### API Keys Required
- **News API**: Get from [newsapi.org](https://newsapi.org/)
- **Pinecone**: Get from [pinecone.io](https://pinecone.io/)  
- **OpenRouter**: Get from [openrouter.ai](https://openrouter.ai/)

### System Requirements
- Python 3.8+
- Docker & Docker Compose
- 4GB+ RAM (8GB recommended for GPU)
- CUDA GPU (optional, for faster processing)

## 🔧 Configuration

### Pinecone Index Settings
- **Name**: `ev-market-intelligence-multimodal`
- **Dimension**: 512 (CLIP ViT-Base-Patch32)
- **Metric**: Cosine similarity
- **Type**: Serverless (AWS us-east-1)

### CLIP Model
- **Model**: `openai/clip-vit-base-patch32`
- **Embedding Dimension**: 512
- **Supported**: Text + Images (PNG, JPG, JPEG, GIF, WebP)

### Data Sources
- Electric vehicle news articles
- Tesla updates and announcements  
- EV charging infrastructure news
- Battery technology developments

## 🎯 Usage Examples

### Text-Only Query
```
Question: "What are the latest Tesla charging innovations?"
→ Searches through text embeddings for relevant articles
```

### Image-Only Query  
```
Upload: [Image of EV charging station]
Question: "What is this charging technology?"
→ Uses image embedding to find similar images and related content
```

### Multimodal Query
```
Upload: [Image of Tesla Model S]
Question: "What are the latest updates about this car model?"
→ Combines image and text embeddings for comprehensive search
```

## 🔍 How It Works

### 1. Data Ingestion
- **Producer**: Fetches EV news from News API
- **Image Extraction**: Finds and validates image URLs in articles
- **Quality Control**: Filters out invalid/corrupted images

### 2. Processing Pipeline  
- **Text Processing**: Splits articles into chunks
- **Image Download**: Retrieves and processes images
- **CLIP Encoding**: Creates unified embeddings for text and images
- **Vector Storage**: Stores in Pinecone with metadata

### 3. Query Processing
- **Input Handling**: Accepts text questions ± image uploads
- **Embedding Creation**: Generates query embeddings using CLIP
- **Similarity Search**: Finds relevant content in Pinecone
- **Multimodal Response**: Uses Gemini 1.5 Flash for comprehensive answers

## 📁 Project Structure

```
real-time-rag/
├── app.py                          # Main Streamlit application
├── setup_multimodal_index.py       # Pinecone index setup script
├── validate_setup.py               # Environment validation script (NEW)
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Kafka infrastructure
├── Makefile                        # Common commands helper (NEW)
├── .env.example                    # Example environment file (NEW)
├── .env                            # Your API keys (create this, not in git)
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── CONTRIBUTING.md                 # Contribution guidelines (NEW)
├── ingestion_scripts/
│   ├── __init__.py
│   └── producer.py                 # Kafka producer with image extraction
├── data_processor/
│   ├── __init__.py
│   └── consumer_and_embedder.py    # CLIP-based consumer
├── check_openrouter_models.py      # OpenRouter model checker
├── debug_pinecone.py               # Pinecone debugging utility
└── files.txt                       # Project files list
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Use CPU or reduce batch sizes
export CUDA_VISIBLE_DEVICES=""
```

**2. Pinecone Dimension Mismatch**
```bash
# Solution: Delete and recreate index
python setup_multimodal_index.py
```

**3. Kafka Connection Refused**
```bash
# Solution: Make sure Kafka is running
docker-compose up -d
docker-compose ps  # Check if container is running
```

**4. No Results in Search**
```bash
# Solution: Check if data pipeline is running and has data
python debug_pinecone.py  # If this file exists to check index
```

**5. OpenRouter API Errors**
- Check your API key is valid
- Verify you have credits on OpenRouter
- The system has fallback models configured

**6. Image Processing Fails**
- Some images may be too large or invalid
- Check logs for specific image URLs that fail
- System automatically skips invalid images

## 🚀 Performance Tips

### GPU Acceleration
- Use CUDA-enabled GPU for 5-10x faster processing
- CLIP model benefits significantly from GPU
- Set `CUDA_VISIBLE_DEVICES` to control GPU usage

### Batch Processing
- Consumer processes images in batches for efficiency
- Adjust batch sizes in consumer code based on memory

### Index Optimization
- Use cosine similarity for best results with CLIP
- Keep vectors at 512 dimensions (CLIP standard)
- Consider pod-based index for very large datasets

## 🔒 Security Best Practices

### API Keys
- Never commit `.env` file to version control
- Use environment variables in production
- Rotate API keys regularly
- Set up API key restrictions where possible

### Data Privacy
- Be aware that images are stored as base64 in metadata
- Consider data retention policies
- Implement access controls for production deployments

### Network Security
- Kafka is exposed on localhost only by default
- Consider using TLS for production Kafka
- Implement authentication for production deployments

## 🤝 Contributing

We welcome contributions! Here's how you can help:

**Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.**

### Quick Start for Contributors
```bash
# Clone the repository
git clone https://github.com/Mukul2425/real-time-rag.git
cd real-time-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup pre-commit hooks (if you add them)
# pre-commit install
```

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

### Testing
- Test your changes with different query types
- Verify both text and image processing
- Check error handling edge cases

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Project Metrics

- **Embedding Model**: CLIP ViT-Base-Patch32 (512-dim)
- **Vector Database**: Pinecone Serverless
- **Message Queue**: Apache Kafka 3.7.0
- **LLM Provider**: OpenRouter (6 model fallbacks)
- **Data Source**: News API

## 🔮 Roadmap

### Planned Features
- [ ] Support for video content processing
- [ ] Advanced filtering by date, source, sentiment
- [ ] Caching layer for frequently asked questions
- [ ] Real-time dashboard for data pipeline monitoring
- [ ] Support for multiple languages
- [ ] Export functionality for search results
- [ ] User authentication and personal query history

### Potential Improvements
- [ ] Add vector quantization for reduced storage
- [ ] Implement semantic caching
- [ ] Add A/B testing for different embedding models
- [ ] Create Docker image for easy deployment
- [ ] Add comprehensive test suite
- [ ] Implement rate limiting and quotas

**📖 For detailed code improvement suggestions, see [CODE_IMPROVEMENTS.md](CODE_IMPROVEMENTS.md)**

## 📚 Additional Resources

### Documentation
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Tutorials
- [RAG Implementation Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Multimodal ML Guide](https://huggingface.co/blog/vision-language-pretraining)
- [CLIP Usage Examples](https://huggingface.co/docs/transformers/model_doc/clip)

## ❓ FAQ

**Q: Why use CLIP instead of separate text/image embeddings?**
A: CLIP creates unified embeddings in the same vector space, enabling multimodal search where images and text can be compared directly.

**Q: Can I use this with sources other than News API?**
A: Yes! Modify `producer.py` to fetch from any source - RSS feeds, web scrapers, or custom APIs.

**Q: How much does it cost to run?**
A: Costs depend on usage:
- Pinecone: Free tier available (100k vectors)
- OpenRouter: Pay per use (~$0.50-2.00 per 1M tokens)
- News API: Free tier available (100 requests/day)

**Q: Can I run this without GPU?**
A: Yes, but it will be slower. CPU mode works fine for small-scale usage.

**Q: How do I add more news sources?**
A: Edit the `search_queries` list in `producer.py` to add more topics.

**Q: What image formats are supported?**
A: PNG, JPG, JPEG, GIF, and WebP formats are supported.

**Q: How long does it take to process articles?**
A: Typical processing time:
- Text chunk: ~0.1-0.5 seconds
- Image download + embedding: ~1-3 seconds per image
- Total: Depends on article length and image count

**Q: Can I use a different LLM?**
A: Yes! The system uses OpenRouter which supports 100+ models. Modify the `models_to_try` list in `app.py`.

## 📸 Screenshots

The repository includes visual examples:
- `rag_ui.png` - Main Streamlit interface
- `pinecone.png` - Pinecone index configuration

## 🙏 Acknowledgments

This project uses amazing open-source technologies:
- **OpenAI CLIP** for multimodal embeddings
- **Pinecone** for vector search
- **Apache Kafka** for streaming data
- **Streamlit** for the web interface
- **Hugging Face** for model hosting
- **OpenRouter** for LLM access

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Reach out to the repository owner

## ⚡ Quick Commands Reference

```bash
# Setup
python setup_multimodal_index.py

# Start services
docker-compose up -d

# Run data pipeline
python ingestion_scripts/producer.py      # Terminal 1
python data_processor/consumer_and_embedder.py  # Terminal 2

# Start app
streamlit run app.py

# Stop services
docker-compose down

# Check Kafka status
docker-compose ps

# View Kafka logs
docker-compose logs kafka

# Check Pinecone stats
# Use the sidebar in app.py or create a debug script
```

---

**Built with ❤️ for the EV community and multimodal AI enthusiasts**