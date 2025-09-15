# 🚗🖼️ Multimodal EV RAG Assistant

A Retrieval-Augmented Generation (RAG) system that processes both **text** and **images** from electric vehicle news articles using CLIP embeddings and multimodal LLMs.

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

### 1. Clone and Setup
```bash
git clone <your-repo>
cd multimodal-ev-rag
pip install -r requirements.txt
```

### 2. Environment Configuration
Create `.env` file:
```env
NEWS_API_KEY=your_news_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here  
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Setup Pinecone Index
```bash
python setup_multimodal_index.py
```

### 4. Start Infrastructure
```bash
docker-compose up -d
```

### 5. Run Data Pipeline
```bash
# Terminal 1: Producer
python ingestion_scripts/producer.py

# Terminal 2: Consumer  
python data_processor/consumer_and_embedder.py
```

### 6. Launch Application
```bash
streamlit run app.py
```

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
multimodal-ev-rag/
├── app.py                      # Main Streamlit application
├── setup_multimodal_index.py   # Pinecone index setup
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Kafka infrastructure
├── .env                        # Environment variables
├── ingestion_scripts/
│   └── producer.py            # Enhanced Kafka producer
├── data_processor/
│   └── consumer_and_embedder.py # CLIP-based consumer
└── config/
    └── settings.py            # Configuration (optional)
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

**3. Kafka Connection