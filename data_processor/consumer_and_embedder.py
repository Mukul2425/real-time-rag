import os
import json
import requests
import io
import hashlib
from dotenv import load_dotenv
from kafka import KafkaConsumer
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Fallback to a simple splitter if langchain is not installed or API differs
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text):
            if not text:
                return []
            chunks = []
            start = 0
            length = len(text)
            while start < length:
                end = min(start + self.chunk_size, length)
                chunks.append(text[start:end])
                start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
            return chunks
from pinecone import Pinecone, ServerlessSpec
import time
from uuid import uuid4
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from urllib.parse import urlparse
from shared_runtime import DeadLetterStore, DedupStore, LocalObjectStore, MetricsStore, retry_with_backoff, setup_logging

# Load environment variables
load_dotenv()
logger = setup_logging("consumer")
dedup_store = DedupStore(os.getenv("DEDUP_DB_PATH", ".rag_state/processed_items.sqlite"))
metrics = MetricsStore(os.getenv("METRICS_DB_PATH", ".rag_state/metrics.sqlite"))
object_store = LocalObjectStore(os.getenv("OBJECT_STORE_DIR", ".rag_state/object_store"))
dead_letters = DeadLetterStore(os.getenv("DEAD_LETTER_FILE", ".rag_state/dead_letters.jsonl"))

# Initialize CLIP model for multimodal embeddings
logger.info("Loading CLIP model for multimodal embeddings")
model_name = "openai/clip-vit-base-patch32"  # This produces 512-dim embeddings
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)
print(f"✅ CLIP model loaded on {device}")
logger.info("CLIP model loaded on %s", device)


def _model_output_to_numpy(output):
    """Robustly convert HF model outputs or tensors to numpy 1D array.

    Handles cases where the model returns a tensor directly or a
    BaseModelOutputWithPooling-like object with attributes such as
    `pooler_output` or `last_hidden_state`.
    """
    try:
        # If it's already a tensor
        if hasattr(output, 'cpu'):
            tensor = output
        # HuggingFace model output with pooler_output
        elif hasattr(output, 'pooler_output'):
            tensor = output.pooler_output
        # Fallback: mean-pool last_hidden_state
        elif hasattr(output, 'last_hidden_state'):
            tensor = output.last_hidden_state.mean(dim=1)
        else:
            raise ValueError("Unsupported model output type for embedding conversion")

        return tensor.detach().cpu().numpy()[0]
    except Exception as e:
        raise

# Connect to Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ev-market-intelligence-multimodal"

# Check if index exists, and create if it doesn't
try:
    existing_indexes = [index.name for index in pinecone_client.list_indexes()]
    
    if index_name not in existing_indexes:
        logger.info("Creating new multimodal index: %s", index_name)
        pinecone_client.create_index(
            name=index_name,
            dimension=512,  # CLIP ViT-Base produces 512-dim embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Change to your region if needed
            )
        )
        while not pinecone_client.describe_index(index_name).status['ready']:
            logger.info("Waiting for index to be ready")
            time.sleep(1)
        logger.info("Index %s created successfully", index_name)
    else:
        logger.info("Index %s already exists", index_name)
        
except Exception as e:
    logger.exception("Error with Pinecone index: %s", e)
    logger.error("Please check your Pinecone API key and try again.")
    exit(1)

try:
    index = pinecone_client.Index(index_name)
    logger.info("Connected to Pinecone index: %s", index_name)
except Exception as e:
    logger.exception("Error connecting to Pinecone index: %s", e)
    exit(1)

# Initialize Kafka Consumer
try:
    consumer = KafkaConsumer(
        'market-news-raw',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=30000  # 30 second timeout
    )
    logger.info("Kafka consumer initialized successfully")
except Exception as e:
    logger.exception("Error initializing Kafka consumer: %s", e)
    exit(1)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

@retry_with_backoff(attempts=3, base_delay=1.0, max_delay=6.0)
def download_image(url, max_size_mb=5):
    """Download and process image from URL"""
    try:
        # Add timeout and size limit
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logger.warning("Image too large: %s bytes", content_length)
            return None
        
        # Read image data
        image_data = response.content
        if len(image_data) > max_size_mb * 1024 * 1024:
            logger.warning("Image too large after download: %s bytes", len(image_data))
            return None
        
        # Open and process image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (CLIP can handle up to 224x224 efficiently)
        if image.size[0] > 512 or image.size[1] > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Return image PIL object, raw bytes, and the source URL
        return image, image_data, url
        
    except Exception as e:
        logger.warning("Failed to download image from %s: %s", url, e)
        metrics.increment("consumer_image_download_failures_total", 1)
        return None

def create_multimodal_embeddings(text, images=None):
    """Create embeddings for text and/or images using CLIP"""
    embeddings = []
    
    try:
        with torch.no_grad():
            # Create text embedding
            text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_out = clip_model.get_text_features(**text_inputs)
            text_embedding = _model_output_to_numpy(text_out)
            
            embeddings.append({
                'type': 'text',
                'embedding': text_embedding.tolist(),
                'content': text
            })
            
            # Create image embeddings if images are provided
            if images:
                for i, (image, image_data, image_path) in enumerate(images):
                    try:
                        image_inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
                        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                        image_out = clip_model.get_image_features(**image_inputs)
                        image_embedding = _model_output_to_numpy(image_out)
                        embeddings.append({
                            'type': 'image',
                            'embedding': image_embedding.tolist(),
                            'image_index': i,
                            'image_path': image_path
                        })
                        
                    except Exception as img_error:
                        print(f"   ❌ Error processing image {i}: {img_error}")
                        continue
                        
    except Exception as e:
        print(f"   ❌ Error creating embeddings: {e}")
        return []
    
    return embeddings

logger.info("Starting Multimodal Kafka Consumer")
logger.info("Processing both text and images from news articles")
logger.info("Using CLIP for unified text-image embeddings")
logger.info("Waiting for messages")

articles_processed = 0
vectors_created = 0
images_processed = 0

try:
    for message in consumer:
        article = message.value
        articles_processed += 1
        logger.info("%s", "=" * 60)
        logger.info("Processing article %s", articles_processed)
        logger.info("Title: %s", article.get('title', 'Unknown title'))
        logger.info("Images found: %s", article.get('image_count', 0))
        
        try:
            content = article.get('content')
            if not content or content.lower() in ['[removed]', 'null', '']:
                logger.info("Skipping article with no content")
                dead_letters.write(article, "missing_or_removed_content", "consume")
                metrics.increment("consumer_dead_letter_total", 1)
                continue

            article_url = article.get('url', '')
            dedup_key_source = article_url or (article.get('title', '') + content[:200])
            dedup_key = hashlib.sha256(dedup_key_source.encode('utf-8')).hexdigest()
            if dedup_store.has(dedup_key):
                logger.info("Skipping already processed article: %s", article.get('title', 'Unknown title'))
                metrics.increment("consumer_skipped_duplicates_total", 1)
                continue
            
            # Download images if available
            downloaded_images = []
            image_urls = article.get('image_urls', [])
            
            if image_urls:
                logger.info("Downloading %s images", len(image_urls))
                for i, img_url in enumerate(image_urls):
                    logger.info("Downloading image %s: %s", i + 1, img_url)
                    img_result = download_image(img_url)
                    if img_result:
                        image, image_data, image_url = img_result
                        image_key = hashlib.sha256(image_data).hexdigest()
                        image_suffix = os.path.splitext(urlparse(img_url).path)[1] or ".bin"
                        image_path = object_store.save_bytes(image_data, image_key, suffix=image_suffix)
                        downloaded_images.append((image, image_data, image_path))
                        images_processed += 1
                        metrics.increment("consumer_images_downloaded_total", 1)
                        logger.info("Image %s downloaded successfully", i + 1)
                    else:
                        logger.warning("Failed to download image %s", i + 1)
            
            # 1. Split content into chunks
            chunks = text_splitter.split_text(content)
            if not chunks:
                logger.info("No chunks created from content")
                continue
            
            logger.info("Created %s text chunks", len(chunks))
            logger.info("Downloaded %s images", len(downloaded_images))
            
            # 2. Create multimodal embeddings for each chunk
            vectors_to_upsert = []
            
            for chunk_idx, chunk in enumerate(chunks):
                # For the first chunk, include all images
                # For other chunks, only include text
                chunk_images = downloaded_images if chunk_idx == 0 else None
                
                embeddings = create_multimodal_embeddings(chunk, chunk_images)
                
                if not embeddings:
                    logger.warning("No embeddings created for chunk %s", chunk_idx)
                    continue
                
                # Create vectors for each embedding
                for emb_idx, embedding_data in enumerate(embeddings):
                    vector_id = str(uuid4())
                    
                    metadata = {
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "title": article.get('title', 'Unknown title'),
                        "url": article.get('url', ''),
                        "published_at": article.get('publishedAt', ''),
                        "chunk_index": chunk_idx,
                        "embedding_type": embedding_data['type'],
                        "embedding_model": "clip-vit-base-patch32"
                    }
                    
                    if embedding_data['type'] == 'text':
                        metadata.update({
                            "text": chunk,
                            "content_type": "text"
                        })
                    else:  # image
                        image_path = chunk_images[embedding_data['image_index']][2] if chunk_images and embedding_data['image_index'] < len(chunk_images) else ''
                        metadata.update({
                            "image_path": image_path,
                            "image_index": embedding_data['image_index'],
                            "content_type": "image",
                            "text": f"Image {embedding_data['image_index'] + 1} from article: {article.get('title', 'Unknown')}"
                        })
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding_data['embedding'],
                        "metadata": metadata
                    })
            
            # 3. Upsert to Pinecone
            if vectors_to_upsert:
                try:
                    # Upsert in batches to avoid timeouts
                    batch_size = 100
                    for i in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[i:i + batch_size]
                        index.upsert(vectors=batch)
                    
                    vectors_created += len(vectors_to_upsert)
                    text_vectors = sum(1 for v in vectors_to_upsert if v['metadata']['content_type'] == 'text')
                    image_vectors = sum(1 for v in vectors_to_upsert if v['metadata']['content_type'] == 'image')
                    
                    logger.info("Upserted %s vectors (text=%s image=%s)", len(vectors_to_upsert), text_vectors, image_vectors)
                    logger.info("Total stats: %s articles, %s vectors, %s images", articles_processed, vectors_created, images_processed)
                    metrics.increment("consumer_articles_processed_total", 1)
                    metrics.increment("consumer_vectors_upserted_total", len(vectors_to_upsert))
                    metrics.increment("consumer_upsert_batches_total", 1)
                    dedup_store.add(dedup_key)
                    logger.info("Marked article as processed: %s", article.get('title', 'Unknown title'))
                    
                except Exception as upsert_error:
                    logger.exception("Error upserting to Pinecone: %s", upsert_error)
                    dead_letters.write(article, f"pinecone_upsert_failed: {upsert_error}", "upsert")
                    metrics.increment("consumer_upsert_failures_total", 1)
                    continue
            else:
                logger.info("No vectors to upsert")
                dead_letters.write(article, "no_vectors_created", "embed")
                metrics.increment("consumer_no_vectors_total", 1)
                
        except Exception as e:
            logger.exception("Error processing article '%s': %s", article.get('title', 'Unknown'), str(e))
            dead_letters.write(article, f"consumer_error: {e}", "consume")
            metrics.increment("consumer_processing_failures_total", 1)
            continue

except KeyboardInterrupt:
    logger.info("Shutting down consumer")
    logger.info("Final stats: %s articles processed, %s vectors created, %s images processed", articles_processed, vectors_created, images_processed)
except Exception as e:
    logger.exception("Consumer error: %s", e)
finally:
    try:
        consumer.close()
        dedup_store.close()
        metrics.close()
        logger.info("Consumer closed successfully")
    except:
        pass