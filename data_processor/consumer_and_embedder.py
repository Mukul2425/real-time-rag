import os
import json
import requests
import io
from dotenv import load_dotenv
from kafka import KafkaConsumer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import time
from uuid import uuid4
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Initialize CLIP model for multimodal embeddings
print("🤖 Loading CLIP model for multimodal embeddings...")
model_name = "openai/clip-vit-base-patch32"  # This produces 512-dim embeddings
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)
print(f"✅ CLIP model loaded on {device}")

# Connect to Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ev-market-intelligence-multimodal"

# Check if index exists, and create if it doesn't
try:
    existing_indexes = [index.name for index in pinecone_client.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new multimodal index: {index_name}...")
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
            print("Waiting for index to be ready...")
            time.sleep(1)
        print(f"✅ Index {index_name} created successfully!")
    else:
        print(f"✅ Index {index_name} already exists.")
        
except Exception as e:
    print(f"❌ Error with Pinecone index: {e}")
    print("Please check your Pinecone API key and try again.")
    exit(1)

try:
    index = pinecone_client.Index(index_name)
    print(f"✅ Connected to Pinecone index: {index_name}")
except Exception as e:
    print(f"❌ Error connecting to Pinecone index: {e}")
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
    print("✅ Kafka consumer initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Kafka consumer: {e}")
    exit(1)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def download_image(url, max_size_mb=5):
    """Download and process image from URL"""
    try:
        # Add timeout and size limit
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            print(f"   ⚠️  Image too large: {content_length} bytes")
            return None
        
        # Read image data
        image_data = response.content
        if len(image_data) > max_size_mb * 1024 * 1024:
            print(f"   ⚠️  Image too large after download: {len(image_data)} bytes")
            return None
        
        # Open and process image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (CLIP can handle up to 224x224 efficiently)
        if image.size[0] > 512 or image.size[1] > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        return image, image_data
        
    except Exception as e:
        print(f"   ❌ Failed to download image from {url}: {e}")
        return None

def create_multimodal_embeddings(text, images=None):
    """Create embeddings for text and/or images using CLIP"""
    embeddings = []
    
    try:
        with torch.no_grad():
            # Create text embedding
            text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_embedding = clip_model.get_text_features(**text_inputs)
            text_embedding = text_embedding.cpu().numpy()[0]  # Shape: (512,)
            
            embeddings.append({
                'type': 'text',
                'embedding': text_embedding.tolist(),
                'content': text
            })
            
            # Create image embeddings if images are provided
            if images:
                for i, (image, image_data) in enumerate(images):
                    try:
                        image_inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
                        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                        image_embedding = clip_model.get_image_features(**image_inputs)
                        image_embedding = image_embedding.cpu().numpy()[0]  # Shape: (512,)
                        
                        # Convert image to base64 for storage
                        image_b64 = base64.b64encode(image_data).decode('utf-8')
                        
                        embeddings.append({
                            'type': 'image',
                            'embedding': image_embedding.tolist(),
                            'image_data': image_b64,
                            'image_index': i
                        })
                        
                    except Exception as img_error:
                        print(f"   ❌ Error processing image {i}: {img_error}")
                        continue
                        
    except Exception as e:
        print(f"   ❌ Error creating embeddings: {e}")
        return []
    
    return embeddings

print("🔄 Starting Multimodal Kafka Consumer...")
print("📊 Processing both text and images from news articles")
print("🎯 Using CLIP for unified text-image embeddings")
print("⏳ Waiting for messages...")

articles_processed = 0
vectors_created = 0
images_processed = 0

try:
    for message in consumer:
        article = message.value
        articles_processed += 1
        print(f"\n{'='*60}")
        print(f"📰 Processing article {articles_processed}")
        print(f"Title: {article.get('title', 'Unknown title')}")
        print(f"Images found: {article.get('image_count', 0)}")
        
        try:
            content = article.get('content')
            if not content or content.lower() in ['[removed]', 'null', '']:
                print("❌ Skipping article with no content.")
                continue
            
            # Download images if available
            downloaded_images = []
            image_urls = article.get('image_urls', [])
            
            if image_urls:
                print(f"📸 Downloading {len(image_urls)} images...")
                for i, img_url in enumerate(image_urls):
                    print(f"   Downloading image {i+1}: {img_url}")
                    img_result = download_image(img_url)
                    if img_result:
                        downloaded_images.append(img_result)
                        images_processed += 1
                        print(f"   ✅ Image {i+1} downloaded successfully")
                    else:
                        print(f"   ❌ Failed to download image {i+1}")
            
            # 1. Split content into chunks
            chunks = text_splitter.split_text(content)
            if not chunks:
                print("❌ No chunks created from content.")
                continue
            
            print(f"✅ Created {len(chunks)} text chunks")
            print(f"✅ Downloaded {len(downloaded_images)} images")
            
            # 2. Create multimodal embeddings for each chunk
            vectors_to_upsert = []
            
            for chunk_idx, chunk in enumerate(chunks):
                # For the first chunk, include all images
                # For other chunks, only include text
                chunk_images = downloaded_images if chunk_idx == 0 else None
                
                embeddings = create_multimodal_embeddings(chunk, chunk_images)
                
                if not embeddings:
                    print(f"   ❌ No embeddings created for chunk {chunk_idx}")
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
                        metadata.update({
                            "image_data": embedding_data['image_data'],
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
                    
                    print(f"✅ Upserted {len(vectors_to_upsert)} vectors:")
                    print(f"   📄 Text vectors: {text_vectors}")
                    print(f"   🖼️  Image vectors: {image_vectors}")
                    print(f"📊 Total stats: {articles_processed} articles, {vectors_created} vectors, {images_processed} images")
                    
                except Exception as upsert_error:
                    print(f"❌ Error upserting to Pinecone: {upsert_error}")
                    continue
            else:
                print("❌ No vectors to upsert.")
                
        except Exception as e:
            print(f"❌ Error processing article '{article.get('title', 'Unknown')}': {str(e)}")
            continue

except KeyboardInterrupt:
    print(f"\n🛑 Shutting down consumer...")
    print(f"📊 Final stats: {articles_processed} articles processed, {vectors_created} vectors created, {images_processed} images processed")
except Exception as e:
    print(f"❌ Consumer error: {e}")
finally:
    try:
        consumer.close()
        print("✅ Consumer closed successfully.")
    except:
        pass