import os
import json
from dotenv import load_dotenv
from kafka import KafkaConsumer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time
from uuid import uuid4

# Load environment variables
load_dotenv()

# Connect to Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ev-market-intelligence-openrouter"

# Check if index exists, and create if it doesn't
try:
    existing_indexes = [index.name for index in pinecone_client.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}...")
        pinecone_client.create_index(
            name=index_name,
            dimension=384, # Dimension for all-MiniLM-L6-v2 model (free)
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # Change to your region if needed
            )
        )
        while not pinecone_client.describe_index(index_name).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)
        print(f"Index {index_name} created successfully!")
    else:
        print(f"Index {index_name} already exists.")
        
except Exception as e:
    print(f"Error with Pinecone index: {e}")
    print("Please check your Pinecone API key and try again.")
    exit(1)

try:
    index = pinecone_client.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")
except Exception as e:
    print(f"Error connecting to Pinecone index: {e}")
    exit(1)

# Initialize the embedding model (FREE - no API key needed)
try:
    print("Loading SentenceTransformer model (free, local embeddings)...")
    print("This may take a moment on first run to download the model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded successfully!")
    print("Note: We'll use OpenRouter for the RAG query system, not embeddings.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please install: pip install sentence-transformers")
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
    print("Kafka consumer initialized successfully.")
except Exception as e:
    print(f"Error initializing Kafka consumer: {e}")
    exit(1)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

print("Starting Kafka consumer...")
print("Using FREE local embeddings (SentenceTransformer)")
print("OpenRouter will be used later for RAG queries")
print("Waiting for messages...")

articles_processed = 0
vectors_created = 0

try:
    for message in consumer:
        article = message.value
        articles_processed += 1
        print(f"\n--- Processing article {articles_processed} ---")
        print(f"Title: {article.get('title', 'Unknown title')}")
        
        try:
            content = article.get('content')
            if not content or content.lower() in ['[removed]', 'null', '']:
                print("❌ Skipping article with no content.")
                continue
            
            # 1. Split content into chunks
            chunks = text_splitter.split_text(content)
            if not chunks:
                print("❌ No chunks created from content.")
                continue
            
            print(f"✅ Created {len(chunks)} chunks from article.")
            
            # 2. Generate embeddings using FREE local model
            try:
                embeddings = embedding_model.encode(chunks)
                print(f"✅ Generated {len(embeddings)} embeddings using SentenceTransformer.")
            except Exception as embed_error:
                print(f"❌ Error generating embeddings: {embed_error}")
                continue
            
            # 3. Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                vector_id = str(uuid4())
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embeddings[i].tolist(),  # Convert numpy array to list
                    "metadata": {
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "title": article.get('title', 'Unknown title'),
                        "url": article.get('url', ''),
                        "text": chunk,
                        "published_at": article.get('publishedAt', ''),
                        "chunk_index": i,
                        "embedding_model": "all-MiniLM-L6-v2"  # Track which model we used
                    }
                })
            
            # 4. Upsert to Pinecone
            if vectors_to_upsert:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_created += len(vectors_to_upsert)
                    print(f"✅ Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")
                    print(f"📊 Total: {articles_processed} articles processed, {vectors_created} vectors created")
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
    print(f"📊 Final stats: {articles_processed} articles processed, {vectors_created} vectors created")
except Exception as e:
    print(f"❌ Consumer error: {e}")
finally:
    try:
        consumer.close()
        print("✅ Consumer closed successfully.")
    except:
        pass