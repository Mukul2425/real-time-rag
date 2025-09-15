import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

def setup_multimodal_pinecone():
    """Setup Pinecone index for multimodal RAG"""
    print("🔧 Setting up Multimodal Pinecone Index...")
    print("=" * 60)
    
    # Initialize Pinecone
    try:
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("✅ Connected to Pinecone")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return False
    
    # List existing indexes
    print("\n📋 Current indexes:")
    existing_indexes = pinecone_client.list_indexes()
    for idx in existing_indexes:
        print(f"  - {idx.name} (dim: {idx.dimension}, vectors: unknown)")
    
    # Define index names
    old_index = "ev-market-intelligence-openrouter"
    new_index = "ev-market-intelligence-multimodal"
    
    # Check if we need to handle old index
    old_exists = any(idx.name == old_index for idx in existing_indexes)
    new_exists = any(idx.name == new_index for idx in existing_indexes)
    
    print(f"\n🎯 Target setup:")
    print(f"  Old index ({old_index}): {'EXISTS' if old_exists else 'NOT FOUND'}")
    print(f"  New index ({new_index}): {'EXISTS' if new_exists else 'NOT FOUND'}")
    
    # Handle old index
    if old_exists:
        print(f"\n⚠️  Found existing index: {old_index}")
        response = input("Do you want to DELETE the old index? This will remove all your data! (yes/no): ")
        
        if response.lower() == 'yes':
            try:
                pinecone_client.delete_index(old_index)
                print(f"✅ Deleted old index: {old_index}")
                
                # Wait for deletion to complete
                print("⏳ Waiting for deletion to complete...")
                time.sleep(10)
            except Exception as e:
                print(f"❌ Error deleting old index: {e}")
        else:
            print("⏭️  Keeping old index. You can delete it manually later.")
    
    # Create or check new index
    if new_exists:
        print(f"\n✅ Multimodal index already exists: {new_index}")
        
        # Check if it has the right dimensions
        try:
            index = pinecone_client.Index(new_index)
            stats = index.describe_index_stats()
            print(f"   Dimension: {stats.dimension}")
            print(f"   Vector count: {stats.total_vector_count}")
            
            if stats.dimension != 512:
                print(f"⚠️  WARNING: Index has dimension {stats.dimension}, but CLIP needs 512!")
                response = input("Do you want to DELETE and recreate this index? (yes/no): ")
                
                if response.lower() == 'yes':
                    pinecone_client.delete_index(new_index)
                    print(f"✅ Deleted incompatible index")
                    time.sleep(10)
                    new_exists = False  # Force recreation
        except Exception as e:
            print(f"❌ Error checking index: {e}")
    
    if not new_exists:
        print(f"\n🔨 Creating new multimodal index: {new_index}")
        try:
            pinecone_client.create_index(
                name=new_index,
                dimension=512,  # CLIP ViT-Base produces 512-dim embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Change if needed
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            while True:
                try:
                    status = pinecone_client.describe_index(new_index).status['ready']
                    if status:
                        break
                    time.sleep(2)
                except:
                    time.sleep(2)
            
            print(f"✅ Successfully created multimodal index: {new_index}")
            print("   Dimension: 512 (CLIP compatible)")
            print("   Metric: cosine")
            print("   Type: Serverless")
            
        except Exception as e:
            print(f"❌ Error creating new index: {e}")
            return False
    
    print(f"\n🎉 Setup complete!")
    print(f"✅ Your multimodal index '{new_index}' is ready")
    print(f"📊 Dimension: 512 (perfect for CLIP embeddings)")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Update your app.py to use: {new_index}")
    print(f"2. Run your enhanced data pipeline:")
    print(f"   - python ingestion_scripts/producer.py")
    print(f"   - python data_processor/consumer_and_embedder.py")
    print(f"3. Test your multimodal app: streamlit run app.py")
    
    return True

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('PIL', 'Pillow (PIL)'),
        ('pinecone', 'Pinecone'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n📦 To install missing packages:")
        print(f"pip install torch transformers pillow pinecone-client")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Multimodal RAG Setup Script")
    print("This script will help you set up Pinecone for multimodal (text + image) RAG")
    print("")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing packages first!")
        exit(1)
    
    # Setup Pinecone
    success = setup_multimodal_pinecone()
    
    if success:
        print("\n🎯 Ready for multimodal RAG!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")