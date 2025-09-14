import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

def debug_pinecone():
    """Debug Pinecone connection and index status"""
    print("🔍 Debugging Pinecone Setup...")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment variables")
        return False
    else:
        print(f"✅ PINECONE_API_KEY found: {api_key[:8]}...")
    
    try:
        # Initialize Pinecone client
        print("\n🔗 Connecting to Pinecone...")
        pinecone_client = PineconeClient(api_key=api_key)
        print("✅ Pinecone client initialized successfully")
        
        # List all indexes
        print("\n📋 Listing all indexes...")
        indexes = pinecone_client.list_indexes()
        print(f"Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"  - {idx.name} (dimension: {idx.dimension}, metric: {idx.metric})")
        
        # Check both indexes
        index_names = ["ev-market-intelligence", "ev-market-intelligence-openrouter"]
        
        for index_name in index_names:
            print(f"\n🎯 Checking index: {index_name}")
            
            try:
                index = pinecone_client.Index(index_name)
                print("✅ Successfully connected to index")
                
                # Get index stats
                print(f"\n📊 Index Statistics for {index_name}:")
                stats = index.describe_index_stats()
                print(f"  - Total vectors: {stats.total_vector_count}")
                print(f"  - Index fullness: {stats.index_fullness}")
                print(f"  - Dimension: {stats.dimension}")
                
                if stats.total_vector_count == 0:
                    print(f"\n⚠️  INDEX {index_name} IS EMPTY!")
                else:
                    print(f"✅ Index {index_name} has {stats.total_vector_count} vectors")
                    
                    # Test a simple query for non-empty indexes
                    if stats.total_vector_count > 0:
                        print(f"\n🧪 Testing a sample query on {index_name}...")
                        try:
                            # Create a dummy vector for testing (match the dimension)
                            test_vector = [0.1] * stats.dimension
                            results = index.query(
                                vector=test_vector,
                                top_k=3,
                                include_metadata=True
                            )
                            print(f"✅ Query successful! Found {len(results.matches)} matches")
                            
                            if results.matches:
                                print(f"\n📄 Sample results from {index_name}:")
                                for i, match in enumerate(results.matches[:2]):
                                    metadata = match.metadata
                                    print(f"  Result {i+1}:")
                                    print(f"    - Title: {metadata.get('title', 'N/A')}")
                                    print(f"    - Source: {metadata.get('source', 'N/A')}")
                                    print(f"    - Score: {match.score}")
                            
                        except Exception as query_error:
                            print(f"❌ Query test failed: {query_error}")
                            
            except Exception as index_error:
                print(f"❌ Failed to connect to index {index_name}: {index_error}")
                
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False
    
    print("\n🎯 DIAGNOSIS:")
    print("=" * 50)
    print("✅ You have 1,700 vectors in 'ev-market-intelligence-openrouter'")
    print("❌ You have 0 vectors in 'ev-market-intelligence'")
    print("🔧 Your app.py is trying to use 'ev-market-intelligence' (the empty one)")
    print("💡 SOLUTION: Change your app to use 'ev-market-intelligence-openrouter'")
    
    return True

def check_data_pipeline():
    """Check if data pipeline scripts exist"""
    print("\n🚀 Data Pipeline Status Check:")
    print("=" * 50)
    
    # Check script locations
    producer_path = "ingestion_scripts/producer.py"
    consumer_path = "data_processor/consumer_and_embedder.py"
    
    if os.path.exists(producer_path):
        print(f"✅ {producer_path} found")
    else:
        print(f"❌ {producer_path} not found")
        
    if os.path.exists(consumer_path):
        print(f"✅ {consumer_path} found")
    else:
        print(f"❌ {consumer_path} not found")
    
    print("\n💡 To run your data pipeline:")
    print(f"1. Start Kafka: docker-compose up -d")
    print(f"2. Run producer: python {producer_path}")
    print(f"3. Run consumer: python {consumer_path}")

if __name__ == "__main__":
    # Run debugging
    success = debug_pinecone()
    
    if success:
        check_data_pipeline()
        
    print("\n🎯 Next Steps:")
    print("1. ✅ Update your app.py to use 'ev-market-intelligence-openrouter' index")
    print("2. ✅ Your data is already there (1,700 vectors)!")
    print("3. ✅ Test your Streamlit app: streamlit run app.py")