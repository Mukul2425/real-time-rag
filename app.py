import streamlit as st
import os
from dotenv import load_dotenv
import requests

# Updated import to avoid deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    USE_HUGGINGFACE_EMBEDDINGS = True
except ImportError:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    USE_HUGGINGFACE_EMBEDDINGS = False

from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# --- Page configuration ---
st.set_page_config(page_title="Real-Time EV News RAG")
st.title("🚗 EV News RAG Assistant")
st.markdown("Ask questions about electric vehicle market trends and news!")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Pinecone and Embeddings Setup ---
@st.cache_resource
def get_components():
    """Initialize Pinecone client and embedding model"""
    # Initialize Pinecone client
    pinecone_client = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 🔧 FIXED: Use the correct index name that has your data
    index_name = "ev-market-intelligence-openrouter"  # This has 1,700 records!
    index = pinecone_client.Index(index_name)

    # Initialize the SentenceTransformer embeddings model
    if USE_HUGGINGFACE_EMBEDDINGS:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    return index, embeddings

def check_index_status(index):
    """Check if index has data"""
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count, stats
    except Exception as e:
        st.error(f"Error checking index status: {e}")
        return 0, None

def search_similar_content(query, index, embeddings, top_k=5):
    """Search for similar content in Pinecone"""
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def query_openrouter(messages, model="mistralai/mistral-7b-instruct:free"):
    """Query OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",  # Streamlit default port
        "X-Title": "EV Market Intelligence RAG"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error querying OpenRouter: {str(e)}"

def rag_query(user_question, index, embeddings):
    """Perform RAG query combining search + generation"""
    # 1. Search for relevant content
    search_results = search_similar_content(user_question, index, embeddings, top_k=5)
    
    if not search_results.matches:
        # Enhanced error message
        vector_count, stats = check_index_status(index)
        if vector_count == 0:
            return """🚨 **Database is Empty!**
            
Your Pinecone index has no data. To fix this:

1. **Start Kafka**: `docker-compose up -d`
2. **Run News Producer**: `python ingestion_scripts/producer.py` 
3. **Run Data Consumer**: `python data_processor/consumer_and_embedder.py`
4. **Wait for articles to be processed** (you'll see logs)
5. **Then ask your question again**

The system needs news articles to answer your questions about EVs!""", []
        else:
            return f"I searched through {vector_count:,} articles but couldn't find relevant information about: '{user_question}'. Try asking about Tesla, EV sales, charging infrastructure, or electric vehicle market trends.", []
    
    # 2. Extract context from search results
    context_pieces = []
    sources = []
    for match in search_results.matches:
        metadata = match.metadata
        context_pieces.append(f"Source: {metadata.get('source', 'Unknown')}\n"
                            f"Title: {metadata.get('title', 'Unknown')}\n"
                            f"Content: {metadata.get('text', '')}")
        
        # Collect sources for citation
        source_info = {
            'title': metadata.get('title', 'Unknown'),
            'source': metadata.get('source', 'Unknown'),
            'url': metadata.get('url', '')
        }
        if source_info not in sources:
            sources.append(source_info)
    
    context = "\n---\n".join(context_pieces)
    
    # 3. Create prompt for OpenRouter
    messages = [
        {
            "role": "system",
            "content": """You are an expert analyst for electric vehicle market intelligence. 
            You will be provided with relevant news articles and context about EVs. 
            Use this information to provide comprehensive, accurate answers to user questions.
            Always cite your sources when possible and mention if information is from recent news articles.
            Be concise but informative."""
        },
        {
            "role": "user",
            "content": f"""Based on the following recent EV market news and information, please answer this question: {user_question}

Context from recent EV news articles:
{context}

Please provide a comprehensive answer based on this information."""
        }
    ]
    
    # 4. Query OpenRouter
    response = query_openrouter(messages)
    
    return response, sources

# --- Main application logic ---
def main():
    # Initialize components
    try:
        index, embeddings = get_components()
        
        # Check index status
        vector_count, stats = check_index_status(index)
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.error("Please check your Pinecone API key and make sure the index exists.")
        return

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("This RAG system searches through real-time EV news articles to answer your questions.")
        st.write("**Data Sources:** News articles processed through Kafka")
        st.write("**Embeddings:** SentenceTransformer (local)")
        st.write("**LLM:** OpenRouter API")
        st.write("**Index:** ev-market-intelligence-openrouter")
        
        # Database status
        st.header("📊 Database Status")
        if vector_count > 0:
            st.success(f"✅ {vector_count:,} articles indexed")
            st.info(f"Dimension: {stats.dimension if stats else 'Unknown'}")
        else:
            st.error("❌ Database is empty!")
            st.warning("Run your data pipeline:")
            st.code("""
# 1. Start Kafka
docker-compose up -d

# 2. Run producer
python ingestion_scripts/producer.py

# 3. Run consumer  
python data_processor/consumer_and_embedder.py
            """)
        
        # Check API keys
        st.header("🔑 API Keys")
        if os.getenv("OPENROUTER_API_KEY"):
            st.success("✅ OpenRouter API key loaded")
        else:
            st.error("❌ OpenRouter API key missing")
            
        if os.getenv("PINECONE_API_KEY"):
            st.success("✅ Pinecone API key loaded")
        else:
            st.error("❌ Pinecone API key missing")

    # Main content area
    if vector_count > 0:
        st.success(f"🎯 Ready to search through {vector_count:,} EV news articles!")
    else:
        st.warning("⚠️ Database is empty. Run the data pipeline first.")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"**{source['title']}**")
                        st.write(f"Source: {source['source']}")
                        if source['url']:
                            st.write(f"[Read more]({source['url']})")
                        st.write("---")

    # Accept user input
    if prompt := st.chat_input("Ask a question about EVs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching EV news database..."):
                try:
                    response, sources = rag_query(prompt, index, embeddings)
                    st.markdown(response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.write(f"**{source['title']}**")
                                st.write(f"Source: {source['source']}")
                                if source['url']:
                                    st.write(f"[Read more]({source['url']})")
                                st.write("---")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sources": []
                    })

    # Sample questions
    st.markdown("### 💡 Try asking:")
    sample_questions = [
        "What are the latest Tesla developments?",
        "What challenges are EV companies facing?",
        "Tell me about new EV models being released",
        "How are EV sales performing?",
        "What's happening with EV charging infrastructure?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(question, key=f"sample_{i}"):
                # Trigger the query by adding to session state and rerunning
                st.session_state.messages.append({"role": "user", "content": question})
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

if __name__ == "__main__":
    main()