import streamlit as st
import os
from dotenv import load_dotenv
import requests
import base64
import io
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

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
st.set_page_config(page_title="Multimodal EV News RAG", layout="wide")
st.title("🚗🖼️ Multimodal EV News RAG Assistant")
st.markdown("Ask questions about electric vehicles using both **text** and **images**!")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CLIP Model Setup ---
@st.cache_resource
def load_clip_model():
    """Load CLIP model for multimodal embeddings"""
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)
    return clip_model, clip_processor, device

# --- Pinecone Setup ---
@st.cache_resource
def get_components():
    """Initialize Pinecone client and CLIP model"""
    # Initialize Pinecone client
    pinecone_client = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "ev-market-intelligence-multimodal"  # New multimodal index
    index = pinecone_client.Index(index_name)

    # Load CLIP model
    clip_model, clip_processor, device = load_clip_model()
    
    return index, clip_model, clip_processor, device

def create_query_embedding(query_text, query_image=None, clip_model=None, clip_processor=None, device=None):
    """Create embedding for text and/or image query using CLIP"""
    try:
        with torch.no_grad():
            if query_image is not None and clip_model is not None:
                # Multimodal query: combine text and image
                # Process image
                image_inputs = clip_processor(images=[query_image], return_tensors="pt", padding=True)
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                image_embedding = clip_model.get_image_features(**image_inputs)
                image_embedding = image_embedding.cpu().numpy()[0]
                
                # Process text
                text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                text_embedding = clip_model.get_text_features(**text_inputs)
                text_embedding = text_embedding.cpu().numpy()[0]
                
                # Average the embeddings for multimodal search
                combined_embedding = (image_embedding + text_embedding) / 2
                return combined_embedding.tolist()
            else:
                # Text-only query
                text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                text_embedding = clip_model.get_text_features(**text_inputs)
                return text_embedding.cpu().numpy()[0].tolist()
                
    except Exception as e:
        st.error(f"Error creating query embedding: {e}")
        return None

def search_similar_content(query_text, query_image, index, clip_model, clip_processor, device, top_k=8):
    """Search for similar content in Pinecone using multimodal embeddings"""
    # Generate query embedding
    query_embedding = create_query_embedding(query_text, query_image, clip_model, clip_processor, device)
    
    if query_embedding is None:
        return None
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def check_index_status(index):
    """Check if index has data"""
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count, stats
    except Exception as e:
        st.error(f"Error checking index status: {e}")
        return 0, None

def query_multimodal_llm(messages, model="google/gemini-2.5-flash-image-preview"):
    """Query a multimodal LLM via OpenRouter API with fallback models"""
    
    # List of multimodal models to try (in order of preference)
    # Based on your successful test results
    models_to_try = [
        "google/gemini-2.5-flash-image-preview",  # Best Google model from your test
        "openai/gpt-5-chat",                      # GPT-5 with vision capabilities
        "stepfun-ai/step3",                       # Fast and reliable
        "mistralai/mistral-medium-3.1",           # Good Mistral model
        "openai/gpt-5-nano",                      # Faster GPT-5 variant
        "z-ai/glm-4.5v",                          # Backup option
    ]
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Multimodal EV RAG Assistant"
    }
    
    # Try each model until one works
    for attempt, model_id in enumerate(models_to_try):
        try:
            data = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                if attempt > 0:  # Used fallback model
                    result = f"*[Using {model_id}]*\n\n{result}"
                return result
            else:
                error_msg = response.text
                if attempt < len(models_to_try) - 1:  # Not last attempt
                    continue
                else:
                    return f"All models failed. Last error: {response.status_code} - {error_msg}"
                    
        except Exception as e:
            if attempt < len(models_to_try) - 1:  # Not last attempt
                continue
            else:
                return f"Error querying multimodal LLMs: {str(e)}"

def multimodal_rag_query(user_question, user_image, index, clip_model, clip_processor, device):
    """Perform multimodal RAG query"""
    # 1. Search for relevant content
    search_results = search_similar_content(user_question, user_image, index, clip_model, clip_processor, device, top_k=8)
    
    if not search_results or not search_results.matches:
        vector_count, stats = check_index_status(index)
        if vector_count == 0:
            return """🚨 **Multimodal Database is Empty!**
            
Your Pinecone index has no data. To populate it:

1. **Delete your old index** and create the new multimodal one
2. **Start Kafka**: `docker-compose up -d`
3. **Run Enhanced Producer**: `python ingestion_scripts/producer.py` 
4. **Run Multimodal Consumer**: `python data_processor/consumer_and_embedder.py`
5. **Wait for articles AND images to be processed**

The new system processes both text and images from EV news!""", []
        else:
            return f"I searched through {vector_count:,} multimodal vectors but couldn't find relevant information about: '{user_question}'", []
    
    # 2. Extract context and images from search results
    text_contexts = []
    image_contexts = []
    sources = []
    
    for match in search_results.matches:
        metadata = match.metadata
        content_type = metadata.get('content_type', 'text')
        
        if content_type == 'text':
            text_contexts.append({
                'source': metadata.get('source', 'Unknown'),
                'title': metadata.get('title', 'Unknown'),
                'text': metadata.get('text', ''),
                'score': match.score
            })
        elif content_type == 'image':
            image_contexts.append({
                'source': metadata.get('source', 'Unknown'),
                'title': metadata.get('title', 'Unknown'),
                'image_data': metadata.get('image_data', ''),
                'image_index': metadata.get('image_index', 0),
                'text': metadata.get('text', ''),
                'score': match.score
            })
        
        # Collect unique sources
        source_info = {
            'title': metadata.get('title', 'Unknown'),
            'source': metadata.get('source', 'Unknown'),
            'url': metadata.get('url', ''),
            'content_type': content_type
        }
        if source_info not in sources:
            sources.append(source_info)
    
    # 3. Create multimodal prompt
    text_context = "\n---\n".join([
        f"Source: {ctx['source']}\nTitle: {ctx['title']}\nContent: {ctx['text']}"
        for ctx in text_contexts[:4]  # Limit text contexts
    ])
    
    # Prepare message with text context
    messages = [
        {
            "role": "system",
            "content": """You are an expert multimodal analyst for electric vehicle market intelligence. 
            You analyze both text articles and images related to EVs.
            Use the provided context to give comprehensive, accurate answers.
            When referring to images, describe what you see and how it relates to the question.
            Always mention your sources and be specific about whether information comes from text or images."""
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": f"""Based on the following EV market context, please answer the question: "{user_question}"

TEXT CONTEXT:
{text_context}

IMAGE CONTEXT:
{"Found " + str(len(image_contexts)) + " relevant images from EV articles" if image_contexts else "No relevant images found"}

Please provide a comprehensive answer using both text and image context where available."""
                }
            ]
        }
    ]
    
    # Add user's uploaded image if provided
    if user_image is not None:
        # Convert PIL image to base64
        buffered = io.BytesIO()
        user_image.save(buffered, format="PNG")
        user_img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{user_img_b64}"
            }
        })
    
    # Add retrieved images to the context
    for img_ctx in image_contexts[:2]:  # Limit to 2 images to avoid token limits
        if img_ctx['image_data']:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_ctx['image_data']}"
                }
            })
    
    # 4. Query the multimodal LLM
    response = query_multimodal_llm(messages)
    
    return response, sources

# --- Main App Interface ---
try:
    # Initialize components
    with st.spinner("🔄 Loading multimodal models..."):
        index, clip_model, clip_processor, device = get_components()
    
    st.success(f"✅ Connected to Pinecone and loaded CLIP model on {device}")
    
    # Check index status
    vector_count, stats = check_index_status(index)
    
    if vector_count > 0:
        st.info(f"📊 Database contains {vector_count:,} multimodal vectors")
    else:
        st.warning("⚠️ Database is empty. Please run the data ingestion pipeline.")
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask your question")
        user_question = st.text_input(
            "Question about Electric Vehicles:",
            placeholder="e.g., What are the latest Tesla charging innovations?"
        )
    
    with col2:
        st.subheader("🖼️ Upload an image (optional)")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image related to your question for multimodal analysis"
        )
    
    # Display uploaded image
    user_image = None
    if uploaded_file is not None:
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="Uploaded Image", width=300)
    
    # Query button
    if st.button("🔍 Search", type="primary", disabled=not user_question):
        if user_question:
            with st.spinner("🧠 Analyzing with multimodal AI..."):
                try:
                    response, sources = multimodal_rag_query(
                        user_question, 
                        user_image, 
                        index, 
                        clip_model, 
                        clip_processor, 
                        device
                    )
                    
                    # Display response
                    st.subheader("🤖 AI Response")
                    st.write(response)
                    
                    # Display sources
                    if sources:
                        st.subheader("📚 Sources")
                        for i, source in enumerate(sources[:5], 1):
                            with st.expander(f"Source {i}: {source['title'][:50]}..."):
                                st.write(f"**Source:** {source['source']}")
                                st.write(f"**Title:** {source['title']}")
                                st.write(f"**Type:** {source['content_type']}")
                                if source['url']:
                                    st.write(f"**URL:** {source['url']}")
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question!")
    
    # Display chat history
    if st.session_state.messages:
        st.subheader("💬 Chat History")
        for i, (q, r) in enumerate(st.session_state.messages[-3:], 1):  # Show last 3
            with st.expander(f"Query {i}: {q[:50]}..."):
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {r}")

except Exception as e:
    st.error(f"""❌ **Setup Error:** {str(e)}

**Possible solutions:**
1. **Check your .env file** with required API keys:
   - PINECONE_API_KEY
   - OPENROUTER_API_KEY

2. **Verify Pinecone index exists:**
   - Run `python setup_multimodal_index.py`

3. **Install missing packages:**
   - `pip install -r requirements.txt`

4. **Check if data pipeline is running:**
   - `docker-compose up -d`
   - `python ingestion_scripts/producer.py`
   - `python data_processor/consumer_and_embedder.py`
""")

# --- Sidebar with system info ---
with st.sidebar:
    st.header("🔧 System Status")
    
    # Model info
    st.subheader("🤖 Models")
    st.write("**Embedding:** CLIP ViT-Base-Patch32")
    st.write("**LLM:** Gemini 2.5 Flash (with 5 fallbacks)")
    st.write(f"**Device:** {device if 'device' in locals() else 'Unknown'}")
    
    # Show available models
    with st.expander("Available Models"):
        st.write("Primary: Google Gemini 2.5 Flash Image Preview")
        st.write("Fallback 1: OpenAI GPT-5 Chat")  
        st.write("Fallback 2: StepFun Step3")
        st.write("Fallback 3: Mistral Medium 3.1")
        st.write("+ 2 more fallbacks")
    
    # Check OpenRouter connectivity
    if st.button("🔍 Test OpenRouter Connection"):
        with st.spinner("Testing API..."):
            test_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            result = query_multimodal_llm(test_messages)
            if "Error" in result or "failed" in result.lower():
                st.error(f"❌ OpenRouter Error: {result}")
            else:
                st.success("✅ OpenRouter Connected")
                st.write(f"Response: {result[:100]}...")
    
    # Database info
    st.subheader("💾 Database")
    if 'vector_count' in locals():
        st.write(f"**Vectors:** {vector_count:,}")
        if 'stats' in locals() and stats:
            st.write(f"**Dimension:** {stats.dimension}")
    else:
        st.write("**Status:** Not connected")
    
    # Instructions
    st.subheader("📖 How to Use")
    st.write("""
    1. **Text Query:** Ask any EV-related question
    2. **Image Upload:** Add an image for multimodal analysis
    3. **Search:** Get AI-powered answers from news data
    4. **Sources:** Review the sources used for answers
    """)
    
    st.subheader("🚀 Setup Guide")
    with st.expander("First Time Setup"):
        st.code("""
# 1. Setup Pinecone index
python setup_multimodal_index.py

# 2. Start Kafka
docker-compose up -d

# 3. Run data pipeline
python ingestion_scripts/producer.py
python data_processor/consumer_and_embedder.py

# 4. Start app
streamlit run app.py
        """)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()