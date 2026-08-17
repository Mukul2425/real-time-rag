import streamlit as st
import os
from dotenv import load_dotenv
import requests
import base64
import io
import socket
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
from shared_runtime import MetricsStore, retry_with_backoff

# Load environment variables
load_dotenv()

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
metrics = MetricsStore(os.getenv("METRICS_DB_PATH", ".rag_state/metrics.sqlite"))

OPENROUTER_MODELS = [
    "google/gemini-2.5-flash-image-preview",
    "openai/gpt-5-chat",
    "stepfun-ai/step3",
    "mistralai/mistral-medium-3.1",
    "openai/gpt-5-nano",
    "z-ai/glm-4.5v",
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

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
        def _model_output_to_numpy(output):
            # Same robust conversion as used in consumer
            if hasattr(output, 'cpu'):
                tensor = output
            elif hasattr(output, 'pooler_output'):
                tensor = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                tensor = output.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Unsupported model output type for embedding conversion")
            return tensor.detach().cpu().numpy()[0]

        with torch.no_grad():
            if query_image is not None and clip_model is not None:
                # Multimodal query: combine text and image
                # Process image
                image_inputs = clip_processor(images=[query_image], return_tensors="pt", padding=True)
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                image_out = clip_model.get_image_features(**image_inputs)
                image_embedding = _model_output_to_numpy(image_out)
                
                # Process text
                text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                text_out = clip_model.get_text_features(**text_inputs)
                text_embedding = _model_output_to_numpy(text_out)
                
                # Average the embeddings for multimodal search
                combined_embedding = (image_embedding + text_embedding) / 2
                return combined_embedding.tolist()
            else:
                # Text-only query
                text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                text_out = clip_model.get_text_features(**text_inputs)
                text_embedding = _model_output_to_numpy(text_out)
                return text_embedding.tolist()
                
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


def check_kafka_reachable(host="localhost", port=9092, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def load_pipeline_metrics():
    try:
        return metrics.all()
    except Exception:
        return {}

def _infer_mime_type(data_url):
    if data_url.startswith("data:") and ";base64," in data_url:
        return data_url.split(";base64,", 1)[0].replace("data:", "") or "image/png"
    return "image/png"


def _extract_base64_from_data_url(data_url):
    if data_url.startswith("data:") and ";base64," in data_url:
        return data_url.split(";base64,", 1)[1]
    return data_url


def _image_to_data_url(image_path: str) -> str | None:
    try:
        path = image_path.strip()
        if not path:
            return None
        with open(path, "rb") as handle:
            image_bytes = handle.read()
        mime_type = "image/png"
        lower_path = path.lower()
        if lower_path.endswith(".jpg") or lower_path.endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif lower_path.endswith(".webp"):
            mime_type = "image/webp"
        elif lower_path.endswith(".gif"):
            mime_type = "image/gif"
        return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
    except Exception:
        return None


def _messages_to_gemini_payload(messages):
    system_instruction = ""
    user_message = None
    for message in messages:
        if message.get("role") == "system":
            system_instruction = message.get("content", "")
        elif message.get("role") == "user":
            user_message = message

    parts = []
    if user_message:
        for item in user_message.get("content", []):
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                parts.append({
                    "inline_data": {
                        "mime_type": _infer_mime_type(image_url),
                        "data": _extract_base64_from_data_url(image_url),
                    }
                })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": 1000,
            "temperature": 0.7,
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    return payload


@retry_with_backoff(attempts=3, base_delay=1.5, max_delay=8.0)
def _post_json(url, headers, payload, timeout=60):
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 500 or response.status_code == 429:
        raise requests.RequestException(f"Transient LLM error {response.status_code}: {response.text}")
    return response


def _call_openrouter(messages, model=None):
    models_to_try = [model] if model else []
    for candidate in OPENROUTER_MODELS:
        if candidate not in models_to_try:
            models_to_try.append(candidate)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Multimodal EV RAG Assistant",
    }

    for attempt, model_id in enumerate(models_to_try):
        try:
            response = _post_json(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                payload={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                if attempt > 0:
                    result = f"*[Using {model_id}]*\n\n{result}"
                return result
            error_msg = response.text
            if attempt == len(models_to_try) - 1:
                return f"All OpenRouter models failed. Last error: {response.status_code} - {error_msg}"
        except Exception as e:
            if attempt == len(models_to_try) - 1:
                return f"Error querying OpenRouter: {str(e)}"


def _call_gemini(messages, model=None):
    models_to_try = [model] if model else []
    for candidate in GEMINI_MODELS:
        if candidate not in models_to_try:
            models_to_try.append(candidate)

    payload = _messages_to_gemini_payload(messages)
    for attempt, model_id in enumerate(models_to_try):
        try:
            response = _post_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                payload=payload,
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                    if attempt > 0:
                        text = f"*[Using {model_id}]*\n\n{text}"
                    return text or "Gemini returned an empty response."
                return "Gemini returned no candidates."
            error_msg = response.text
            if attempt == len(models_to_try) - 1:
                return f"All Gemini models failed. Last error: {response.status_code} - {error_msg}"
        except Exception as e:
            if attempt == len(models_to_try) - 1:
                return f"Error querying Gemini: {str(e)}"


def query_multimodal_llm(messages, provider="auto", model=None):
    """Query a multimodal LLM through OpenRouter or Gemini direct API."""
    provider = (provider or DEFAULT_LLM_PROVIDER or "auto").lower()
    if model and str(model).lower() == "auto":
        model = None
    model = model or DEFAULT_LLM_MODEL or None

    if provider == "auto":
        if GEMINI_API_KEY:
            provider = "gemini"
        elif OPENROUTER_API_KEY:
            provider = "openrouter"
        else:
            return "No LLM API key found. Set GEMINI_API_KEY or OPENROUTER_API_KEY in your .env file."

    if provider == "gemini":
        if not GEMINI_API_KEY:
            return "GEMINI_API_KEY is missing from your .env file."
        return _call_gemini(messages, model=model)

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            return "OPENROUTER_API_KEY is missing from your .env file."
        return _call_openrouter(messages, model=model)

    return f"Unsupported LLM provider: {provider}"

def multimodal_rag_query(user_question, user_image, index, clip_model, clip_processor, device, provider="auto", model=None):
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
                'image_path': metadata.get('image_path', ''),
                'image_url': metadata.get('image_url', ''),
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
        # If the index stored a local image path, load it for the LLM.
        image_path = img_ctx.get('image_path')
        if image_path:
            data_url = _image_to_data_url(image_path)
            if data_url:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
                continue

        # Backward compatibility: fall back to remote URLs if present.
        img_url = img_ctx.get('image_url')
        if img_url:
            try:
                resp = requests.get(img_url, timeout=8)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode()
                ctype = resp.headers.get('Content-Type', 'image/png')
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{ctype};base64,{b64}"}
                })
            except Exception:
                continue
    
    # 4. Query the multimodal LLM
    response = query_multimodal_llm(messages, provider=provider, model=model)
    
    return response, sources

# --- Main App Interface ---
try:
    # Initialize components
    with st.spinner("🔄 Loading multimodal models..."):
        index, clip_model, clip_processor, device = get_components()
    
    st.success(f"✅ Connected to Pinecone and loaded CLIP model on {device}")
    
    # Check index status
    vector_count, stats = check_index_status(index)
    pipeline_metrics = load_pipeline_metrics()
    kafka_ok = check_kafka_reachable()
    producer_count = pipeline_metrics.get("producer_articles_enqueued_total", 0)
    consumer_count = pipeline_metrics.get("consumer_articles_processed_total", 0)
    backlog_estimate = max(0, int(producer_count - consumer_count))
    
    if vector_count > 0:
        st.info(f"📊 Database contains {vector_count:,} multimodal vectors")
    else:
        st.warning("⚠️ Database is empty. Please run the data ingestion pipeline.")

    st.subheader("📈 Pipeline Health")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        st.metric("Kafka", "Healthy" if kafka_ok else "Down")
    with status_col2:
        st.metric("Producer Articles", f"{int(producer_count):,}")
    with status_col3:
        st.metric("Consumer Articles", f"{int(consumer_count):,}")
    with status_col4:
        st.metric("Estimated Backlog", f"{backlog_estimate:,}")

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.caption(f"Image downloads failed: {int(pipeline_metrics.get('consumer_image_download_failures_total', 0)):,}")
    with metrics_col2:
        st.caption(f"Pinecone upserts: {int(pipeline_metrics.get('consumer_vectors_upserted_total', 0)):,}")
    with metrics_col3:
        st.caption(f"Dead letters: {int(pipeline_metrics.get('consumer_dead_letter_total', 0)):,}")
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])

    provider_options = ["auto", "gemini", "openrouter"]
    model_options_map = {
        "auto": [DEFAULT_LLM_MODEL or "auto"],
        "gemini": GEMINI_MODELS,
        "openrouter": OPENROUTER_MODELS,
    }
    
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

    with st.sidebar:
        st.subheader("🎛️ LLM Options")
        provider_choice = st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(DEFAULT_LLM_PROVIDER) if DEFAULT_LLM_PROVIDER in provider_options else 0,
            help="Choose Gemini direct API or OpenRouter. Auto selects an available key.",
        )

        model_options = model_options_map.get(provider_choice, OPENROUTER_MODELS)
        default_model = DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in model_options else model_options[0]
        model_choice = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(default_model),
            help="Pick the model family for the selected provider.",
        )
    
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
                        device,
                        provider=provider_choice,
                        model=model_choice,
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
    st.write(f"**LLM Provider:** {provider_choice}")
    st.write(f"**LLM Model:** {model_choice}")
    st.write(f"**Device:** {device if 'device' in locals() else 'Unknown'}")
    
    # Show available models
    with st.expander("Available Models"):
        st.write("Gemini direct: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash")
        st.write("OpenRouter: google/gemini-2.5-flash-image-preview, openai/gpt-5-chat, stepfun-ai/step3, mistralai/mistral-medium-3.1, openai/gpt-5-nano, z-ai/glm-4.5v")
    
    # Check LLM connectivity
    if st.button("🔍 Test LLM Connection"):
        with st.spinner("Testing API..."):
            test_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            result = query_multimodal_llm(test_messages, provider=provider_choice, model=model_choice)
            if "Error" in result or "failed" in result.lower():
                st.error(f"❌ LLM Error: {result}")
            else:
                st.success("✅ LLM Connected")
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