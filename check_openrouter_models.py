import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

def get_available_models():
    """Get list of available models from OpenRouter"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models = response.json()["data"]
            return models
        else:
            print(f"Error fetching models: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def find_multimodal_models(models):
    """Filter models that support vision/multimodal capabilities"""
    multimodal_models = []
    
    keywords = ['vision', 'multimodal', 'gemini', 'gpt-4', 'claude-3', 'gpt-4o']
    
    for model in models:
        model_id = model.get("id", "")
        model_name = model.get("name", "")
        description = model.get("description", "")
        
        # Check if model likely supports vision
        is_multimodal = any(keyword in model_id.lower() or 
                          keyword in model_name.lower() or
                          keyword in description.lower() 
                          for keyword in keywords)
        
        if is_multimodal:
            multimodal_models.append({
                "id": model_id,
                "name": model_name,
                "pricing": model.get("pricing", {}),
                "context_length": model.get("context_length", "Unknown")
            })
    
    return multimodal_models

def test_model(model_id):
    """Test if a specific model works"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Model Test"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, can you see images?"
                    }
                ]
            }
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()["choices"][0]["message"]["content"]
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Checking OpenRouter Models for Multimodal Capabilities")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        return
    
    print("✅ API key found")
    
    # Get all models
    print("\n📡 Fetching available models...")
    models = get_available_models()
    
    if not models:
        print("❌ Failed to fetch models")
        return
    
    print(f"✅ Found {len(models)} total models")
    
    # Filter multimodal models
    print("\n🔍 Filtering multimodal models...")
    multimodal_models = find_multimodal_models(models)
    
    print(f"✅ Found {len(multimodal_models)} potential multimodal models")
    
    # Display results
    print(f"\n📋 Multimodal Models Available:")
    print("-" * 60)
    
    working_models = []
    
    for i, model in enumerate(multimodal_models[:10], 1):  # Test first 10
        model_id = model["id"]
        print(f"\n{i}. {model_id}")
        print(f"   Name: {model['name']}")
        print(f"   Context: {model['context_length']}")
        
        # Test the model
        print(f"   Testing...", end=" ")
        works, response = test_model(model_id)
        
        if works:
            print("✅ WORKS")
            print(f"   Response: {response[:100]}...")
            working_models.append(model_id)
        else:
            print("❌ Failed")
            print(f"   Error: {response[:100]}...")
    
    # Summary
    print(f"\n🎉 SUMMARY")
    print("=" * 60)
    print(f"Working multimodal models:")
    
    if working_models:
        for i, model_id in enumerate(working_models, 1):
            print(f"{i}. {model_id}")
        
        print(f"\n💡 Recommended models for your app.py:")
        print("Update the models_to_try list with:")
        print(f"models_to_try = {working_models}")
    else:
        print("❌ No working multimodal models found")
        print("\nTry these common model IDs:")
        print("- google/gemini-flash-1.5")
        print("- google/gemini-pro-1.5")
        print("- openai/gpt-4o-mini")
        print("- openai/gpt-4-vision-preview")
        print("- anthropic/claude-3-haiku")

if __name__ == "__main__":
    main()