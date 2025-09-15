import os
import requests
import time
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
import re
from urllib.parse import urljoin, urlparse

# Load environment variables from .env file
load_dotenv()

# Get your News API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def extract_image_urls(article):
    """Extract image URLs from article content and metadata"""
    image_urls = []
    
    # 1. Get the main article image (urlToImage from News API)
    if article.get('urlToImage'):
        image_urls.append(article['urlToImage'])
    
    # 2. Extract images from article content (if any)
    content = article.get('content', '') or ''
    description = article.get('description', '') or ''
    
    # Look for image URLs in content using regex
    img_pattern = r'https?://[^\s<>"]{1,}\.(jpg|jpeg|png|gif|webp)'
    
    # Find images in content and description
    content_images = re.findall(img_pattern, content, re.IGNORECASE)
    desc_images = re.findall(img_pattern, description, re.IGNORECASE)
    
    # Add found images (just the full URL, not the extension tuple)
    for match in content_images:
        if isinstance(match, tuple):
            # re.findall returns tuples when there are groups, get the full match
            full_url = match[0] if len(match) > 1 else match
        else:
            full_url = match
        
        # Reconstruct full URL if needed
        full_match = re.search(r'https?://[^\s<>"]{1,}\.' + (match[1] if isinstance(match, tuple) else 'jpg|jpeg|png|gif|webp'), content)
        if full_match:
            image_urls.append(full_match.group(0))
    
    # Remove duplicates and invalid URLs
    unique_urls = []
    for url in image_urls:
        if url and url not in unique_urls and is_valid_image_url(url):
            unique_urls.append(url)
    
    return unique_urls[:3]  # Limit to 3 images per article

def is_valid_image_url(url):
    """Check if URL is a valid image URL"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        return any(url.lower().endswith(ext) for ext in valid_extensions)
    except:
        return False

def create_kafka_producer(max_retries=5):
    """Create Kafka producer with retry logic"""
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=10000,  # 10 seconds timeout
                retries=3
            )
            print("Successfully connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            print(f"Attempt {attempt + 1}/{max_retries}: Kafka not available, waiting 10 seconds...")
            time.sleep(10)
    
    raise Exception("Failed to connect to Kafka after multiple attempts")

def get_news_and_send_to_kafka(producer, query):
    """Fetches news from News API and sends it to a Kafka topic with image URLs."""
    print(f"Fetching news for query: '{query}'")
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=20'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raises an HTTPError for bad responses
        articles = response.json().get('articles', [])
        
        if not articles:
            print("No new articles found.")
            return
        
        articles_with_images = 0
        total_articles = 0
        
        for article in articles:
            # We only care about articles with a title and content
            if article.get('title') and article.get('content'):
                total_articles += 1
                
                # Extract image URLs from the article
                image_urls = extract_image_urls(article)
                
                # Add image URLs to the article data
                enhanced_article = {
                    **article,  # Include all original article data
                    'image_urls': image_urls,  # Add extracted image URLs
                    'has_images': len(image_urls) > 0,  # Flag for easier filtering
                    'image_count': len(image_urls)
                }
                
                if image_urls:
                    articles_with_images += 1
                    print(f"📸 Article with {len(image_urls)} images: {article['title'][:60]}...")
                    for img_url in image_urls:
                        print(f"   - {img_url}")
                else:
                    print(f"📄 Text-only article: {article['title'][:60]}...")
                
                # Send the enhanced article as a JSON message to Kafka
                producer.send('market-news-raw', value=enhanced_article)
        
        # Flush to ensure all messages are sent
        producer.flush()
        
        print(f"\n📊 Summary:")
        print(f"   - Total articles sent: {total_articles}")
        print(f"   - Articles with images: {articles_with_images}")
        print(f"   - Text-only articles: {total_articles - articles_with_images}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")

if __name__ == "__main__":
    print("🚀 Initializing Multimodal Kafka Producer...")
    print("This version extracts both text and images from news articles")
    
    try:
        # Create producer with retry logic
        producer = create_kafka_producer()
        
        # We'll monitor the electric vehicle market as our example
        search_queries = [
            "electric vehicle",
            "Tesla",
            "EV charging",
            "battery technology"
        ]
        
        # The script will run continuously to mimic real-time ingestion
        while True:
            for query in search_queries:
                print(f"\n{'='*60}")
                get_news_and_send_to_kafka(producer, query)
                time.sleep(30)  # Wait between queries
                
            print(f"\n💤 Sleeping for 10 minutes before next batch...")
            time.sleep(600)  # Sleep for 10 minutes (600 seconds)
            
    except KeyboardInterrupt:
        print("Shutting down producer...")
        if 'producer' in locals():
            producer.close()
    except Exception as e:
        print(f"Error: {e}")
        if 'producer' in locals():
            producer.close()