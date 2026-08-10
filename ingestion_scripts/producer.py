import os
import requests
import time
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib

# Load environment variables from .env file
load_dotenv()

# Get your News API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# Poll interval in seconds between batches (env override)
NEWS_POLL_INTERVAL = int(os.getenv("NEWS_POLL_INTERVAL", "600"))

def extract_image_urls(article):
    """Extract image URLs from article metadata and HTML (robust approach)

    Strategy:
    - Prefer `urlToImage` from News API
    - If not available or to augment, fetch the article page and look for
      OpenGraph `og:image`, `twitter:image`, and first few `<img>` tags.
    - Validate common image extensions and return unique URLs (max 3).
    """
    image_urls = []

    # 1. Prefer News API primary image
    url_to_image = article.get('urlToImage')
    if url_to_image and is_valid_image_url(url_to_image):
        image_urls.append(url_to_image)

    # 2. Try to fetch the article HTML and parse for image tags / OG tags
    article_url = article.get('url')
    if article_url:
        try:
            resp = requests.get(article_url, timeout=6)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, 'html.parser')

            # OpenGraph / Twitter cards
            og = soup.find('meta', property='og:image') or soup.find('meta', attrs={'name': 'og:image'})
            if og and og.get('content') and is_valid_image_url(og.get('content')):
                image_urls.append(og.get('content'))

            tw = soup.find('meta', property='twitter:image') or soup.find('meta', attrs={'name': 'twitter:image'})
            if tw and tw.get('content') and is_valid_image_url(tw.get('content')):
                image_urls.append(tw.get('content'))

            # First few <img> tags (prefer images with width/height attributes or large src)
            imgs = soup.find_all('img', src=True)
            for img in imgs[:6]:
                src = img.get('src')
                # Resolve relative URLs if needed
                if src and src.startswith('//'):
                    src = 'https:' + src
                if src and src.startswith('/') and article_url:
                    # build absolute
                    parsed = urlparse(article_url)
                    src = f"{parsed.scheme}://{parsed.netloc}{src}"
                if src and is_valid_image_url(src):
                    image_urls.append(src)

        except Exception:
            # Ignore HTML fetch failures — we still return what we have
            pass

    # De-duplicate while preserving order and validate extensions
    seen = set()
    unique = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    for u in image_urls:
        if not u:
            continue
        low = u.split('?')[0].lower()
        if any(low.endswith(ext) for ext in valid_extensions) and u not in seen:
            seen.add(u)
            unique.append(u)

    return unique[:3]


def is_valid_image_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        # Quick extension check
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        return any(url.lower().split('?')[0].endswith(ext) for ext in valid_extensions)
    except Exception:
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