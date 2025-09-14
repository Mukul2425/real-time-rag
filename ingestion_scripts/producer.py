import os
import requests
import time
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json

# Load environment variables from .env file
load_dotenv()

# Get your News API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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
    """Fetches news from News API and sends it to a Kafka topic."""
    print(f"Fetching news for query: '{query}'")
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raises an HTTPError for bad responses
        articles = response.json().get('articles', [])
        
        if not articles:
            print("No new articles found.")
            return
        
        for article in articles:
            # We only care about articles with a title and content
            if article.get('title') and article.get('content'):
                # Send the article as a JSON message to Kafka
                producer.send('market-news-raw', value=article)
                print(f"Sent article: {article['title']}")
        
        # Flush to ensure all messages are sent
        producer.flush()
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
    
if __name__ == "__main__":
    print("Initializing Kafka producer...")
    
    try:
        # Create producer with retry logic
        producer = create_kafka_producer()
        
        # We'll monitor the electric vehicle market as our example
        search_query = "electric vehicle"
        
        # The script will run continuously to mimic real-time ingestion
        while True:
            get_news_and_send_to_kafka(producer, search_query)
            print("Sleeping for 15 minutes before next fetch...")
            time.sleep(900) # Sleep for 15 minutes (900 seconds)
            
    except KeyboardInterrupt:
        print("Shutting down producer...")
        if 'producer' in locals():
            producer.close()
    except Exception as e:
        print(f"Error: {e}")
        if 'producer' in locals():
            producer.close()