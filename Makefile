.PHONY: help setup validate start-kafka stop-kafka start-producer start-consumer start-app clean

help: ## Show this help message
	@echo "Multimodal EV RAG Assistant - Available Commands"
	@echo "================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and setup environment
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"
	@echo ""
	@echo "📝 Next steps:"
	@echo "  1. Copy .env.example to .env and add your API keys"
	@echo "  2. Run 'make validate' to check your setup"
	@echo "  3. Run 'make setup-pinecone' to create the Pinecone index"

validate: ## Validate environment setup
	@echo "🔍 Validating environment..."
	python validate_setup.py

setup-pinecone: ## Setup Pinecone index
	@echo "🔧 Setting up Pinecone index..."
	python setup_multimodal_index.py

start-kafka: ## Start Kafka using Docker Compose
	@echo "🚀 Starting Kafka..."
	docker-compose up -d
	@echo "⏳ Waiting for Kafka to be ready..."
	@sleep 5
	@docker-compose ps

stop-kafka: ## Stop Kafka containers
	@echo "🛑 Stopping Kafka..."
	docker-compose down

kafka-logs: ## Show Kafka logs
	docker-compose logs -f kafka

start-producer: ## Start the news producer
	@echo "📰 Starting news producer..."
	@echo "Press Ctrl+C to stop"
	python ingestion_scripts/producer.py

start-consumer: ## Start the consumer and embedder
	@echo "🔄 Starting consumer and embedder..."
	@echo "Press Ctrl+C to stop"
	python data_processor/consumer_and_embedder.py

start-app: ## Start the Streamlit application
	@echo "🎨 Starting Streamlit app..."
	@echo "App will open at http://localhost:8501"
	streamlit run app.py

start-all: ## Start complete pipeline (requires multiple terminals)
	@echo "⚠️  This requires multiple terminal windows!"
	@echo ""
	@echo "Terminal 1: make start-producer"
	@echo "Terminal 2: make start-consumer"
	@echo "Terminal 3: make start-app"
	@echo ""
	@echo "Or run manually:"
	@echo "  1. Start Kafka: make start-kafka"
	@echo "  2. Start producer in one terminal"
	@echo "  3. Start consumer in another terminal"
	@echo "  4. Start app in a third terminal"

status: ## Check status of all services
	@echo "📊 System Status"
	@echo "================"
	@echo ""
	@echo "Docker Containers:"
	@docker-compose ps 2>/dev/null || echo "Docker Compose not running"
	@echo ""
	@echo "Python Environment:"
	@python --version
	@echo ""
	@echo "Disk Space:"
	@df -h . | tail -1

clean: ## Clean up temporary files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .cache 2>/dev/null || true
	rm -rf tmp 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-all: clean stop-kafka ## Clean everything including Docker volumes
	@echo "🧹 Deep cleaning..."
	docker-compose down -v 2>/dev/null || true
	@echo "✅ Deep cleanup complete"

test-connections: ## Test API connections
	@echo "🔌 Testing API connections..."
	@python validate_setup.py --quick 2>/dev/null || echo "Run 'make validate' for comprehensive checks"

.DEFAULT_GOAL := help
