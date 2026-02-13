.PHONY: lint format test test-cov run docker-build clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run all linters (black, flake8, isort)
	black --check src/ tests/ main.py
	flake8 src/ tests/ main.py
	isort --check-only --diff src/ tests/ main.py

format: ## Auto-format code with black and isort
	black src/ tests/ main.py
	isort src/ tests/ main.py

test: ## Run tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

run: ## Run analysis on sample video
	python main.py --input data/input_videos/match.mp4 --use-stubs

api: ## Start the FastAPI server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build Docker image
	docker compose -f docker/docker-compose.yml build

clean: ## Remove cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage
