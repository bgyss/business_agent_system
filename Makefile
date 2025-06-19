# Business Agent System - Development Makefile

.PHONY: help dev-setup install clean test test-web test-smoke lint format type-check build run-restaurant run-retail dashboard

# Default target
help:
	@echo "Business Agent Management System - Development Commands"
	@echo "======================================================="
	@echo ""
	@echo "Setup:"
	@echo "  dev-setup    - Set up development environment with Nix and uv"
	@echo "  install      - Install dependencies with uv"
	@echo "  clean        - Clean build artifacts and caches"
	@echo ""
	@echo "Development:"
	@echo "  test         - Run all tests"
	@echo "  test-web     - Run web/UI tests for dashboard"
	@echo "  test-smoke   - Run quick smoke tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run mypy type checking"
	@echo ""
	@echo "Build & Run:"
	@echo "  build        - Build the package with Nix"
	@echo "  run-restaurant - Run with restaurant configuration"
	@echo "  run-restaurant-fast - Run 5-minute restaurant simulation at 3x speed"
	@echo "  run-retail   - Run with retail configuration"
	@echo "  dashboard    - Launch the Streamlit dashboard"
	@echo ""
	@echo "Nix Commands:"
	@echo "  nix develop  - Enter development shell"
	@echo "  nix build    - Build the package"
	@echo "  nix run      - Run the default application"

# Development environment setup
dev-setup:
	@echo "🔧 Setting up development environment..."
	@if ! command -v nix >/dev/null 2>&1; then \
		echo "❌ Nix is not installed. Please install Nix first."; \
		echo "Visit: https://nixos.org/download.html"; \
		exit 1; \
	fi
	@if ! command -v direnv >/dev/null 2>&1; then \
		echo "📦 Installing direnv..."; \
		nix profile install nixpkgs#direnv; \
	fi
	@echo "🔄 Allowing direnv for this directory..."
	@direnv allow
	@echo "📋 Copying environment template..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "✅ Development environment ready!"
	@echo "💡 Edit .env and add your ANTHROPIC_API_KEY"

# Install dependencies
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync --all-extras

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .uv-cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Run tests
test:
	@echo "🧪 Running tests..."
	uv run pytest -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	uv run pytest --cov --cov-report=html --cov-report=term

# Web testing
test-web:
	@echo "🌐 Running web tests for dashboard..."
	@echo "📋 Ensuring required directories exist..."
	@mkdir -p test_screenshots test_artifacts
	uv run pytest tests/test_dashboard_web.py -v --html=test_artifacts/web_test_report.html --self-contained-html

test-smoke:
	@echo "💨 Running smoke tests..."
	@mkdir -p test_screenshots test_artifacts
	uv run pytest tests/test_dashboard_smoke.py -v --html=test_artifacts/smoke_test_report.html --self-contained-html

test-web-headless:
	@echo "🌐 Running web tests in headless mode..."
	@mkdir -p test_screenshots test_artifacts
	uv run pytest tests/test_dashboard_web.py -v --html=test_artifacts/web_test_report.html --self-contained-html -m "not slow"

test-web-full:
	@echo "🌐 Running full web test suite..."
	@mkdir -p test_screenshots test_artifacts
	uv run pytest tests/test_dashboard_web.py tests/test_dashboard_smoke.py -v --html=test_artifacts/full_web_test_report.html --self-contained-html

# Linting
lint:
	@echo "🔍 Running linting checks..."
	uv run ruff check .
	uv run flake8 agents/ models/ simulation/ dashboard/

# Code formatting
format:
	@echo "🎨 Formatting code..."
	uv run black .
	uv run isort .
	uv run ruff check --fix .

# Type checking
type-check:
	@echo "🔍 Running type checks..."
	uv run mypy agents/ models/ simulation/

# Build package
build:
	@echo "🔨 Building package with Nix..."
	nix build

# Run applications
run-restaurant:
	@echo "🍽️ Starting restaurant business system..."
	uv run python main.py --config config/restaurant_config.yaml

run-restaurant-fast:
	@echo "🍽️ Starting restaurant business system (5 min, 3x speed)..."
	uv run python main.py --config config/restaurant_fast_test.yaml

run-retail:
	@echo "🛍️ Starting retail business system..."
	uv run python main.py --config config/retail_config.yaml

# Launch dashboard
dashboard:
	@echo "📊 Launching dashboard..."
	uv run streamlit run dashboard/app.py

# Generate historical data
generate-data-restaurant:
	@echo "📈 Generating 90 days of restaurant data..."
	uv run python main.py --config config/restaurant_config.yaml --generate-historical 90

generate-data-retail:
	@echo "📈 Generating 90 days of retail data..."
	uv run python main.py --config config/retail_config.yaml --generate-historical 90

# Development checks (run before committing)
check: lint type-check test
	@echo "✅ All checks passed!"

# CI pipeline (comprehensive checks)
ci: clean install check test-cov
	@echo "🚀 CI pipeline completed successfully!"

# Documentation
docs:
	@echo "📚 Building documentation..."
	@echo "Documentation generation not yet implemented"

# Docker-related targets (if using Docker alongside Nix)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t business-agent-system .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm -p 8501:8501 business-agent-system

# Nix-specific targets
nix-shell:
	@echo "🐚 Entering Nix development shell..."
	nix develop

nix-build:
	@echo "🔨 Building with Nix..."
	nix build

nix-run:
	@echo "🚀 Running with Nix..."
	nix run

nix-check:
	@echo "🔍 Running Nix checks..."
	nix flake check

# Update dependencies
update:
	@echo "⬆️ Updating dependencies..."
	uv sync --upgrade
	nix flake update

# Show system status
status:
	@echo "📊 System Status:"
	@echo "=================="
	@echo "Python: $$(python --version 2>/dev/null || echo 'Not available')"
	@echo "uv: $$(uv --version 2>/dev/null || echo 'Not available')"
	@echo "Nix: $$(nix --version 2>/dev/null || echo 'Not available')"
	@echo "Virtual env: $$(if [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV"; else echo "Not activated"; fi)"
	@echo "Environment: $$(if [ -f .env ]; then echo "✅ .env exists"; else echo "❌ .env missing"; fi)"
	@echo "Dependencies: $$(if [ -d .venv ]; then echo "✅ Installed"; else echo "❌ Run 'make install'"; fi)"