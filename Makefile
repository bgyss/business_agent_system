# Business Agent System - Development Makefile

.PHONY: help help-setup help-dev help-quality help-ci help-build help-fix help-validate help-watch
.PHONY: dev-setup install clean test test-unit test-integration test-integration-smoke test-integration-file
.PHONY: test-web test-smoke test-performance perf-test perf-test-quick perf-report lint format type-check 
.PHONY: build run-restaurant run-retail dashboard ci-check status watch
.PHONY: quality-gate quality-gate-strict quality-gate-quick quality-gate-emergency validate-file
.PHONY: format-check lint-check type-check-strict security-check complexity-check docstring-check
.PHONY: fix-all fix-file fix-format fix-imports fix-lint fix-docstrings fix-auto
.PHONY: validate-file-with-tests validate-changed-files validate-branch validate-env
.PHONY: pre-commit pre-commit-install pre-commit-update pre-commit-clean pre-commit-autoupdate
.PHONY: watch-quality watch-tests watch-all setup-ide validate-ide test-discovery
.PHONY: env-check env-validate env-setup env-clean deps-check deps-update deps-audit
.PHONY: ci-simulate ci-validate-all ci-test-local branch-validate github-validate

# Default target - show comprehensive help
help:
	@echo "Business Agent Management System - Development Commands"
	@echo "======================================================="
	@echo ""
	@echo "📋 Quick Help Categories:"
	@echo "  help-setup   - Environment setup and dependencies"
	@echo "  help-dev     - Development and testing workflows"
	@echo "  help-quality - Quality gates and validation tools"
	@echo "  help-fix     - Auto-fix and formatting commands"
	@echo "  help-ci      - CI/CD and automation tools"
	@echo "  help-build   - Build and deployment commands"
	@echo ""
	@echo "🚀 Most Common Commands:"
	@echo "  dev-setup         - Set up complete development environment"
	@echo "  quality-gate      - Run standard quality checks before commit"
	@echo "  fix-all           - Auto-fix all formatting and linting issues"
	@echo "  test              - Run full test suite"
	@echo "  watch             - Start continuous validation mode"
	@echo "  validate-file FILE=path - Validate specific file"
	@echo ""
	@echo "💡 For detailed help on any category, run: make help-<category>"
	@echo "   Example: make help-quality"

# Detailed help sections
help-setup:
	@echo "🔧 Environment Setup Commands:"
	@echo "==============================================="
	@echo "  dev-setup           - Complete development environment setup"
	@echo "  install             - Install all dependencies with uv"
	@echo "  setup-ide           - Configure and validate IDE integration"
	@echo "  env-setup           - Set up environment variables and config"
	@echo "  env-validate        - Validate environment configuration"
	@echo "  pre-commit-install  - Install pre-commit hooks"
	@echo "  deps-check          - Check dependency status and conflicts"
	@echo "  deps-update         - Update all dependencies safely"
	@echo "  clean               - Clean all build artifacts and caches"

help-dev:
	@echo "🧪 Development & Testing Commands:"
	@echo "============================================="
	@echo "Testing:"
	@echo "  test                - Run complete test suite"
	@echo "  test-unit           - Run unit tests only"
	@echo "  test-integration    - Run integration tests"
	@echo "  test-web            - Run web/dashboard tests"
	@echo "  test-smoke          - Run quick smoke tests"
	@echo "  test-cov            - Run tests with coverage report"
	@echo ""
	@echo "Performance:"
	@echo "  perf-test           - Comprehensive performance testing"
	@echo "  perf-test-quick     - Quick performance checks"
	@echo "  perf-report         - Generate performance reports"
	@echo ""
	@echo "Development:"
	@echo "  watch               - Continuous validation (all files)"
	@echo "  watch-quality       - Watch mode for quality checks only"
	@echo "  watch-tests         - Watch mode for tests only"
	@echo "  test-discovery      - Discover and validate test structure"

help-quality:
	@echo "🚪 Quality Gates & Validation Commands:"
	@echo "================================================"
	@echo "Quality Gates (comprehensive checks):"
	@echo "  quality-gate             - Standard quality gate (normal mode)"
	@echo "  quality-gate-strict      - Strict quality gate (CI/production)"
	@echo "  quality-gate-quick       - Quick quality gate (development)"
	@echo "  quality-gate-emergency   - Minimal emergency quality gate"
	@echo ""
	@echo "Individual Checks:"
	@echo "  format-check        - Check code formatting (black, isort)"
	@echo "  lint-check          - Check linting (ruff, flake8)"
	@echo "  type-check          - Check types (mypy)"
	@echo "  type-check-strict   - Strict type checking with extra rules"
	@echo "  security-check      - Security vulnerability scanning"
	@echo "  complexity-check    - Code complexity analysis"
	@echo "  docstring-check     - Documentation string validation"
	@echo ""
	@echo "File Validation:"
	@echo "  validate-file FILE=path     - Validate specific file"
	@echo "  validate-file-with-tests    - Validate file and run related tests"
	@echo "  validate-changed-files      - Validate only changed files (git)"
	@echo "  validate-branch             - Validate entire branch changes"

help-fix:
	@echo "🔧 Auto-fix & Formatting Commands:"
	@echo "==========================================="
	@echo "Auto-fix All:"
	@echo "  fix-all             - Run all auto-fixes (format, imports, lint)"
	@echo "  fix-auto            - Smart auto-fix based on detected issues"
	@echo ""
	@echo "Specific Fixes:"
	@echo "  fix-format          - Auto-fix formatting (black, isort)"
	@echo "  fix-imports         - Auto-fix import sorting and unused imports"
	@echo "  fix-lint            - Auto-fix linting issues (ruff --fix)"
	@echo "  fix-docstrings      - Auto-fix docstring formatting"
	@echo "  fix-file FILE=path  - Auto-fix specific file"
	@echo ""
	@echo "Manual Formatting:"
	@echo "  format              - Format code (non-destructive preview)"
	@echo "  lint                - Show linting issues without fixing"

help-ci:
	@echo "🚀 CI/CD & Automation Commands:"
	@echo "====================================="
	@echo "CI Simulation:"
	@echo "  ci-simulate              - Simulate complete CI pipeline locally"
	@echo "  ci-test-local            - Test CI configuration locally"
	@echo "  test-ci-incremental      - Test incremental CI workflow"
	@echo "  ci-quality-gate          - Run strict CI quality gate"
	@echo ""
	@echo "Validation:"
	@echo "  validate-ci-workflows    - Validate GitHub Actions YAML files"
	@echo "  branch-validate          - Validate branch protection setup"
	@echo "  github-validate          - Validate GitHub repository settings"
	@echo ""
	@echo "Setup:"
	@echo "  setup-branch-protection  - Configure branch protection rules"
	@echo "  pre-commit-update        - Update pre-commit hook versions"
	@echo "  pre-commit-autoupdate    - Auto-update pre-commit dependencies"

help-build:
	@echo "🔨 Build & Deployment Commands:"
	@echo "===================================="
	@echo "Build:"
	@echo "  build               - Build package with Nix"
	@echo "  nix-build           - Build with Nix (alternative)"
	@echo "  docker-build        - Build Docker image"
	@echo ""
	@echo "Run Applications:"
	@echo "  run-restaurant      - Start restaurant business system"
	@echo "  run-restaurant-fast - Start 5-minute restaurant simulation"
	@echo "  run-retail          - Start retail business system"
	@echo "  dashboard           - Launch Streamlit monitoring dashboard"
	@echo ""
	@echo "Data Generation:"
	@echo "  generate-data-restaurant - Generate 90 days restaurant data"
	@echo "  generate-data-retail     - Generate 90 days retail data"
	@echo ""
	@echo "Status & Info:"
	@echo "  status              - Show system and environment status"
	@echo "  nix-check           - Validate Nix configuration"

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
	@echo "🧪 Running all tests..."
	uv run pytest -v

test-unit:
	@echo "🧪 Running unit tests..."
	uv run pytest tests/unit/ -v

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
	uv run ruff check agents/ models/ simulation/ dashboard/ main.py tests/ scripts/
	uv run flake8 agents/ models/ simulation/ dashboard/

# Code formatting
format:
	@echo "🎨 Formatting code..."
	uv run black agents/ models/ simulation/ dashboard/ main.py tests/ scripts/
	uv run isort agents/ models/ simulation/ dashboard/ main.py tests/ scripts/
	uv run ruff check --fix agents/ models/ simulation/ dashboard/ main.py tests/ scripts/

# Type checking
type-check:
	@echo "🔍 Running type checks..."
	uv run mypy agents/ models/ simulation/

# Individual quality check targets (non-fixing)
format-check:
	@echo "🎨 Checking code formatting (black, isort)..."
	@uv run black --check --diff agents/ models/ simulation/ dashboard/ main.py tests/ scripts/ || (echo "❌ Formatting issues found. Run 'make fix-format' to fix."; exit 1)
	@uv run isort --check-only --diff agents/ models/ simulation/ dashboard/ main.py tests/ scripts/ || (echo "❌ Import sorting issues found. Run 'make fix-imports' to fix."; exit 1)

lint-check:
	@echo "🔍 Checking linting (ruff, flake8)..."
	@uv run ruff check agents/ models/ simulation/ dashboard/ main.py tests/ scripts/ || (echo "❌ Linting issues found. Run 'make fix-lint' to fix."; exit 1)
	@uv run flake8 agents/ models/ simulation/ dashboard/ || (echo "❌ Additional linting issues found."; exit 1)

type-check-strict:
	@echo "🔍 Running strict type checking..."
	@uv run mypy agents/ models/ simulation/ --strict --show-error-codes || (echo "❌ Type checking failed with strict mode."; exit 1)

security-check:
	@echo "🔒 Running security vulnerability scanning..."
	@uv run bandit -r agents/ models/ simulation/ dashboard/ main.py -f json -o security-report.json || true
	@uv run bandit -r agents/ models/ simulation/ dashboard/ main.py || (echo "⚠️ Security issues found. Check security-report.json"; exit 1)

complexity-check:
	@echo "📊 Analyzing code complexity..."
	@uv run radon cc agents/ models/ simulation/ -a -s || (echo "⚠️ High complexity found. Consider refactoring."; exit 1)
	@uv run radon mi agents/ models/ simulation/ -s || (echo "⚠️ Maintainability issues found."; exit 1)

docstring-check:
	@echo "📝 Checking docstring coverage and quality..."
	@uv run pydocstyle agents/ models/ simulation/ || (echo "⚠️ Docstring issues found. Run 'make fix-docstrings' to fix."; exit 1)
	@uv run interrogate agents/ models/ simulation/ --verbose --fail-under=80 || (echo "⚠️ Docstring coverage below 80%."; exit 1)

# Pre-commit
pre-commit-install:
	@echo "🔧 Installing pre-commit hooks..."
	uv run pre-commit install

pre-commit:
	@echo "🔍 Running pre-commit hooks on all files..."
	uv run pre-commit run --all-files

# Enhanced pre-commit targets
pre-commit-update:
	@echo "🔄 Updating pre-commit hook versions..."
	uv run pre-commit autoupdate

pre-commit-autoupdate:
	@echo "🔄 Auto-updating pre-commit dependencies..."
	uv run pre-commit autoupdate
	@echo "📋 Updated hooks. Review .pre-commit-config.yaml for changes."

pre-commit-clean:
	@echo "🧹 Cleaning pre-commit cache..."
	uv run pre-commit clean

# Auto-fix targets (destructive - make changes to files)
fix-all:
	@echo "🔧 Running all auto-fixes (format, imports, lint)..."
	@echo "⚠️  This will modify your files. Ensure you have committed or backed up your changes."
	@sleep 2
	$(MAKE) fix-format
	$(MAKE) fix-imports
	$(MAKE) fix-lint
	@echo "✅ All auto-fixes completed!"

fix-auto:
	@echo "🤖 Smart auto-fix based on detected issues..."
	@echo "🔍 Detecting issues and applying appropriate fixes..."
	@python -c "\
import subprocess; \
import sys; \
fixes_applied = []; \
result = subprocess.run(['uv', 'run', 'black', '--check', '.'], capture_output=True); \
fixes_applied.append('formatting') if result.returncode != 0 and subprocess.run(['uv', 'run', 'black', '.']).returncode == 0 else None; \
result = subprocess.run(['uv', 'run', 'isort', '--check-only', '.'], capture_output=True); \
fixes_applied.append('imports') if result.returncode != 0 and subprocess.run(['uv', 'run', 'isort', '.']).returncode == 0 else None; \
result = subprocess.run(['uv', 'run', 'ruff', 'check', '.'], capture_output=True); \
fixes_applied.append('linting') if result.returncode != 0 and subprocess.run(['uv', 'run', 'ruff', 'check', '--fix', '.']).returncode == 0 else None; \
print(f'✅ Applied fixes for: {', '.join([f for f in fixes_applied if f])}') if fixes_applied else print('✅ No issues detected - code is already clean!')"

fix-format:
	@echo "🎨 Auto-fixing formatting (black, isort)..."
	uv run black agents/ models/ simulation/ dashboard/ main.py tests/ scripts/
	uv run isort agents/ models/ simulation/ dashboard/ main.py tests/ scripts/

fix-imports:
	@echo "📦 Auto-fixing imports (sorting and removing unused)..."
	uv run isort agents/ models/ simulation/ dashboard/ main.py tests/ scripts/
	uv run autoflake --remove-all-unused-imports --recursive --in-place agents/ models/ simulation/ dashboard/ main.py tests/ scripts/

fix-lint:
	@echo "🔍 Auto-fixing linting issues (ruff)..."
	uv run ruff check --fix agents/ models/ simulation/ dashboard/ main.py tests/ scripts/

fix-docstrings:
	@echo "📝 Auto-fixing docstring formatting..."
	uv run docformatter --in-place --recursive agents/ models/ simulation/ dashboard/ main.py

fix-file:
	@echo "🔧 Auto-fixing specific file: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify FILE=path/to/file.py"; \
		echo "Example: make fix-file FILE=agents/base_agent.py"; \
		exit 1; \
	fi
	@echo "🎨 Formatting $(FILE)..."
	@uv run black "$(FILE)"
	@uv run isort "$(FILE)"
	@echo "🔍 Fixing linting issues in $(FILE)..."
	@uv run ruff check --fix "$(FILE)"
	@echo "📦 Removing unused imports from $(FILE)..."
	@uv run autoflake --remove-all-unused-imports --in-place "$(FILE)"
	@echo "✅ Fixed $(FILE)"

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

# IDE Setup and Validation
setup-ide:
	@echo "🔧 Setting up IDE integration..."
	python scripts/setup-ide.py

# Development checks (run before committing)
check: lint type-check test
	@echo "✅ All checks passed!"

# CI checks (comprehensive for CI/CD)
ci-check: clean install lint type-check test-unit test-integration test-smoke
	@echo "🚀 CI checks completed successfully!"

# Test performance alias
test-performance: perf-test-quick
	@echo "📊 Performance tests completed!"

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

# Performance testing
perf-test:
	@echo "🚀 Running comprehensive performance tests..."
	@mkdir -p tests/performance/results
	uv run python tests/performance/performance_runner.py

perf-test-quick:
	@echo "⚡ Running quick performance tests..."
	@mkdir -p tests/performance/results
	uv run python tests/performance/performance_runner.py --categories agent database --no-baseline

perf-test-agent:
	@echo "🤖 Running agent performance tests..."
	uv run pytest tests/performance/test_agent_performance.py -v --benchmark-json=tests/performance/results/agent_benchmarks.json

perf-test-database:
	@echo "🗄️ Running database performance tests..."
	uv run pytest tests/performance/test_database_performance.py -v --benchmark-json=tests/performance/results/database_benchmarks.json

perf-test-simulation:
	@echo "🎮 Running simulation performance tests..."
	uv run pytest tests/performance/test_simulation_performance.py -v --benchmark-json=tests/performance/results/simulation_benchmarks.json

perf-test-dashboard:
	@echo "📊 Running dashboard performance tests..."
	uv run pytest tests/performance/test_dashboard_performance.py -v --benchmark-json=tests/performance/results/dashboard_benchmarks.json

perf-test-stress:
	@echo "💪 Running stress tests..."
	uv run pytest tests/performance/test_stress.py -v --benchmark-json=tests/performance/results/stress_benchmarks.json

perf-report:
	@echo "📈 Generating performance reports..."
	@mkdir -p tests/performance/results
	uv run python -c "from tests.performance.benchmark_utils import PerformanceTracker, PerformanceReporter; tracker = PerformanceTracker(); reporter = PerformanceReporter(tracker); reporter.generate_summary_report('tests/performance/results/performance_summary.html'); print('Performance report generated: tests/performance/results/performance_summary.html')"

# Environment and dependency management targets
env-setup:
	@echo "🔧 Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "📋 Created .env from .env.example"; \
		echo "💡 Edit .env and add your ANTHROPIC_API_KEY"; \
	else \
		echo "✅ .env already exists"; \
	fi
	@echo "🔍 Validating environment setup..."
	@$(MAKE) env-validate

env-validate:
	@echo "🔍 Validating environment configuration..."
	@uv run python scripts/validate-env.py --env-only || python scripts/validate-env.py --env-only

env-check:
	@echo "📊 Environment Status Check:"
	@echo "============================"
	@$(MAKE) env-validate
	@$(MAKE) deps-check

env-clean:
	@echo "🧹 Cleaning environment..."
	@echo "🔄 Removing .env (backup will be created)"
	@if [ -f .env ]; then \
		cp .env .env.backup; \
		rm .env; \
		echo "✅ .env removed (backup: .env.backup)"; \
	else \
		echo "✅ .env not found - nothing to clean"; \
	fi

deps-check:
	@echo "📦 Checking dependency status..."
	@uv pip check || echo "⚠️ Dependency conflicts detected"
	@echo "🔍 Python dependencies:"
	@uv pip list --format=columns | head -10
	@echo "📊 Virtual environment:"
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "✅ Virtual environment active: $$VIRTUAL_ENV"; \
	else \
		echo "❌ No virtual environment active"; \
	fi

deps-update:
	@echo "⬆️ Updating dependencies safely..."
	@echo "📋 Creating backup of current lock file..."
	@cp uv.lock uv.lock.backup || echo "No uv.lock to backup"
	@echo "🔄 Updating dependencies..."
	@uv sync --upgrade || (echo "❌ Update failed. Restoring backup..."; mv uv.lock.backup uv.lock 2>/dev/null || true; exit 1)
	@echo "✅ Dependencies updated successfully"

deps-audit:
	@echo "🔒 Auditing dependencies for security vulnerabilities..."
	@echo "📊 Basic dependency audit..."
	@uv pip list --format=columns | head -10
	@echo "💡 Run: uv run safety check for comprehensive vulnerability scanning"

test-discovery:
	@echo "🔍 Discovering and validating test structure..."
	@uv run python scripts/test-discovery.py

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

# Integration testing
test-integration:
	@echo "🔗 Running integration tests..."
	cd tests/integration && uv run python test_runner.py

test-integration-smoke:
	@echo "💨 Running integration smoke tests..."
	cd tests/integration && uv run python test_runner.py --smoke

test-integration-file:
	@echo "🔗 Running specific integration test file: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify FILE=test_filename.py"; \
		exit 1; \
	fi
	cd tests/integration && uv run python test_runner.py --file $(FILE)

# Watch mode for continuous validation
watch:
	@echo "👀 Starting watch mode for continuous validation..."
	@echo "🔍 Monitoring agents/, models/, simulation/ for changes"
	@echo "🛑 Press Ctrl+C to stop"
	uv run python scripts/watch-validate.py

# Enhanced watch targets
watch-quality:
	@echo "👀 Starting watch mode for quality checks only..."
	@echo "🔍 Monitoring for changes - running quality gates only"
	@echo "🛑 Press Ctrl+C to stop"
	uv run python scripts/watch-validate.py --mode quality

watch-tests:
	@echo "👀 Starting watch mode for tests only..."
	@echo "🧪 Monitoring for changes - running tests only"
	@echo "🛑 Press Ctrl+C to stop"
	uv run python scripts/watch-validate.py --mode tests

watch-all:
	@echo "👀 Starting comprehensive watch mode..."
	@echo "🔍 Monitoring all files - full validation pipeline"
	@echo "🛑 Press Ctrl+C to stop"
	uv run python scripts/watch-validate.py --mode all

# Quality Gate System
quality-gate:
	@echo "🚪 Running normal quality gate checks..."
	uv run python scripts/quality-gate.py --mode normal

quality-gate-strict:
	@echo "🚪 Running strict quality gate checks..."
	uv run python scripts/quality-gate.py --mode strict

quality-gate-quick:
	@echo "🚪 Running quick quality gate checks..."
	uv run python scripts/quality-gate.py --mode quick

quality-gate-emergency:
	@echo "🚪 Running emergency quality gate checks..."
	uv run python scripts/quality-gate.py --mode emergency

validate-file:
	@echo "🔍 Validating file: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify FILE=path/to/file.py"; \
		echo "Example: make validate-file FILE=agents/base_agent.py"; \
		exit 1; \
	fi
	uv run python scripts/validate-file.py $(FILE)

validate-file-with-tests:
	@echo "🔍 Validating file with tests: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify FILE=path/to/file.py"; \
		echo "Example: make validate-file-with-tests FILE=agents/base_agent.py"; \
		exit 1; \
	fi
	uv run python scripts/validate-file.py $(FILE) --with-tests

# Enhanced validation targets
validate-changed-files:
	@echo "🔍 Validating only changed files (git diff)..."
	@CHANGED_FILES=$$(git diff --name-only HEAD | grep -E '\.py$$' | tr '\n' ' '); \
	if [ -n "$$CHANGED_FILES" ]; then \
		echo "📋 Changed files: $$CHANGED_FILES"; \
		uv run python scripts/quality-gate.py --mode normal --files $$CHANGED_FILES; \
	else \
		echo "✅ No Python files changed."; \
	fi

validate-branch:
	@echo "🔍 Validating entire branch changes..."
	@BRANCH_FILES=$$(git diff main...HEAD --name-only | grep -E '\.py$$' | tr '\n' ' '); \
	if [ -n "$$BRANCH_FILES" ]; then \
		echo "📋 Branch changed files: $$BRANCH_FILES"; \
		uv run python scripts/quality-gate.py --mode strict --files $$BRANCH_FILES; \
	else \
		echo "✅ No Python files changed in branch."; \
	fi

validate-env:
	@echo "🔍 Validating environment setup..."
	@python scripts/validate-env.py || uv run python scripts/validate-env.py

validate-ide:
	@echo "🔍 Validating IDE integration..."
	@python scripts/setup-ide.py --validate || uv run python scripts/setup-ide.py --validate

# Pre-commit quality gate
pre-commit-quality:
	@echo "🔍 Running pre-commit quality gate..."
	uv run python scripts/quality-gate.py --mode quick

# CI/CD quality gate 
ci-quality-gate:
	@echo "🚀 Running CI/CD quality gate..."
	uv run python scripts/quality-gate.py --mode strict --output quality-gate-results.json

# CI/CD workflow testing
test-ci-incremental:
	@echo "🧪 Testing incremental CI workflow locally..."
	@echo "📋 Simulating change detection..."
	@git diff --name-only HEAD~1 HEAD | grep -E '^(agents/|models/|simulation/).*\.py$$' || echo "No Python changes detected"
	@echo "🔍 Running quality gate on recent changes..."
	@CHANGED_FILES=$$(git diff --name-only HEAD~1 HEAD | grep -E '^(agents/|models/|simulation/).*\.py$$' | head -3 | tr '\n' ' '); \
	if [ -n "$$CHANGED_FILES" ]; then \
		uv run python scripts/quality-gate.py --mode normal --files $$CHANGED_FILES --no-color; \
	else \
		echo "No files to validate"; \
	fi

validate-ci-workflows:
	@echo "✅ Validating CI workflow YAML syntax..."
	@python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-incremental.yml')); print('✅ ci-incremental.yml is valid')"
	@python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('✅ ci.yml is valid')"

setup-branch-protection:
	@echo "🔒 Setting up branch protection..."
	@./scripts/setup-branch-protection.sh

setup-branch-protection-dry-run:
	@echo "🔍 Testing branch protection setup (dry run)..."
	@./scripts/setup-branch-protection.sh --dry-run

# Enhanced CI/CD targets
ci-simulate:
	@echo "🚀 Simulating complete CI pipeline locally..."
	@echo "📋 Step 1: Environment validation"
	@$(MAKE) env-check
	@echo "📋 Step 2: Dependency check"
	@$(MAKE) deps-check
	@echo "📋 Step 3: Code quality checks"
	@$(MAKE) quality-gate-strict
	@echo "📋 Step 4: Security scanning"
	@$(MAKE) security-check
	@echo "📋 Step 5: Full test suite"
	@$(MAKE) test
	@echo "📋 Step 6: Performance tests"
	@$(MAKE) perf-test-quick
	@echo "✅ CI pipeline simulation completed successfully!"

ci-test-local:
	@echo "🧪 Testing CI configuration locally..."
	@echo "📋 Validating CI workflow files..."
	@$(MAKE) validate-ci-workflows
	@echo "📋 Testing incremental workflow..."
	@$(MAKE) test-ci-incremental
	@echo "📋 Testing quality gates..."
	@$(MAKE) ci-quality-gate
	@echo "✅ CI configuration tests completed!"

ci-validate-all:
	@echo "✅ Comprehensive CI validation..."
	@$(MAKE) validate-ci-workflows
	@$(MAKE) branch-validate
	@$(MAKE) github-validate

branch-validate:
	@echo "🔍 Validating branch protection configuration..."
	@echo "📋 Checking if branch protection is properly configured..."
	@./scripts/setup-branch-protection.sh --validate || echo "⚠️ Branch protection needs configuration"

github-validate:
	@echo "🔍 Validating GitHub repository settings..."
	@if command -v gh >/dev/null 2>&1; then \
		echo "✅ GitHub CLI is available"; \
		gh repo view --json name,private,hasIssues 2>/dev/null || echo "⚠️ Could not fetch repository information"; \
		gh workflow list >/dev/null 2>&1 && echo "✅ GitHub Actions workflows are accessible" || echo "⚠️ Could not access workflows"; \
	else \
		echo "❌ GitHub CLI (gh) not installed. Install it for full validation."; \
	fi