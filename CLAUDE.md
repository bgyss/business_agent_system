# CLAUDE.md - Business Agent Management System

This document serves as the main guide for Claude AI to understand the Business Agent Management System. For detailed module-specific guidance, see the CLAUDE.md files in each subdirectory.

## Project Overview

**Business Agent Management System** is an autonomous business management platform powered by AI agents using Anthropic's Claude API. The system monitors and optimizes three core business areas through intelligent agents that make contextual decisions based on real-time data analysis.

### Core Philosophy
- **Autonomous Operation**: Agents make decisions independently with minimal human intervention
- **Contextual Intelligence**: Each agent uses Claude AI to understand business context and make appropriate recommendations
- **Extensible Architecture**: Modular design allows easy addition of new agents and business types
- **Reproducible Deployment**: Nix ensures consistent environments across development and production

## System Architecture

### Agent Framework
```
BaseAgent (abstract)
├── AccountingAgent - Financial monitoring and anomaly detection
├── InventoryAgent - Stock management and reorder optimization  
└── HRAgent - Employee scheduling and labor cost optimization
```

**Key Components:**
- **Message Queue**: Asyncio-based inter-agent communication
- **Decision Logging**: All agent decisions stored with reasoning and confidence scores
- **Claude Integration**: Each agent has specialized prompts for domain expertise
- **Database Layer**: SQLAlchemy ORM with support for SQLite/PostgreSQL

### Data Models
- **Financial**: Transactions, accounts, receivables, payables (QuickBooks-compatible)
- **Inventory**: Items, stock movements, suppliers, purchase orders
- **Employee**: Staff records, time tracking, schedules, payroll

### Simulation Engine
- **Business Profiles**: Configurable templates for different business types
- **Realistic Data Generation**: Seasonal patterns, customer behavior, supplier dynamics
- **Anomaly Injection**: Introduces edge cases for agent testing
- **Real-time Mode**: Continuous data generation during operation

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary language with async/await patterns
- **Anthropic Claude**: AI reasoning engine for agent decisions
- **SQLAlchemy**: Database ORM with Pydantic for validation
- **Streamlit**: Real-time monitoring dashboard
- **FastAPI**: REST API endpoints (optional)

### Development Environment
- **Nix**: Reproducible build system and package management
- **uv**: Fast Python dependency management
- **direnv**: Automatic environment activation
- **Make**: Development workflow automation

### Quality Assurance
- **pytest**: Test framework with async support
- **mypy**: Static type checking
- **black/isort/ruff**: Code formatting and linting
- **GitHub Actions**: CI/CD pipeline

## File Structure and Key Components

### Critical Files
```
business_agent_system/
├── main.py                    # Application entry point and orchestration
├── flake.nix                  # Nix development environment and package definition
├── pyproject.toml             # Python project configuration and dependencies
├── Makefile                   # Development workflow commands
├── agents/                    # AI agents and agent framework (see agents/CLAUDE.md)
├── models/                    # SQLAlchemy and Pydantic data models (see models/CLAUDE.md)
├── simulation/                # Business data generators and simulators (see simulation/CLAUDE.md)
├── tests/                     # Test infrastructure and LLM testing (see tests/CLAUDE.md)
├── config/                    # Business-specific YAML configurations
└── dashboard/                 # Streamlit monitoring interface
```

### Configuration System
- **YAML-based**: Business-specific settings in `config/`
- **Environment Variables**: API keys and sensitive data in `.env`
- **Agent Parameters**: Thresholds, intervals, and behavior settings
- **Simulation Settings**: Data generation parameters and business profiles

## Development Workflow

**IMPORTANT**: Always use `uv` to enter the virtual environment before starting any work:
```bash
source .venv/bin/activate
# or alternatively, prefix commands with: uv run
```

### Setting Up Development Environment
```bash
# Nix approach (recommended)
make dev-setup
nix develop
make install

# Alternative: uv only
uv sync --all-extras
```

### Common Development Tasks
```bash
# Code quality
make lint          # Check code quality
make format        # Auto-format code
make type-check    # Run mypy type checking
make test          # Run test suite

# Testing (IMPORTANT: Always use uv for running tests)
uv run pytest tests/unit/              # Run unit tests
uv run pytest tests/integration/       # Run integration tests
uv run pytest tests/unit/ -v --tb=short # Verbose unit tests with short traceback
make test                              # Run full test suite via make

# Running applications
make run-restaurant   # Start restaurant system
make run-retail      # Start retail system  
make dashboard       # Launch monitoring dashboard

# Data management
make generate-data-restaurant  # Create test data
make clean                    # Remove build artifacts
```

## Module-Specific Documentation

For detailed guidance on specific components, see:

- **agents/CLAUDE.md** - Agent development patterns, decision framework, creating new agents
- **models/CLAUDE.md** - Data models, database management, schema updates
- **simulation/CLAUDE.md** - Business simulation, data generation, configuration
- **tests/CLAUDE.md** - Testing strategy, LLM testing, coverage requirements
- **integrations/CLAUDE.md** - External system integrations (POS, accounting, etc.)
- **deployment/CLAUDE.md** - Deployment, monitoring, maintenance, and troubleshooting

## Security Considerations

### API Key Management
- Store Anthropic API key in environment variables only
- Use different keys for development and production
- Rotate keys regularly
- Monitor API usage and costs

### Data Security
- Encrypt sensitive financial data
- Use secure database connections
- Implement proper access controls
- Regular security dependency updates

### Code Security
- Run security scans with bandit
- Keep dependencies updated
- Use type checking to prevent runtime errors
- Validate all external inputs

## Contact and Support

### Getting Help
- Review this CLAUDE.md for guidance
- Check the README.md for setup instructions
- Examine agent decision logs for troubleshooting
- Use `make help` for available commands

### Code Review Guidelines
- Ensure all new agents follow the BaseAgent pattern
- Add comprehensive type hints for all functions
- Include unit tests for new functionality
- Update configuration examples for new features
- Document all public APIs and agent behaviors

### Contributing
- Follow the established coding standards (black, mypy, ruff)
- Add tests for new functionality
- Update documentation for any API changes
- Use conventional commit messages
- Ensure Nix builds work on all supported platforms

---

*This document should be updated whenever significant changes are made to the system architecture. Keep it current to ensure effective maintenance and development.*