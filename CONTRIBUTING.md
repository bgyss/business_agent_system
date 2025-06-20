# Contributing to Business Agent Management System

Thank you for your interest in contributing to the Business Agent Management System! This guide will help you get started with contributing to this AI-powered business management platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Types](#contribution-types)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful, constructive, and professional in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- **Nix** (recommended) or Python 3.8+
- **Git** for version control
- **Anthropic API key** for testing AI agent functionality
- Basic understanding of Python, asyncio, and AI concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git fork https://github.com/briangyss/business-agent-system.git
   git clone https://github.com/YOUR_USERNAME/business-agent-system.git
   cd business-agent-system
   ```

2. **Set Up Development Environment**
   ```bash
   # Using Nix (recommended)
   make dev-setup
   nix develop
   
   # Or using uv directly
   uv sync --all-extras
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY to .env
   ```

4. **Verify Setup**
   ```bash
   make test
   make lint
   make type-check
   ```

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   make format      # Auto-format code
   make lint        # Check code quality
   make type-check  # Run mypy type checking
   make test        # Run full test suite
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new agent feature"
   ```

## Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, installation method
- **Steps to Reproduce**: Clear, numbered steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Logs**: Relevant error messages and logs
- **Configuration**: Anonymized config files if relevant

### 💡 Feature Requests

For new features:

- **Use Case**: Describe the business problem this solves
- **Proposed Solution**: How you envision the feature working
- **Alternatives**: Other approaches you've considered
- **Impact**: Who would benefit from this feature

### 🧪 Code Contributions

We welcome contributions in these areas:

#### New AI Agents
- **Domain Expertise**: Marketing, Operations, Customer Service agents
- **Industry Specialization**: Retail, Restaurant, Service business variants
- **Integration Agents**: Connect with external APIs and services

#### Business Simulators
- **New Business Types**: Manufacturing, E-commerce, Professional Services
- **Enhanced Realism**: More sophisticated simulation patterns
- **Stress Testing**: Edge cases and anomaly scenarios

#### Dashboard & Visualization
- **New Charts**: Additional business metrics visualization
- **Interactive Features**: Drill-down capabilities, filtering
- **Mobile Responsiveness**: Improved mobile dashboard experience

#### Infrastructure & DevOps
- **Docker Improvements**: Multi-stage builds, optimization
- **CI/CD Enhancements**: GitHub Actions workflows
- **Documentation**: Setup guides, tutorials, API documentation

## Code Standards

### Python Code Style

We use strict code quality standards:

```bash
# Formatting
make format  # Uses black, isort

# Linting  
make lint    # Uses ruff, flake8

# Type Checking
make type-check  # Uses mypy
```

### Code Quality Requirements

- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Public functions need clear docstrings
- **Error Handling**: Proper exception handling with logging
- **Async Patterns**: Use async/await consistently for I/O operations

### AI Agent Development Guidelines

When creating new agents:

#### 1. Inherit from BaseAgent
```python
from agents.base_agent import BaseAgent

class YourAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return "Your specialized AI prompt..."
    
    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        # Your agent logic here
        pass
```

#### 2. Decision Making Patterns
- Use structured `AgentDecision` objects
- Include confidence scores (0.0 to 1.0)
- Provide clear reasoning for decisions
- Log all decisions with appropriate context

#### 3. Configuration Integration
- Add agent settings to YAML configuration files
- Support configurable thresholds and parameters
- Include enable/disable toggles

## Testing Guidelines

### Test Coverage Requirements

- **Minimum Coverage**: 95% for new code
- **Agent Logic**: 100% coverage for decision-making code
- **Integration Tests**: Test agent interactions
- **Performance Tests**: Benchmark critical paths

### Testing Types

#### Unit Tests
```bash
# Run specific test files
uv run pytest tests/unit/test_accounting_agent.py -v

# Test with coverage
make test-unit
```

#### Integration Tests
```bash
# Test agent coordination
make test-integration
```

#### Performance Tests
```bash
# Benchmark agent performance
make test-performance
```

### Writing Tests

#### Agent Testing Pattern
```python
import pytest
from agents.your_agent import YourAgent

@pytest.mark.asyncio
async def test_agent_decision_making():
    agent = YourAgent(config=test_config)
    
    # Test normal operation
    decision = await agent.process_data(sample_data)
    assert decision.confidence >= 0.7
    assert decision.action == "expected_action"
    
    # Test edge cases
    edge_data = create_edge_case_data()
    decision = await agent.process_data(edge_data)
    assert decision is not None
```

#### Simulation Testing
```python
def test_business_simulation():
    simulator = BusinessSimulator(business_config)
    
    # Generate realistic data
    data = simulator.generate_day_data()
    
    # Verify data quality
    assert len(data.transactions) > 0
    assert all(t.amount > 0 for t in data.transactions)
```

## Pull Request Process

### Before Submitting

1. **Ensure All Tests Pass**
   ```bash
   make test
   ```

2. **Code Quality Checks**
   ```bash
   make lint
   make type-check
   make format
   ```

3. **Update Documentation**
   - Update README.md if adding new features
   - Add docstrings for new public functions
   - Update CLAUDE.md for significant changes

### PR Template

When submitting a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Code documentation updated
- [ ] README updated (if applicable)
- [ ] CLAUDE.md updated (if applicable)

## Screenshots
(If applicable)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline must pass
2. **Code Review**: At least one maintainer review
3. **Testing**: Manual testing for significant changes
4. **Documentation**: Verify documentation updates

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community chat
- **Email**: Direct contact at bgyss@hey.com

### Getting Help

- **Setup Issues**: Check the troubleshooting section in README.md
- **Development Questions**: Open a GitHub Discussion
- **Bug Reports**: Create a detailed GitHub Issue

### Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Special mentions for major features

## Development Resources

### Useful Commands

```bash
# Complete development workflow
make dev-setup     # Initial setup
make install       # Install dependencies
make test          # Run all tests
make lint          # Code quality checks
make format        # Auto-format code
make type-check    # Type checking

# Application commands
make run-restaurant  # Test restaurant system
make run-retail     # Test retail system
make dashboard      # Launch dashboard
make generate-data-restaurant  # Create test data
```

### Architecture Understanding

Before contributing, familiarize yourself with:

- **Agent Framework**: BaseAgent pattern and decision logging
- **Business Simulation**: Data generation and business modeling
- **Configuration System**: YAML-based business settings
- **Database Models**: SQLAlchemy models for business data

### Key Files to Understand

- `agents/base_agent.py` - Core agent framework
- `main.py` - Application orchestration
- `simulation/business_simulator.py` - Data generation
- `models/` - Database and validation models
- `config/` - Business configuration templates

## Questions?

If you have questions about contributing:

1. Check existing [GitHub Issues](https://github.com/briangyss/business-agent-system/issues)
2. Review [GitHub Discussions](https://github.com/briangyss/business-agent-system/discussions)
3. Read the [README.md](README.md) and [CLAUDE.md](CLAUDE.md)
4. Contact the maintainer: bgyss@hey.com

Thank you for contributing to Business Agent Management System! 🚀