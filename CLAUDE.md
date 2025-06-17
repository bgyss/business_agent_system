# CLAUDE.md - Business Agent Management System

This document serves as a comprehensive guide for Claude AI to understand, maintain, and extend the Business Agent Management System. It contains essential context, architectural decisions, and maintenance procedures.

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
├── agents/
│   ├── base_agent.py          # Abstract agent framework
│   ├── accounting_agent.py    # Financial monitoring agent
│   ├── inventory_agent.py     # Inventory management agent
│   └── hr_agent.py           # Human resources agent
├── models/                    # SQLAlchemy and Pydantic data models
├── simulation/                # Business data generators and simulators
├── config/                    # Business-specific YAML configurations
└── dashboard/                 # Streamlit monitoring interface
```

### Configuration System
- **YAML-based**: Business-specific settings in `config/`
- **Environment Variables**: API keys and sensitive data in `.env`
- **Agent Parameters**: Thresholds, intervals, and behavior settings
- **Simulation Settings**: Data generation parameters and business profiles

## Development Workflow

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

# Running applications
make run-restaurant   # Start restaurant system
make run-retail      # Start retail system  
make dashboard       # Launch monitoring dashboard

# Data management
make generate-data-restaurant  # Create test data
make clean                    # Remove build artifacts
```

### Testing Strategy
- **Unit Tests**: Core agent logic and decision making
- **Integration Tests**: Agent communication and data flow
- **Simulation Tests**: Data generation and business scenarios
- **Type Checking**: Comprehensive mypy coverage for type safety

## Agent Development Guidelines

### Creating New Agents

1. **Inherit from BaseAgent**:
```python
class NewAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return "Your specialized AI prompt for this domain..."
    
    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        # Core agent logic here
        pass
    
    async def generate_report(self) -> Dict[str, Any]:
        # Reporting functionality
        pass
```

2. **Key Requirements**:
   - Implement all abstract methods from BaseAgent
   - Use structured decision logging with reasoning
   - Handle errors gracefully with proper logging
   - Include confidence scores for decisions
   - Provide clear, actionable recommendations

3. **Integration Steps**:
   - Add agent configuration to YAML files
   - Register agent in `main.py` initialization
   - Update dashboard to display agent metrics
   - Add tests for agent functionality

### Agent Decision Framework

**Decision Structure**:
```python
AgentDecision(
    agent_id="agent_name",
    decision_type="category_of_decision",
    context={"relevant": "data"},
    reasoning="AI-generated explanation",
    action="specific_action_to_take", 
    confidence=0.85  # 0.0 to 1.0
)
```

**Best Practices**:
- Use descriptive decision types for categorization
- Include all relevant context for decision auditing
- Ensure reasoning is clear and actionable
- Set appropriate confidence levels based on data quality
- Log decisions immediately after making them

## Business Configuration

### Adding New Business Types

1. **Create Configuration File**:
   - Copy existing config from `config/` directory
   - Adjust business-specific parameters
   - Update agent thresholds and intervals
   - Configure simulation profiles

2. **Key Configuration Sections**:
   ```yaml
   business:
     name: "Business Name"
     type: "business_type"
   
   agents:
     accounting:
       anomaly_threshold: 0.25
       alert_thresholds:
         cash_low: 1000
   
   simulation:
     business_profile:
       avg_daily_revenue: 2500
       seasonal_factors: {...}
   ```

3. **Business Profile Requirements**:
   - Revenue patterns (daily, seasonal, weekly)
   - Expense categories and typical amounts
   - Customer behavior patterns
   - Inventory consumption rates (if applicable)
   - Staffing requirements and labor costs

### Simulation Customization

**Financial Data Generation**:
- Realistic transaction patterns
- Seasonal revenue variations
- Expense category distributions
- Accounts receivable/payable timing
- Anomaly injection for testing

**Inventory Simulation**:
- Consumption based on sales volume
- Supplier delivery patterns
- Waste and spoilage modeling
- Reorder point calculations
- Purchase order generation

**HR Data Simulation**:
- Employee scheduling patterns
- Time tracking and attendance
- Labor cost calculations
- Overtime management
- Leave request handling

## Maintenance Procedures

### Regular Maintenance Tasks

**Weekly**:
- Review agent decision logs for accuracy
- Check system health metrics
- Update dependencies with `uv sync --upgrade`
- Monitor database performance

**Monthly**:
- Analyze agent performance trends
- Review and update business configurations
- Update Nix flake dependencies (`nix flake update`)
- Performance optimization review

**Quarterly**:
- Comprehensive testing of all business scenarios
- Review and update agent prompts for accuracy
- Security audit of dependencies
- Documentation updates

### Troubleshooting Common Issues

**Agent Performance Issues**:
1. Check API key configuration and rate limits
2. Review agent decision logs for error patterns
3. Verify database connectivity and performance
4. Monitor memory and CPU usage

**Simulation Problems**:
1. Validate business configuration parameters
2. Check data generation for realistic patterns
3. Verify database schema compatibility
4. Review simulation interval settings

**Environment Issues**:
1. Rebuild Nix environment: `nix develop --rebuild`
2. Clear uv cache: `uv cache clean`
3. Reset virtual environment: `rm -rf .venv && uv sync`
4. Check environment variables in `.env`

### Database Management

**Schema Updates**:
```python
# Use SQLAlchemy migrations for schema changes
# Add new models to appropriate files in models/
# Update business simulator to generate data for new fields
# Test with both SQLite and PostgreSQL
```

**Performance Optimization**:
- Index frequently queried columns
- Archive old transaction data periodically
- Monitor query performance with SQLAlchemy echo
- Use connection pooling for production deployments

## Deployment Guide

### Development Deployment
```bash
# Local development
nix develop
make install
make run-restaurant

# Docker development
docker-compose up business-agent-restaurant dashboard
```

### Production Deployment
```bash
# Using Nix
nix build
nix run .#restaurant

# Using Docker
docker-compose -f docker-compose.yml up -d

# Environment variables required:
# - ANTHROPIC_API_KEY
# - DATABASE_URL (for PostgreSQL)
# - REDIS_URL (for message queue)
```

### Monitoring and Observability

**Application Metrics**:
- Agent decision frequency and accuracy
- Database query performance
- API response times
- Memory and CPU usage

**Business Metrics**:
- Transaction processing volume
- Inventory optimization effectiveness
- Labor cost management success
- Cash flow prediction accuracy

**Alerting**:
- Agent failures or errors
- Database connection issues
- API rate limit violations
- Unusual business patterns detected

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

## Integration Points

### External Systems
- **Accounting Software**: QuickBooks-compatible data formats
- **POS Systems**: Transaction import capabilities
- **Payroll Systems**: Employee data synchronization
- **Inventory Systems**: Stock level monitoring

### API Endpoints
```python
# Health check
GET /health

# Agent status
GET /agents/status
GET /agents/{agent_id}/decisions

# Business reports
GET /reports/financial
GET /reports/inventory
GET /reports/hr

# Configuration
GET /config
POST /config/reload
```

### Webhook Integration
- Real-time transaction notifications
- Inventory level alerts
- Employee schedule changes
- System health notifications

## Performance Optimization

### Agent Optimization
- Batch similar decisions to reduce API calls
- Cache frequently accessed data
- Use appropriate check intervals for different data types
- Implement decision confidence thresholds

### Database Optimization
- Use appropriate indexes for query patterns
- Implement connection pooling
- Consider read replicas for reporting
- Archive historical data regularly

### System Resources
- Monitor memory usage with large datasets
- Use async patterns for I/O operations
- Implement proper error handling and retries
- Scale horizontally with multiple agent instances

## Future Enhancements

### Planned Features
- **ML Integration**: Time series forecasting for demand prediction
- **Advanced Analytics**: Business intelligence and reporting dashboards
- **Multi-tenant Support**: Multiple businesses in single deployment
- **Mobile Interface**: Native mobile app for business monitoring

### Technical Improvements
- **Event Sourcing**: Complete audit trail of all business events
- **CQRS Pattern**: Separate read/write models for better performance
- **Microservices**: Split agents into independent services
- **GraphQL API**: More flexible data querying interface

### AI Enhancement Opportunities
- **Fine-tuned Models**: Business-specific Claude fine-tuning
- **Multi-agent Collaboration**: Agents working together on complex decisions
- **Predictive Analytics**: Proactive problem identification
- **Natural Language Interface**: Chat-based business queries

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

*This document should be updated whenever significant changes are made to the system architecture, agent behavior, or deployment procedures. Keep it current to ensure effective maintenance and development.*