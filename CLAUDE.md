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

**Comprehensive Test Coverage (98%+ achieved)**
- **Unit Tests**: Core agent logic and decision making with extensive edge case coverage
- **Integration Tests**: Agent communication and data flow between components
- **Simulation Tests**: Business data generation and realistic scenarios
- **Performance Tests**: Agent scalability and simulation performance under load
- **Type Checking**: Comprehensive mypy coverage for type safety

**Test Architecture**
- **Unit Tests** (`tests/unit/`): Focus on individual component functionality
  - Agent decision logic and exception handling
  - Model validation and business rule enforcement  
  - Simulation data generation with realistic patterns
- **Integration Tests** (`tests/integration/`): Cross-component interactions
  - Agent coordination and message passing
  - Database operations and data consistency
  - End-to-end business workflows
- **Performance Tests** (`tests/performance/`): System scalability
  - Agent performance under high data volume
  - Database query optimization validation
  - Memory usage and resource management

**Coverage Standards**
- **Target Coverage**: 98%+ overall test coverage
- **Critical Components**: 100% coverage for decision logic
- **Simulation Modules**: Comprehensive scenario testing
- **Agent Exception Handling**: All error paths tested
- **Edge Cases**: Boundary conditions and malformed data handling

**Test Execution**
```bash
# Run all tests with coverage
make test

# Run specific test suites  
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-performance   # Performance benchmarks

# Generate coverage reports
make coverage-html      # HTML coverage report
make coverage-xml       # XML coverage for CI/CD
```

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
- **Accounting Software**: Integration with major SMB accounting platforms via REST/GraphQL APIs
- **POS Systems**: Transaction import capabilities via REST APIs
- **Payroll Systems**: Employee data synchronization
- **Inventory Systems**: Stock level monitoring

### Restaurant POS Integrations

The system supports integration with major restaurant POS platforms through their REST APIs. Each integration provides real-time transaction data, menu management, and operational metrics to enhance agent decision-making.

#### Supported POS Platforms

**Toast**
- **Developer Portal**: [https://doc.toasttab.com/doc/devguide/index.html](https://doc.toasttab.com/doc/devguide/index.html)
- **API Reference**: [https://toastintegrations.redoc.ly/](https://toastintegrations.redoc.ly/)
- **Integration Capabilities**: Sales data, menu items, employee time tracking, inventory levels
- **Authentication**: OAuth 2.0 with restaurant-specific tokens

**Square for Restaurants**
- **Developer Portal**: [https://developer.squareup.com/us/en](https://developer.squareup.com/us/en)
- **API Reference**: [https://developer.squareup.com/reference/square](https://developer.squareup.com/reference/square)
- **Integration Capabilities**: Payment processing, order management, customer data, inventory
- **Authentication**: Bearer tokens with application-level permissions

**Clover**
- **Developer Portal**: [https://docs.clover.com/dev/reference/api-reference-overview](https://docs.clover.com/dev/reference/api-reference-overview)
- **Integration Capabilities**: Order management, payment processing, employee management, inventory
- **Authentication**: OAuth 2.0 with merchant-specific access tokens

**Lightspeed Restaurant**
- **K-Series API**: [https://api-docs.lsk.lightspeed.app/](https://api-docs.lsk.lightspeed.app/)
- **O-Series API**: [https://o-series-support.lightspeedhq.com/hc/en-us/articles/31329318935067-API-Documentation](https://o-series-support.lightspeedhq.com/hc/en-us/articles/31329318935067-API-Documentation)
- **Integration Capabilities**: Sales reporting, menu management, employee scheduling, inventory tracking
- **Authentication**: API keys with location-specific access

**SpotOn**
- **Developer Portal**: [https://developers.spoton.com/restaurant/docs/api-access](https://developers.spoton.com/restaurant/docs/api-access)
- **Integration Capabilities**: Export API for sales data, customer information, menu items
- **Authentication**: API keys with rate limiting

**TouchBistro** *(Partner-only)*
- **Integrations Page**: [https://www.touchbistro.com/features/integrations/](https://www.touchbistro.com/features/integrations/)
- **API Catalog**: [https://apitracker.io/a/touchbistro](https://apitracker.io/a/touchbistro)
- **Integration Capabilities**: Requires certified partner status for API access
- **Authentication**: Partner-specific credentials with signed agreements

**Revel Systems**
- **Developer Portal**: [https://developer.revelsystems.com/revelsystems/](https://developer.revelsystems.com/revelsystems/)
- **API Reference**: [https://developer.revelsystems.com/revelsystems/docs/rest-api](https://developer.revelsystems.com/revelsystems/docs/rest-api)
- **Integration Capabilities**: Sales data, inventory management, employee data, reporting
- **Authentication**: API keys with establishment-level permissions

#### POS Integration Implementation Guidelines

**Data Synchronization Strategy**:
1. **Real-time Webhooks**: Configure POS webhooks for immediate transaction processing
2. **Scheduled Polling**: Hourly sync for menu updates and inventory changes
3. **Daily Batch**: End-of-day reports and comprehensive reconciliation
4. **Error Handling**: Retry logic with exponential backoff for API failures

**Agent Enhancement with POS Data**:
- **Accounting Agent**: Real-time transaction processing, sales categorization, payment reconciliation
- **Inventory Agent**: Automatic stock deduction, popular item tracking, waste reduction analysis
- **HR Agent**: Employee performance metrics, labor cost optimization, schedule effectiveness

**Security Considerations**:
- Store API credentials in environment variables or secure key management
- Implement rate limiting to respect POS platform constraints
- Use sandbox environments for development and testing
- Encrypt sensitive transaction data in transit and at rest

**Monitoring and Observability**:
- Track API response times and success rates for each POS integration
- Monitor data sync lag and alert on significant delays
- Log integration errors with sufficient context for debugging
- Maintain audit trails for all POS data modifications

### Restaurant Management & Purchasing Platform Integrations

The system supports integration with restaurant management platforms that handle purchasing, inventory management, and supplier relationships. These integrations enable comprehensive food cost tracking, automated purchase order management, and supply chain optimization for restaurant operations.

#### Dedicated Purchasing & Inventory Platforms

**MarketMan**
- **API Documentation**: JSON REST API v3 with Swagger specification
- **Integration Capabilities**: Item management, vendor relationships, purchase orders, invoices, stock counts
- **Authentication**: Bearer token authentication
- **Rate Limits**: 120 requests/minute
- **Sandbox**: Available upon request
- **Key Features**: Multi-vendor PO transmission, delivery tracking, order guides

**MarginEdge**
- **Developer Portal**: Public REST API with comprehensive documentation
- **Integration Capabilities**: OCR invoice processing, theoretical vs actual food cost analysis, re-ordering workflows
- **Authentication**: OAuth 2.0
- **Rate Limits**: 300 requests/minute
- **API Access**: Read-only for invoices/products; write endpoints (vendors, PO) in beta
- **Key Features**: Automated cost variance analysis, one-click reordering

**BlueCart**
- **API Reference**: REST API with OpenAPI specification
- **Integration Capabilities**: Mobile/web marketplace connecting restaurants to distributors
- **Authentication**: API key header authentication
- **Rate Limits**: 240 requests/minute
- **Pagination**: Limited to 25 orders per API call
- **Key Features**: Direct PO transmission to distributors, marketplace functionality

**Choco**
- **API Status**: No public API; proprietary integration workflows only
- **Integration Capabilities**: Chat-style ordering between chefs and distributors
- **Access**: Partner-only integrations
- **Key Features**: Conversational ordering interface, distributor relationship management
- **Note**: Limited integration options for automated data extraction

#### Comprehensive Restaurant Management Suites

**Restaurant365**
- **API Style**: OData + REST connectors with vendor integration hub
- **Integration Capabilities**: Full CRUD operations on vendors, items, purchase orders; automated PO to AP matching
- **Authentication**: OAuth 2.0
- **Rate Limits**: 600 requests/minute (highest in category)
- **Sandbox**: Available for development
- **Key Features**: Comprehensive back-office suite with integrated accounting

**Toast + xtraCHEF**
- **API Access**: Partner REST endpoints within xtraCHEF platform
- **Integration Capabilities**: Bid-sheet comparison, order guides, invoice digitization
- **Authentication**: OAuth 2.0 (requires Toast partner program membership)
- **Rate Limits**: 300 requests/minute
- **Sandbox**: Partner sandbox environment
- **Key Features**: Integrated POS and purchasing with automated invoice processing

**Oracle MICROS Simphony Cloud**
- **API Documentation**: Configuration & Content REST API (CCAPI) with OpenAPI specification
- **Integration Capabilities**: Procurement management, recipe costing, stock management, multi-site operations
- **Authentication**: Token-based with granular role scoping
- **Rate Limits**: 400 requests/minute
- **Sandbox**: Available for development
- **Key Features**: Enterprise-grade multi-location management

**Revel Systems**
- **Developer Portal**: REST API with comprehensive purchasing documentation
- **Integration Capabilities**: iPad POS integration, inventory management, PO creation and receiving
- **Authentication**: OAuth 2.0
- **Rate Limits**: 300 requests/minute (default)
- **Sandbox**: Dedicated sandbox organization per partner
- **Key Features**: Mobile-first POS with integrated purchasing workflows

**Lightspeed Restaurant (U-Series/Upserve)**
- **API Access**: OLO API for ordering, POS Core API, inventory data via partner endpoints
- **Integration Capabilities**: Upserve Inventory with costed recipes, auto-replenishment, electronic POs
- **Authentication**: OAuth 2.0 (write access requires Lightspeed partner approval)
- **Rate Limits**: 300 requests/minute
- **Sandbox**: Partner sandbox environment
- **Key Features**: Automated replenishment based on sales velocity

**Compeat (R365 Division)**
- **API Documentation**: Legacy REST Web API documentation
- **Integration Capabilities**: Back-office operations, inventory management, recipe costing, purchasing workflows
- **Authentication**: Basic token authentication
- **Rate Limits**: 150 requests/minute
- **Access**: Keys issued by account representatives to existing customers only
- **Key Features**: Integrated with Restaurant365 ecosystem

#### Restaurant Management Integration Implementation Guidelines

**Integration Architecture Strategy**:
1. **Bolt-on vs All-in-one**: Choose between dedicated purchasing platforms (MarketMan, MarginEdge) or comprehensive suites (Restaurant365, Simphony)
2. **API-First Approach**: Prioritize platforms with published OpenAPI specifications and sandbox environments
3. **Webhook Utilization**: Leverage real-time event notifications (invoice.created, stockcount.completed) to minimize polling
4. **Data Normalization**: Standardize unit codes (CS, EA, LB) before integration with business agents

**Data Synchronization Patterns**:
- **Real-time Events**: Webhook-driven updates for purchase orders, invoices, and inventory changes
- **Batch Processing**: Scheduled bulk imports for vendor catalogs and historical data
- **Incremental Sync**: Delta updates for inventory levels and pricing changes
- **Throttled Operations**: Batch PO syncs in 100-record chunks respecting rate limits

**Agent Enhancement with Restaurant Management Data**:
- **Inventory Agent**: Real-time stock level updates, automated reorder point calculation, supplier performance tracking
- **Accounting Agent**: Purchase order to invoice matching, food cost variance analysis, vendor payment optimization
- **HR Agent**: Labor cost correlation with food cost for comprehensive P&L optimization

**Platform-Specific Integration Considerations**:

**MarketMan**: Ideal for multi-vendor environments requiring vendor-neutral purchasing hub with strong API support.

**MarginEdge**: Best for operations focused on cost control and variance analysis with OCR-driven invoice processing.

**Restaurant365**: Comprehensive solution for multi-unit operators requiring integrated accounting and purchasing.

**Toast + xtraCHEF**: Optimal for Toast POS users seeking seamless integration between front-of-house and purchasing.

**Oracle Simphony**: Enterprise-grade solution for large restaurant groups with complex procurement needs.

**Error Handling and Resilience**:
- **Rate Limit Management**: Implement exponential backoff and respect Retry-After headers
- **API Version Management**: Handle deprecation cycles and maintain compatibility across platform updates
- **Data Validation**: Validate unit codes, vendor IDs, and item mappings before processing
- **Audit Trails**: Maintain complete transaction history for food cost analysis and compliance

**Security and Compliance**:
- **OAuth 2.0 Implementation**: Secure token management for partner-level integrations
- **Data Encryption**: Protect sensitive vendor and pricing information
- **Access Control**: Role-based permissions for purchasing and inventory data
- **Audit Logging**: Complete audit trails for purchase orders and inventory transactions

### SMB Accounting & Bookkeeping Integrations

The system supports comprehensive integration with major small and mid-sized business accounting platforms. These integrations enable automatic financial data synchronization, reducing manual data entry and improving accuracy of financial agent decisions.

#### Supported Accounting Platforms

**QuickBooks Online (Intuit)**
- **Developer Portal**: [https://developer.intuit.com/app/developer/qbo/docs/develop](https://developer.intuit.com/app/developer/qbo/docs/develop)
- **API Reference**: [https://developer.intuit.com/app/developer/qbo/docs/api/accounting](https://developer.intuit.com/app/developer/qbo/docs/api/accounting)
- **Integration Capabilities**: Transactions, accounts, invoices, payments, customer data, vendor management
- **Authentication**: OAuth 2.0 with sandbox environment
- **Rate Limits**: 500 requests/minute per realm
- **Official SDKs**: Java, .NET, PHP, Node.js

**Xero**
- **Developer Portal**: [https://developer.xero.com/](https://developer.xero.com/)
- **API Reference**: [https://developer.xero.com/documentation/api/accounting/overview](https://developer.xero.com/documentation/api/accounting/overview)
- **Integration Capabilities**: Accounting, payroll, assets, files APIs with shared OAuth token
- **Authentication**: OAuth 2.0 with webhook support
- **Rate Limits**: 60 requests/minute, 5,000/day (default tier)
- **Official SDKs**: Node.js, Java, .NET, PHP, Ruby

**FreshBooks**
- **Developer Portal**: [https://www.freshbooks.com/developers](https://www.freshbooks.com/developers)
- **API Reference**: [https://www.freshbooks.com/api/start](https://www.freshbooks.com/api/start)
- **Integration Capabilities**: Invoices, expenses, time tracking, client management, project management
- **Authentication**: OAuth 2.0
- **Test Environment**: 30-day trial accounts for development
- **Official SDKs**: Node.js

**Wave Accounting**
- **API Documentation**: [https://developer.waveapps.com/hc/en-us/articles/360018570992-Building-on-GraphQL](https://developer.waveapps.com/hc/en-us/articles/360018570992-Building-on-GraphQL)
- **Integration Capabilities**: Customers, invoices, payments, bank transactions via GraphQL
- **Authentication**: OAuth 2.0 to JWT flow
- **API Style**: GraphQL (single endpoint)
- **Community SDKs**: PHP, JavaScript

**Zoho Books**
- **API Documentation**: [https://www.zoho.com/books/api/v3/introduction/](https://www.zoho.com/books/api/v3/introduction/)
- **Integration Capabilities**: Complete accounting features with granular scope control
- **Authentication**: OAuth 2.0 with per-organization limits
- **Rate Limits**: 1,000 calls/day (free tier)
- **Official SDKs**: Java, .NET, PHP, Python

**Sage Business Cloud Accounting**
- **Developer Portal**: [https://developer.sage.com/accounting/reference/](https://developer.sage.com/accounting/reference/)
- **API Reference**: [https://developer.sage.com/accounting/guides/concepts/overview/](https://developer.sage.com/accounting/guides/concepts/overview/)
- **Integration Capabilities**: Full REST API with OpenAPI/Swagger specification
- **Authentication**: OAuth 2.0
- **Rate Limits**: 60 calls/minute
- **Community SDKs**: Node.js, .NET

**NetSuite (SuiteTalk)**
- **API Documentation**: [https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/book_1559132836.html](https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/book_1559132836.html)
- **Integration Capabilities**: Enterprise-grade ERP with REST and SOAP endpoints
- **Authentication**: Token-based or OAuth 2.0
- **Target Market**: Growing SMBs transitioning to enterprise
- **Community SDKs**: Ruby, Python

**Odoo (Open Source)**
- **API Documentation**: [https://www.odoo.com/documentation/18.0/developer/reference/external_api.html](https://www.odoo.com/documentation/18.0/developer/reference/external_api.html)
- **Integration Capabilities**: XML-RPC and JSON-RPC endpoints with no formal rate limits
- **Deployment**: Self-hosted or cloud-based
- **Development**: Local instance testing capability
- **Community SDKs**: Python (OdooRPC)

#### Accounting Integration Implementation Guidelines

**API Integration Strategy**:
1. **OAuth 2.0 First**: All platforms require OAuth 2.0 authentication (except Odoo XML-RPC)
2. **Rate Limit Compliance**: Respect platform-specific throttling (QuickBooks: 500 rpm, Xero: 60 rpm)
3. **SDK Utilization**: Use official SDKs when available for better error handling and retries
4. **Sandbox Testing**: Leverage staging environments to avoid production data contamination

**Data Synchronization Patterns**:
- **Real-time Sync**: Webhook-based updates for critical transactions
- **Batch Processing**: Daily/hourly bulk imports for historical data
- **Incremental Updates**: Change detection and delta synchronization
- **Conflict Resolution**: Last-write-wins with audit trails

**Agent Enhancement with Accounting Data**:
- **Accounting Agent**: Direct API integration for real-time anomaly detection and cash flow monitoring
- **Inventory Agent**: Cost of goods sold analysis and purchase order automation
- **HR Agent**: Payroll integration and labor cost analysis against revenue

**Error Handling and Resilience**:
- **Retry Logic**: Exponential backoff for transient API failures
- **Circuit Breakers**: Prevent cascade failures during extended outages
- **Data Validation**: Schema validation before API submission
- **Audit Logging**: Complete transaction history for compliance and debugging

**Security and Compliance**:
- **Credential Management**: Secure storage of OAuth tokens and API keys
- **Data Encryption**: TLS in transit, AES-256 at rest for sensitive financial data
- **Access Control**: Role-based permissions for financial data access
- **Compliance**: SOC 2, PCI DSS considerations for financial data handling

#### Platform-Specific Integration Notes

**QuickBooks Online**: Most comprehensive ecosystem with robust sandbox environment. Best for businesses requiring detailed financial reporting and tax compliance.

**Xero**: Strong international presence with excellent webhook support. Ideal for businesses with complex multi-currency requirements.

**Wave**: Free platform with GraphQL API makes it excellent for startups and small businesses. Limited advanced features but cost-effective.

**FreshBooks**: Strong focus on service businesses with excellent time tracking and project management integration.

**NetSuite**: Enterprise-grade solution for growing businesses that need advanced ERP capabilities alongside accounting.

**Odoo**: Open-source flexibility allows for complete customization. Best for businesses with specific workflow requirements.

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