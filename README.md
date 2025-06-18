# Business Agent Management System

A comprehensive autonomous business management system powered by AI agents that monitor and optimize three key business areas: accounting/bookkeeping, inventory management, and human resources.

Built with **Nix** for reproducible builds and **uv** for fast Python dependency management.

## Features

### 🤖 AI Agents
- **Accounting Agent**: Monitors financial transactions, detects anomalies, tracks cash flow, and manages accounts receivable/payable
- **Inventory Agent**: Tracks stock levels, predicts reorder needs, analyzes consumption patterns, and optimizes inventory costs
- **HR Agent**: Manages employee schedules, monitors labor costs, tracks overtime, and optimizes staffing levels

### 📊 Business Simulation
- **Real-time Data Generation**: Simulates realistic business transactions, inventory movements, and employee activities
- **Historical Data**: Generates months of historical data for analysis and testing
- **Business Profiles**: Pre-configured templates for restaurants and retail stores
- **Anomaly Injection**: Introduces realistic business scenarios for agent testing

### 📈 Monitoring Dashboard
- **Real-time Metrics**: Live dashboard showing financial performance, inventory levels, and HR metrics
- **Interactive Charts**: Revenue trends, expense breakdowns, stock levels, and more
- **Alert System**: Visual indicators for issues requiring attention
- **Multi-business Support**: Switch between different business configurations

## Quick Start

### Prerequisites
- **Nix** (recommended) or Python 3.8+
- **Anthropic API key**

### Option 1: Using Nix (Recommended)

1. **Install Nix** (if not already installed)
   ```bash
   # Multi-user installation (recommended)
   sh <(curl -L https://nixos.org/nix/install) --daemon
   
   # Enable flakes (required)
   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
   ```

2. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd business_agent_system
   
   # Set up development environment (installs direnv if needed)
   make dev-setup
   
   # Or manually enter the Nix development shell
   nix develop
   ```

3. **Configure environment**
   ```bash
   # Edit .env file (created from template)
   # Add your Anthropic API key
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. **Install dependencies and run**
   ```bash
   # Install Python dependencies with uv
   make install
   
   # Run restaurant business system
   make run-restaurant
   
   # Or run retail business system  
   make run-retail
   
   # Launch dashboard (in separate terminal)
   make dashboard
   ```

### Option 2: Using uv directly

1. **Install uv**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd business_agent_system
   
   # Create virtual environment and install dependencies
   uv sync
   
   # Copy environment template
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Run the system**
   ```bash
   # For a restaurant business
   uv run python main.py --config config/restaurant_config.yaml

   # For a retail business
   uv run python main.py --config config/retail_config.yaml
   
   # Launch dashboard (in separate terminal)
   uv run streamlit run dashboard/app.py
   ```

### Option 3: Traditional Python setup

1. **Setup Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure and run**
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   
   python main.py --config config/restaurant_config.yaml
   ```

## Usage Examples

### Using Make commands (with Nix)
```bash
# Generate historical data
make generate-data-restaurant  # 90 days for restaurant
make generate-data-retail      # 90 days for retail

# Development workflow
make install                   # Install dependencies  
make test                     # Run tests
make lint                     # Check code quality
make format                   # Format code
make type-check              # Type checking

# Run applications
make run-restaurant          # Start restaurant system
make run-retail             # Start retail system  
make dashboard              # Launch dashboard
```

### Using uv directly
```bash
# Generate historical data
uv run python main.py --config config/restaurant_config.yaml --generate-historical 90

# Run in production mode
uv run python main.py --config config/restaurant_config.yaml --mode production

# Development commands
uv run pytest               # Run tests
uv run black .              # Format code
uv run mypy agents/         # Type checking
```

### Using Nix commands
```bash
# Build the package
nix build

# Run applications directly
nix run .#restaurant        # Restaurant system
nix run .#retail           # Retail system
nix run .#dashboard        # Dashboard

# Enter development shell
nix develop
```

### Custom Configuration
Create your own configuration file based on the provided templates in the `config/` directory.

## System Architecture

```
business_agent_system/
├── agents/                    # AI agent implementations
│   ├── base_agent.py         # Abstract base agent class
│   ├── accounting_agent.py   # Financial monitoring agent
│   ├── inventory_agent.py    # Inventory management agent
│   └── hr_agent.py           # Human resources agent
├── models/                    # Data models (SQLAlchemy + Pydantic)
│   ├── financial.py          # Financial transaction models
│   ├── inventory.py          # Inventory and supplier models
│   └── employee.py           # Employee and HR models
├── simulation/                # Business data simulators
│   ├── business_simulator.py # Main simulation orchestrator
│   ├── financial_generator.py # Financial data generator
│   └── inventory_simulator.py # Inventory data generator
├── config/                    # Business configuration files
│   ├── restaurant_config.yaml
│   └── retail_config.yaml
├── dashboard/                 # Streamlit monitoring dashboard
│   └── app.py
├── main.py                   # Main application entry point
└── requirements.txt          # Python dependencies
```

## Configuration

The system uses YAML configuration files to define business parameters, agent settings, and simulation parameters. Key configuration sections:

### Business Settings
```yaml
business:
  name: "Sample Restaurant"
  type: "restaurant"
  description: "Full-service restaurant with dine-in and takeout"
```

### Agent Configuration
```yaml
agents:
  accounting:
    enabled: true
    check_interval: 300  # seconds
    anomaly_threshold: 0.25  # 25% variance triggers alert
  inventory:
    enabled: true
    reorder_lead_time: 3  # days
  hr:
    enabled: true
    max_labor_cost_percentage: 0.32  # 32% of revenue
```

### Simulation Settings
```yaml
simulation:
  enabled: true
  mode: "real_time"  # "historical", "real_time", or "off"
  historical_days: 90
```

## Agent Decision Making

Each agent uses Claude AI to analyze data and make intelligent decisions:

1. **Data Processing**: Agents continuously monitor their respective data streams
2. **Pattern Recognition**: AI identifies anomalies, trends, and optimization opportunities  
3. **Decision Logging**: All agent decisions are logged with reasoning and confidence scores
4. **Action Recommendations**: Agents provide specific, actionable recommendations

### Example Agent Decision
```python
AgentDecision(
    agent_id="accounting_agent",
    decision_type="transaction_anomaly", 
    context={"transaction_amount": 5000, "variance": 0.45},
    reasoning="Transaction amount is 45% higher than typical...",
    action="Flag transaction for review",
    confidence=0.85
)
```

## Business Scenarios

The system includes pre-built scenarios for testing agent responses:

- **Cash Flow Crisis**: Simulates low cash situations
- **Seasonal Rush**: Models holiday/peak season increases
- **Equipment Failure**: Tests response to unexpected expenses
- **New Competitor**: Simulates market pressure scenarios

## API Reference

### Agent Health Check
```python
health_status = await agent.health_check()
# Returns: {"agent_id": "...", "status": "running", "decisions_count": 42}
```

### Generate Reports
```python
report = await agent.generate_report()
# Returns comprehensive summary with metrics and recent decisions
```

### System Status
```python
status = system.get_system_status()
# Returns overall system health and performance metrics
```

## Extending the System

### Adding New Agents
1. Create a new agent class inheriting from `BaseAgent`
2. Implement required abstract methods: `system_prompt`, `process_data`, `generate_report`
3. Add agent configuration to your YAML config file
4. Register the agent in `main.py`

### Adding New Business Types
1. Create a new configuration file in `config/`
2. Define business-specific parameters and thresholds
3. Update simulation profiles if needed
4. Customize agent prompts for the business domain

### Custom Data Models
1. Add new SQLAlchemy models in the `models/` directory
2. Create corresponding Pydantic models for validation
3. Update database initialization in the simulator
4. Add new data generators as needed

## Troubleshooting

### Common Issues

**"ANTHROPIC_API_KEY not found"**
- Ensure you've set the environment variable: `export ANTHROPIC_API_KEY=your_key`

**"Configuration file not found"**  
- Check the file path and ensure the config file exists
- Use absolute paths if needed

**"Database connection error"**
- Check database URL in configuration
- Ensure SQLite database directory exists
- For PostgreSQL, verify connection parameters

**"No historical data"**
- Run with `--generate-historical N` to create sample data
- Check simulation settings in configuration

### Debug Mode
```bash
python main.py --config config/restaurant_config.yaml --debug
```

### Logs
Check logs in the `logs/` directory for detailed error information and agent decision history.

## Performance Considerations

- **Database**: SQLite is suitable for development; use PostgreSQL for production
- **API Rate Limits**: Monitor Anthropic API usage and adjust agent check intervals
- **Memory Usage**: Historical data generation can be memory-intensive for large datasets
- **Concurrent Agents**: All agents run concurrently; consider resource limits

## Security Notes

- Store API keys securely using environment variables
- Use secure database connections in production
- Implement proper authentication for API endpoints
- Regularly review agent decision logs for unexpected behavior

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## License

PROPRIETARY SOFTWARE LICENSE

Copyright (c) 2025 Brian Gyss. All rights reserved.

NOTICE: This software and associated documentation files (the "Software") are 
proprietary and confidential. The Software is protected by copyright laws and 
international copyright treaties, as well as other intellectual property laws 
and treaties.

RESTRICTIONS:
1. The Software is licensed, not sold. You may not copy, modify, distribute, 
   sell, or lease any part of the Software without explicit written permission 
   from the copyright holder.

2. You may not reverse engineer, decompile, disassemble, or attempt to derive 
   the source code of the Software.

3. You may not use the Software for any commercial purposes without a valid 
   commercial license agreement.

4. You may not remove or alter any proprietary notices or labels on the Software.

5. This license does not grant you any rights to use the copyright holder's 
   trademarks or service marks.

PERMITTED USE:
- Authorized users may use the Software solely for internal evaluation, 
  development, and testing purposes.
- Any use beyond evaluation requires a separate commercial license agreement.

TERMINATION:
This license is effective until terminated. Your rights under this license 
will terminate automatically without notice if you fail to comply with any 
term of this license.

DISCLAIMER:
THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

For commercial licensing inquiries, please contact: 
bgyss@hey.com

## Support

For issues and questions:
- Check the troubleshooting section above
- Review agent decision logs for insights
- Open an issue with system configuration and error details