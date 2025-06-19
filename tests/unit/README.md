# Unit Tests for Business Agent System

This directory contains comprehensive unit tests for all agent classes and core components of the business agent management system.

## Test Structure

### Test Files

- **`test_base_agent.py`** - Tests for the abstract BaseAgent class and common functionality
- **`test_accounting_agent.py`** - Tests for AccountingAgent decision-making logic
- **`test_inventory_agent.py`** - Tests for InventoryAgent stock management logic  
- **`test_hr_agent.py`** - Tests for HRAgent employee management logic
- **`test_agent_decisions.py`** - Tests for AgentDecision data models and serialization

### Configuration Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_runner.py`** - Standalone test runner with coverage reporting
- **`README.md`** - This documentation file

## Test Coverage

### BaseAgent Tests (`test_base_agent.py`)

**Core Functionality:**
- Agent initialization and configuration
- System prompt property implementation
- Claude API integration with mocking
- Message queue operations (send/receive)
- Decision logging to database
- Health check reporting
- Agent lifecycle (start/stop)
- Error handling and graceful failures

**Key Test Scenarios:**
- ✅ Successful Claude API calls
- ✅ API error handling
- ✅ Message filtering by recipient
- ✅ Database transaction failures
- ✅ Agent state management
- ✅ Decision history tracking

### AccountingAgent Tests (`test_accounting_agent.py`)

**Decision-Making Logic:**
- Transaction anomaly detection
- Daily financial analysis
- Cash flow monitoring
- Aging analysis for receivables/payables
- Alert threshold management

**Key Test Scenarios:**
- ✅ High variance transaction detection
- ✅ Low cash balance alerts  
- ✅ Overdue invoice tracking
- ✅ Normal vs. anomalous patterns
- ✅ Revenue/expense analysis
- ✅ Confidence score calculation

**Edge Cases:**
- ✅ Zero variance transactions
- ✅ Missing transaction history
- ✅ Database query failures
- ✅ Multiple account types

### InventoryAgent Tests (`test_inventory_agent.py`)

**Decision-Making Logic:**
- Stock movement analysis
- Low stock alerts
- Reorder recommendations
- Expiry date monitoring
- Supplier performance tracking

**Key Test Scenarios:**
- ✅ Low stock threshold detection
- ✅ Unusual consumption patterns
- ✅ Reorder quantity calculations
- ✅ Expiring item alerts
- ✅ Daily inventory checks
- ✅ Urgency level assignment

**Edge Cases:**
- ✅ Zero current stock handling
- ✅ Items without consumption history
- ✅ Multiple urgency calculations
- ✅ Lead time optimization

### HRAgent Tests (`test_hr_agent.py`)

**Decision-Making Logic:**
- Time record analysis
- Overtime detection
- Labor cost monitoring
- Staffing level analysis
- Leave request processing

**Key Test Scenarios:**
- ✅ Unusual clock-in times
- ✅ Overtime calculations
- ✅ Labor cost percentage analysis
- ✅ Staffing gap detection
- ✅ Leave request conflict analysis
- ✅ Business multiplier logic

**Edge Cases:**
- ✅ Complex time calculations
- ✅ Multi-break work days
- ✅ Weekend/holiday scheduling
- ✅ Emergency leave handling

### AgentDecision Tests (`test_agent_decisions.py`)

**Data Model Validation:**
- Pydantic model validation
- Database model conversion
- Context serialization
- Field requirement enforcement

**Key Test Scenarios:**
- ✅ Confidence score validation (0.0-1.0)
- ✅ Required field enforcement
- ✅ Decimal to float conversion
- ✅ DateTime serialization
- ✅ Nested context handling
- ✅ Database round-trip integrity

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# OR using uv
uv sync --all-extras
```

### Run All Unit Tests

```bash
# From project root
python -m pytest tests/unit/ -v

# With coverage
python -m pytest tests/unit/ --cov=agents --cov-report=term-missing

# Using the test runner
python tests/unit/test_runner.py
```

### Run Specific Test Files

```bash
# Test a specific agent
python -m pytest tests/unit/test_accounting_agent.py -v

# Test a specific method
python -m pytest tests/unit/test_base_agent.py::TestBaseAgent::test_agent_initialization -v

# Using the test runner
python tests/unit/test_runner.py test_accounting_agent.py
```

### Test with Different Verbosity

```bash
# Minimal output
python -m pytest tests/unit/ -q

# Verbose with details
python -m pytest tests/unit/ -v -s

# Show local variables in failures
python -m pytest tests/unit/ -vv -l
```

## Test Patterns and Best Practices

### Mocking Strategy

**External Dependencies:**
- **Anthropic API**: Mocked to return predictable responses
- **Database**: SQLAlchemy sessions mocked with query chains
- **Logging**: Suppressed to reduce test noise
- **Time/Dates**: Controlled for reproducible tests

**Mock Patterns:**
```python
# API Response Mocking
@pytest.fixture
def mock_anthropic(self):
    with patch('agents.base_agent.Anthropic') as mock_client:
        mock_response = Mock()
        mock_response.content = [Mock(text="Expected response")]
        mock_client.return_value.messages.create.return_value = mock_response
        yield mock_client

# Database Session Mocking  
@pytest.fixture
def mock_db_session(self):
    with patch('agents.base_agent.sessionmaker') as mock_sessionmaker:
        mock_session = Mock()
        mock_sessionmaker.return_value = mock_session
        yield mock_session
```

### Async Test Handling

All async tests use `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_async_method(self, agent):
    result = await agent.some_async_method()
    assert result is not None
```

### Decision Validation

Common pattern for validating agent decisions:

```python
def assert_valid_decision(decision, expected_type=None):
    assert decision is not None
    assert isinstance(decision, AgentDecision)
    assert decision.agent_id is not None
    assert 0.0 <= decision.confidence <= 1.0
    if expected_type:
        assert decision.decision_type == expected_type
```

### Configuration Testing

Each agent tests both custom and default configurations:

```python
def test_config_defaults(self):
    agent = AccountingAgent(
        agent_id="test", api_key="test", 
        config={}, db_url="sqlite:///:memory:"
    )
    assert agent.anomaly_threshold == 0.2  # Default value
```

## Coverage Goals

- **Line Coverage**: >90% for all agent classes
- **Branch Coverage**: >85% for decision logic
- **Function Coverage**: 100% for public methods

### Current Coverage Status

Run `python tests/unit/test_runner.py` to see current coverage metrics.

## Adding New Tests

### For New Agents

1. Create `test_new_agent.py` following existing patterns
2. Include all decision-making scenarios  
3. Test configuration handling
4. Add edge cases and error conditions
5. Update this README

### For New Features

1. Add tests to existing agent test files
2. Mock new external dependencies
3. Test both success and failure paths
4. Verify confidence score calculations
5. Update coverage expectations

### Test Structure Template

```python
class TestNewFeature:
    @pytest.fixture
    def feature_config(self):
        return {"param": "value"}
    
    @pytest.mark.asyncio
    async def test_normal_case(self, agent, mock_db):
        # Arrange
        data = {"type": "test_data"}
        
        # Act  
        result = await agent.process_data(data)
        
        # Assert
        assert result is not None
        assert result.decision_type == "expected_type"
    
    @pytest.mark.asyncio  
    async def test_edge_case(self, agent, mock_db):
        # Test edge case scenarios
        pass
        
    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_db):
        # Test error conditions
        pass
```

## Troubleshooting

### Common Issues

**Import Errors:**
- Ensure PYTHONPATH includes project root
- Check sys.path modifications in test files

**Async Test Failures:**
- Verify `@pytest.mark.asyncio` decorator
- Check event loop configuration in conftest.py

**Mock Issues:**
- Ensure mocks are properly scoped (function vs class)
- Verify mock return values match expected types

**Database Test Issues:**
- Check session mock setup
- Verify query chain mocking patterns

### Debug Commands

```bash
# Run with debug output
python -m pytest tests/unit/ -v -s --tb=long

# Run specific failing test
python -m pytest tests/unit/test_agent.py::TestClass::test_method -vv

# Check test discovery
python -m pytest --collect-only tests/unit/
```

## Integration with CI/CD

These unit tests are designed to run in CI/CD environments:

- No external service dependencies
- Deterministic timing with mocked dates
- Clean test isolation
- Coverage reporting integration
- Exit codes for build status

Example GitHub Actions integration:

```yaml
- name: Run Unit Tests
  run: |
    python -m pytest tests/unit/ \
      --cov=agents \
      --cov-fail-under=90 \
      --junit-xml=test-results.xml
```