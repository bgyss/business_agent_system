# Integration Tests for Business Agent System

This directory contains comprehensive integration tests for the Business Agent Management System. These tests verify end-to-end functionality, agent coordination, database operations, and error handling scenarios.

## Test Structure

### Test Files

1. **test_system_initialization.py**
   - System startup and configuration loading
   - Agent initialization and database setup
   - Configuration validation and error handling

2. **test_business_simulation.py**
   - Business simulation workflows
   - Historical data generation
   - Real-time simulation testing
   - Multiple business type support

3. **test_agent_coordination.py**
   - Agent startup/shutdown coordination
   - Message routing and communication
   - Multi-agent decision making
   - System monitoring functionality

4. **test_database_operations.py**
   - Database schema creation and integrity
   - CRUD operations for all models
   - Complex queries and aggregations
   - Concurrent access and transaction handling

5. **test_end_to_end_workflows.py**
   - Complete business workflows
   - Agent decision persistence
   - Business scenario testing (cash flow crisis, inventory shortage, etc.)
   - Normal operations validation

6. **test_error_scenarios.py**
   - Configuration error handling
   - Database failure recovery
   - Agent error isolation
   - Network and communication errors
   - System recovery mechanisms

### Test Fixtures

The integration tests use several key fixtures defined in `conftest.py`:

- **temp_db**: Temporary SQLite database for isolated testing
- **test_config**: Test configuration with fast intervals for quick testing
- **mock_anthropic_client**: Mocked Claude API to avoid external calls
- **business_system**: Fully initialized BusinessAgentSystem instance
- **running_system**: Started system with all agents running
- **integration_helper**: Helper utilities for common test operations

## Running the Tests

### Prerequisites

1. Install test dependencies:
   ```bash
   uv sync --all-extras
   ```

2. Ensure you're in the project root directory:
   ```bash
   cd /path/to/business_agent_system
   ```

### Running All Integration Tests

```bash
# Run all integration tests
python tests/integration/test_runner.py

# Or use pytest directly
pytest tests/integration/ -v --asyncio-mode=auto
```

### Running Specific Test Files

```bash
# Run specific test file
python tests/integration/test_runner.py --file test_system_initialization.py

# Or with pytest
pytest tests/integration/test_system_initialization.py -v
```

### Running Quick Smoke Test

```bash
# Run basic functionality test quickly
python tests/integration/test_runner.py --smoke
```

### Running Tests with Make

```bash
# If available in Makefile
make test-integration
```

## Test Configuration

The tests use a fast test configuration that:
- Uses 1-second intervals for quick execution
- Uses temporary SQLite databases for isolation
- Mocks external API calls to avoid dependencies
- Generates small datasets for speed
- Uses high speed multipliers for simulations

### Environment Variables

Tests automatically set up required environment variables:
- `ANTHROPIC_API_KEY`: Set to a test value (API calls are mocked)
- `PYTHONPATH`: Configured to include project root

## Test Patterns

### Async Test Pattern
```python
@pytest.mark.asyncio
async def test_async_functionality(running_system, integration_helper):
    """Test async functionality."""
    # Send test message
    await integration_helper.send_test_message(system, message)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Verify results
    assert system.is_running is True
```

### Database Test Pattern
```python
def test_database_operations(temp_db):
    """Test database operations."""
    engine = create_engine(temp_db)
    session = sessionmaker(bind=engine)()
    
    try:
        # Perform database operations
        session.add(test_object)
        session.commit()
        
        # Verify operations
        result = session.query(Model).first()
        assert result is not None
        
    finally:
        session.close()
```

### Error Handling Test Pattern
```python
def test_error_handling(system):
    """Test error handling."""
    with pytest.raises(ExpectedError):
        # Trigger error condition
        system.do_something_invalid()
    
    # Verify system remains stable
    assert system.is_running is True
```

## Test Data

Tests use various types of test data:

### Business Configurations
- Restaurant configuration with fast intervals
- Retail configuration for different business type testing
- Invalid configurations for error testing

### Simulated Data
- Financial transactions (credits, debits, transfers)
- Inventory items and stock movements
- Employee records and schedules
- Agent decisions with various confidence levels

### Test Messages
- Transaction notifications
- Cash flow alerts
- Inventory updates
- Schedule changes
- System status requests

## Coverage Areas

### Core Functionality
- ✅ System initialization and configuration
- ✅ Agent lifecycle management
- ✅ Database operations and persistence
- ✅ Message routing and communication
- ✅ Business simulation workflows

### Business Scenarios
- ✅ Normal daily operations
- ✅ Cash flow crisis handling
- ✅ Inventory shortage management
- ✅ Seasonal demand fluctuations
- ✅ Multi-agent coordination

### Error Handling
- ✅ Configuration errors
- ✅ Database failures
- ✅ Network timeouts
- ✅ Agent crashes
- ✅ Resource exhaustion
- ✅ Recovery mechanisms

### Performance and Scalability
- ✅ Concurrent database access
- ✅ Message queue overflow
- ✅ Large dataset handling
- ✅ Memory usage patterns

## Test Utilities

### IntegrationTestHelper
Provides common testing utilities:
- `wait_for_decisions()`: Wait for agents to make decisions
- `send_test_message()`: Send messages through the system
- `verify_database_tables()`: Verify database schema

### Mock Objects
- Anthropic API client (avoids external calls)
- Database connections (for error simulation)
- Message queues (for failure testing)

## Debugging Tests

### Logging
Tests use reduced logging to minimize noise, but you can increase verbosity:
```python
import logging
logging.getLogger("BusinessAgentSystem").setLevel(logging.DEBUG)
```

### Test Isolation
Each test uses:
- Temporary databases for complete isolation
- Fresh configuration files
- Independent agent instances
- Separate message queues

### Common Issues
1. **Async timeout errors**: Increase sleep times in tests
2. **Database locks**: Ensure proper session cleanup
3. **Port conflicts**: Tests use different ports from main app
4. **Memory usage**: Tests clean up resources in teardown

## Contributing Test Cases

When adding new integration tests:

1. **Follow naming conventions**: `test_[feature]_[scenario].py`
2. **Use appropriate fixtures**: Leverage existing fixtures when possible
3. **Test both success and failure cases**: Include error scenarios
4. **Verify cleanup**: Ensure tests don't leave state behind
5. **Document complex scenarios**: Add docstrings explaining test purpose
6. **Use realistic data**: Generate data that resembles real business scenarios

### Test Categories

Add tests to appropriate categories:
- **Unit-like**: Test single component in isolation
- **Integration**: Test component interactions
- **End-to-end**: Test complete user workflows
- **Performance**: Test under load or with large datasets
- **Error**: Test failure modes and recovery

## Continuous Integration

These tests are designed to run in CI environments:
- No external dependencies (API calls mocked)
- Fast execution (optimized intervals and datasets)
- Proper cleanup (no file system pollution)
- Clear pass/fail indicators
- Detailed error reporting

For CI configuration, ensure:
- Required Python packages are installed
- Database permissions are correct
- Temporary directory access is available
- Sufficient memory for concurrent tests