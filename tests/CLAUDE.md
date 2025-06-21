# tests/CLAUDE.md - Testing Strategy and Infrastructure

This document provides comprehensive guidance for testing the Business Agent Management System, including LLM testing patterns and coverage requirements.

## Testing Strategy

**Comprehensive Test Coverage (98%+ achieved)**
- **Unit Tests**: Core agent logic and decision making with extensive edge case coverage
- **Integration Tests**: Agent communication and data flow between components
- **Simulation Tests**: Business data generation and realistic scenarios
- **Performance Tests**: Agent scalability and simulation performance under load
- **Type Checking**: Comprehensive mypy coverage for type safety
- **LLM Testing**: Deterministic testing with record & replay patterns and semantic caching

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

## LLM Testing & Caching Infrastructure

The system implements advanced testing and caching strategies for LLM interactions to ensure deterministic, cost-effective, and reliable agent testing. This infrastructure addresses the inherent challenges of testing AI-powered agents that make external API calls.

**Core Testing Challenges Addressed**:
- **Non-deterministic Responses**: LLM responses vary between identical prompts
- **API Cost Management**: Minimize expensive API calls during development and testing
- **Offline Testing**: Enable testing without internet connectivity
- **Performance Optimization**: Reduce latency through intelligent caching

### VCR.py Record & Replay Pattern

**Implementation Overview**:
The system uses VCR.py (Video Cassette Recorder) to record real LLM API interactions during initial test runs and replay them consistently in subsequent executions.

**Configuration** (`tests/conftest.py`):
```python
import vcr
import pytest

# VCR configuration for LLM testing
vcr_config = vcr.VCR(
    cassette_library_dir="tests/cassettes",
    record_mode="once",  # Record only on first run
    match_on=["method", "scheme", "host", "port", "path", "body"],
    filter_headers=[
        ("Authorization", "DUMMY"),
        ("X-API-Key", "REDACTED")
    ],  # Redact API keys in cassettes
    decode_compressed_response=True,
    serializer="yaml"
)

@pytest.fixture()
def vcr_cassette_dir(request):
    """Dynamic cassette directory per test module"""
    return f"tests/cassettes/{request.module.__name__}"
```

**Test Implementation Patterns**:

```python
# Decorator-based cassette usage
@vcr_config.use_cassette("agent_decisions/accounting_anomaly.yaml")
def test_accounting_agent_anomaly_detection():
    """Test accounting agent with recorded LLM responses"""
    agent = AccountingAgent(config)
    result = await agent.process_data(anomaly_data)
    assert result.decision_type == "anomaly_alert"
    assert result.confidence > 0.8

# Pytest fixture integration
@pytest.mark.vcr
def test_inventory_agent_reorder():
    """Test inventory agent with automatic cassette naming"""
    agent = InventoryAgent(config)
    result = await agent.process_data(low_stock_data)
    assert "reorder" in result.action.lower()
```

**Cassette Organization Structure**:
```
tests/cassettes/
├── agents/
│   ├── accounting/
│   │   ├── anomaly_detection.yaml
│   │   ├── cash_flow_analysis.yaml
│   │   └── fraud_detection.yaml
│   ├── inventory/
│   │   ├── reorder_optimization.yaml
│   │   ├── demand_forecasting.yaml
│   │   └── supplier_evaluation.yaml
│   └── hr/
│       ├── schedule_optimization.yaml
│       ├── labor_cost_analysis.yaml
│       └── performance_evaluation.yaml
└── integration/
    ├── multi_agent_coordination.yaml
    └── business_simulation.yaml
```

**CI/CD Integration**:
```python
# CI environment configuration
if os.getenv("CI"):
    vcr_config.record_mode = "none"  # Fail if cassette missing
    vcr_config.allow_playback_repeats = True
```

### Mock LLM Testing Framework

**Purpose**: Enable unit testing of agent orchestration logic without LLM API dependencies.

**FakeListLLM Integration**:
```python
from langchain_core.language_models.fake import FakeListLLM
from typing import List, Dict, Any

class MockAgentLLM:
    """Mock LLM for agent testing with predefined responses"""
    
    def __init__(self, responses: List[str]):
        self.llm = FakeListLLM(responses=responses)
        self.call_count = 0
    
    async def generate_decision(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate mock decision based on predefined responses"""
        response = self.llm.invoke([])
        self.call_count += 1
        return response

# Test implementation
def test_agent_decision_logic():
    """Test agent logic without external LLM calls"""
    mock_responses = [
        "ANOMALY_DETECTED: Unusual transaction pattern detected",
        "CONFIDENCE: 0.85",
        "ACTION: Alert finance team and investigate transactions"
    ]
    
    mock_llm = MockAgentLLM(mock_responses)
    agent = AccountingAgent(config, llm=mock_llm)
    
    result = await agent.process_data(test_data)
    
    assert mock_llm.call_count == 1
    assert result.decision_type == "anomaly_alert"
    assert result.confidence == 0.85
```

### Semantic Caching Infrastructure

**GPTCache Integration**:
The system implements semantic caching to improve cache hit rates for similar but not identical prompts, reducing API costs and improving response times.

**Cache Configuration**:
```python
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import CacheBase, VectorBase

def init_semantic_cache():
    """Initialize semantic cache for LLM responses"""
    
    # Embedding model for semantic similarity
    embedding_model = Onnx()
    
    # Vector database for similarity search
    vector_base = VectorBase("faiss", dimension=embedding_model.dimension)
    
    # Cache manager
    cache_base = CacheBase("sqlite", sql_url="sqlite:///./llm_cache.db")
    
    # Similarity evaluation (threshold: 0.8)
    evaluation = SearchDistanceEvaluation(
        distance_threshold=0.2,  # Lower = more similar required
        max_distance=1.0
    )
    
    cache.init(
        embedding_func=embedding_model.to_embeddings,
        data_manager=cache_base,
        similarity_evaluation=evaluation,
        vector_base=vector_base
    )
```

### Advanced Testing Patterns

**Agent-Specific Test Suites**:
```python
class TestAccountingAgentLLM:
    """Comprehensive LLM testing for Accounting Agent"""
    
    @pytest.fixture
    def agent_cassettes(self) -> Dict[str, str]:
        """Mapping of test scenarios to cassette files"""
        return {
            "normal_transactions": "accounting/normal_flow.yaml",
            "anomaly_detection": "accounting/anomaly_alert.yaml", 
            "fraud_investigation": "accounting/fraud_analysis.yaml",
            "cash_flow_forecast": "accounting/cash_projection.yaml"
        }
    
    @pytest.mark.parametrize("scenario,cassette", [
        ("normal_transactions", "accounting/normal_flow.yaml"),
        ("anomaly_detection", "accounting/anomaly_alert.yaml")
    ])
    @vcr_config.use_cassette()
    def test_agent_scenarios(self, scenario: str, cassette: str):
        """Parameterized testing across multiple scenarios"""
        agent = AccountingAgent(config)
        test_data = self.load_test_data(scenario)
        
        result = await agent.process_data(test_data)
        
        self.validate_decision_structure(result)
        self.validate_scenario_specific_logic(result, scenario)
```

## Unit Testing Best Practices

### Agent Testing Patterns
```python
@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client for consistent testing"""
    with patch('agents.base_agent.Anthropic') as mock_client:
        mock_response = Mock()
        mock_response.content = [Mock(text="Test agent response")]
        mock_client.return_value.messages.create.return_value = mock_response
        yield mock_client

@pytest.fixture
def mock_db_session():
    """Mock database session"""
    with patch('agents.base_agent.create_engine'), \
         patch('agents.base_agent.sessionmaker') as mock_sessionmaker:
        mock_session = Mock()
        mock_sessionmaker.return_value = mock_session
        yield mock_session

class TestInventoryAgent:
    """Comprehensive test suite for InventoryAgent"""
    
    async def test_low_stock_detection(self, agent, mock_db_session):
        """Test low stock detection logic"""
        # Mock item with low stock
        mock_item = Mock(current_stock=5, reorder_point=10)
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_item]
        
        result = await agent._check_low_stock(mock_db_session)
        
        assert result is not None
        assert "low_stock" in result.decision_type
        assert result.confidence > 0.7
```

### Model Testing
```python
class TestFinancialModels:
    """Test financial data models"""
    
    def test_transaction_validation(self):
        """Test transaction model validation"""
        transaction = TransactionModel(
            amount=Decimal('100.00'),
            description="Test transaction",
            account_id=1,
            transaction_date=datetime.now()
        )
        
        assert transaction.amount == Decimal('100.00')
        assert transaction.is_valid()
    
    def test_invalid_transaction_amount(self):
        """Test validation of invalid transaction amounts"""
        with pytest.raises(ValidationError):
            TransactionModel(
                amount=Decimal('-100.00'),  # Invalid negative amount
                description="Invalid transaction",
                account_id=1
            )
```

## Integration Testing

### Cross-Agent Communication Testing
```python
class TestAgentCommunication:
    """Test agent communication patterns"""
    
    async def test_message_passing(self):
        """Test message passing between agents"""
        accounting_agent = AccountingAgent(config)
        inventory_agent = InventoryAgent(config)
        
        # Create message queue
        message_queue = asyncio.Queue()
        
        # Test message sending
        message = AgentMessage(
            type="cash_flow_inquiry",
            data={"amount": 5000, "timing": datetime.now()}
        )
        
        await inventory_agent.send_message_to_agent("accounting_agent", message)
        
        # Verify message received
        received_message = await message_queue.get()
        assert received_message["target"] == "accounting_agent"
        assert received_message["sender"] == "inventory_agent"
```

### Database Integration Testing
```python
class TestDatabaseIntegration:
    """Test database operations and transactions"""
    
    @pytest.fixture
    def test_db(self):
        """Create test database"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine
    
    async def test_transaction_rollback(self, test_db):
        """Test database transaction rollback on error"""
        session = sessionmaker(bind=test_db)()
        
        try:
            # Start transaction
            item = Item(name="Test Item", sku="TEST-001")
            session.add(item)
            
            # Simulate error
            raise Exception("Simulated error")
            
        except Exception:
            session.rollback()
            
        # Verify rollback
        items = session.query(Item).all()
        assert len(items) == 0
```

## Performance Testing

### Agent Load Testing
```python
class TestAgentPerformance:
    """Performance tests for agent scalability"""
    
    @pytest.mark.benchmark
    async def test_agent_decision_performance(self, benchmark):
        """Benchmark agent decision making performance"""
        agent = InventoryAgent(config)
        test_data = generate_test_inventory_data(1000)  # Large dataset
        
        result = await benchmark(agent.process_data, test_data)
        
        assert result is not None
        assert benchmark.stats.mean < 0.5  # Sub-500ms response time
    
    async def test_concurrent_agent_processing(self):
        """Test multiple agents processing data concurrently"""
        agents = [
            AccountingAgent(config),
            InventoryAgent(config),
            HRAgent(config)
        ]
        
        test_data = generate_test_business_data()
        
        # Process data concurrently
        tasks = [agent.process_data(test_data) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        assert all(result is not None for result in results)
        assert len(results) == 3
```

### Memory and Resource Testing
```python
class TestResourceUsage:
    """Test memory and resource consumption"""
    
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow unbounded"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        agent = InventoryAgent(config)
        
        # Process large amounts of data
        for i in range(1000):
            data = generate_large_inventory_dataset()
            asyncio.run(agent.process_data(data))
            
            if i % 100 == 0:
                gc.collect()  # Force garbage collection
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be bounded
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/testing.yml
name: Comprehensive Testing Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --all-extras
    
    - name: Run unit tests
      run: |
        source .venv/bin/activate
        make test-unit
    
    - name: Run integration tests
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        source .venv/bin/activate
        make test-integration
    
    - name: Run performance tests
      run: |
        source .venv/bin/activate
        make test-performance
    
    - name: Generate coverage report
      run: |
        source .venv/bin/activate
        make coverage-xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Data Management

### Test Data Generation
```python
def generate_test_financial_data(num_transactions: int = 100) -> List[Dict]:
    """Generate realistic financial test data"""
    transactions = []
    for i in range(num_transactions):
        transactions.append({
            "amount": Decimal(random.uniform(10.0, 1000.0)),
            "description": f"Test transaction {i}",
            "account_id": random.randint(1, 5),
            "transaction_date": datetime.now() - timedelta(days=random.randint(0, 365))
        })
    return transactions

def generate_test_inventory_data(num_items: int = 50) -> List[Dict]:
    """Generate realistic inventory test data"""
    items = []
    for i in range(num_items):
        items.append({
            "sku": f"ITEM-{i:03d}",
            "name": f"Test Item {i}",
            "current_stock": random.randint(0, 100),
            "reorder_point": random.randint(10, 30),
            "unit_cost": Decimal(random.uniform(5.0, 50.0))
        })
    return items
```

### Test Database Fixtures
```python
@pytest.fixture(scope="session")
def test_database():
    """Create and populate test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    # Populate with test data
    session = sessionmaker(bind=engine)()
    
    # Add test items
    for item_data in generate_test_inventory_data():
        item = Item(**item_data)
        session.add(item)
    
    session.commit()
    return engine
```

---

*This document should be updated when new testing patterns are introduced or when coverage requirements change.*