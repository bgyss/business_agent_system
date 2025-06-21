# agents/CLAUDE.md - Agent Development Guide

This document provides comprehensive guidance for developing and maintaining AI agents in the Business Agent Management System.

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

## Agent Types and Specializations

### AccountingAgent
**Purpose**: Financial monitoring and anomaly detection
**Key Methods**:
- `_analyze_transaction_anomalies()` - Detect unusual financial patterns
- `_check_cash_flow()` - Monitor liquidity and cash positions
- `_analyze_receivables()` - Track outstanding customer payments
- `_analyze_payables()` - Monitor supplier payment obligations

**Configuration Parameters**:
```yaml
accounting:
  anomaly_threshold: 0.25
  alert_thresholds:
    cash_low: 1000
    receivables_overdue: 30  # days
    payables_overdue: 15     # days
```

### InventoryAgent
**Purpose**: Stock management and reorder optimization
**Key Methods**:
- `_analyze_stock_movement()` - Process inventory transactions
- `_check_low_stock()` - Monitor reorder points
- `_analyze_supplier_performance()` - Evaluate delivery reliability
- `predict_demand()` - Forecast future inventory needs

**Advanced Analytics**:
- Demand forecasting with seasonality
- Optimal reorder point calculations
- Bulk purchase optimization
- Supplier diversification analysis

**Configuration Parameters**:
```yaml
inventory:
  low_stock_multiplier: 1.2
  reorder_lead_time: 7
  consumption_analysis_days: 30
  forecast_horizon_days: 30
```

### HRAgent
**Purpose**: Employee scheduling and labor cost optimization
**Key Methods**:
- `_analyze_schedule_efficiency()` - Optimize staff scheduling
- `_monitor_labor_costs()` - Track labor expenses vs revenue
- `_check_overtime_patterns()` - Identify excessive overtime
- `_analyze_performance_metrics()` - Evaluate employee productivity

**Configuration Parameters**:
```yaml
hr:
  max_overtime_percentage: 0.15
  min_staff_level: 2
  scheduling_horizon_days: 14
```

## Agent Communication Patterns

### Message Queue Integration
Agents communicate through an asyncio-based message queue system:

```python
async def send_message_to_agent(self, target_agent: str, message: AgentMessage):
    """Send message to another agent"""
    await self.message_queue.put({
        "target": target_agent,
        "sender": self.agent_id,
        "message": message,
        "timestamp": datetime.now()
    })

async def receive_messages(self):
    """Process incoming messages from other agents"""
    while True:
        message = await self.message_queue.get()
        if message["target"] == self.agent_id:
            await self._process_message(message)
```

### Cross-Agent Decision Coordination
Agents can coordinate decisions for complex business scenarios:

```python
# Example: Inventory agent consults with accounting agent before large purchase
if purchase_amount > self.large_purchase_threshold:
    cash_flow_message = AgentMessage(
        type="cash_flow_inquiry",
        data={"amount": purchase_amount, "timing": proposed_date}
    )
    await self.send_message_to_agent("accounting_agent", cash_flow_message)
```

## Error Handling and Resilience

### Exception Handling Patterns
```python
async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
    session = self.SessionLocal()
    try:
        # Agent processing logic
        result = await self._perform_analysis(session, data)
        return result
    except DatabaseError as e:
        self.logger.error(f"Database error in {self.agent_id}: {e}")
        return None
    except APIError as e:
        self.logger.error(f"Claude API error in {self.agent_id}: {e}")
        return None
    except Exception as e:
        self.logger.error(f"Unexpected error in {self.agent_id}: {e}")
        return None
    finally:
        session.close()
```

### Recovery Strategies
- **Database Connection Issues**: Retry with exponential backoff
- **API Rate Limits**: Queue requests and throttle
- **Data Validation Errors**: Log and skip problematic records
- **Memory Issues**: Implement batch processing for large datasets

## Agent Configuration Management

### YAML Configuration Structure
```yaml
agents:
  accounting:
    check_interval: 300  # seconds
    anomaly_threshold: 0.25
    alert_thresholds:
      cash_low: 1000
      receivables_overdue: 30
    
  inventory:
    check_interval: 600
    low_stock_multiplier: 1.2
    reorder_lead_time: 7
    
  hr:
    check_interval: 3600
    max_overtime_percentage: 0.15
```

### Dynamic Configuration Updates
Agents support runtime configuration updates:

```python
async def update_config(self, new_config: Dict[str, Any]):
    """Update agent configuration at runtime"""
    self.config.update(new_config)
    self.logger.info(f"Configuration updated for {self.agent_id}")
    
    # Validate critical parameters
    if not self._validate_config():
        self.logger.error("Invalid configuration detected")
        raise ConfigurationError("Configuration validation failed")
```

## Performance Optimization

### Agent Optimization Guidelines
- Batch similar decisions to reduce API calls
- Cache frequently accessed data using Redis
- Use appropriate check intervals for different data types
- Implement decision confidence thresholds to avoid low-value decisions

### Memory Management
```python
class BaseAgent:
    def __init__(self, agent_id: str, api_key: str, config: Dict[str, Any], db_url: str):
        # Limit decision history to prevent memory growth
        self.decision_history = deque(maxlen=1000)
        
        # Use connection pooling for database efficiency
        self.engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
```

### Claude API Optimization
- Use system prompts for consistent context
- Implement response caching for repeated queries
- Monitor token usage and costs
- Use confidence thresholds to avoid unnecessary API calls

## Testing Agent Functionality

### Unit Testing Patterns
```python
@pytest.fixture
def mock_anthropic():
    with patch('agents.base_agent.Anthropic') as mock_client:
        mock_response = Mock()
        mock_response.content = [Mock(text="Test decision response")]
        mock_client.return_value.messages.create.return_value = mock_response
        yield mock_client

async def test_agent_decision_making(mock_anthropic):
    agent = InventoryAgent(
        agent_id="test_agent",
        api_key="test_key",
        config=test_config,
        db_url="sqlite:///:memory:"
    )
    
    result = await agent.process_data(test_data)
    assert isinstance(result, AgentDecision)
    assert result.confidence > 0.5
```

### Integration Testing
- Test agent communication patterns
- Verify database transaction handling
- Validate decision logging and retrieval
- Test error recovery scenarios

## Agent Monitoring and Observability

### Decision Logging
All agent decisions are automatically logged with:
- Timestamp and agent ID
- Input data context
- Decision reasoning and confidence
- Execution time and resource usage

### Performance Metrics
- Decision frequency and accuracy
- API response times and costs
- Database query performance
- Memory and CPU usage patterns

### Health Checks
```python
async def health_check(self) -> Dict[str, Any]:
    """Agent health check for monitoring"""
    return {
        "agent_id": self.agent_id,
        "status": "healthy",
        "last_decision": self.last_decision_time,
        "decisions_today": len(self.recent_decisions),
        "api_calls_today": self.api_call_count,
        "avg_response_time": self.avg_response_time
    }
```

---

*This document should be updated when new agent types are added or when agent patterns change significantly.*