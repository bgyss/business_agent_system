# Performance Testing Suite

This directory contains comprehensive performance tests and benchmarking utilities for the Business Agent Management System. The suite is designed to establish baseline metrics, track performance trends over time, and identify potential performance regressions.

## Overview

The performance testing suite includes:

- **Agent Performance Tests** - Measure agent decision processing speed and memory usage
- **Database Performance Tests** - Benchmark database queries and operations under various loads
- **Simulation Performance Tests** - Test data generation and simulation efficiency
- **Dashboard Performance Tests** - Measure dashboard loading times and responsiveness
- **Stress Tests** - High-volume scenarios and memory leak detection
- **Benchmarking Utilities** - Tools for baseline tracking and performance reporting

## Quick Start

### Running All Performance Tests

```bash
# Install performance testing dependencies
uv sync --extra performance

# Run all performance tests
python tests/performance/performance_runner.py

# Run specific categories
python tests/performance/performance_runner.py --categories agent database

# Run with coverage
COVERAGE=true python tests/performance/performance_runner.py
```

### Running Individual Test Categories

```bash
# Agent performance tests
pytest tests/performance/test_agent_performance.py -v

# Database performance tests  
pytest tests/performance/test_database_performance.py -v

# Simulation performance tests
pytest tests/performance/test_simulation_performance.py -v

# Dashboard performance tests
pytest tests/performance/test_dashboard_performance.py -v

# Stress tests
pytest tests/performance/test_stress.py -v
```

### Using Pytest Markers

```bash
# Run only benchmark tests
pytest tests/performance/ -m benchmark

# Run only memory tests
pytest tests/performance/ -m memory

# Run only stress tests
pytest tests/performance/ -m stress

# Run database-related tests
pytest tests/performance/ -m database
```

## Test Categories

### Agent Performance Tests (`test_agent_performance.py`)

Tests agent decision processing performance under various conditions:

- **Decision Processing Speed** - Benchmark how quickly agents process decisions
- **Memory Usage** - Monitor memory consumption during agent operations
- **Concurrent Processing** - Test agent performance with multiple simultaneous decisions
- **Decision Logging Performance** - Benchmark decision persistence speed
- **Message Handling** - Test inter-agent communication performance
- **Health Check Performance** - Measure agent status checking speed
- **Memory Leak Detection** - Detect potential memory leaks in long-running operations
- **Database Query Performance** - Test agent database interaction speed
- **High Volume Processing** - Stress test with large numbers of decisions
- **Startup Performance** - Benchmark agent initialization time

**Key Metrics:**
- Decisions per second
- Memory usage per decision
- Database query response time
- Agent startup time

### Database Performance Tests (`test_database_performance.py`)

Comprehensive database performance testing:

- **Insert Performance** - Benchmark record insertion speed
- **Query Performance** - Test query response times with various dataset sizes
- **Aggregation Performance** - Benchmark complex queries and aggregations
- **Update Performance** - Test record update operations
- **Batch Operations** - Measure bulk insert/update performance
- **Concurrent Access** - Test database performance under concurrent load
- **Index Performance** - Compare performance with and without indexes
- **Memory Usage** - Monitor database memory consumption

**Key Metrics:**
- Transactions per second
- Query response time
- Memory usage during operations
- Concurrent operation throughput

### Simulation Performance Tests (`test_simulation_performance.py`)

Test data generation and simulation efficiency:

- **Business Simulator Initialization** - Benchmark simulator startup time
- **Financial Data Generation** - Test financial data creation speed
- **Daily Transaction Generation** - Measure daily data generation performance
- **Inventory Simulation** - Test inventory data simulation speed
- **Large Scale Generation** - Test performance with extensive historical data
- **Anomaly Generation** - Benchmark anomaly injection performance
- **Real-time Simulation** - Test continuous data generation performance
- **Memory Usage** - Monitor simulation memory consumption
- **Concurrent Simulation** - Test multiple simultaneous simulations

**Key Metrics:**
- Records generated per second
- Memory usage during generation
- Simulation cycle time
- Data persistence speed

### Dashboard Performance Tests (`test_dashboard_performance.py`)

Measure dashboard loading and rendering performance:

- **Dashboard Initialization** - Benchmark dashboard startup time
- **Component Loading** - Test individual dashboard component performance
- **Chart Generation** - Measure chart creation and rendering speed
- **Data Processing** - Test dashboard data preparation performance
- **Concurrent Access** - Test dashboard under multiple user load
- **Rapid Refresh** - Test performance during frequent updates
- **Large Dataset Handling** - Test dashboard with extensive data

**Key Metrics:**
- Page load time
- Chart generation time
- Data processing speed
- Concurrent user support

### Stress Tests (`test_stress.py`)

High-volume scenarios and stability testing:

- **High Volume Agent Decisions** - Process thousands of decisions
- **Database Concurrent Access** - Many simultaneous database operations
- **Continuous Simulation** - Extended data generation periods
- **Multi-Agent Stress** - Multiple agents operating simultaneously
- **Memory Leak Detection** - Extended monitoring for memory issues
- **Error Recovery** - Test system recovery from failures
- **Large Data Volume** - Test with very large datasets

**Key Metrics:**
- System stability under load
- Memory leak detection
- Error recovery capability
- Maximum throughput

## Benchmarking Utilities

### Performance Tracker (`benchmark_utils.py`)

The `PerformanceTracker` class provides:

- **Metrics Storage** - Persistent storage of performance metrics
- **Baseline Calculation** - Automatic baseline metric calculation
- **Regression Detection** - Identify performance regressions
- **Trend Analysis** - Track performance changes over time

### Benchmark Runner

The `BenchmarkRunner` class offers:

- **Automated Benchmarking** - Run and record benchmark tests
- **Resource Monitoring** - Track CPU and memory usage
- **Result Storage** - Persist benchmark results
- **Baseline Comparison** - Compare results with historical baselines

### Performance Reporter

The `PerformanceReporter` class generates:

- **HTML Reports** - Interactive performance summaries
- **Trend Charts** - Visual performance trend analysis
- **Comparison Charts** - Multi-test performance comparisons
- **Markdown Reports** - Summary reports for documentation

## Performance Baselines

### Expected Performance Ranges

Based on testing with typical hardware (4-core CPU, 16GB RAM):

#### Agent Performance
- **Decision Processing**: < 100ms per decision (with mocked API)
- **Memory Usage**: < 50MB for 100 decisions
- **Database Operations**: < 50ms per query
- **Startup Time**: < 100ms

#### Database Performance
- **Insert Operations**: > 100 transactions/second
- **Simple Queries**: < 100ms response time
- **Complex Aggregations**: < 200ms response time
- **Concurrent Operations**: > 5 operations/second per thread

#### Simulation Performance
- **Data Generation**: > 1000 records/second
- **Daily Simulation**: < 200ms per cycle
- **Historical Data**: < 30 seconds for 90 days
- **Memory Usage**: < 500MB for large datasets

#### Dashboard Performance
- **Component Loading**: < 500ms per component
- **Chart Generation**: < 300ms per chart
- **Page Load**: < 2 seconds complete load
- **Refresh Operations**: < 500ms

## Configuration

### Environment Variables

```bash
# Enable coverage reporting
export COVERAGE=true

# Set Anthropic API key for agent tests (optional, will use mocks if not set)
export ANTHROPIC_API_KEY=your_api_key_here

# Set custom database URL for testing
export TEST_DATABASE_URL=sqlite:///custom_test.db
```

### Pytest Configuration

The tests use pytest markers for categorization:

- `@pytest.mark.benchmark` - Benchmark tests
- `@pytest.mark.memory` - Memory usage tests
- `@pytest.mark.stress` - Stress tests
- `@pytest.mark.database` - Database tests
- `@pytest.mark.agent` - Agent tests
- `@pytest.mark.simulation` - Simulation tests
- `@pytest.mark.dashboard` - Dashboard tests

### Custom Configuration

Modify `conftest.py` to adjust:

- Dataset sizes (small, medium, large)
- Test timeouts
- Benchmark thresholds
- Memory limits

## Interpreting Results

### Benchmark Output

Pytest-benchmark provides detailed statistics:

```
Name (time in ms)                    Min     Max     Mean   StdDev   Median
test_agent_decision_speed         48.21   52.34   49.87    1.23    49.45
```

### Performance Reports

Generated reports include:

- **Summary Statistics** - Overall performance metrics
- **Trend Analysis** - Performance changes over time
- **Regression Detection** - Identification of performance degradation
- **System Information** - Hardware and software environment details

### Key Performance Indicators (KPIs)

Monitor these critical metrics:

1. **Agent Decision Latency** - Time to process decisions
2. **Database Query Response Time** - Database operation speed
3. **Simulation Throughput** - Data generation rate
4. **Memory Efficiency** - Memory usage per operation
5. **System Stability** - Error rates and recovery capability

## Troubleshooting

### Common Issues

#### Slow Test Execution
- Check system resources (CPU, memory, disk)
- Verify database performance
- Consider reducing dataset sizes for development

#### Memory Issues
- Monitor for memory leaks in long-running tests
- Ensure proper cleanup in test fixtures
- Check for reference cycles in agent implementations

#### Inconsistent Results
- Run tests multiple times to establish consistent baselines
- Consider system load and background processes
- Use appropriate test isolation

#### Database Performance
- Ensure proper indexing on test databases
- Monitor disk I/O during tests
- Consider using in-memory databases for faster tests

### Performance Optimization Tips

1. **Agent Optimization**
   - Cache frequently accessed data
   - Batch similar operations
   - Optimize database queries
   - Use appropriate check intervals

2. **Database Optimization**
   - Add indexes for frequently queried columns
   - Use connection pooling
   - Optimize query patterns
   - Consider read replicas for reporting

3. **Simulation Optimization**
   - Batch database operations
   - Use efficient data structures
   - Minimize memory allocations
   - Optimize random number generation

## Continuous Integration

### Running in CI/CD

```yaml
# Example GitHub Actions step
- name: Run Performance Tests
  run: |
    uv sync --extra performance
    python tests/performance/performance_runner.py --no-baseline
  env:
    COVERAGE: true
```

### Performance Regression Detection

Set up automated monitoring:

1. Run performance tests on every commit
2. Compare results with baseline metrics
3. Fail builds if performance degrades significantly
4. Generate trend reports for performance tracking

## Contributing

### Adding New Performance Tests

1. Follow the existing test structure
2. Use appropriate pytest markers
3. Include memory usage monitoring
4. Add baseline expectations
5. Document expected performance ranges

### Updating Baselines

When legitimate performance improvements are made:

1. Run tests multiple times to establish new baseline
2. Update baseline expectations in test code
3. Document performance improvements
4. Update this README with new expected ranges

### Performance Test Guidelines

- **Isolation** - Tests should not interfere with each other
- **Repeatability** - Results should be consistent across runs
- **Measurement** - Include relevant performance metrics
- **Documentation** - Clearly document what is being measured
- **Thresholds** - Set appropriate performance thresholds