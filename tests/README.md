# Web Testing Framework for Business Agent Dashboard

This directory contains a comprehensive web testing framework for the Streamlit dashboard using Selenium WebDriver.

## Quick Start

1. Install dependencies:
```bash
make install
# or
uv sync --all-extras
```

2. Run smoke tests (quick verification):
```bash
make test-smoke
```

3. Run full web test suite:
```bash
make test-web
```

## Test Structure

### Files Overview

- `conftest.py` - Pytest fixtures and configuration for web testing
- `test_dashboard_smoke.py` - Quick smoke tests for basic functionality
- `test_dashboard_web.py` - Comprehensive web tests for dashboard features
- `web_test_utils.py` - Utility classes and helper functions
- `README.md` - This documentation

### Test Categories

#### Smoke Tests (`test_dashboard_smoke.py`)
Quick tests to verify basic functionality:
- Dashboard loads without errors
- Main components are present
- No obvious JavaScript errors
- Basic navigation works

#### Comprehensive Web Tests (`test_dashboard_web.py`)
Detailed tests for all dashboard features:
- Live monitoring view functionality
- Historical analytics view
- Interactive elements (buttons, dropdowns)
- Chart rendering
- Data display accuracy
- Navigation between views
- Accessibility basics

## Test Commands

### Make Commands

```bash
# Quick smoke tests
make test-smoke

# Full web test suite
make test-web

# Web tests without slow tests
make test-web-headless

# All web tests with detailed reporting
make test-web-full
```

### Direct Pytest Commands

```bash
# Run all web tests with verbose output
uv run pytest tests/ -v

# Run only smoke tests
uv run pytest tests/test_dashboard_smoke.py -v

# Run web tests with HTML report
uv run pytest tests/test_dashboard_web.py --html=report.html --self-contained-html

# Run tests with screenshots on failure
uv run pytest tests/ -v --capture=no
```

## Configuration

### Browser Configuration

The tests use Chrome in headless mode by default. You can modify browser settings in `conftest.py`:

```python
options = Options()
options.add_argument("--headless")  # Remove for visual debugging
options.add_argument("--window-size=1920,1080")
```

### Test Data

Tests use the existing configuration files:
- `config/restaurant_config.yaml`
- `config/retail_config.yaml`

### Environment Setup

The framework automatically:
1. Starts a Streamlit server on port 8502
2. Waits for server to be ready
3. Creates browser instance
4. Cleans up after tests

## Writing New Tests

### Basic Test Template

```python
def test_my_feature(dashboard_page, streamlit_helper):
    """Test a specific dashboard feature"""
    # Navigate to specific view
    streamlit_helper.select_sidebar_radio("Historical Analytics")
    
    # Verify expected content
    streamlit_helper.wait_for_text("Expected Content")
    
    # Interact with elements
    streamlit_helper.click_sidebar_button("Refresh Now")
    
    # Assert results
    assert "Expected Result" in dashboard_page.page_source
```

### Using Page Objects

```python
def test_with_page_object(dashboard_page):
    """Test using page object pattern"""
    from web_test_utils import StreamlitPageObject
    
    page = StreamlitPageObject(dashboard_page)
    page.navigate_to_live_monitoring()
    
    # Get structured data
    metrics = page.get_financial_metrics()
    assert len(metrics) > 0
```

### Helper Functions

The `StreamlitTestHelper` class provides methods for common interactions:

- `wait_for_element(by, value)` - Wait for element to appear
- `wait_for_text(text)` - Wait for text to appear
- `click_sidebar_button(text)` - Click sidebar button
- `select_sidebar_radio(option)` - Select radio button
- `get_metric_value(label)` - Extract metric values

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```
   Port 8502 is already in use. Please stop any running Streamlit instances.
   ```
   Solution: Stop existing Streamlit processes or change port in `conftest.py`

2. **ChromeDriver not found**
   ```
   selenium.common.exceptions.WebDriverException: chromedriver executable needs to be in PATH
   ```
   Solution: The framework auto-installs ChromeDriver, but ensure Chrome browser is installed

3. **Tests timeout**
   ```
   selenium.common.exceptions.TimeoutException: Message: 
   ```
   Solution: Increase timeout values or check if dashboard is loading properly

4. **Elements not found**
   ```
   selenium.common.exceptions.NoSuchElementException
   ```
   Solution: Dashboard structure may have changed, update selectors in tests

### Debugging Tips

1. **Run tests with visible browser** (remove headless mode):
   ```python
   # In conftest.py, comment out:
   # options.add_argument("--headless")
   ```

2. **Add debugging pauses**:
   ```python
   import time
   time.sleep(5)  # Pause to inspect browser state
   ```

3. **Screenshot on failure**:
   ```python
   # Tests automatically take screenshots in test_screenshots/
   # Check this directory after test failures
   ```

4. **Enable verbose logging**:
   ```bash
   uv run pytest tests/ -v -s --log-cli-level=DEBUG
   ```

## Test Reports

### HTML Reports
Test results are saved in `test_artifacts/`:
- `web_test_report.html` - Full web test results
- `smoke_test_report.html` - Smoke test results

### Screenshots
Screenshots are saved in `test_screenshots/`:
- Automatic screenshots on test failures
- Manual screenshots for visual verification

### Performance Data
Basic performance metrics are collected:
- Page load times
- DOM ready times
- Navigation performance

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Web Tests
on: [push, pull_request]

jobs:
  web-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Chrome
        run: |
          wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
          sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
          sudo apt-get update
          sudo apt-get install google-chrome-stable
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync --all-extras
      - name: Run smoke tests
        run: make test-smoke
      - name: Upload test artifacts
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: |
            test_artifacts/
            test_screenshots/
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on others
2. **Cleanup**: Use fixtures for setup/teardown to ensure clean state
3. **Explicit Waits**: Always use explicit waits instead of sleep()
4. **Page Objects**: Use page object pattern for complex interactions
5. **Error Handling**: Test both success and failure scenarios
6. **Performance**: Monitor test execution time and optimize slow tests
7. **Accessibility**: Include basic accessibility checks in tests

## Extending the Framework

### Adding New Test Categories

1. Create new test file: `test_dashboard_[category].py`
2. Import fixtures from `conftest.py`
3. Add new Make targets in `Makefile`
4. Update this README

### Adding New Helper Methods

1. Add methods to `StreamlitTestHelper` class in `conftest.py`
2. Or create new utility classes in `web_test_utils.py`
3. Document usage in this README

### Supporting New Browsers

1. Add browser fixtures in `conftest.py`
2. Install appropriate WebDriver manager
3. Update browser options as needed

## Dependencies

Key testing dependencies:
- `selenium>=4.15.0` - WebDriver for browser automation
- `pytest-selenium>=4.1.0` - Pytest integration for Selenium
- `webdriver-manager>=4.0.0` - Automatic WebDriver management
- `pytest-html>=4.1.0` - HTML test reports

All dependencies are automatically installed with `make install` or `uv sync --all-extras`.