"""
Pytest configuration and fixtures for web testing
"""
import pytest
import subprocess
import time
import os
import signal
import socket
from typing import Generator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


@pytest.fixture(scope="session")
def streamlit_server() -> Generator[str, None, None]:
    """
    Start a Streamlit server for testing and return the URL.
    This fixture runs once per test session.
    """
    port = 8502  # Use different port from default to avoid conflicts
    
    # Check if port is already in use
    if is_port_in_use(port):
        pytest.skip(f"Port {port} is already in use. Please stop any running Streamlit instances.")
    
    # Start Streamlit server in background
    cmd = [
        "streamlit", "run", 
        "dashboard/app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Wait for server to start
        url = f"http://localhost:{port}"
        max_attempts = 30  # 30 seconds timeout
        for _ in range(max_attempts):
            if is_port_in_use(port):
                # Give a bit more time for full initialization
                time.sleep(2)
                break
            time.sleep(1)
        else:
            pytest.fail(f"Streamlit server failed to start on port {port}")
        
        yield url
        
    finally:
        # Clean up: terminate the process group
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture(scope="function")
def browser() -> Generator[webdriver.Chrome, None, None]:
    """
    Create a Chrome browser instance for testing.
    This fixture runs for each test function.
    """
    # Configure Chrome options
    options = Options()
    options.add_argument("--headless")  # Run in headless mode for CI/CD
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")  # Speed up page loads
    
    # Install and setup ChromeDriver
    service = Service(ChromeDriverManager().install())
    
    # Create browser instance
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)  # Wait up to 10 seconds for elements
    
    yield driver
    
    # Cleanup
    driver.quit()


@pytest.fixture(scope="function")
def dashboard_page(browser: webdriver.Chrome, streamlit_server: str):
    """
    Navigate to the dashboard page and wait for it to load.
    Returns the browser instance positioned on the dashboard.
    """
    browser.get(streamlit_server)
    
    # Wait for Streamlit to load (look for the main title)
    wait = WebDriverWait(browser, 30)
    wait.until(
        EC.presence_of_element_located((By.TAG_NAME, "h1"))
    )
    
    # Additional wait for dynamic content to load
    time.sleep(3)
    
    return browser


@pytest.fixture
def sample_config_path() -> str:
    """Return path to a sample configuration file for testing"""
    return "config/restaurant_config.yaml"


class StreamlitTestHelper:
    """Helper class for interacting with Streamlit elements"""
    
    def __init__(self, browser: webdriver.Chrome):
        self.browser = browser
        self.wait = WebDriverWait(browser, 20)
    
    def wait_for_element(self, by: By, value: str, timeout: int = 20):
        """Wait for an element to be present"""
        wait = WebDriverWait(self.browser, timeout)
        return wait.until(EC.presence_of_element_located((by, value)))
    
    def wait_for_text(self, text: str, timeout: int = 20):
        """Wait for specific text to appear on the page"""
        wait = WebDriverWait(self.browser, timeout)
        wait.until(EC.text_to_be_present_in_element((By.TAG_NAME, "body"), text))
    
    def click_sidebar_button(self, button_text: str):
        """Click a button in the Streamlit sidebar"""
        button = self.wait_for_element(
            By.XPATH, 
            f"//button[contains(text(), '{button_text}')]"
        )
        button.click()
        time.sleep(1)  # Wait for UI to update
    
    def select_sidebar_radio(self, option_text: str):
        """Select a radio button option in the sidebar"""
        radio_option = self.wait_for_element(
            By.XPATH,
            f"//label[contains(text(), '{option_text}')]"
        )
        radio_option.click()
        time.sleep(2)  # Wait for view to change
    
    def select_dropdown(self, dropdown_value: str):
        """Select an option from a dropdown"""
        dropdown = self.wait_for_element(By.CSS_SELECTOR, "select")
        dropdown.click()
        
        option = self.wait_for_element(
            By.XPATH,
            f"//option[contains(text(), '{dropdown_value}')]"
        )
        option.click()
        time.sleep(2)  # Wait for data to load
    
    def get_metric_value(self, metric_label: str) -> str:
        """Get the value of a Streamlit metric"""
        metric_element = self.wait_for_element(
            By.XPATH,
            f"//div[contains(@data-testid, 'metric')]//div[contains(text(), '{metric_label}')]/following-sibling::div"
        )
        return metric_element.text
    
    def check_chart_exists(self, chart_title: str) -> bool:
        """Check if a chart with the given title exists"""
        try:
            self.wait_for_element(
                By.XPATH,
                f"//div[contains(text(), '{chart_title}')]",
                timeout=10
            )
            return True
        except:
            return False
    
    def get_table_data(self, table_selector: str = "[data-testid='stDataFrame']"):
        """Get data from a Streamlit dataframe/table"""
        table = self.wait_for_element(By.CSS_SELECTOR, table_selector)
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        data = []
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells:  # Skip header row
                data.append([cell.text for cell in cells])
        
        return data


@pytest.fixture
def streamlit_helper(browser: webdriver.Chrome) -> StreamlitTestHelper:
    """Create a StreamlitTestHelper instance"""
    return StreamlitTestHelper(browser)