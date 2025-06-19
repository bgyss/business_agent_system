"""
Utility functions and classes for web testing
"""
import time
import subprocess
import psutil
from typing import List, Dict, Any, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import TimeoutException, WebDriverException


class DashboardTestData:
    """Test data provider for dashboard testing"""
    
    @staticmethod
    def get_sample_financial_data() -> Dict[str, Any]:
        """Get sample financial data for testing"""
        return {
            "revenue": 25000.00,
            "expenses": 18000.00,
            "net_income": 7000.00,
            "cash_balance": 15000.00
        }
    
    @staticmethod
    def get_sample_inventory_data() -> Dict[str, Any]:
        """Get sample inventory data for testing"""
        return {
            "total_items": 150,
            "total_value": 12000.00,
            "low_stock_items": 5,
            "out_of_stock_items": 2
        }
    
    @staticmethod
    def get_sample_hr_data() -> Dict[str, Any]:
        """Get sample HR data for testing"""
        return {
            "total_employees": 12,
            "active_employees": 10,
            "recent_time_entries": 45
        }


class StreamlitPageObject:
    """Page Object Model for Streamlit dashboard"""
    
    def __init__(self, driver: WebDriver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 20)
    
    # Navigation methods
    def navigate_to_live_monitoring(self):
        """Navigate to live monitoring view"""
        radio_button = self.wait.until(
            EC.element_to_be_clickable((
                By.XPATH, 
                "//label[contains(text(), 'Live Agent Monitoring')]"
            ))
        )
        radio_button.click()
        time.sleep(2)
    
    def navigate_to_historical_analytics(self):
        """Navigate to historical analytics view"""
        radio_button = self.wait.until(
            EC.element_to_be_clickable((
                By.XPATH, 
                "//label[contains(text(), 'Historical Analytics')]"
            ))
        )
        radio_button.click()
        time.sleep(2)
    
    def select_business_config(self, config_name: str):
        """Select a business configuration from dropdown"""
        dropdown = self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "select"))
        )
        
        for option in dropdown.find_elements(By.TAG_NAME, "option"):
            if config_name.lower() in option.text.lower():
                option.click()
                time.sleep(3)  # Wait for data to reload
                break
    
    def select_time_period(self, days: int):
        """Select time period for analysis"""
        time_dropdown = self.wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//label[contains(text(), 'Analysis Period')]/following-sibling::div//select"
            ))
        )
        
        for option in time_dropdown.find_elements(By.TAG_NAME, "option"):
            if str(days) in option.text:
                option.click()
                time.sleep(2)
                break
    
    # Content verification methods
    def is_live_monitoring_active(self) -> bool:
        """Check if live monitoring view is active"""
        try:
            self.wait.until(
                EC.presence_of_element_located((
                    By.XPATH, 
                    "//text()[contains(., 'LIVE - Agent Activity Monitor')]"
                ))
            )
            return True
        except TimeoutException:
            return False
    
    def is_historical_analytics_active(self) -> bool:
        """Check if historical analytics view is active"""
        try:
            self.wait.until(
                EC.presence_of_element_located((
                    By.XPATH, 
                    "//h2[contains(text(), 'Historical Analytics')]"
                ))
            )
            return True
        except TimeoutException:
            return False
    
    def get_financial_metrics(self) -> Dict[str, str]:
        """Extract financial metrics from the dashboard"""
        metrics = {}
        
        try:
            # Look for metric components
            metric_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "[data-testid='metric-container']"
            )
            
            for element in metric_elements:
                label = element.find_element(By.CSS_SELECTOR, "[data-testid='metric-label']").text
                value = element.find_element(By.CSS_SELECTOR, "[data-testid='metric-value']").text
                metrics[label] = value
                
        except Exception:
            # Fallback: try to extract metrics by text content
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                if "Revenue" in page_text:
                    metrics["extracted"] = "Found metrics in page text"
            except Exception:
                pass
        
        return metrics
    
    def get_agent_status_info(self) -> List[Dict[str, str]]:
        """Get agent status information"""
        agents = []
        
        try:
            # Look for agent status cards or sections
            agent_elements = self.driver.find_elements(
                By.XPATH,
                "//div[contains(@class, 'agent-status') or contains(text(), 'Agent')]"
            )
            
            for element in agent_elements:
                agent_info = {
                    "text": element.text,
                    "status": "unknown"
                }
                
                # Determine status based on text content or styling
                if "✅" in element.text or "running" in element.text.lower():
                    agent_info["status"] = "running"
                elif "❌" in element.text or "disabled" in element.text.lower():
                    agent_info["status"] = "disabled"
                elif "⚠️" in element.text or "warning" in element.text.lower():
                    agent_info["status"] = "warning"
                
                agents.append(agent_info)
                
        except Exception:
            pass
        
        return agents
    
    def get_chart_titles(self) -> List[str]:
        """Get titles of all charts on the page"""
        titles = []
        
        try:
            # Look for chart titles in various possible locations
            title_selectors = [
                "//div[contains(@class, 'plotly-graph-div')]//text()[contains(@class, 'gtitle')]",
                "//div[contains(text(), 'Revenue')]",
                "//div[contains(text(), 'Expense')]",
                "//div[contains(text(), 'Chart')]",
                "//div[contains(text(), 'Trend')]"
            ]
            
            for selector in title_selectors:
                elements = self.driver.find_elements(By.XPATH, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and text not in titles:
                        titles.append(text)
                        
        except Exception:
            pass
        
        return titles
    
    def refresh_dashboard(self):
        """Click the refresh button"""
        try:
            refresh_button = self.wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[contains(text(), 'Refresh')]"
                ))
            )
            refresh_button.click()
            time.sleep(3)  # Wait for refresh to complete
        except TimeoutException:
            # Fallback: refresh the page
            self.driver.refresh()
            time.sleep(5)
    
    def wait_for_loading_complete(self, timeout: int = 30):
        """Wait for the dashboard to finish loading"""
        # Wait for Streamlit to finish loading
        try:
            self.wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Check if there are any loading spinners
            loading_selectors = [
                "[data-testid='stSpinner']",
                ".stSpinner",
                "//div[contains(text(), 'Loading')]"
            ]
            
            for selector in loading_selectors:
                try:
                    WebDriverWait(self.driver, 5).until_not(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                except TimeoutException:
                    pass  # Loading element not found or already gone
                    
        except TimeoutException:
            raise TimeoutException("Dashboard failed to load within timeout period")


class WebTestReporter:
    """Utility for generating test reports and capturing screenshots"""
    
    def __init__(self, driver: WebDriver):
        self.driver = driver
    
    def capture_screenshot(self, filename: str) -> str:
        """Capture a screenshot and save to file"""
        try:
            full_path = f"test_screenshots/{filename}"
            self.driver.save_screenshot(full_path)
            return full_path
        except Exception as e:
            print(f"Failed to capture screenshot: {e}")
            return ""
    
    def capture_page_source(self, filename: str) -> str:
        """Capture page source and save to file"""
        try:
            full_path = f"test_artifacts/{filename}"
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            return full_path
        except Exception as e:
            print(f"Failed to capture page source: {e}")
            return ""
    
    def get_page_performance_info(self) -> Dict[str, Any]:
        """Get basic page performance information"""
        try:
            # Execute JavaScript to get performance information
            perf_data = self.driver.execute_script("""
                return {
                    loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
                    domReady: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                    url: window.location.href,
                    title: document.title
                };
            """)
            return perf_data
        except WebDriverException:
            return {"error": "Could not retrieve performance data"}


class StreamlitProcessManager:
    """Utility for managing Streamlit process during testing"""
    
    @staticmethod
    def find_streamlit_processes() -> List[psutil.Process]:
        """Find running Streamlit processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'streamlit' in proc.info['name'].lower():
                    processes.append(proc)
                elif proc.info['cmdline'] and any('streamlit' in arg for arg in proc.info['cmdline']):
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes
    
    @staticmethod
    def kill_streamlit_processes():
        """Kill all running Streamlit processes"""
        processes = StreamlitProcessManager.find_streamlit_processes()
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
    
    @staticmethod
    def start_streamlit_for_testing(port: int = 8502, config_file: str = "config/restaurant_config.yaml") -> subprocess.Popen:
        """Start Streamlit process for testing"""
        cmd = [
            "streamlit", "run", 
            "dashboard/app.py",
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.enableXsrfProtection", "false"
        ]
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )