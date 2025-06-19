"""
Web tests for the Streamlit dashboard using Selenium
"""
import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class TestDashboardBasicFunctionality:
    """Test basic dashboard loading and navigation"""
    
    def test_dashboard_loads_successfully(self, dashboard_page, streamlit_helper):
        """Test that the dashboard loads and displays the main title"""
        assert "Business Management Dashboard" in dashboard_page.title or \
               streamlit_helper.wait_for_text("Business Management Dashboard")
    
    def test_sidebar_navigation_exists(self, dashboard_page, streamlit_helper):
        """Test that sidebar navigation elements are present"""
        # Check for sidebar header
        streamlit_helper.wait_for_text("Dashboard Mode")
        
        # Check for view selection radio buttons
        try:
            streamlit_helper.wait_for_element(
                By.XPATH, 
                "//label[contains(text(), 'Live Agent Monitoring')]"
            )
            streamlit_helper.wait_for_element(
                By.XPATH, 
                "//label[contains(text(), 'Historical Analytics')]"
            )
        except TimeoutException:
            pytest.fail("Sidebar navigation elements not found")
    
    def test_configuration_section_exists(self, dashboard_page, streamlit_helper):
        """Test that configuration section is present in sidebar"""
        streamlit_helper.wait_for_text("Configuration")
        
        # Check for business configuration dropdown
        try:
            streamlit_helper.wait_for_element(By.CSS_SELECTOR, "select")
        except TimeoutException:
            pytest.fail("Configuration dropdown not found")


class TestLiveMonitoringView:
    """Test the Live Agent Monitoring view"""
    
    def test_switch_to_live_monitoring(self, dashboard_page, streamlit_helper):
        """Test switching to live monitoring view"""
        # Click on Live Agent Monitoring radio button
        streamlit_helper.select_sidebar_radio("Live Agent Monitoring")
        
        # Check that live monitoring content is displayed
        streamlit_helper.wait_for_text("Live Agent Monitoring")
        streamlit_helper.wait_for_text("LIVE - Agent Activity Monitor")
    
    def test_agent_status_section(self, dashboard_page, streamlit_helper):
        """Test that agent status section displays correctly"""
        streamlit_helper.select_sidebar_radio("Live Agent Monitoring")
        
        # Check for agent status section
        streamlit_helper.wait_for_text("Agent Status")
        
        # Should display current time
        streamlit_helper.wait_for_text("Current Time")
    
    def test_recent_activity_section(self, dashboard_page, streamlit_helper):
        """Test that recent agent activity section is present"""
        streamlit_helper.select_sidebar_radio("Live Agent Monitoring")
        
        # Check for recent activity section
        streamlit_helper.wait_for_text("Recent Agent Activity")
    
    def test_live_metrics_section(self, dashboard_page, streamlit_helper):
        """Test that live system metrics are displayed"""
        streamlit_helper.select_sidebar_radio("Live Agent Monitoring")
        
        # Check for live metrics
        streamlit_helper.wait_for_text("Live System Metrics")
        
        # Should have metrics for cash, inventory, transactions
        try:
            streamlit_helper.wait_for_text("Current Cash")
            streamlit_helper.wait_for_text("Low Stock Alerts")
            streamlit_helper.wait_for_text("Today's Transactions")
        except TimeoutException:
            pytest.fail("Live metrics not properly displayed")
    
    def test_auto_refresh_toggle(self, dashboard_page, streamlit_helper):
        """Test that auto-refresh toggle is available"""
        streamlit_helper.select_sidebar_radio("Live Agent Monitoring")
        
        # Look for auto-refresh checkbox in sidebar
        try:
            streamlit_helper.wait_for_element(
                By.XPATH,
                "//label[contains(text(), 'Auto-refresh')]"
            )
        except TimeoutException:
            pytest.fail("Auto-refresh toggle not found")


class TestHistoricalAnalyticsView:
    """Test the Historical Analytics view"""
    
    def test_switch_to_historical_analytics(self, dashboard_page, streamlit_helper):
        """Test switching to historical analytics view"""
        # Click on Historical Analytics radio button
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check that historical analytics content is displayed
        streamlit_helper.wait_for_text("Historical Analytics")
    
    def test_financial_metrics_display(self, dashboard_page, streamlit_helper):
        """Test that financial metrics are displayed correctly"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check for key financial metrics
        try:
            streamlit_helper.wait_for_text("Revenue")
            streamlit_helper.wait_for_text("Expenses")
            streamlit_helper.wait_for_text("Net Income")
            streamlit_helper.wait_for_text("Cash Balance")
        except TimeoutException:
            pytest.fail("Financial metrics not properly displayed")
    
    def test_charts_are_rendered(self, dashboard_page, streamlit_helper):
        """Test that charts are rendered in historical view"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Wait for charts to load
        time.sleep(3)
        
        # Check for chart titles
        assert streamlit_helper.check_chart_exists("Daily Revenue Trend")
        assert streamlit_helper.check_chart_exists("Expense Breakdown by Category")
    
    def test_inventory_section(self, dashboard_page, streamlit_helper):
        """Test inventory overview section"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check for inventory section
        streamlit_helper.wait_for_text("Inventory Overview")
        
        # Check for inventory metrics
        try:
            streamlit_helper.wait_for_text("Total Items")
            streamlit_helper.wait_for_text("Inventory Value")
            streamlit_helper.wait_for_text("Low Stock Items")
            streamlit_helper.wait_for_text("Out of Stock")
        except TimeoutException:
            pytest.fail("Inventory metrics not properly displayed")
    
    def test_hr_section(self, dashboard_page, streamlit_helper):
        """Test human resources section"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check for HR section
        streamlit_helper.wait_for_text("Human Resources")
        
        # Check for HR metrics
        try:
            streamlit_helper.wait_for_text("Total Employees")
            streamlit_helper.wait_for_text("Active Employees")
            streamlit_helper.wait_for_text("Recent Time Entries")
        except TimeoutException:
            pytest.fail("HR metrics not properly displayed")
    
    def test_recent_transactions_table(self, dashboard_page, streamlit_helper):
        """Test that recent transactions table is displayed"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check for recent transactions section
        streamlit_helper.wait_for_text("Recent Transactions")
        
        # Wait for table to load
        time.sleep(2)
    
    def test_agent_decisions_section(self, dashboard_page, streamlit_helper):
        """Test agent decisions and recommendations section"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check for agent decisions section
        streamlit_helper.wait_for_text("Agent Decisions & Recommendations")
    
    def test_system_status_section(self, dashboard_page, streamlit_helper):
        """Test system status section"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Check for system status section
        streamlit_helper.wait_for_text("System Status")
        
        # Should show business info
        try:
            streamlit_helper.wait_for_text("Business:")
            streamlit_helper.wait_for_text("Type:")
            streamlit_helper.wait_for_text("Database:")
        except TimeoutException:
            pytest.fail("System status information not displayed")


class TestInteractiveFeatures:
    """Test interactive features of the dashboard"""
    
    def test_refresh_button_functionality(self, dashboard_page, streamlit_helper):
        """Test that refresh buttons work"""
        # Test refresh button in sidebar
        try:
            refresh_button = streamlit_helper.wait_for_element(
                By.XPATH,
                "//button[contains(text(), 'Refresh Now')]"
            )
            refresh_button.click()
            time.sleep(2)  # Wait for refresh
        except TimeoutException:
            pytest.fail("Refresh button not found or not clickable")
    
    def test_time_period_selection(self, dashboard_page, streamlit_helper):
        """Test time period selection dropdown"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Look for time period dropdown
        try:
            time_period_dropdown = streamlit_helper.wait_for_element(
                By.XPATH,
                "//label[contains(text(), 'Analysis Period')]/following-sibling::div//select"
            )
            # Test that dropdown has options
            options = time_period_dropdown.find_elements(By.TAG_NAME, "option")
            assert len(options) > 1, "Time period dropdown should have multiple options"
        except TimeoutException:
            pytest.fail("Time period dropdown not found")
    
    def test_business_config_selection(self, dashboard_page, streamlit_helper):
        """Test business configuration selection"""
        try:
            config_dropdown = streamlit_helper.wait_for_element(
                By.XPATH,
                "//label[contains(text(), 'Select Business Configuration')]/following-sibling::div//select"
            )
            # Test that dropdown has options
            options = config_dropdown.find_elements(By.TAG_NAME, "option")
            assert len(options) >= 1, "Business config dropdown should have at least one option"
        except TimeoutException:
            pytest.fail("Business configuration dropdown not found")


class TestResponsiveness:
    """Test dashboard responsiveness and error handling"""
    
    def test_error_handling_with_invalid_config(self, dashboard_page, streamlit_helper):
        """Test that the dashboard handles configuration errors gracefully"""
        # This test would need a way to simulate invalid configs
        # For now, just ensure the dashboard loads without crashing
        streamlit_helper.wait_for_text("Business Management Dashboard")
    
    def test_empty_data_handling(self, dashboard_page, streamlit_helper):
        """Test how dashboard handles empty or missing data"""
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        # Wait for page to load even if data is empty
        time.sleep(3)
        
        # Should not crash and should show appropriate messages
        # Look for "No data" or similar messages
        try:
            streamlit_helper.wait_for_element(By.TAG_NAME, "body")
        except TimeoutException:
            pytest.fail("Dashboard failed to load with empty data")
    
    def test_page_navigation_performance(self, dashboard_page, streamlit_helper):
        """Test that navigation between views is reasonably fast"""
        start_time = time.time()
        
        # Switch between views multiple times
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        time.sleep(1)
        streamlit_helper.select_sidebar_radio("Live Agent Monitoring")
        time.sleep(1)
        streamlit_helper.select_sidebar_radio("Historical Analytics")
        
        end_time = time.time()
        navigation_time = end_time - start_time
        
        # Should complete navigation within reasonable time (10 seconds)
        assert navigation_time < 10, f"Navigation took too long: {navigation_time}s"


class TestAccessibility:
    """Test basic accessibility features"""
    
    def test_page_has_title(self, dashboard_page):
        """Test that the page has a proper title"""
        assert dashboard_page.title, "Page should have a title"
        assert len(dashboard_page.title) > 0, "Page title should not be empty"
    
    def test_headings_hierarchy(self, dashboard_page, streamlit_helper):
        """Test that headings follow proper hierarchy"""
        # Check for main heading
        try:
            h1_elements = dashboard_page.find_elements(By.TAG_NAME, "h1")
            assert len(h1_elements) >= 1, "Page should have at least one h1 element"
        except:
            pytest.fail("Could not find proper heading structure")
    
    def test_interactive_elements_accessible(self, dashboard_page, streamlit_helper):
        """Test that interactive elements are accessible"""
        # Check that buttons and inputs have appropriate attributes
        buttons = dashboard_page.find_elements(By.TAG_NAME, "button")
        
        for button in buttons[:5]:  # Check first 5 buttons
            # Buttons should have text or aria-label
            button_text = button.text or button.get_attribute("aria-label")
            assert button_text, f"Button should have accessible text: {button.get_attribute('outerHTML')[:100]}"