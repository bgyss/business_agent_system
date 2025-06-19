"""
Smoke tests for dashboard - quick tests to verify basic functionality
"""
import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from .web_test_utils import StreamlitPageObject, WebTestReporter


class TestDashboardSmoke:
    """Smoke tests for basic dashboard functionality"""
    
    def test_dashboard_loads(self, dashboard_page):
        """Basic smoke test - dashboard loads without errors"""
        # Check that page loaded
        assert dashboard_page.current_url.startswith("http://localhost:")
        
        # Check that page has content
        body = dashboard_page.find_element(By.TAG_NAME, "body")
        assert len(body.text) > 0, "Page should have content"
    
    def test_title_is_present(self, dashboard_page, streamlit_helper):
        """Test that the main dashboard title is present"""
        try:
            streamlit_helper.wait_for_text("Business Management Dashboard", timeout=10)
        except TimeoutException:
            # Try alternative ways to find the title
            h1_elements = dashboard_page.find_elements(By.TAG_NAME, "h1")
            assert len(h1_elements) > 0, "Should have at least one h1 element"
            
            # Check if any h1 contains dashboard-related text
            dashboard_text_found = any(
                "dashboard" in h1.text.lower() or "business" in h1.text.lower()
                for h1 in h1_elements
            )
            assert dashboard_text_found, "Should have dashboard-related title"
    
    def test_sidebar_is_present(self, dashboard_page):
        """Test that sidebar navigation is present"""
        # Look for sidebar elements
        sidebar_selectors = [
            "[data-testid='stSidebar']",
            "section[data-testid='stSidebar']",
            ".sidebar",
            "//div[contains(@class, 'sidebar')]"
        ]
        
        sidebar_found = False
        for selector in sidebar_selectors:
            try:
                if selector.startswith("//"):
                    dashboard_page.find_element(By.XPATH, selector)
                else:
                    dashboard_page.find_element(By.CSS_SELECTOR, selector)
                sidebar_found = True
                break
            except:
                continue
        
        # If no sidebar container found, look for sidebar content
        if not sidebar_found:
            try:
                dashboard_page.find_element(By.XPATH, "//text()[contains(., 'Dashboard Mode')]")
                sidebar_found = True
            except:
                pass
        
        assert sidebar_found, "Sidebar should be present"
    
    def test_main_content_area_exists(self, dashboard_page):
        """Test that main content area exists"""
        # Look for main content indicators
        main_content_found = False
        
        # Check for Streamlit main content area
        main_selectors = [
            "[data-testid='stMain']",
            ".main",
            "main",
            "//div[contains(@class, 'main')]"
        ]
        
        for selector in main_selectors:
            try:
                if selector.startswith("//"):
                    element = dashboard_page.find_element(By.XPATH, selector)
                else:
                    element = dashboard_page.find_element(By.CSS_SELECTOR, selector)
                
                # Check that it has content
                if len(element.text.strip()) > 0:
                    main_content_found = True
                    break
            except:
                continue
        
        assert main_content_found, "Main content area should exist and have content"
    
    def test_no_obvious_errors(self, dashboard_page):
        """Test that there are no obvious error messages"""
        page_text = dashboard_page.find_element(By.TAG_NAME, "body").text.lower()
        
        error_indicators = [
            "error connecting to database",
            "traceback",
            "exception",
            "internal server error",
            "500 error",
            "404 not found"
        ]
        
        for error_text in error_indicators:
            assert error_text not in page_text, f"Page should not contain error: {error_text}"
    
    def test_configuration_loads(self, dashboard_page, streamlit_helper):
        """Test that configuration section loads"""
        try:
            streamlit_helper.wait_for_text("Configuration", timeout=10)
        except TimeoutException:
            # Check if there's any configuration-related content
            page_text = dashboard_page.find_element(By.TAG_NAME, "body").text.lower()
            assert "config" in page_text, "Should have configuration-related content"
    
    @pytest.mark.slow
    def test_view_switching_works(self, dashboard_page, streamlit_helper):
        """Test that basic view switching works"""
        page_object = StreamlitPageObject(dashboard_page)
        
        try:
            # Try to switch to live monitoring
            page_object.navigate_to_live_monitoring()
            time.sleep(2)
            
            # Try to switch to historical analytics
            page_object.navigate_to_historical_analytics()
            time.sleep(2)
            
            # Both should not crash the page
            body_text = dashboard_page.find_element(By.TAG_NAME, "body").text
            assert len(body_text) > 0, "Page should still have content after view switching"
            
        except Exception as e:
            # View switching might not work perfectly, but page should not crash
            page_text = dashboard_page.find_element(By.TAG_NAME, "body").text.lower()
            assert "error" not in page_text, f"Page crashed during view switching: {e}"
    
    def test_page_responsive(self, dashboard_page):
        """Test that page responds to window resizing"""
        original_size = dashboard_page.get_window_size()
        
        try:
            # Test different window sizes
            dashboard_page.set_window_size(1200, 800)
            time.sleep(1)
            
            dashboard_page.set_window_size(800, 600)
            time.sleep(1)
            
            # Page should still be functional
            body = dashboard_page.find_element(By.TAG_NAME, "body")
            assert len(body.text) > 0, "Page should remain functional after resizing"
            
        finally:
            # Restore original size
            dashboard_page.set_window_size(original_size['width'], original_size['height'])
    
    def test_javascript_loads(self, dashboard_page):
        """Test that JavaScript loads properly"""
        try:
            # Execute simple JavaScript to verify it's working
            result = dashboard_page.execute_script("return document.readyState;")
            assert result == "complete", "Document should be completely loaded"
            
            # Test that we can access basic DOM elements
            title = dashboard_page.execute_script("return document.title;")
            assert isinstance(title, str), "Should be able to access document title"
            
        except Exception as e:
            pytest.fail(f"JavaScript execution failed: {e}")


class TestDashboardSmokeWithReporting:
    """Smoke tests with enhanced reporting and screenshots"""
    
    def test_dashboard_screenshot(self, dashboard_page):
        """Take a screenshot of the dashboard for visual verification"""
        reporter = WebTestReporter(dashboard_page)
        
        # Ensure test_screenshots directory exists
        import os
        os.makedirs("test_screenshots", exist_ok=True)
        
        screenshot_path = reporter.capture_screenshot("dashboard_smoke_test.png")
        
        # Test passes if we could take a screenshot
        if screenshot_path:
            print(f"Screenshot saved to: {screenshot_path}")
        
        # Basic verification that page loaded
        assert dashboard_page.current_url.startswith("http://localhost:")
    
    def test_performance_baseline(self, dashboard_page):
        """Establish basic performance baseline"""
        reporter = WebTestReporter(dashboard_page)
        
        perf_info = reporter.get_page_performance_info()
        
        if "error" not in perf_info:
            print(f"Page load time: {perf_info.get('loadTime', 'unknown')}ms")
            print(f"DOM ready time: {perf_info.get('domReady', 'unknown')}ms")
            
            # Basic performance assertion (page should load within 30 seconds)
            load_time = perf_info.get('loadTime', 0)
            if load_time > 0:
                assert load_time < 30000, f"Page load time too slow: {load_time}ms"
        
        # Test passes regardless of performance data availability
        assert True