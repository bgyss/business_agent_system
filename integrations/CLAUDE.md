# integrations/CLAUDE.md - External System Integrations Guide

This document provides comprehensive guidance for integrating the Business Agent Management System with external platforms and services.

## Integration Architecture

### Integration Patterns
- **REST API Integration**: Standard HTTP-based communication
- **Webhook Endpoints**: Real-time event notifications
- **OAuth 2.0 Authentication**: Secure token-based authentication
- **Rate Limiting**: Respect external platform constraints
- **Error Handling**: Retry logic with exponential backoff
- **Data Validation**: Schema validation before processing

### Common Integration Structure
```python
class BaseIntegration:
    """Base class for external system integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.rate_limiter = RateLimiter(config.get("rate_limit", 60))  # requests per minute
        
    async def authenticate(self) -> str:
        """Authenticate and return access token"""
        pass
        
    async def make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make rate-limited API request with retry logic"""
        await self.rate_limiter.acquire()
        
        for attempt in range(3):
            try:
                response = await self._execute_request(method, endpoint, data)
                return response
            except RateLimitError:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except APIError as e:
                if attempt == 2:  # Last attempt
                    raise e
                await asyncio.sleep(1)
        
    async def _execute_request(self, method: str, endpoint: str, data: Dict) -> Dict:
        """Execute HTTP request"""
        # Implementation depends on HTTP client library
        pass
```

## Restaurant POS Integrations

### Toast POS Integration
```python
class ToastIntegration(BaseIntegration):
    """Integration with Toast POS system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.restaurant_guid = config["restaurant_guid"]
        
    async def get_sales_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch sales data from Toast"""
        endpoint = f"/orders"
        params = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "pageSize": 100
        }
        
        all_orders = []
        page_token = None
        
        while True:
            if page_token:
                params["pageToken"] = page_token
                
            response = await self.make_request("GET", endpoint, params)
            
            orders = response.get("orders", [])
            all_orders.extend(orders)
            
            page_token = response.get("nextPageToken")
            if not page_token:
                break
                
        return self._transform_toast_orders(all_orders)
    
    def _transform_toast_orders(self, toast_orders: List[Dict]) -> List[Dict]:
        """Transform Toast order format to internal format"""
        transactions = []
        
        for order in toast_orders:
            transaction = {
                "amount": order["totalAmount"],
                "description": f"Toast Order #{order['orderNumber']}",
                "transaction_date": datetime.fromisoformat(order["orderDate"]),
                "reference_number": f"TOAST-{order['orderNumber']}",
                "external_id": order["guid"],
                "payment_method": order.get("paymentMethod", "unknown"),
                "items": self._extract_order_items(order)
            }
            transactions.append(transaction)
            
        return transactions
    
    async def get_menu_items(self) -> List[Dict]:
        """Fetch menu items for inventory tracking"""
        endpoint = f"/menus"
        response = await self.make_request("GET", endpoint)
        
        menu_items = []
        for menu in response.get("menus", []):
            for item in menu.get("menuItems", []):
                menu_items.append({
                    "external_id": item["guid"],
                    "name": item["name"],
                    "price": item["price"],
                    "category": item.get("category", ""),
                    "sku": item.get("sku", item["guid"]),
                    "ingredients": self._extract_ingredients(item)
                })
                
        return menu_items
```

### Square Integration
```python
class SquareIntegration(BaseIntegration):
    """Integration with Square for Restaurants"""
    
    async def get_payments(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch payment data from Square"""
        endpoint = "/v2/payments"
        
        params = {
            "begin_time": start_date.isoformat(),
            "end_time": end_date.isoformat(),
            "sort_order": "ASC",
            "limit": 100
        }
        
        payments = []
        cursor = None
        
        while True:
            if cursor:
                params["cursor"] = cursor
                
            response = await self.make_request("GET", endpoint, params)
            
            batch_payments = response.get("payments", [])
            payments.extend(batch_payments)
            
            cursor = response.get("cursor")
            if not cursor:
                break
                
        return self._transform_square_payments(payments)
    
    async def get_inventory_counts(self) -> List[Dict]:
        """Fetch inventory counts from Square"""
        endpoint = "/v2/inventory/counts"
        
        response = await self.make_request("GET", endpoint)
        
        inventory_items = []
        for count in response.get("counts", []):
            inventory_items.append({
                "catalog_object_id": count["catalog_object_id"],
                "location_id": count["location_id"],
                "quantity": count["quantity"],
                "calculated_at": count["calculated_at"]
            })
            
        return inventory_items
```

## Accounting System Integrations

### QuickBooks Online Integration
```python
class QuickBooksIntegration(BaseIntegration):
    """Integration with QuickBooks Online"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.company_id = config["company_id"]
        self.discovery_document = config["discovery_document"]
        
    async def authenticate(self) -> str:
        """OAuth 2.0 authentication with QuickBooks"""
        auth_url = f"{self.discovery_document['authorization_endpoint']}?" \
                  f"client_id={self.config['client_id']}&" \
                  f"scope=com.intuit.quickbooks.accounting&" \
                  f"redirect_uri={self.config['redirect_uri']}&" \
                  f"response_type=code&access_type=offline"
        
        # This would typically redirect user to auth_url
        # and handle the callback to get access token
        return auth_url
    
    async def sync_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Sync transactions to QuickBooks"""
        results = {"created": 0, "updated": 0, "errors": []}
        
        for transaction in transactions:
            try:
                # Check if transaction already exists
                existing = await self._find_existing_transaction(transaction)
                
                if existing:
                    # Update existing transaction
                    updated = await self._update_transaction(existing["Id"], transaction)
                    if updated:
                        results["updated"] += 1
                else:
                    # Create new transaction
                    created = await self._create_transaction(transaction)
                    if created:
                        results["created"] += 1
                        
            except Exception as e:
                results["errors"].append({
                    "transaction": transaction,
                    "error": str(e)
                })
                
        return results
    
    async def _create_transaction(self, transaction: Dict) -> Dict:
        """Create transaction in QuickBooks"""
        qb_transaction = {
            "TxnDate": transaction["transaction_date"].strftime("%Y-%m-%d"),
            "Amount": transaction["amount"],
            "Description": transaction["description"]
        }
        
        if transaction["transaction_type"] == "credit":
            # Create as Sales Receipt or Invoice
            endpoint = f"/v3/company/{self.company_id}/salesreceipt"
            qb_transaction.update({
                "CustomerRef": {"value": "1"},  # Default customer
                "Line": [{
                    "Amount": transaction["amount"],
                    "DetailType": "SalesItemLineDetail",
                    "SalesItemLineDetail": {
                        "ItemRef": {"value": "1"}  # Default item
                    }
                }]
            })
        else:
            # Create as Expense
            endpoint = f"/v3/company/{self.company_id}/purchase"
            qb_transaction.update({
                "AccountRef": {"value": transaction.get("account_id", "1")},
                "Line": [{
                    "Amount": transaction["amount"],
                    "DetailType": "AccountBasedExpenseLineDetail",
                    "AccountBasedExpenseLineDetail": {
                        "AccountRef": {"value": transaction.get("account_id", "1")}
                    }
                }]
            })
        
        response = await self.make_request("POST", endpoint, qb_transaction)
        return response
    
    async def get_chart_of_accounts(self) -> List[Dict]:
        """Fetch chart of accounts from QuickBooks"""
        endpoint = f"/v3/company/{self.company_id}/accounts"
        
        response = await self.make_request("GET", endpoint)
        
        accounts = []
        for account in response.get("QueryResponse", {}).get("Account", []):
            accounts.append({
                "id": account["Id"],
                "name": account["Name"],
                "account_type": account["AccountType"],
                "account_sub_type": account.get("AccountSubType"),
                "is_active": account.get("Active", True)
            })
            
        return accounts
```

### Xero Integration
```python
class XeroIntegration(BaseIntegration):
    """Integration with Xero accounting system"""
    
    async def sync_bank_transactions(self, transactions: List[Dict]) -> Dict:
        """Sync bank transactions to Xero"""
        endpoint = "/api.xro/2.0/BankTransactions"
        
        results = {"created": 0, "errors": []}
        
        # Batch transactions (Xero allows up to 100 per request)
        batch_size = 100
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            
            xero_transactions = []
            for transaction in batch:
                xero_transaction = {
                    "Type": "RECEIVE" if transaction["transaction_type"] == "credit" else "SPEND",
                    "Contact": {"Name": "Default Contact"},
                    "LineItems": [{
                        "Description": transaction["description"],
                        "Quantity": 1,
                        "UnitAmount": transaction["amount"],
                        "AccountCode": transaction.get("account_code", "200")
                    }],
                    "BankAccount": {"Code": self.config["bank_account_code"]},
                    "Date": transaction["transaction_date"].strftime("%Y-%m-%d"),
                    "Reference": transaction.get("reference_number", "")
                }
                xero_transactions.append(xero_transaction)
            
            try:
                batch_data = {"BankTransactions": xero_transactions}
                response = await self.make_request("POST", endpoint, batch_data)
                
                if response.get("BankTransactions"):
                    results["created"] += len(response["BankTransactions"])
                    
            except Exception as e:
                results["errors"].append({
                    "batch_start": i,
                    "error": str(e)
                })
        
        return results
```

## Inventory Management Integrations

### MarketMan Integration
```python
class MarketManIntegration(BaseIntegration):
    """Integration with MarketMan inventory management"""
    
    async def get_inventory_levels(self) -> List[Dict]:
        """Fetch current inventory levels"""
        endpoint = "/api/v3/items"
        
        response = await self.make_request("GET", endpoint)
        
        inventory_items = []
        for item in response.get("items", []):
            inventory_items.append({
                "external_id": item["id"],
                "name": item["name"],
                "current_stock": item["current_stock_quantity"],
                "unit": item["unit"],
                "cost_per_unit": item.get("cost_per_unit", 0),
                "supplier_id": item.get("vendor_id"),
                "category": item.get("category", "")
            })
            
        return inventory_items
    
    async def create_purchase_order(self, order_data: Dict) -> Dict:
        """Create purchase order in MarketMan"""
        endpoint = "/api/v3/purchase_orders"
        
        marketman_order = {
            "vendor_id": order_data["supplier_id"],
            "delivery_date": order_data["delivery_date"].isoformat(),
            "notes": order_data.get("notes", ""),
            "items": []
        }
        
        for item in order_data["items"]:
            marketman_order["items"].append({
                "item_id": item["item_id"],
                "quantity": item["quantity"],
                "unit_cost": float(item["unit_cost"])
            })
        
        response = await self.make_request("POST", endpoint, marketman_order)
        return response
    
    async def receive_delivery(self, delivery_data: Dict) -> Dict:
        """Record delivery receipt in MarketMan"""
        endpoint = f"/api/v3/purchase_orders/{delivery_data['po_id']}/receive"
        
        receipt_data = {
            "received_date": delivery_data["received_date"].isoformat(),
            "items": []
        }
        
        for item in delivery_data["items"]:
            receipt_data["items"].append({
                "item_id": item["item_id"],
                "quantity_received": item["quantity_received"],
                "quality_rating": item.get("quality_rating", 5)
            })
        
        response = await self.make_request("POST", endpoint, receipt_data)
        return response
```

## HR and Payroll Integrations

### Generic Time Tracking Integration
```python
class TimeTrackingIntegration(BaseIntegration):
    """Generic time tracking system integration"""
    
    async def sync_time_entries(self, time_entries: List[Dict]) -> Dict:
        """Sync time entries to external system"""
        results = {"synced": 0, "errors": []}
        
        for entry in time_entries:
            try:
                external_entry = {
                    "employee_id": entry["employee_id"],
                    "date": entry["clock_in"].strftime("%Y-%m-%d"),
                    "clock_in": entry["clock_in"].isoformat(),
                    "clock_out": entry["clock_out"].isoformat() if entry["clock_out"] else None,
                    "break_minutes": entry.get("break_minutes", 0),
                    "hours_worked": self._calculate_hours(entry),
                    "department": entry.get("department"),
                    "notes": entry.get("notes", "")
                }
                
                endpoint = "/api/timeentries"
                response = await self.make_request("POST", endpoint, external_entry)
                
                if response.get("success"):
                    results["synced"] += 1
                    
            except Exception as e:
                results["errors"].append({
                    "entry": entry,
                    "error": str(e)
                })
        
        return results
    
    def _calculate_hours(self, entry: Dict) -> float:
        """Calculate hours worked from time entry"""
        if not entry.get("clock_out"):
            return 0.0
            
        time_diff = entry["clock_out"] - entry["clock_in"]
        minutes_worked = time_diff.total_seconds() / 60 - entry.get("break_minutes", 0)
        
        return round(minutes_worked / 60, 2)
```

## Webhook Integration Framework

### Webhook Handler
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

class WebhookHandler:
    """Handle incoming webhooks from external systems"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.handlers = {}
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup webhook endpoints"""
        
        @self.app.post("/webhooks/{provider}")
        async def handle_webhook(provider: str, request: Request):
            """Generic webhook handler"""
            try:
                # Verify webhook signature
                if not self._verify_signature(provider, request):
                    raise HTTPException(status_code=401, detail="Invalid signature")
                
                # Parse webhook data
                webhook_data = await request.json()
                
                # Route to appropriate handler
                if provider in self.handlers:
                    result = await self.handlers[provider](webhook_data)
                    return JSONResponse(content={"status": "success", "result": result})
                else:
                    raise HTTPException(status_code=404, detail="Provider not found")
                    
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": str(e)}
                )
    
    def register_handler(self, provider: str, handler_func):
        """Register webhook handler for provider"""
        self.handlers[provider] = handler_func
    
    async def _verify_signature(self, provider: str, request: Request) -> bool:
        """Verify webhook signature"""
        # Implementation depends on provider's signature method
        return True  # Simplified for example

# Example usage
webhook_handler = WebhookHandler(app)

@webhook_handler.register_handler("toast")
async def handle_toast_webhook(data: Dict) -> Dict:
    """Handle Toast POS webhooks"""
    event_type = data.get("eventType")
    
    if event_type == "ORDER_CREATED":
        # Process new order
        order_data = data["order"]
        await process_new_order(order_data)
        
    elif event_type == "PAYMENT_PROCESSED":
        # Process payment
        payment_data = data["payment"]
        await process_payment(payment_data)
    
    return {"processed": True}
```

## Integration Monitoring and Error Handling

### Integration Health Monitoring
```python
class IntegrationMonitor:
    """Monitor integration health and performance"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
    
    async def check_integration_health(self, integration_name: str) -> Dict:
        """Check health of specific integration"""
        try:
            integration = self._get_integration(integration_name)
            
            # Test basic connectivity
            start_time = time.time()
            await integration.health_check()
            response_time = time.time() - start_time
            
            # Update metrics
            self.metrics[integration_name] = {
                "status": "healthy",
                "last_check": datetime.utcnow(),
                "response_time": response_time,
                "success_rate": self._calculate_success_rate(integration_name)
            }
            
            return self.metrics[integration_name]
            
        except Exception as e:
            self.metrics[integration_name] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow(),
                "error": str(e)
            }
            
            # Trigger alert
            await self._trigger_alert(integration_name, str(e))
            
            return self.metrics[integration_name]
    
    async def _trigger_alert(self, integration_name: str, error: str):
        """Trigger alert for integration failure"""
        alert = {
            "integration": integration_name,
            "error": error,
            "timestamp": datetime.utcnow(),
            "severity": "high"
        }
        
        self.alerts.append(alert)
        
        # Send notification (email, Slack, etc.)
        await self._send_notification(alert)
```

### Rate Limiting Implementation
```python
import asyncio
from collections import deque
from time import time

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make request"""
        async with self._lock:
            now = time()
            
            # Remove old requests outside time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make request
            if len(self.requests) >= self.max_requests:
                # Wait until we can make request
                wait_time = self.requests[0] + self.time_window - now
                await asyncio.sleep(wait_time)
                return await self.acquire()  # Retry
            
            # Record request time
            self.requests.append(now)
```

## Security Best Practices

### Secure Credential Management
```python
import os
from cryptography.fernet import Fernet

class CredentialManager:
    """Secure credential storage and retrieval"""
    
    def __init__(self):
        self.cipher_suite = Fernet(self._get_or_create_key())
    
    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or create new one"""
        key = os.getenv("INTEGRATION_ENCRYPTION_KEY")
        if not key:
            key = Fernet.generate_key()
            # In production, store this securely
        return key.encode() if isinstance(key, str) else key
    
    def store_credential(self, integration_name: str, credential_data: Dict):
        """Encrypt and store integration credentials"""
        encrypted_data = self.cipher_suite.encrypt(
            json.dumps(credential_data).encode()
        )
        
        # Store in secure location (database, key vault, etc.)
        self._save_encrypted_credential(integration_name, encrypted_data)
    
    def get_credential(self, integration_name: str) -> Dict:
        """Retrieve and decrypt integration credentials"""
        encrypted_data = self._load_encrypted_credential(integration_name)
        
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
```

---

*This document should be updated when new integrations are added or when integration patterns change significantly.*