# deployment/CLAUDE.md - Deployment, Monitoring, and Maintenance Guide

This document provides comprehensive guidance for deploying, monitoring, and maintaining the Business Agent Management System in production environments.

## Deployment Guide

### Development Deployment
```bash
# Local development
nix develop
make install
make run-restaurant

# Docker development
docker-compose up business-agent-restaurant dashboard
```

### Production Deployment
```bash
# Using Nix
nix build
nix run .#restaurant

# Using Docker
docker-compose -f docker-compose.yml up -d

# Environment variables required:
# - ANTHROPIC_API_KEY
# - DATABASE_URL (for PostgreSQL)
# - REDIS_URL (for message queue)
```

### Environment Configuration

#### Production Environment Variables
```bash
# Core Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/business_agents
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration (for message queue)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# Claude API Configuration
ANTHROPIC_API_KEY=your_claude_api_key
CLAUDE_MODEL=claude-3-sonnet-20240229

# Security Configuration
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# Monitoring Configuration
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090

# External Integrations
QUICKBOOKS_CLIENT_ID=your_qb_client_id
QUICKBOOKS_CLIENT_SECRET=your_qb_client_secret
TOAST_API_KEY=your_toast_api_key
```

#### Docker Deployment
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  business-agent-app:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/business_agents
      - REDIS_URL=redis://redis:6379/0
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - db
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: business_agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    restart: unless-stopped
  
  redis:
    image: redis:7
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
  
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/business_agents
    depends_on:
      - db
    ports:
      - "8501:8501"
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - business-agent-app
      - dashboard
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: business-agent-system
  labels:
    app: business-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: business-agent-system
  template:
    metadata:
      labels:
        app: business-agent-system
    spec:
      containers:
      - name: business-agent-app
        image: business-agent-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: business-agent-service
spec:
  selector:
    app: business-agent-system
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring and Observability

### Application Metrics

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Agent decision metrics
agent_decisions_total = Counter(
    'agent_decisions_total',
    'Total number of agent decisions made',
    ['agent_id', 'decision_type']
)

agent_decision_duration = Histogram(
    'agent_decision_duration_seconds',
    'Time spent processing agent decisions',
    ['agent_id']
)

agent_confidence_score = Gauge(
    'agent_confidence_score',
    'Current confidence score of agent decisions',
    ['agent_id']
)

# API metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request processing time'
)

# Database metrics
db_connections_active = Gauge(
    'db_connections_active',
    'Number of active database connections'
)

db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query execution time',
    ['query_type']
)

# Integration metrics
integration_requests_total = Counter(
    'integration_requests_total',
    'Total requests to external integrations',
    ['integration', 'status']
)

integration_response_time = Histogram(
    'integration_response_time_seconds',
    'Response time for external integrations',
    ['integration']
)

class MetricsCollector:
    """Collect and expose application metrics"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        start_http_server(port)
    
    def record_agent_decision(self, agent_id: str, decision_type: str, duration: float, confidence: float):
        """Record agent decision metrics"""
        agent_decisions_total.labels(agent_id=agent_id, decision_type=decision_type).inc()
        agent_decision_duration.labels(agent_id=agent_id).observe(duration)
        agent_confidence_score.labels(agent_id=agent_id).set(confidence)
    
    def record_api_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record API request metrics"""
        api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        api_request_duration.observe(duration)
    
    def record_db_operation(self, query_type: str, duration: float, connections: int):
        """Record database operation metrics"""
        db_query_duration.labels(query_type=query_type).observe(duration)
        db_connections_active.set(connections)
```

#### Health Check Endpoints
```python
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI()

class HealthChecker:
    """Application health checking"""
    
    @staticmethod
    async def check_database() -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Test database connection
            async with get_db_session() as session:
                result = await session.execute("SELECT 1")
                return {"status": "healthy", "latency": "low"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @staticmethod
    async def check_claude_api() -> Dict[str, Any]:
        """Check Claude API connectivity"""
        try:
            # Test Claude API with minimal request
            client = get_claude_client()
            response = await client.messages.create(
                model="claude-3-haiku-20240307",  # Use cheaper model for health checks
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return {"status": "healthy", "model": "accessible"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @staticmethod
    async def check_redis() -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            # Test Redis connection
            redis_client = get_redis_client()
            await redis_client.ping()
            return {"status": "healthy", "connection": "active"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @staticmethod
    async def check_agents() -> Dict[str, Any]:
        """Check agent health"""
        agent_health = {}
        
        for agent_id in ["accounting_agent", "inventory_agent", "hr_agent"]:
            try:
                agent = get_agent(agent_id)
                health = await agent.health_check()
                agent_health[agent_id] = health
            except Exception as e:
                agent_health[agent_id] = {"status": "unhealthy", "error": str(e)}
        
        return agent_health

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    checker = HealthChecker()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": await checker.check_database(),
            "claude_api": await checker.check_claude_api(),
            "redis": await checker.check_redis(),
            "agents": await checker.check_agents()
        }
    }
    
    # Determine overall health
    component_statuses = [
        comp["status"] for comp in health_status["components"].values()
    ]
    
    if "unhealthy" in component_statuses:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    elif any(isinstance(comp, dict) and any(
        agent["status"] == "unhealthy" for agent in comp.values()
    ) for comp in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Quick checks for readiness
        checker = HealthChecker()
        db_status = await checker.check_database()
        
        if db_status["status"] != "healthy":
            raise HTTPException(status_code=503, detail="Not ready")
        
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Not ready")
```

### Logging Configuration

#### Structured Logging
```python
import structlog
import logging
from pythonjsonlogger import jsonlogger

def configure_logging():
    """Configure structured logging for production"""
    
    # Configure standard library logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

class AgentLogger:
    """Specialized logger for agent operations"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = structlog.get_logger("agent").bind(agent_id=agent_id)
    
    def log_decision(self, decision: AgentDecision, execution_time: float):
        """Log agent decision with context"""
        self.logger.info(
            "Agent decision made",
            decision_type=decision.decision_type,
            confidence=decision.confidence,
            execution_time=execution_time,
            action=decision.action[:100] + "..." if len(decision.action) > 100 else decision.action
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log agent error with context"""
        self.logger.error(
            "Agent error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration=duration,
            **kwargs
        )
```

### Error Tracking and Alerting

#### Sentry Integration
```python
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

def configure_sentry():
    """Configure Sentry for error tracking"""
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("ENVIRONMENT", "development"),
        integrations=[
            SqlalchemyIntegration(),
            RedisIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
        ],
        traces_sample_rate=0.1,  # Sample 10% of transactions
        before_send=filter_sensitive_data
    )

def filter_sensitive_data(event, hint):
    """Filter sensitive data from Sentry events"""
    # Remove API keys and credentials
    if 'extra' in event:
        for key in list(event['extra'].keys()):
            if any(sensitive in key.lower() for sensitive in ['key', 'password', 'token', 'secret']):
                event['extra'][key] = '[REDACTED]'
    
    return event

class ErrorHandler:
    """Centralized error handling and reporting"""
    
    @staticmethod
    def capture_agent_error(agent_id: str, error: Exception, context: Dict = None):
        """Capture agent-specific errors"""
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("component", "agent")
            scope.set_tag("agent_id", agent_id)
            scope.set_context("agent_context", context or {})
            sentry_sdk.capture_exception(error)
    
    @staticmethod
    def capture_integration_error(integration: str, error: Exception, request_data: Dict = None):
        """Capture integration-specific errors"""
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("component", "integration")
            scope.set_tag("integration", integration)
            scope.set_context("request_data", request_data or {})
            sentry_sdk.capture_exception(error)
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Maintenance
```python
class DailyMaintenance:
    """Daily maintenance tasks"""
    
    @staticmethod
    async def cleanup_old_logs():
        """Clean up log files older than 30 days"""
        log_directory = Path("logs")
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for log_file in log_directory.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                logger.info(f"Deleted old log file: {log_file}")
    
    @staticmethod
    async def vacuum_database():
        """Vacuum database to reclaim space"""
        async with get_db_session() as session:
            if "postgresql" in str(session.bind.url):
                await session.execute("VACUUM ANALYZE;")
                logger.info("Database vacuum completed")
    
    @staticmethod
    async def check_agent_health():
        """Check and report agent health"""
        health_checker = HealthChecker()
        agent_health = await health_checker.check_agents()
        
        unhealthy_agents = [
            agent_id for agent_id, health in agent_health.items()
            if health.get("status") != "healthy"
        ]
        
        if unhealthy_agents:
            await send_alert(f"Unhealthy agents detected: {', '.join(unhealthy_agents)}")
    
    @staticmethod
    async def backup_database():
        """Create daily database backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backups/daily_backup_{timestamp}.sql"
        
        # PostgreSQL backup
        import subprocess
        result = subprocess.run([
            "pg_dump",
            os.getenv("DATABASE_URL"),
            "-f", backup_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Database backup created: {backup_file}")
        else:
            logger.error(f"Database backup failed: {result.stderr}")
```

#### Weekly Maintenance
```python
class WeeklyMaintenance:
    """Weekly maintenance tasks"""
    
    @staticmethod
    async def analyze_agent_performance():
        """Analyze agent performance trends"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        performance_report = {}
        
        for agent_id in ["accounting_agent", "inventory_agent", "hr_agent"]:
            decisions = await get_agent_decisions(agent_id, start_date, end_date)
            
            performance_report[agent_id] = {
                "total_decisions": len(decisions),
                "avg_confidence": sum(d.confidence for d in decisions) / len(decisions) if decisions else 0,
                "decision_types": {}
            }
            
            # Group by decision type
            for decision in decisions:
                decision_type = decision.decision_type
                if decision_type not in performance_report[agent_id]["decision_types"]:
                    performance_report[agent_id]["decision_types"][decision_type] = 0
                performance_report[agent_id]["decision_types"][decision_type] += 1
        
        # Generate report
        await generate_performance_report(performance_report)
    
    @staticmethod
    async def optimize_database():
        """Optimize database performance"""
        async with get_db_session() as session:
            # Update table statistics
            await session.execute("ANALYZE;")
            
            # Reindex tables if needed
            tables = ["agent_decisions", "transactions", "stock_movements"]
            for table in tables:
                await session.execute(f"REINDEX TABLE {table};")
        
        logger.info("Database optimization completed")
    
    @staticmethod
    async def review_integration_health():
        """Review integration health and performance"""
        integrations = ["quickbooks", "toast", "square", "xero"]
        integration_report = {}
        
        for integration in integrations:
            try:
                health = await check_integration_health(integration)
                integration_report[integration] = health
            except Exception as e:
                integration_report[integration] = {"status": "error", "error": str(e)}
        
        # Send weekly integration report
        await send_integration_report(integration_report)
```

#### Monthly Maintenance
```python
class MonthlyMaintenance:
    """Monthly maintenance tasks"""
    
    @staticmethod
    async def archive_old_data():
        """Archive data older than 1 year"""
        cutoff_date = datetime.now() - timedelta(days=365)
        
        async with get_db_session() as session:
            # Archive old agent decisions
            old_decisions = await session.execute(
                select(AgentDecision).where(AgentDecision.timestamp < cutoff_date)
            )
            decisions_to_archive = old_decisions.fetchall()
            
            if decisions_to_archive:
                # Export to archive
                await export_to_archive("agent_decisions", decisions_to_archive)
                
                # Delete from main table
                await session.execute(
                    delete(AgentDecision).where(AgentDecision.timestamp < cutoff_date)
                )
                
                logger.info(f"Archived {len(decisions_to_archive)} old decisions")
    
    @staticmethod
    async def update_dependencies():
        """Check for dependency updates"""
        # This would typically integrate with dependency management tools
        logger.info("Checking for dependency updates...")
        
        # Run security audit
        result = subprocess.run(["uv", "audit"], capture_output=True, text=True)
        if result.returncode != 0:
            await send_alert(f"Security vulnerabilities detected: {result.stdout}")
    
    @staticmethod
    async def performance_review():
        """Comprehensive performance review"""
        # Generate monthly performance metrics
        metrics = await collect_monthly_metrics()
        
        # Check for performance degradation
        issues = []
        if metrics["avg_response_time"] > 2.0:  # 2 second threshold
            issues.append("High response times detected")
        
        if metrics["error_rate"] > 0.05:  # 5% error rate threshold
            issues.append("High error rate detected")
        
        if issues:
            await send_alert(f"Performance issues: {', '.join(issues)}")
```

### Troubleshooting Common Issues

#### Agent Performance Issues
```python
class TroubleshootingGuide:
    """Common troubleshooting procedures"""
    
    @staticmethod
    async def diagnose_slow_agents():
        """Diagnose and fix slow agent performance"""
        # Check Claude API response times
        api_metrics = await get_claude_api_metrics()
        if api_metrics["avg_response_time"] > 10:
            logger.warning("Claude API response times are high")
            # Implement circuit breaker or fallback logic
        
        # Check database query performance
        db_metrics = await get_db_metrics()
        slow_queries = [q for q in db_metrics["queries"] if q["duration"] > 5]
        if slow_queries:
            logger.warning(f"Slow database queries detected: {len(slow_queries)}")
            # Suggest index optimization
        
        # Check memory usage
        memory_usage = await get_memory_usage()
        if memory_usage > 0.8:  # 80% memory usage
            logger.warning("High memory usage detected")
            # Suggest increasing memory or optimizing code
    
    @staticmethod
    async def fix_integration_failures():
        """Diagnose and fix integration failures"""
        failed_integrations = await get_failed_integrations()
        
        for integration, error in failed_integrations.items():
            if "rate limit" in error.lower():
                # Implement exponential backoff
                await implement_rate_limiting(integration)
            elif "authentication" in error.lower():
                # Refresh authentication tokens
                await refresh_integration_auth(integration)
            elif "network" in error.lower():
                # Check network connectivity
                await check_network_connectivity(integration)
    
    @staticmethod
    async def resolve_database_issues():
        """Resolve common database issues"""
        # Check connection pool
        pool_stats = await get_connection_pool_stats()
        if pool_stats["active_connections"] >= pool_stats["max_connections"]:
            logger.error("Database connection pool exhausted")
            # Increase pool size or find connection leaks
        
        # Check for long-running transactions
        long_transactions = await get_long_running_transactions()
        if long_transactions:
            logger.warning(f"Long-running transactions detected: {len(long_transactions)}")
            # Consider killing long transactions or optimizing queries
        
        # Check disk space
        disk_usage = await get_database_disk_usage()
        if disk_usage > 0.9:  # 90% disk usage
            logger.error("Database disk space running low")
            await send_alert("Database disk space critical")
```

### Disaster Recovery

#### Backup and Recovery Procedures
```python
class DisasterRecovery:
    """Disaster recovery procedures"""
    
    @staticmethod
    async def create_full_backup():
        """Create complete system backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Database backup
        db_backup = f"backups/full_db_backup_{timestamp}.sql"
        await backup_database(db_backup)
        
        # Redis backup
        redis_backup = f"backups/redis_backup_{timestamp}.rdb"
        await backup_redis(redis_backup)
        
        # Configuration backup
        config_backup = f"backups/config_backup_{timestamp}.tar.gz"
        await backup_configuration(config_backup)
        
        # Upload to cloud storage
        await upload_to_cloud_storage([db_backup, redis_backup, config_backup])
        
        logger.info(f"Full backup completed: {timestamp}")
    
    @staticmethod
    async def restore_from_backup(backup_timestamp: str):
        """Restore system from backup"""
        logger.info(f"Starting restore from backup: {backup_timestamp}")
        
        try:
            # Stop all services
            await stop_all_services()
            
            # Download backup from cloud storage
            backup_files = await download_from_cloud_storage(backup_timestamp)
            
            # Restore database
            await restore_database(backup_files["database"])
            
            # Restore Redis
            await restore_redis(backup_files["redis"])
            
            # Restore configuration
            await restore_configuration(backup_files["config"])
            
            # Start services
            await start_all_services()
            
            # Verify restoration
            await verify_system_health()
            
            logger.info("System restoration completed successfully")
            
        except Exception as e:
            logger.error(f"Restoration failed: {e}")
            await send_alert(f"Disaster recovery failed: {e}")
            raise
    
    @staticmethod
    async def failover_to_secondary():
        """Failover to secondary environment"""
        logger.info("Initiating failover to secondary environment")
        
        # Update DNS to point to secondary
        await update_dns_records("secondary")
        
        # Start services on secondary
        await start_secondary_services()
        
        # Verify secondary is healthy
        await verify_secondary_health()
        
        logger.info("Failover completed")
```

---

*This document should be updated when deployment procedures change or when new monitoring requirements are added.*