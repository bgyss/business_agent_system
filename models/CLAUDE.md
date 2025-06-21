# models/CLAUDE.md - Data Models and Database Guide

This document provides comprehensive guidance for working with data models, database operations, and schema management in the Business Agent Management System.

## Data Model Architecture

### Core Model Categories

**Financial Models** (`financial.py`):
- **TransactionModel**: Individual financial transactions
- **AccountModel**: Chart of accounts structure
- **ReceivableModel**: Outstanding customer payments
- **PayableModel**: Outstanding supplier payments

**Inventory Models** (`inventory.py`):
- **Item**: Product/service catalog items
- **StockMovement**: Inventory transaction history
- **PurchaseOrder**: Supplier purchase orders
- **Supplier**: Vendor information and performance
- **InventorySummary**: Aggregate inventory reporting

**Employee Models** (`employee.py`):
- **Employee**: Staff records and information
- **TimeEntry**: Work time tracking
- **Schedule**: Staff scheduling and shifts
- **PayrollRecord**: Compensation and benefits

**Agent Decision Models** (`agent_decisions.py`):
- **AgentDecision**: AI agent decision logging
- **AgentMessage**: Inter-agent communication
- **DecisionContext**: Decision reasoning and metadata

## Database Design Patterns

### SQLAlchemy ORM Integration
```python
from sqlalchemy import Column, Integer, String, Decimal, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class BaseModel(Base):
    """Abstract base model with common fields"""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Pydantic Validation Integration
```python
from pydantic import BaseModel, validator
from decimal import Decimal
from typing import Optional

class TransactionCreate(BaseModel):
    """Pydantic model for transaction creation"""
    amount: Decimal
    description: str
    account_id: int
    transaction_date: Optional[datetime] = None
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }
```

## Financial Data Models

### Transaction Model
```python
class TransactionModel(BaseModel):
    """Core financial transaction model"""
    id: int
    amount: Decimal
    description: str
    account_id: int
    transaction_date: datetime
    transaction_type: str  # "credit" or "debit"
    reference_number: Optional[str] = None
    
    # Relationships
    account: Optional[AccountModel] = None
    
    class Config:
        orm_mode = True
        
    def is_credit(self) -> bool:
        """Check if transaction is a credit"""
        return self.transaction_type == "credit"
    
    def is_debit(self) -> bool:
        """Check if transaction is a debit"""
        return self.transaction_type == "debit"
```

### Account Model
```python
class AccountModel(BaseModel):
    """Chart of accounts model"""
    id: int
    account_number: str
    account_name: str
    account_type: str  # "asset", "liability", "equity", "revenue", "expense"
    parent_account_id: Optional[int] = None
    is_active: bool = True
    
    # Relationships
    transactions: List[TransactionModel] = []
    sub_accounts: List['AccountModel'] = []
    
    def get_balance(self) -> Decimal:
        """Calculate current account balance"""
        credits = sum(t.amount for t in self.transactions if t.is_credit())
        debits = sum(t.amount for t in self.transactions if t.is_debit())
        
        if self.account_type in ["asset", "expense"]:
            return debits - credits
        else:  # liability, equity, revenue
            return credits - debits
```

## Inventory Data Models

### Item Model
```python
class Item(BaseModel):
    """Inventory item model"""
    id: int
    sku: str
    name: str
    description: Optional[str] = None
    current_stock: int = 0
    reorder_point: int = 0
    reorder_quantity: int = 0
    unit_cost: Decimal
    selling_price: Optional[Decimal] = None
    supplier_id: Optional[int] = None
    category: Optional[str] = None
    status: ItemStatus = ItemStatus.ACTIVE
    
    # Inventory tracking
    stock_movements: List[StockMovement] = []
    
    class Config:
        orm_mode = True
    
    @property
    def is_low_stock(self) -> bool:
        """Check if item is below reorder point"""
        return self.current_stock <= self.reorder_point
    
    @property
    def stock_value(self) -> Decimal:
        """Calculate total stock value"""
        return Decimal(self.current_stock) * self.unit_cost
```

### Stock Movement Model
```python
class StockMovement(BaseModel):
    """Inventory movement tracking"""
    id: int
    item_id: int
    movement_type: StockMovementType  # IN, OUT, ADJUSTMENT
    quantity: int
    movement_date: datetime
    reference_number: Optional[str] = None
    notes: Optional[str] = None
    
    # Cost tracking
    unit_cost: Optional[Decimal] = None
    total_cost: Optional[Decimal] = None
    
    # Relationships
    item: Optional[Item] = None
    
    def update_item_stock(self):
        """Update item stock based on movement"""
        if self.movement_type == StockMovementType.IN:
            self.item.current_stock += self.quantity
        elif self.movement_type == StockMovementType.OUT:
            self.item.current_stock -= self.quantity
        # ADJUSTMENT movements set absolute quantity
```

## Employee Data Models

### Employee Model
```python
class Employee(BaseModel):
    """Employee information model"""
    id: int
    employee_id: str
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    hire_date: datetime
    hourly_rate: Decimal
    department: Optional[str] = None
    position: Optional[str] = None
    status: EmployeeStatus = EmployeeStatus.ACTIVE
    
    # Relationships
    time_entries: List[TimeEntry] = []
    schedules: List[Schedule] = []
    
    @property
    def full_name(self) -> str:
        """Get employee full name"""
        return f"{self.first_name} {self.last_name}"
    
    def calculate_weekly_hours(self, week_start: datetime) -> Decimal:
        """Calculate hours worked in a specific week"""
        week_end = week_start + timedelta(days=7)
        
        weekly_entries = [
            entry for entry in self.time_entries
            if week_start <= entry.clock_in <= week_end
        ]
        
        return sum(entry.hours_worked for entry in weekly_entries)
```

### Time Entry Model
```python
class TimeEntry(BaseModel):
    """Employee time tracking"""
    id: int
    employee_id: int
    clock_in: datetime
    clock_out: Optional[datetime] = None
    break_minutes: int = 0
    notes: Optional[str] = None
    
    # Relationships
    employee: Optional[Employee] = None
    
    @property
    def hours_worked(self) -> Decimal:
        """Calculate hours worked"""
        if not self.clock_out:
            return Decimal('0')
        
        total_time = self.clock_out - self.clock_in
        total_minutes = total_time.total_seconds() / 60 - self.break_minutes
        
        return Decimal(total_minutes / 60).quantize(Decimal('0.01'))
    
    @property
    def is_overtime(self) -> bool:
        """Check if this entry includes overtime"""
        return self.hours_worked > Decimal('8')
```

## Agent Decision Models

### AgentDecision Model
```python
class AgentDecision(BaseModel):
    """AI agent decision logging"""
    id: Optional[int] = None
    agent_id: str
    decision_type: str
    context: Dict[str, Any]
    reasoning: str
    action: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Execution tracking
    executed: bool = False
    execution_result: Optional[str] = None
    execution_timestamp: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
    
    @validator('confidence')
    def confidence_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
```

## Database Management

### Schema Updates and Migrations
```python
# Use SQLAlchemy migrations for schema changes
from alembic import command
from alembic.config import Config

def run_migrations():
    """Run database migrations"""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

def create_migration(message: str):
    """Create new migration"""
    alembic_cfg = Config("alembic.ini")
    command.revision(alembic_cfg, message=message, autogenerate=True)
```

### Database Session Management
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
```

### Query Optimization Patterns
```python
class OptimizedQueries:
    """Database query optimization patterns"""
    
    @staticmethod
    def get_low_stock_items(session, threshold_multiplier: float = 1.2):
        """Efficiently query low stock items"""
        return session.query(Item).filter(
            Item.current_stock <= Item.reorder_point * threshold_multiplier,
            Item.status == ItemStatus.ACTIVE
        ).options(
            selectinload(Item.stock_movements),  # Eager load relationships
            selectinload(Item.supplier)
        ).all()
    
    @staticmethod
    def get_recent_transactions(session, days: int = 30):
        """Get recent transactions with pagination"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return session.query(TransactionModel).filter(
            TransactionModel.transaction_date >= cutoff_date
        ).options(
            selectinload(TransactionModel.account)
        ).order_by(
            TransactionModel.transaction_date.desc()
        ).limit(1000).all()  # Limit to prevent memory issues
```

### Performance Optimization

#### Indexing Strategy
```sql
-- Critical indexes for agent queries
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_account ON transactions(account_id);
CREATE INDEX idx_stock_movements_item_date ON stock_movements(item_id, movement_date);
CREATE INDEX idx_stock_movements_type ON stock_movements(movement_type);
CREATE INDEX idx_items_status ON items(status);
CREATE INDEX idx_agent_decisions_timestamp ON agent_decisions(timestamp);
CREATE INDEX idx_agent_decisions_agent_type ON agent_decisions(agent_id, decision_type);
```

#### Connection Pooling
```python
# Production database configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Number of persistent connections
    max_overflow=30,        # Additional connections when needed
    pool_timeout=30,        # Timeout for getting connection from pool
    pool_recycle=3600,      # Recycle connections every hour
    pool_pre_ping=True,     # Validate connections before use
    echo=False              # Disable SQL logging in production
)
```

### Data Validation and Constraints

#### Model-Level Validation
```python
class ValidatedTransactionModel(TransactionModel):
    """Transaction model with enhanced validation"""
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Transaction amount must be positive')
        if v > Decimal('1000000'):
            raise ValueError('Transaction amount exceeds maximum limit')
        return v.quantize(Decimal('0.01'))  # Round to 2 decimal places
    
    @validator('account_id')
    def validate_account_exists(cls, v, values):
        # In a real implementation, this would check database
        if v <= 0:
            raise ValueError('Invalid account ID')
        return v
    
    @validator('transaction_date')
    def validate_transaction_date(cls, v):
        if v > datetime.utcnow():
            raise ValueError('Transaction date cannot be in the future')
        if v < datetime.utcnow() - timedelta(days=365):
            raise ValueError('Transaction date too far in the past')
        return v
```

#### Database-Level Constraints
```python
class ConstrainedModels:
    """Models with database-level constraints"""
    
    class Item(Base):
        __tablename__ = 'items'
        
        id = Column(Integer, primary_key=True)
        sku = Column(String(50), unique=True, nullable=False, index=True)
        current_stock = Column(Integer, CheckConstraint('current_stock >= 0'))
        reorder_point = Column(Integer, CheckConstraint('reorder_point >= 0'))
        unit_cost = Column(Numeric(10, 2), CheckConstraint('unit_cost > 0'))
        
        __table_args__ = (
            CheckConstraint('reorder_quantity > 0', name='positive_reorder_qty'),
            Index('idx_item_stock_status', 'current_stock', 'status'),
        )
```

### Data Export and Import

#### Export Utilities
```python
class DataExporter:
    """Export data for reporting and analysis"""
    
    @staticmethod
    def export_financial_summary(session, start_date: datetime, end_date: datetime) -> Dict:
        """Export financial summary for date range"""
        transactions = session.query(TransactionModel).filter(
            TransactionModel.transaction_date.between(start_date, end_date)
        ).all()
        
        return {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_transactions": len(transactions),
            "total_credits": sum(t.amount for t in transactions if t.is_credit()),
            "total_debits": sum(t.amount for t in transactions if t.is_debit()),
            "transactions": [t.dict() for t in transactions]
        }
    
    @staticmethod
    def export_inventory_report(session) -> Dict:
        """Export current inventory status"""
        items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()
        
        return {
            "report_date": datetime.utcnow().isoformat(),
            "total_items": len(items),
            "total_value": sum(item.stock_value for item in items),
            "low_stock_items": [item.dict() for item in items if item.is_low_stock],
            "inventory_details": [item.dict() for item in items]
        }
```

### Backup and Recovery

#### Automated Backup
```python
class BackupManager:
    """Database backup and recovery utilities"""
    
    def __init__(self, database_url: str, backup_location: str):
        self.database_url = database_url
        self.backup_location = backup_location
    
    def create_backup(self) -> str:
        """Create database backup"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{self.backup_location}/backup_{timestamp}.sql"
        
        # SQLite backup
        if "sqlite" in self.database_url:
            import shutil
            db_file = self.database_url.replace("sqlite:///", "")
            shutil.copy2(db_file, backup_file)
        
        # PostgreSQL backup
        elif "postgresql" in self.database_url:
            import subprocess
            subprocess.run([
                "pg_dump",
                self.database_url,
                "-f", backup_file
            ], check=True)
        
        return backup_file
    
    def restore_backup(self, backup_file: str):
        """Restore from backup file"""
        # Implementation depends on database type
        pass
```

---

*This document should be updated when new models are added or when database schema changes significantly.*