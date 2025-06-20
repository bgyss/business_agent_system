import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Column, DateTime, Numeric, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TransactionType(str, Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"


class AccountType(str, Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit"
    REVENUE = "revenue"
    EXPENSE = "expense"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"


class Account(Base):
    __tablename__ = "accounts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    account_type = Column(String(50), nullable=False)
    description = Column(Text)
    balance = Column(Numeric(10, 2), default=0.00)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    description = Column(String(500), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    transaction_type = Column(String(50), nullable=False)
    from_account_id = Column(String, nullable=True)
    to_account_id = Column(String, nullable=True)
    category = Column(String(100))
    reference_number = Column(String(100))
    transaction_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_reconciled = Column(Boolean, default=False)


class AccountModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    name: str
    account_type: AccountType
    description: Optional[str] = None
    balance: Decimal = Field(default=Decimal("0.00"))
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True


class TransactionModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    description: str
    amount: Decimal
    transaction_type: TransactionType
    from_account_id: Optional[str] = None
    to_account_id: Optional[str] = None
    category: Optional[str] = None
    reference_number: Optional[str] = None
    transaction_date: datetime
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_reconciled: bool = False


class FinancialSummary(BaseModel):
    total_revenue: Decimal
    total_expenses: Decimal
    net_income: Decimal
    cash_balance: Decimal
    accounts_receivable: Decimal
    accounts_payable: Decimal
    period_start: datetime
    period_end: datetime
    transaction_count: int


class CashFlowStatement(BaseModel):
    operating_activities: Decimal
    investing_activities: Decimal
    financing_activities: Decimal
    net_cash_flow: Decimal
    beginning_cash: Decimal
    ending_cash: Decimal
    period_start: datetime
    period_end: datetime


class AccountsReceivable(Base):
    __tablename__ = "accounts_receivable"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_name = Column(String(255), nullable=False)
    invoice_number = Column(String(100), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    due_date = Column(DateTime, nullable=False)
    invoice_date = Column(DateTime, nullable=False)
    status = Column(String(50), default="unpaid")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AccountsPayable(Base):
    __tablename__ = "accounts_payable"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    vendor_name = Column(String(255), nullable=False)
    invoice_number = Column(String(100), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    due_date = Column(DateTime, nullable=False)
    invoice_date = Column(DateTime, nullable=False)
    status = Column(String(50), default="unpaid")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AccountsReceivableModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    customer_name: str
    invoice_number: str
    amount: Decimal
    due_date: datetime
    invoice_date: datetime
    status: str = "unpaid"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AccountsPayableModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    vendor_name: str
    invoice_number: str
    amount: Decimal
    due_date: datetime
    invoice_date: datetime
    status: str = "unpaid"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
