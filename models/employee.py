from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Numeric, DateTime, Integer, Text, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class EmployeeStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TERMINATED = "terminated"
    ON_LEAVE = "on_leave"


class TimeRecordType(str, Enum):
    CLOCK_IN = "clock_in"
    CLOCK_OUT = "clock_out"
    BREAK_START = "break_start"
    BREAK_END = "break_end"


class LeaveType(str, Enum):
    SICK = "sick"
    VACATION = "vacation"
    PERSONAL = "personal"
    EMERGENCY = "emergency"


class Employee(Base):
    __tablename__ = "employees"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True)
    phone = Column(String(50))
    address = Column(Text)
    hire_date = Column(Date, nullable=False)
    position = Column(String(100), nullable=False)
    department = Column(String(100))
    hourly_rate = Column(Numeric(10, 2), nullable=False)
    salary = Column(Numeric(10, 2))
    status = Column(String(50), default="active")
    is_full_time = Column(Boolean, default=True)
    emergency_contact_name = Column(String(255))
    emergency_contact_phone = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TimeRecord(Base):
    __tablename__ = "time_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    record_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    notes = Column(Text)
    location = Column(String(255))
    approved_by = Column(String(255))
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Schedule(Base):
    __tablename__ = "schedules"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    work_date = Column(Date, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    break_duration_minutes = Column(Integer, default=30)
    notes = Column(Text)
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LeaveRequest(Base):
    __tablename__ = "leave_requests"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    leave_type = Column(String(50), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    reason = Column(Text)
    status = Column(String(50), default="pending")
    requested_date = Column(DateTime, default=datetime.utcnow)
    approved_by = Column(String(255))
    approved_date = Column(DateTime)
    notes = Column(Text)


class PayrollRecord(Base):
    __tablename__ = "payroll_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    pay_period_start = Column(Date, nullable=False)
    pay_period_end = Column(Date, nullable=False)
    regular_hours = Column(Numeric(5, 2), default=0)
    overtime_hours = Column(Numeric(5, 2), default=0)
    regular_pay = Column(Numeric(10, 2), default=0)
    overtime_pay = Column(Numeric(10, 2), default=0)
    gross_pay = Column(Numeric(10, 2), nullable=False)
    tax_deductions = Column(Numeric(10, 2), default=0)
    other_deductions = Column(Numeric(10, 2), default=0)
    net_pay = Column(Numeric(10, 2), nullable=False)
    pay_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)


class EmployeeModel(BaseModel):
    id: Optional[str] = None
    employee_id: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    hire_date: date
    position: str
    department: Optional[str] = None
    hourly_rate: Decimal
    salary: Optional[Decimal] = None
    status: EmployeeStatus = EmployeeStatus.ACTIVE
    is_full_time: bool = True
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TimeRecordModel(BaseModel):
    id: Optional[str] = None
    employee_id: str
    record_type: TimeRecordType
    timestamp: datetime
    notes: Optional[str] = None
    location: Optional[str] = None
    approved_by: Optional[str] = None
    is_approved: bool = False
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ScheduleModel(BaseModel):
    id: Optional[str] = None
    employee_id: str
    work_date: date
    start_time: datetime
    end_time: datetime
    break_duration_minutes: int = 30
    notes: Optional[str] = None
    is_published: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class LeaveRequestModel(BaseModel):
    id: Optional[str] = None
    employee_id: str
    leave_type: LeaveType
    start_date: date
    end_date: date
    reason: Optional[str] = None
    status: str = "pending"
    requested_date: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None
    notes: Optional[str] = None
    
    class Config:
        from_attributes = True


class PayrollRecordModel(BaseModel):
    id: Optional[str] = None
    employee_id: str
    pay_period_start: date
    pay_period_end: date
    regular_hours: Decimal = Field(default=Decimal("0.00"))
    overtime_hours: Decimal = Field(default=Decimal("0.00"))
    regular_pay: Decimal = Field(default=Decimal("0.00"))
    overtime_pay: Decimal = Field(default=Decimal("0.00"))
    gross_pay: Decimal
    tax_deductions: Decimal = Field(default=Decimal("0.00"))
    other_deductions: Decimal = Field(default=Decimal("0.00"))
    net_pay: Decimal
    pay_date: Optional[date] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class HRSummary(BaseModel):
    total_employees: int
    active_employees: int
    total_hours_worked: Decimal
    total_labor_cost: Decimal
    overtime_hours: Decimal
    pending_leave_requests: int
    schedule_conflicts: int
    period_start: date
    period_end: date


class StaffingRecommendation(BaseModel):
    date: date
    shift_start: datetime
    shift_end: datetime
    recommended_staff: int
    current_scheduled: int
    gap: int
    estimated_revenue: Decimal
    labor_cost_percentage: Decimal
    recommendation_reason: str
    
    
class LaborCostAnalysis(BaseModel):
    period_start: date
    period_end: date
    total_labor_cost: Decimal
    total_revenue: Decimal
    labor_cost_percentage: Decimal
    overtime_cost: Decimal
    overtime_percentage: Decimal
    avg_hourly_rate: Decimal
    total_hours: Decimal
    productivity_score: Decimal