import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class StockMovementType(str, Enum):
    IN = "in"
    OUT = "out"
    ADJUSTMENT = "adjustment"
    WASTE = "waste"


class ItemStatus(str, Enum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


class Item(Base):
    __tablename__ = "items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sku = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    unit_cost = Column(Numeric(10, 2), nullable=False)
    selling_price = Column(Numeric(10, 2))
    current_stock = Column(Integer, default=0)
    minimum_stock = Column(Integer, default=0)
    maximum_stock = Column(Integer, default=1000)
    reorder_point = Column(Integer, default=0)
    reorder_quantity = Column(Integer, default=0)
    unit_of_measure = Column(String(50), default="each")
    status = Column(String(50), default="active")
    supplier_id = Column(String, ForeignKey("suppliers.id"), nullable=True)
    expiry_days = Column(Integer)  # Days until expiry (for perishables)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    supplier = relationship("Supplier", back_populates="items")
    stock_movements = relationship("StockMovement", back_populates="item")


class StockMovement(Base):
    __tablename__ = "stock_movements"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    item_id = Column(String, ForeignKey("items.id"), nullable=False)
    movement_type = Column(String(50), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_cost = Column(Numeric(10, 2))
    reference_number = Column(String(100))
    notes = Column(Text)
    movement_date = Column(DateTime, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)  # For compatibility
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    item = relationship("Item", back_populates="stock_movements")

    def __init__(self, **kwargs: Any) -> None:
        # Handle backward compatibility for 'reference' -> 'reference_number'
        if "reference" in kwargs:
            kwargs["reference_number"] = kwargs.pop("reference")

        # Handle backward compatibility for 'timestamp' -> 'movement_date'
        if "timestamp" in kwargs and "movement_date" not in kwargs:
            kwargs["movement_date"] = kwargs["timestamp"]

        # Set default movement_date if not provided
        if "movement_date" not in kwargs:
            kwargs["movement_date"] = datetime.utcnow()

        super().__init__(**kwargs)


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    contact_person = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    address = Column(Text)
    lead_time_days = Column(Integer, default=7)
    minimum_order = Column(Numeric(10, 2), default=0)
    payment_terms = Column(String(100))
    rating = Column(Numeric(3, 2), default=5.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    items = relationship("Item", back_populates="supplier")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    po_number = Column(String(100), unique=True, nullable=False)
    supplier_id = Column(String, ForeignKey("suppliers.id"), nullable=False)
    status = Column(String(50), default="pending")
    order_date = Column(DateTime, nullable=False)
    expected_delivery_date = Column(DateTime)
    total_amount = Column(Numeric(10, 2), default=0)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PurchaseOrderItem(Base):
    __tablename__ = "purchase_order_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    purchase_order_id = Column(String, ForeignKey("purchase_orders.id"), nullable=False)
    item_id = Column(String, ForeignKey("items.id"), nullable=False)
    quantity_ordered = Column(Integer, nullable=False)
    quantity_received = Column(Integer, default=0)
    unit_cost = Column(Numeric(10, 2), nullable=False)
    total_cost = Column(Numeric(10, 2), nullable=False)


class ItemModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    sku: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    unit_cost: Decimal
    selling_price: Optional[Decimal] = None
    current_stock: int = 0
    minimum_stock: int = 0
    maximum_stock: int = 1000
    reorder_point: int = 0
    reorder_quantity: int = 0
    unit_of_measure: str = "each"
    status: ItemStatus = ItemStatus.ACTIVE
    expiry_days: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StockMovementModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    item_id: str
    movement_type: StockMovementType
    quantity: int
    unit_cost: Optional[Decimal] = None
    reference_number: Optional[str] = None
    notes: Optional[str] = None
    movement_date: datetime
    created_at: Optional[datetime] = None


class SupplierModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    name: str
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    lead_time_days: int = 7
    minimum_order: Decimal = Field(default=Decimal("0.00"))
    payment_terms: Optional[str] = None
    rating: Decimal = Field(default=Decimal("5.0"))
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PurchaseOrderModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    po_number: str
    supplier_id: str
    status: str = "pending"
    order_date: datetime
    expected_delivery_date: Optional[datetime] = None
    total_amount: Decimal = Field(default=Decimal("0.00"))
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PurchaseOrderItemModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    purchase_order_id: str
    item_id: str
    quantity_ordered: int
    quantity_received: int = 0
    unit_cost: Decimal
    total_cost: Decimal


class InventorySummary(BaseModel):
    total_items: int
    total_value: Decimal
    low_stock_items: int
    out_of_stock_items: int
    items_to_reorder: List[str]
    expiring_soon: List[str]
    top_moving_items: List[str]
    slow_moving_items: List[str]


class ReorderSuggestion(BaseModel):
    item_id: str
    item_name: str
    current_stock: int
    reorder_point: int
    suggested_quantity: int
    estimated_cost: Decimal
    preferred_supplier: Optional[str] = None
    urgency: str  # low, medium, high, critical
    reason: str
