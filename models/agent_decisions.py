from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

# Import Base from existing models
from models.financial import Base


def serialize_context(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert Decimal objects to float for JSON serialization"""
    if not context:
        return context

    def convert_value(value):
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return value

    return {k: convert_value(v) for k, v in context.items()}


class AgentDecisionModel(Base):
    """SQLAlchemy model for storing agent decisions"""
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    decision_type = Column(String(100), nullable=False, index=True)
    action = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    context = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AgentDecision(BaseModel):
    """Pydantic model for agent decisions"""
    agent_id: str = Field(..., description="ID of the agent that made the decision")
    decision_type: str = Field(..., description="Type of decision made")
    action: str = Field(..., description="Action taken by the agent")
    reasoning: str = Field(..., description="Reasoning behind the decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the decision was made")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "agent_id": self.agent_id,
            "decision_type": self.decision_type,
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp
        }

    def to_db_model(self) -> AgentDecisionModel:
        """Convert to SQLAlchemy model for database storage"""
        return AgentDecisionModel(
            agent_id=self.agent_id,
            decision_type=self.decision_type,
            action=self.action,
            reasoning=self.reasoning,
            confidence=self.confidence,
            context=serialize_context(self.context),
            timestamp=self.timestamp
        )

    @classmethod
    def from_db_model(cls, db_model: AgentDecisionModel) -> "AgentDecision":
        """Create from SQLAlchemy model"""
        return cls(
            agent_id=db_model.agent_id,
            decision_type=db_model.decision_type,
            action=db_model.action,
            reasoning=db_model.reasoning,
            confidence=db_model.confidence,
            context=db_model.context,
            timestamp=db_model.timestamp
        )


class AgentDecisionSummary(BaseModel):
    """Summary of agent decision activity"""
    agent_id: str
    total_decisions: int
    recent_decisions: int
    last_decision_time: Optional[datetime]
    confidence_avg: float
    decision_types: Dict[str, int]
