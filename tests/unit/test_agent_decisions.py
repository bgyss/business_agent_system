"""
Unit tests for AgentDecision and AgentDecisionModel classes
"""

import os
import sys
from datetime import datetime
from decimal import Decimal

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.agent_decisions import (
    AgentDecision,
    AgentDecisionModel,
    AgentDecisionSummary,
    serialize_context,
)


class TestAgentDecision:
    """Test cases for AgentDecision Pydantic model"""

    def test_agent_decision_creation(self):
        """Test basic AgentDecision creation"""
        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test_decision",
            action="Take test action",
            reasoning="This is test reasoning",
            confidence=0.85,
        )

        assert decision.agent_id == "test_agent"
        assert decision.decision_type == "test_decision"
        assert decision.action == "Take test action"
        assert decision.reasoning == "This is test reasoning"
        assert decision.confidence == 0.85
        assert decision.context is None
        assert isinstance(decision.timestamp, datetime)

    def test_agent_decision_with_context(self):
        """Test AgentDecision creation with context"""
        context = {
            "item_id": 123,
            "amount": 150.75,
            "category": "sales",
            "metadata": {"source": "api", "version": "1.0"},
        }

        decision = AgentDecision(
            agent_id="accounting_agent",
            decision_type="transaction_analysis",
            action="Flag transaction for review",
            reasoning="Transaction amount exceeds normal variance",
            confidence=0.92,
            context=context,
        )

        assert decision.context == context
        assert decision.context["item_id"] == 123
        assert decision.context["metadata"]["source"] == "api"

    def test_agent_decision_confidence_validation(self):
        """Test confidence score validation"""
        # Valid confidence scores
        for confidence in [0.0, 0.5, 1.0, 0.123]:
            decision = AgentDecision(
                agent_id="test_agent",
                decision_type="test",
                action="test",
                reasoning="test",
                confidence=confidence,
            )
            assert decision.confidence == confidence

        # Invalid confidence scores should raise validation error
        with pytest.raises(ValueError):
            AgentDecision(
                agent_id="test_agent",
                decision_type="test",
                action="test",
                reasoning="test",
                confidence=-0.1,  # Below 0
            )

        with pytest.raises(ValueError):
            AgentDecision(
                agent_id="test_agent",
                decision_type="test",
                action="test",
                reasoning="test",
                confidence=1.1,  # Above 1
            )

    def test_agent_decision_required_fields(self):
        """Test that all required fields are validated"""
        # Missing agent_id
        with pytest.raises(ValueError):
            AgentDecision(decision_type="test", action="test", reasoning="test", confidence=0.8)

        # Missing decision_type
        with pytest.raises(ValueError):
            AgentDecision(agent_id="test_agent", action="test", reasoning="test", confidence=0.8)

        # Missing action
        with pytest.raises(ValueError):
            AgentDecision(
                agent_id="test_agent", decision_type="test", reasoning="test", confidence=0.8
            )

        # Missing reasoning
        with pytest.raises(ValueError):
            AgentDecision(
                agent_id="test_agent", decision_type="test", action="test", confidence=0.8
            )

        # Missing confidence
        with pytest.raises(ValueError):
            AgentDecision(
                agent_id="test_agent", decision_type="test", action="test", reasoning="test"
            )

    def test_agent_decision_to_dict(self):
        """Test conversion to dictionary"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        context = {"test": "data"}

        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test_decision",
            action="test_action",
            reasoning="test_reasoning",
            confidence=0.8,
            context=context,
            timestamp=timestamp,
        )

        result_dict = decision.to_dict()

        assert result_dict["agent_id"] == "test_agent"
        assert result_dict["decision_type"] == "test_decision"
        assert result_dict["action"] == "test_action"
        assert result_dict["reasoning"] == "test_reasoning"
        assert result_dict["confidence"] == 0.8
        assert result_dict["context"] == context
        assert result_dict["timestamp"] == timestamp

    def test_agent_decision_to_db_model(self):
        """Test conversion to database model"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        context = {"amount": Decimal("150.75"), "category": "sales"}

        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test_decision",
            action="test_action",
            reasoning="test_reasoning",
            confidence=0.8,
            context=context,
            timestamp=timestamp,
        )

        db_model = decision.to_db_model()

        assert isinstance(db_model, AgentDecisionModel)
        assert db_model.agent_id == "test_agent"
        assert db_model.decision_type == "test_decision"
        assert db_model.action == "test_action"
        assert db_model.reasoning == "test_reasoning"
        assert db_model.confidence == 0.8
        assert db_model.timestamp == timestamp
        # Context should be serialized (Decimal converted to float)
        assert db_model.context["amount"] == 150.75
        assert db_model.context["category"] == "sales"

    def test_agent_decision_from_db_model(self):
        """Test creation from database model"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        context = {"amount": 150.75, "category": "sales"}

        db_model = AgentDecisionModel(
            id=1,
            agent_id="test_agent",
            decision_type="test_decision",
            action="test_action",
            reasoning="test_reasoning",
            confidence=0.8,
            context=context,
            timestamp=timestamp,
            created_at=timestamp,
        )

        decision = AgentDecision.from_db_model(db_model)

        assert decision.agent_id == "test_agent"
        assert decision.decision_type == "test_decision"
        assert decision.action == "test_action"
        assert decision.reasoning == "test_reasoning"
        assert decision.confidence == 0.8
        assert decision.context == context
        assert decision.timestamp == timestamp


class TestSerializeContext:
    """Test cases for context serialization"""

    def test_serialize_none_context(self):
        """Test serializing None context"""
        result = serialize_context(None)
        assert result is None

    def test_serialize_empty_context(self):
        """Test serializing empty context"""
        result = serialize_context({})
        assert result == {}

    def test_serialize_decimal_values(self):
        """Test serializing context with Decimal values"""
        context = {
            "amount": Decimal("150.75"),
            "cost": Decimal("99.99"),
            "percentage": Decimal("0.25"),
        }

        result = serialize_context(context)

        assert result["amount"] == 150.75
        assert result["cost"] == 99.99
        assert result["percentage"] == 0.25
        assert all(isinstance(v, float) for v in result.values())

    def test_serialize_datetime_values(self):
        """Test serializing context with datetime values"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        context = {"created_at": timestamp, "updated_at": timestamp}

        result = serialize_context(context)

        assert result["created_at"] == "2024-01-01T12:00:00"
        assert result["updated_at"] == "2024-01-01T12:00:00"

    def test_serialize_nested_structures(self):
        """Test serializing nested dictionaries and lists"""
        context = {
            "metadata": {
                "amounts": [Decimal("10.00"), Decimal("20.00")],
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),
                "nested": {"value": Decimal("5.50")},
            },
            "list_data": [{"amount": Decimal("100.00")}, {"amount": Decimal("200.00")}],
        }

        result = serialize_context(context)

        assert result["metadata"]["amounts"] == [10.0, 20.0]
        assert result["metadata"]["timestamp"] == "2024-01-01T12:00:00"
        assert result["metadata"]["nested"]["value"] == 5.5
        assert result["list_data"][0]["amount"] == 100.0
        assert result["list_data"][1]["amount"] == 200.0

    def test_serialize_mixed_types(self):
        """Test serializing context with mixed data types"""
        context = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "decimal_value": Decimal("123.45"),
            "datetime_value": datetime(2024, 1, 1),
            "none_value": None,
            "list_value": [1, 2, Decimal("3.0")],
        }

        result = serialize_context(context)

        assert result["string_value"] == "test"
        assert result["int_value"] == 42
        assert result["float_value"] == 3.14
        assert result["bool_value"] is True
        assert result["decimal_value"] == 123.45
        assert result["datetime_value"] == "2024-01-01T00:00:00"
        assert result["none_value"] is None
        assert result["list_value"] == [1, 2, 3.0]


class TestAgentDecisionSummary:
    """Test cases for AgentDecisionSummary"""

    def test_agent_decision_summary_creation(self):
        """Test AgentDecisionSummary creation"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        decision_types = {"transaction_analysis": 5, "cash_flow_check": 3}

        summary = AgentDecisionSummary(
            agent_id="accounting_agent",
            total_decisions=8,
            recent_decisions=3,
            last_decision_time=timestamp,
            confidence_avg=0.85,
            decision_types=decision_types,
        )

        assert summary.agent_id == "accounting_agent"
        assert summary.total_decisions == 8
        assert summary.recent_decisions == 3
        assert summary.last_decision_time == timestamp
        assert summary.confidence_avg == 0.85
        assert summary.decision_types == decision_types

    def test_agent_decision_summary_optional_fields(self):
        """Test AgentDecisionSummary with optional fields"""
        summary = AgentDecisionSummary(
            agent_id="test_agent",
            total_decisions=0,
            recent_decisions=0,
            last_decision_time=None,
            confidence_avg=0.0,
            decision_types={},
        )

        assert summary.last_decision_time is None
        assert summary.decision_types == {}


class TestAgentDecisionModel:
    """Test cases for AgentDecisionModel SQLAlchemy model"""

    def test_agent_decision_model_creation(self):
        """Test AgentDecisionModel creation"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        context = {"test": "data"}

        model = AgentDecisionModel(
            agent_id="test_agent",
            decision_type="test_decision",
            action="test_action",
            reasoning="test_reasoning",
            confidence=0.8,
            context=context,
            timestamp=timestamp,
        )

        assert model.agent_id == "test_agent"
        assert model.decision_type == "test_decision"
        assert model.action == "test_action"
        assert model.reasoning == "test_reasoning"
        assert model.confidence == 0.8
        assert model.context == context
        assert model.timestamp == timestamp

    def test_agent_decision_model_table_name(self):
        """Test AgentDecisionModel table name"""
        assert AgentDecisionModel.__tablename__ == "agent_decisions"


class TestIntegration:
    """Integration tests for AgentDecision workflow"""

    def test_full_decision_workflow(self):
        """Test complete workflow from creation to database conversion and back"""
        original_context = {
            "transaction": {
                "id": 123,
                "amount": Decimal("150.75"),
                "timestamp": datetime(2024, 1, 1, 10, 0, 0),
            },
            "analysis": {"variance": Decimal("0.25"), "threshold": 0.2},
        }

        # Create AgentDecision
        decision = AgentDecision(
            agent_id="accounting_agent",
            decision_type="transaction_anomaly",
            action="Flag transaction for review",
            reasoning="Transaction exceeds variance threshold",
            confidence=0.9,
            context=original_context,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Convert to database model
        db_model = decision.to_db_model()

        # Verify serialization worked
        assert db_model.context["transaction"]["amount"] == 150.75
        assert db_model.context["analysis"]["variance"] == 0.25
        assert db_model.context["transaction"]["timestamp"] == "2024-01-01T10:00:00"

        # Convert back from database model
        restored_decision = AgentDecision.from_db_model(db_model)

        # Verify restoration (note: Decimals become floats after serialization)
        assert restored_decision.agent_id == decision.agent_id
        assert restored_decision.decision_type == decision.decision_type
        assert restored_decision.action == decision.action
        assert restored_decision.reasoning == decision.reasoning
        assert restored_decision.confidence == decision.confidence
        assert restored_decision.timestamp == decision.timestamp

        # Context values should be restored but Decimals become floats
        assert restored_decision.context["transaction"]["id"] == 123
        assert restored_decision.context["transaction"]["amount"] == 150.75
        assert restored_decision.context["analysis"]["variance"] == 0.25

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Very long strings
        long_reasoning = "A" * 10000
        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test",
            action="test",
            reasoning=long_reasoning,
            confidence=0.5,
        )
        assert len(decision.reasoning) == 10000

        # Very complex nested context
        complex_context = {
            "level1": {
                "level2": {
                    "level3": {
                        "amounts": [Decimal(str(i)) for i in range(100)],
                        "dates": [datetime.now() for _ in range(10)],
                    }
                }
            }
        }

        decision_complex = AgentDecision(
            agent_id="test_agent",
            decision_type="complex_test",
            action="test",
            reasoning="test",
            confidence=0.7,
            context=complex_context,
        )

        # Should serialize without errors
        db_model = decision_complex.to_db_model()
        assert db_model.context is not None

        # Should restore without errors
        restored = AgentDecision.from_db_model(db_model)
        assert restored.context is not None
