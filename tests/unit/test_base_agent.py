"""Unit tests for BaseAgent class."""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import AgentDecision, AgentMessage, BaseAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    @property
    def system_prompt(self) -> str:
        return "Test system prompt for concrete agent"

    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        if data.get("type") == "test_data":
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="test_decision",
                context=data,
                reasoning="Test reasoning",
                action="Test action",
                confidence=0.8,
            )
        return None

    async def generate_report(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "decisions_count": len(self.decisions_log),
        }

    async def periodic_check(self):
        """Test implementation of periodic check."""
        self.logger.info(f"Periodic check for test agent {self.agent_id}")


class TestBaseAgent:
    """Test cases for BaseAgent."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client."""
        with patch("agents.base_agent.Anthropic") as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Test Claude response")]
            mock_client.return_value.messages.create.return_value = mock_response
            yield mock_client

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch("agents.base_agent.create_engine"), patch(
            "agents.base_agent.sessionmaker"
        ) as mock_sessionmaker:
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def agent_config(self):
        """Agent configuration for testing."""
        return {"check_interval": 60, "anomaly_threshold": 0.2, "test_param": "test_value"}

    @pytest.fixture
    def concrete_agent(self, mock_anthropic, mock_db_session, agent_config):
        """Create a concrete agent instance for testing."""
        return ConcreteAgent(
            agent_id="test_agent",
            api_key="test_api_key",
            config=agent_config,
            db_url="sqlite:///:memory:",
        )

    def test_agent_initialization(self, concrete_agent, agent_config):
        """Test agent initialization."""
        assert concrete_agent.agent_id == "test_agent"
        assert concrete_agent.config == agent_config
        assert concrete_agent.is_running is False
        assert isinstance(concrete_agent.message_queue, asyncio.Queue)
        assert concrete_agent.decisions_log == []

    def test_system_prompt_property(self, concrete_agent):
        """Test system prompt property."""
        assert concrete_agent.system_prompt == "Test system prompt for concrete agent"

    @pytest.mark.asyncio
    async def test_analyze_with_claude_success(self, concrete_agent, mock_anthropic):
        """Test successful Claude API call."""
        context = {"test": "data"}
        prompt = "Test prompt"

        result = await concrete_agent.analyze_with_claude(prompt, context)

        assert result == "Test Claude response"
        mock_anthropic.return_value.messages.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_anthropic.return_value.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["max_tokens"] == 1000
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_analyze_with_claude_error(self, concrete_agent, mock_anthropic):
        """Test Claude API call with error."""
        mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

        context = {"test": "data"}
        prompt = "Test prompt"

        result = await concrete_agent.analyze_with_claude(prompt, context)

        assert result == "Error: API Error"

    @pytest.mark.asyncio
    async def test_send_message(self, concrete_agent):
        """Test sending message to queue."""
        recipient = "other_agent"
        message_type = "test_message"
        content = {"data": "test"}

        await concrete_agent.send_message(recipient, message_type, content)

        # Check message was added to queue
        assert not concrete_agent.message_queue.empty()
        message = await concrete_agent.message_queue.get()

        assert isinstance(message, AgentMessage)
        assert message.sender == "test_agent"
        assert message.recipient == recipient
        assert message.message_type == message_type
        assert message.content == content

    @pytest.mark.asyncio
    async def test_receive_messages(self, concrete_agent):
        """Test receiving messages from queue."""
        # Add messages to queue
        message1 = AgentMessage(
            sender="other_agent",
            recipient="test_agent",
            message_type="direct",
            content={"data": "test1"},
        )
        message2 = AgentMessage(
            sender="other_agent",
            recipient="all",
            message_type="broadcast",
            content={"data": "test2"},
        )
        message3 = AgentMessage(
            sender="other_agent",
            recipient="different_agent",
            message_type="other",
            content={"data": "test3"},
        )

        await concrete_agent.message_queue.put(message1)
        await concrete_agent.message_queue.put(message2)
        await concrete_agent.message_queue.put(message3)

        messages = await concrete_agent.receive_messages()

        assert len(messages) == 2  # Only messages for this agent or "all"
        assert messages[0].message_type == "direct"
        assert messages[1].message_type == "broadcast"

    def test_log_decision_success(self, concrete_agent, mock_db_session):
        """Test successful decision logging."""
        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test_decision",
            context={"test": "data"},
            reasoning="Test reasoning",
            action="Test action",
            confidence=0.8,
        )

        # Mock successful database operations
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        concrete_agent.log_decision(decision)

        # Check decision was added to memory log
        assert len(concrete_agent.decisions_log) == 1
        assert concrete_agent.decisions_log[0] == decision

        # Check database operations were called
        mock_session_instance.add.assert_called_once()
        mock_session_instance.commit.assert_called_once()
        mock_session_instance.close.assert_called_once()

    def test_log_decision_db_error(self, concrete_agent, mock_db_session):
        """Test decision logging with database error."""
        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test_decision",
            context={"test": "data"},
            reasoning="Test reasoning",
            action="Test action",
            confidence=0.8,
        )

        # Mock database error
        mock_session_instance = Mock()
        mock_session_instance.commit.side_effect = Exception("DB Error")
        mock_db_session.return_value = mock_session_instance

        concrete_agent.log_decision(decision)

        # Decision should still be in memory log
        assert len(concrete_agent.decisions_log) == 1

        # Check rollback was called
        mock_session_instance.rollback.assert_called_once()
        mock_session_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_data_update(self, concrete_agent):
        """Test handling data update message."""
        message = AgentMessage(
            sender="other_agent",
            recipient="test_agent",
            message_type="data_update",
            content={"type": "test_data", "value": 123},
        )

        await concrete_agent.handle_message(message)

        # Should have logged a decision
        assert len(concrete_agent.decisions_log) == 1
        assert concrete_agent.decisions_log[0].decision_type == "test_decision"

    @pytest.mark.asyncio
    async def test_handle_message_report_request(self, concrete_agent):
        """Test handling report request message."""
        message = AgentMessage(
            sender="other_agent", recipient="test_agent", message_type="report_request", content={}
        )

        await concrete_agent.handle_message(message)

        # Should have sent a response message
        assert not concrete_agent.message_queue.empty()
        response = await concrete_agent.message_queue.get()

        assert response.sender == "test_agent"
        assert response.recipient == "other_agent"
        assert response.message_type == "report_response"
        assert "report" in response.content

    @pytest.mark.asyncio
    async def test_process_data_with_valid_data(self, concrete_agent):
        """Test process_data with valid data."""
        data = {"type": "test_data", "value": 123}

        decision = await concrete_agent.process_data(data)

        assert decision is not None
        assert decision.agent_id == "test_agent"
        assert decision.decision_type == "test_decision"
        assert decision.confidence == 0.8

    @pytest.mark.asyncio
    async def test_process_data_with_invalid_data(self, concrete_agent):
        """Test process_data with invalid data."""
        data = {"type": "invalid_data", "value": 123}

        decision = await concrete_agent.process_data(data)

        assert decision is None

    @pytest.mark.asyncio
    async def test_generate_report(self, concrete_agent):
        """Test report generation."""
        report = await concrete_agent.generate_report()

        assert isinstance(report, dict)
        assert report["agent_id"] == "test_agent"
        assert report["status"] == "active"
        assert report["decisions_count"] == 0

    def test_get_decision_history_no_limit(self, concrete_agent):
        """Test getting decision history without limit."""
        # Add some decisions
        for i in range(5):
            decision = AgentDecision(
                agent_id="test_agent",
                decision_type=f"decision_{i}",
                context={"index": i},
                reasoning=f"Reasoning {i}",
                action=f"Action {i}",
                confidence=0.8,
            )
            concrete_agent.decisions_log.append(decision)

        history = concrete_agent.get_decision_history()

        assert len(history) == 5
        assert all(isinstance(d, AgentDecision) for d in history)

    def test_get_decision_history_with_limit(self, concrete_agent):
        """Test getting decision history with limit."""
        # Add some decisions
        for i in range(5):
            decision = AgentDecision(
                agent_id="test_agent",
                decision_type=f"decision_{i}",
                context={"index": i},
                reasoning=f"Reasoning {i}",
                action=f"Action {i}",
                confidence=0.8,
            )
            concrete_agent.decisions_log.append(decision)

        history = concrete_agent.get_decision_history(limit=3)

        assert len(history) == 3
        # Should return last 3 decisions
        assert history[0].decision_type == "decision_2"
        assert history[1].decision_type == "decision_3"
        assert history[2].decision_type == "decision_4"

    @pytest.mark.asyncio
    async def test_health_check(self, concrete_agent, agent_config):
        """Test health check."""
        # Add a decision to test last_decision timestamp
        decision = AgentDecision(
            agent_id="test_agent",
            decision_type="test_decision",
            context={},
            reasoning="Test",
            action="Test",
            confidence=0.8,
        )
        concrete_agent.decisions_log.append(decision)

        health = await concrete_agent.health_check()

        assert health["agent_id"] == "test_agent"
        assert health["status"] == "stopped"  # Agent not running
        assert health["decisions_count"] == 1
        assert health["last_decision"] == decision.timestamp
        assert health["config"] == agent_config

    @pytest.mark.asyncio
    async def test_start_and_stop(self, concrete_agent):
        """Test agent start and stop."""
        # Mock periodic_check to avoid infinite loop in testing
        concrete_agent.periodic_check = AsyncMock()

        # Start agent
        start_task = asyncio.create_task(concrete_agent.start())

        # Give it a moment to start
        await asyncio.sleep(0.1)
        assert concrete_agent.is_running is True

        # Stop agent
        await concrete_agent.stop()
        assert concrete_agent.is_running is False

        # Wait for start task to complete
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_periodic_check_default(self, concrete_agent):
        """Test default periodic_check implementation."""
        # Should not raise an error
        await concrete_agent.periodic_check()

    def test_agent_message_creation(self):
        """Test AgentMessage creation and attributes."""
        message = AgentMessage(
            sender="agent1", recipient="agent2", message_type="test", content={"data": "value"}
        )

        assert message.sender == "agent1"
        assert message.recipient == "agent2"
        assert message.message_type == "test"
        assert message.content == {"data": "value"}
        assert isinstance(message.timestamp, datetime)

    def test_custom_message_queue(self, mock_anthropic, mock_db_session, agent_config):
        """Test agent initialization with custom message queue."""
        custom_queue = asyncio.Queue()
        agent = ConcreteAgent(
            agent_id="test_agent",
            api_key="test_api_key",
            config=agent_config,
            db_url="sqlite:///:memory:",
            message_queue=custom_queue,
        )

        assert agent.message_queue is custom_queue

    @pytest.mark.asyncio
    async def test_receive_messages_exception_handling(self, concrete_agent):
        """Test exception handling in receive_messages (lines 98-99)"""
        # Mock message queue to raise an exception
        from unittest.mock import Mock

        mock_queue = Mock()
        # Return False first time to enter loop, then True to break
        mock_queue.empty.side_effect = [False, True]
        mock_queue.get_nowait.side_effect = Exception("Message queue error")
        concrete_agent.message_queue = mock_queue

        # Should handle the exception gracefully
        with patch.object(concrete_agent, "logger") as mock_logger:
            result = await concrete_agent.receive_messages()
            # The method should still return an empty list despite the error
            assert result == []
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_loop_exception_handling(self, concrete_agent):
        """Test agent main loop handles exceptions gracefully (lines
        133-134)"""
        # Make periodic_check raise an exception
        concrete_agent.periodic_check = AsyncMock(side_effect=Exception("Test error"))

        # Start the agent
        start_task = asyncio.create_task(concrete_agent.start())

        # Give it a moment to encounter the error
        await asyncio.sleep(0.1)

        # Agent should continue running despite the error
        assert concrete_agent.is_running is True

        # Stop the agent
        await concrete_agent.stop()

        # Cancel the task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
