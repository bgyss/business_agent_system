import asyncio
import json
import logging
import os

# Import models from the parent directory
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.agent_decisions import AgentDecision


class AgentMessage(BaseModel):
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = datetime.now()


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        api_key: str,
        config: Dict[str, Any],
        db_url: str,
        message_queue: Optional[asyncio.Queue] = None
    ):
        self.agent_id = agent_id
        self.client = Anthropic(api_key=api_key)
        self.config = config
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.message_queue = message_queue or asyncio.Queue()
        self.is_running = False
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.decisions_log = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        pass

    @abstractmethod
    async def generate_report(self) -> Dict[str, Any]:
        pass

    async def analyze_with_claude(
        self,
        prompt: str,
        context: Dict[str, Any],
        max_tokens: int = 1000
    ) -> str:
        full_prompt = f"{self.system_prompt}\n\nContext: {json.dumps(context, default=str)}\n\nQuery: {prompt}"

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Error calling Claude API: {e}")
            return f"Error: {str(e)}"

    async def send_message(self, recipient: str, message_type: str, content: Dict[str, Any]):
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content
        )
        await self.message_queue.put(message)
        self.logger.info(f"Sent message to {recipient}: {message_type}")

    async def receive_messages(self):
        messages = []
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                if message.recipient == self.agent_id or message.recipient == "all":
                    messages.append(message)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error receiving message: {e}")
                break
        return messages

    def log_decision(self, decision: AgentDecision):
        # Keep in memory for immediate access
        self.decisions_log.append(decision)

        # Persist to database
        session = self.SessionLocal()
        try:
            db_decision = decision.to_db_model()
            session.add(db_decision)
            session.commit()
            self.logger.info(f"Decision logged: {decision.decision_type} - {decision.action}")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to persist decision to database: {e}")
        finally:
            session.close()

    async def start(self):
        self.is_running = True
        self.logger.info(f"Agent {self.agent_id} started")

        # Agent now runs in the background, processing messages via the router
        # and performing periodic checks
        while self.is_running:
            try:
                # Perform periodic analysis based on check_interval
                await self.periodic_check()

                await asyncio.sleep(self.config.get("check_interval", 300))

            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                await asyncio.sleep(60)

    async def periodic_check(self):
        """Perform periodic analysis independent of messages"""
        # This method can be overridden by specific agents for scheduled analysis
        pass

    async def stop(self):
        self.is_running = False
        self.logger.info(f"Agent {self.agent_id} stopped")

    async def handle_message(self, message: AgentMessage):
        self.logger.info(f"Received message from {message.sender}: {message.message_type}")

        if message.message_type == "data_update":
            decision = await self.process_data(message.content)
            if decision:
                self.log_decision(decision)
        elif message.message_type == "report_request":
            report = await self.generate_report()
            await self.send_message(
                message.sender,
                "report_response",
                {"report": report}
            )

    def get_decision_history(self, limit: Optional[int] = None) -> List[AgentDecision]:
        if limit:
            return self.decisions_log[-limit:]
        return self.decisions_log

    async def health_check(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": "running" if self.is_running else "stopped",
            "decisions_count": len(self.decisions_log),
            "last_decision": self.decisions_log[-1].timestamp if self.decisions_log else None,
            "config": self.config
        }
