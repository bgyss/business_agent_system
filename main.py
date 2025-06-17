#!/usr/bin/env python3
import asyncio
import argparse
import logging
import os
import signal
import sys
from typing import Dict, Any, List
import yaml
from datetime import datetime

from dotenv import load_dotenv

# Import agents
from agents.accounting_agent import AccountingAgent
from agents.inventory_agent import InventoryAgent
from agents.hr_agent import HRAgent

# Import simulation
from simulation.business_simulator import BusinessSimulator

# Load environment variables
load_dotenv()


class BusinessAgentSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.agents: Dict[str, Any] = {}
        self.simulator: BusinessSimulator = None
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("BusinessAgentSystem")
        
        # Get API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self):
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create logs directory if it doesn't exist
        log_file = log_config.get("file")
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
    
    def initialize_agents(self):
        """Initialize all enabled agents"""
        agent_configs = self.config.get("agents", {})
        db_url = self.config["database"]["url"]
        
        # Initialize Accounting Agent
        if agent_configs.get("accounting", {}).get("enabled", False):
            self.agents["accounting"] = AccountingAgent(
                agent_id="accounting_agent",
                api_key=self.api_key,
                config=agent_configs["accounting"],
                db_url=db_url
            )
            self.logger.info("Accounting agent initialized")
        
        # Initialize Inventory Agent
        if agent_configs.get("inventory", {}).get("enabled", False):
            self.agents["inventory"] = InventoryAgent(
                agent_id="inventory_agent",
                api_key=self.api_key,
                config=agent_configs["inventory"],
                db_url=db_url
            )
            self.logger.info("Inventory agent initialized")
        
        # Initialize HR Agent
        if agent_configs.get("hr", {}).get("enabled", False):
            self.agents["hr"] = HRAgent(
                agent_id="hr_agent",
                api_key=self.api_key,
                config=agent_configs["hr"],
                db_url=db_url
            )
            self.logger.info("HR agent initialized")
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def initialize_simulator(self):
        """Initialize the business simulator if enabled"""
        sim_config = self.config.get("simulation", {})
        if not sim_config.get("enabled", False):
            self.logger.info("Simulation disabled")
            return
        
        db_url = self.config["database"]["url"]
        business_type = self.config["business"]["type"]
        
        self.simulator = BusinessSimulator(sim_config, db_url)
        self.simulator.initialize_business(business_type)
        
        # Generate historical data if in simulation mode
        mode = sim_config.get("mode", "off")
        if mode in ["historical", "real_time"]:
            historical_days = sim_config.get("historical_days", 90)
            self.logger.info(f"Generating {historical_days} days of historical data...")
            self.simulator.simulate_historical_data(historical_days)
        
        self.logger.info(f"Simulator initialized in {mode} mode")
    
    async def start_agents(self):
        """Start all agents"""
        for agent_name, agent in self.agents.items():
            # Set the shared message queue
            agent.message_queue = self.message_queue
            
            # Start the agent
            task = asyncio.create_task(agent.start())
            self.tasks.append(task)
            self.logger.info(f"Started {agent_name} agent")
    
    async def start_simulator(self):
        """Start the real-time simulator if enabled"""
        if not self.simulator:
            return
        
        sim_config = self.config.get("simulation", {})
        if sim_config.get("mode") == "real_time":
            task = asyncio.create_task(
                self.simulator.start_real_time_simulation(self.message_queue)
            )
            self.tasks.append(task)
            self.logger.info("Started real-time simulation")
    
    async def message_router(self):
        """Route messages between agents"""
        while self.is_running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Route message to appropriate agents
                if isinstance(message, dict):
                    # Broadcast to all agents
                    for agent_name, agent in self.agents.items():
                        await agent.message_queue.put(message)
                    
                    self.logger.debug(f"Routed message: {message.get('type', 'unknown')}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in message router: {e}")
                await asyncio.sleep(1)
    
    async def run(self):
        """Main run loop"""
        self.is_running = True
        self.logger.info("Starting Business Agent System")
        
        try:
            # Initialize components
            self.initialize_agents()
            self.initialize_simulator()
            
            # Start agents
            await self.start_agents()
            
            # Start simulator
            await self.start_simulator()
            
            # Start message router
            router_task = asyncio.create_task(self.message_router())
            self.tasks.append(router_task)
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self.monitor_system())
            self.tasks.append(monitor_task)
            
            self.logger.info("All systems started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def monitor_system(self):
        """Monitor system health and performance"""
        while self.is_running:
            try:
                # Log system status periodically
                active_agents = len([a for a in self.agents.values() if a.is_running])
                total_decisions = sum(len(a.decisions_log) for a in self.agents.values())
                
                self.logger.info(
                    f"System Status: {active_agents}/{len(self.agents)} agents running, "
                    f"{total_decisions} total decisions made"
                )
                
                # Check for agent errors or failures
                for agent_name, agent in self.agents.items():
                    if not agent.is_running:
                        self.logger.warning(f"Agent {agent_name} is not running")
                
                # Wait 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Business Agent System")
        self.is_running = False
        
        # Stop simulator
        if self.simulator:
            await self.simulator.stop_simulation()
        
        # Stop agents
        for agent_name, agent in self.agents.items():
            await agent.stop()
            self.logger.info(f"Stopped {agent_name} agent")
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.logger.info("Shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        agent_status = {}
        for name, agent in self.agents.items():
            agent_status[name] = {
                "running": agent.is_running,
                "decisions": len(agent.decisions_log),
                "last_activity": agent.decisions_log[-1].timestamp if agent.decisions_log else None
            }
        
        simulator_status = None
        if self.simulator:
            simulator_status = self.simulator.get_simulation_status()
        
        return {
            "system_running": self.is_running,
            "agents": agent_status,
            "simulator": simulator_status,
            "business_config": {
                "name": self.config["business"]["name"],
                "type": self.config["business"]["type"]
            },
            "timestamp": datetime.now().isoformat()
        }


def signal_handler(system: BusinessAgentSystem):
    """Handle shutdown signals"""
    def handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        asyncio.create_task(system.shutdown())
    return handler


async def main():
    parser = argparse.ArgumentParser(description="Business Agent Management System")
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to configuration file (e.g., config/restaurant_config.yaml)"
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "production"],
        default="simulation",
        help="Run mode (default: simulation)"
    )
    parser.add_argument(
        "--generate-historical",
        type=int,
        help="Generate N days of historical data and exit"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    
    system = BusinessAgentSystem(args.config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler(system))
    signal.signal(signal.SIGTERM, signal_handler(system))
    
    try:
        if args.generate_historical:
            # Just generate historical data and exit
            system.initialize_simulator()
            print(f"Generated {args.generate_historical} days of historical data")
            return
        
        # Run the system
        await system.run()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is required")
        print("Please set it with: export ANTHROPIC_API_KEY=your_api_key_here")
        sys.exit(1)
    
    asyncio.run(main())