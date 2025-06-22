import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent_decisions import AgentDecisionModel
from models.employee import Employee, TimeRecord
from models.financial import (
    Account,
    AccountsPayable,
    AccountsReceivable,
    Transaction,
    TransactionType,
)
from models.inventory import Item, StockMovement

# Try to import agent classes for decision history
try:
    from agents.accounting_agent import AccountingAgent
    from agents.base_agent import AgentDecision
    from agents.hr_agent import HRAgent
    from agents.inventory_agent import InventoryAgent
except ImportError:
    AgentDecision = None


class BusinessDashboard:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.db_url = self.config["database"]["url"]
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as file:
            return yaml.safe_load(file)

    def get_financial_summary(self, days: int = 30) -> dict:
        session = self.SessionLocal()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get transactions for the period
            transactions = (
                session.query(Transaction).filter(Transaction.transaction_date >= start_date).all()
            )

            total_revenue = sum(
                float(t.amount)
                for t in transactions
                if t.transaction_type == TransactionType.INCOME
            )
            total_expenses = sum(
                float(t.amount)
                for t in transactions
                if t.transaction_type == TransactionType.EXPENSE
            )

            # Get cash balance
            cash_accounts = (
                session.query(Account)
                .filter(Account.account_type.in_(["checking", "savings"]))
                .all()
            )
            cash_balance = sum(float(acc.balance) for acc in cash_accounts)

            # Get receivables and payables
            receivables = (
                session.query(AccountsReceivable)
                .filter(AccountsReceivable.status == "unpaid")
                .all()
            )
            total_receivables = sum(float(ar.amount) for ar in receivables)

            payables = (
                session.query(AccountsPayable).filter(AccountsPayable.status == "unpaid").all()
            )
            total_payables = sum(float(ap.amount) for ap in payables)

            return {
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "net_income": total_revenue - total_expenses,
                "cash_balance": cash_balance,
                "accounts_receivable": total_receivables,
                "accounts_payable": total_payables,
                "transaction_count": len(transactions),
            }
        finally:
            session.close()

    def get_daily_revenue_chart(self, days: int = 30):
        session = self.SessionLocal()
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            # Get daily revenue
            daily_revenue = (
                session.query(
                    func.date(Transaction.transaction_date).label("date"),
                    func.sum(Transaction.amount).label("revenue"),
                )
                .filter(
                    Transaction.transaction_type == TransactionType.INCOME,
                    func.date(Transaction.transaction_date) >= start_date,
                )
                .group_by(func.date(Transaction.transaction_date))
                .all()
            )

            df = pd.DataFrame(daily_revenue, columns=["date", "revenue"])
            df["date"] = pd.to_datetime(df["date"])

            fig = px.line(
                df,
                x="date",
                y="revenue",
                title="Daily Revenue Trend",
                labels={"revenue": "Revenue ($)", "date": "Date"},
            )
            fig.update_layout(height=400)

            return fig
        finally:
            session.close()

    def get_expense_breakdown(self, days: int = 30):
        session = self.SessionLocal()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get expenses by category
            expense_by_category = (
                session.query(Transaction.category, func.sum(Transaction.amount).label("total"))
                .filter(
                    Transaction.transaction_type == TransactionType.EXPENSE,
                    Transaction.transaction_date >= start_date,
                )
                .group_by(Transaction.category)
                .all()
            )

            df = pd.DataFrame(expense_by_category, columns=["category", "total"])

            fig = px.pie(
                df, values="total", names="category", title="Expense Breakdown by Category"
            )
            fig.update_layout(height=400)

            return fig
        finally:
            session.close()

    def get_inventory_summary(self) -> dict:
        session = self.SessionLocal()
        try:
            items = session.query(Item).all()

            total_items = len(items)
            total_value = sum(float(item.current_stock * item.unit_cost) for item in items)
            low_stock_items = len(
                [item for item in items if item.current_stock <= item.reorder_point]
            )
            out_of_stock_items = len([item for item in items if item.current_stock <= 0])

            return {
                "total_items": total_items,
                "total_value": total_value,
                "low_stock_items": low_stock_items,
                "out_of_stock_items": out_of_stock_items,
            }
        finally:
            session.close()

    def get_inventory_levels_chart(self):
        session = self.SessionLocal()
        try:
            items = session.query(Item).filter(Item.current_stock > 0).all()

            item_data = []
            for item in items:
                item_data.append(
                    {
                        "name": item.name,
                        "current_stock": item.current_stock,
                        "reorder_point": item.reorder_point,
                        "status": (
                            "Low Stock" if item.current_stock <= item.reorder_point else "Normal"
                        ),
                    }
                )

            df = pd.DataFrame(item_data)

            if len(df) > 0:
                fig = px.bar(
                    df.head(20),
                    x="name",
                    y="current_stock",
                    color="status",
                    title="Current Stock Levels (Top 20 Items)",
                    labels={"current_stock": "Stock Level", "name": "Item"},
                )
                fig.add_scatter(
                    x=df["name"].head(20),
                    y=df["reorder_point"].head(20),
                    mode="markers",
                    name="Reorder Point",
                    marker={"color": "red", "symbol": "diamond", "size": 8},
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                return fig
            else:
                return px.bar(title="No inventory data available")
        finally:
            session.close()

    def get_hr_summary(self) -> dict:
        session = self.SessionLocal()
        try:
            employees = session.query(Employee).all()
            active_employees = len([emp for emp in employees if emp.status == "active"])

            # Get recent time records (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_records = (
                session.query(TimeRecord).filter(TimeRecord.timestamp >= week_ago).all()
            )

            return {
                "total_employees": len(employees),
                "active_employees": active_employees,
                "recent_time_entries": len(recent_records),
            }
        finally:
            session.close()

    def get_recent_transactions(self, limit: int = 10):
        session = self.SessionLocal()
        try:
            recent_transactions = (
                session.query(Transaction)
                .order_by(Transaction.created_at.desc())
                .limit(limit)
                .all()
            )

            transaction_data = []
            for tx in recent_transactions:
                transaction_data.append(
                    {
                        "Date": tx.transaction_date.strftime("%Y-%m-%d %H:%M"),
                        "Description": tx.description,
                        "Type": tx.transaction_type.title(),
                        "Amount": f"${tx.amount:,.2f}",
                        "Category": tx.category or "N/A",
                    }
                )

            return pd.DataFrame(transaction_data)
        finally:
            session.close()

    def get_recent_stock_movements(self, limit: int = 10):
        """Get recent stock movements for inventory tracking"""
        session = self.SessionLocal()
        try:
            # Join with Item to get item name
            recent_movements = (
                session.query(StockMovement, Item)
                .join(Item, StockMovement.item_id == Item.id)
                .order_by(StockMovement.created_at.desc())
                .limit(limit)
                .all()
            )

            movement_data = []
            for movement, item in recent_movements:
                movement_data.append(
                    {
                        "Date": movement.movement_date.strftime("%Y-%m-%d %H:%M"),
                        "Item": item.name if item else "Unknown",
                        "Type": movement.movement_type.replace("_", " ").title(),
                        "Quantity": f"{movement.quantity:+}",
                        "Notes": movement.notes or "N/A",
                        "Reference": movement.reference_number or "N/A",
                    }
                )

            return pd.DataFrame(movement_data)
        finally:
            session.close()

    def load_agent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Load recent agent decisions from the database
        """
        session = self.SessionLocal()
        try:
            # Get recent decisions ordered by timestamp
            db_decisions = (
                session.query(AgentDecisionModel)
                .order_by(AgentDecisionModel.timestamp.desc())
                .limit(limit)
                .all()
            )

            decisions = []
            for db_decision in db_decisions:
                decisions.append(
                    {
                        "agent_id": db_decision.agent_id,
                        "decision_type": db_decision.decision_type,
                        "timestamp": db_decision.timestamp,
                        "reasoning": db_decision.reasoning,
                        "action": db_decision.action,
                        "confidence": db_decision.confidence,
                        "context": db_decision.context or {},
                    }
                )

            # If no decisions in database, show sample data for demo
            if not decisions:
                decisions = [
                    {
                        "agent_id": "accounting_agent",
                        "decision_type": "cash_flow_alert",
                        "timestamp": datetime.now() - timedelta(minutes=5),
                        "reasoning": "Cash balance is $1,250 which is below the alert threshold of $1,500. The system recommends reviewing upcoming payables and accelerating receivables collection.",
                        "action": "Generate cash flow alert and recommend payment prioritization",
                        "confidence": 0.92,
                        "context": {"cash_balance": 1250, "threshold": 1500},
                    },
                    {
                        "agent_id": "inventory_agent",
                        "decision_type": "reorder_recommendation",
                        "timestamp": datetime.now() - timedelta(minutes=12),
                        "reasoning": "Tomatoes stock level (15 units) has fallen below reorder point (25 units). Historical consumption shows we use 8 units/day on average.",
                        "action": "Recommend reordering 50 units of tomatoes from primary supplier",
                        "confidence": 0.87,
                        "context": {"item": "Tomatoes", "current_stock": 15, "reorder_point": 25},
                    },
                ]

            return decisions

        except Exception as e:
            # Log error and return empty list
            print(f"Error loading agent decisions: {e}")
            return []
        finally:
            session.close()

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all configured agents"""
        agent_status = {}

        # Check which agents are enabled in config
        agent_configs = self.config.get("agents", {})

        for agent_name, config in agent_configs.items():
            if config.get("enabled", False):
                agent_status[agent_name] = {
                    "enabled": True,
                    "status": "running",  # This would be dynamic in a real implementation
                    "last_check": datetime.now() - timedelta(minutes=2),
                    "check_interval": config.get("check_interval", 300),
                    "decision_count": len(
                        [
                            d
                            for d in self.load_agent_decisions()
                            if d["agent_id"] == f"{agent_name}_agent"
                        ]
                    ),
                }
            else:
                agent_status[agent_name] = {"enabled": False, "status": "disabled"}

        return agent_status


def main():
    st.set_page_config(
        page_title="Business Management Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏢 Business Management Dashboard")

    # Sidebar for configuration
    st.sidebar.header("Dashboard Mode")

    # View selection
    view_mode = st.sidebar.radio(
        "Select View", ["🔴 Live Agent Monitoring", "📊 Historical Analytics"], index=0
    )

    st.sidebar.header("Configuration")

    # Config file selection
    config_files = ["config/restaurant_config.yaml", "config/retail_config.yaml"]

    # Check which config files exist
    available_configs = [f for f in config_files if os.path.exists(f)]

    if not available_configs:
        st.error(
            "No configuration files found. Please ensure config files exist in the config/ directory."
        )
        return

    selected_config = st.sidebar.selectbox(
        "Select Business Configuration",
        available_configs,
        format_func=lambda x: x.split("/")[-1].replace("_config.yaml", "").title(),
    )

    # Time period selection
    time_period = st.sidebar.selectbox(
        "Analysis Period", [7, 14, 30, 60, 90], index=2, format_func=lambda x: f"Last {x} days"
    )

    # Refresh controls
    st.sidebar.subheader("Dashboard Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("📊 Reset View", use_container_width=True):
            # Clear any cached data
            st.cache_data.clear()
            st.rerun()

    # Show last update time
    st.sidebar.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Initialize dashboard
    try:
        dashboard = BusinessDashboard(selected_config)
        business_name = dashboard.config["business"]["name"]
        business_type = dashboard.config["business"]["type"].title()

        st.sidebar.success(f"Connected to {business_name} ({business_type})")

    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return

    # Route to different views based on selection
    if view_mode == "🔴 Live Agent Monitoring":
        show_live_monitoring_view(dashboard)
    else:
        show_historical_analytics_view(dashboard, time_period)


def show_live_monitoring_view(dashboard):
    """Display live agent monitoring with real-time updates"""
    st.header("🔴 Live Agent Monitoring")

    # Auto-refresh configuration
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5s)", value=True)

    # Auto-refresh implementation
    if auto_refresh:
        # Use a timer to auto-refresh
        time.sleep(5)
        st.rerun()

    # Live indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("🟢 LIVE - Agent Activity Monitor")

    # Current time
    st.metric("Current Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Agent Status Section
    st.subheader("🤖 Agent Status")
    agent_status = dashboard.get_agent_status()

    if agent_status:
        cols = st.columns(len(agent_status))

        for idx, (agent_name, status) in enumerate(agent_status.items()):
            with cols[idx]:
                agent_display_name = agent_name.replace("_", " ").title()

                if status["enabled"]:
                    if status["status"] == "running":
                        st.success(f"✅ {agent_display_name}")
                        st.write(f"Last check: {status['last_check'].strftime('%H:%M:%S')}")
                        st.write(f"Decisions: {status['decision_count']}")
                        st.write(f"Interval: {status['check_interval']}s")
                    else:
                        st.warning(f"⚠️ {agent_display_name}")
                        st.write("Status: Not running")
                else:
                    st.error(f"❌ {agent_display_name}")
                    st.write("Status: Disabled")
    else:
        st.info("No agent configuration found.")

    # Recent Agent Decisions
    st.subheader("🔥 Recent Agent Activity")
    agent_decisions = dashboard.load_agent_decisions()

    if agent_decisions:
        # Show only the most recent 5 decisions for live monitoring
        for decision in agent_decisions[:5]:
            agent_name = decision["agent_id"].replace("_agent", "").title()

            # Color code by agent type
            agent_colors = {"accounting_agent": "🔴", "inventory_agent": "🟢", "hr_agent": "🔵"}

            color_indicator = agent_colors.get(decision["agent_id"], "⚪")
            time_ago = datetime.now() - decision["timestamp"]

            if time_ago.days > 0:
                time_str = f"{time_ago.days} days ago"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds // 3600} hours ago"
            else:
                time_str = f"{time_ago.seconds // 60} minutes ago"

            # Detailed display matching historical analysis
            with st.expander(
                f"{color_indicator} {agent_name} - {decision['decision_type'].replace('_', ' ').title()} ({time_str})",
                expanded=True,
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write("**Action Taken:**")
                    st.write(decision["action"])

                    st.write("**Reasoning:**")
                    reasoning_text = str(decision["reasoning"])
                    # Use st.text for literal text rendering
                    st.text(reasoning_text)

                    if decision.get("context"):
                        st.write("**Context:**")
                        for key, value in decision["context"].items():
                            st.write(f"- {key.replace('_', ' ').title()}: {value}")

                with col2:
                    confidence_pct = int(decision["confidence"] * 100)
                    st.metric("Confidence", f"{confidence_pct}%")

                    if confidence_pct >= 90:
                        st.success("High confidence")
                    elif confidence_pct >= 70:
                        st.warning("Medium confidence")
                    else:
                        st.error("Low confidence")
    else:
        st.info("No recent agent activity.")

    # Live System Metrics
    st.subheader("📊 Live System Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Get current cash balance
        financial_summary = dashboard.get_financial_summary(1)  # Last day
        st.metric("Current Cash", f"${financial_summary['cash_balance']:,.2f}")

    with col2:
        # Get inventory alerts
        inventory_summary = dashboard.get_inventory_summary()
        low_stock = inventory_summary["low_stock_items"]
        st.metric(
            "Low Stock Alerts",
            low_stock,
            delta=None,
            delta_color="inverse" if low_stock > 0 else "normal",
        )

    with col3:
        # Get recent transaction count
        st.metric("Today's Transactions", financial_summary["transaction_count"])

    # Auto-refresh using Streamlit's built-in refresh
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.rerun()

    # Note: For true auto-refresh, you can use browser refresh or streamlit run with --server.runOnSave true


def show_historical_analytics_view(dashboard, time_period):
    """Display historical analytics and reports"""
    st.header("📊 Historical Analytics")

    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)

    # Get financial summary
    financial_summary = dashboard.get_financial_summary(time_period)

    with col1:
        st.metric("Revenue", f"${financial_summary['total_revenue']:,.2f}", delta=None)

    with col2:
        st.metric("Expenses", f"${financial_summary['total_expenses']:,.2f}", delta=None)

    with col3:
        net_income = financial_summary["net_income"]
        st.metric(
            "Net Income",
            f"${net_income:,.2f}",
            delta=None,
            delta_color="normal" if net_income >= 0 else "inverse",
        )

    with col4:
        st.metric("Cash Balance", f"${financial_summary['cash_balance']:,.2f}", delta=None)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        revenue_chart = dashboard.get_daily_revenue_chart(time_period)
        st.plotly_chart(revenue_chart, use_container_width=True)

    with col2:
        expense_chart = dashboard.get_expense_breakdown(time_period)
        st.plotly_chart(expense_chart, use_container_width=True)

    # Inventory section
    st.header("📦 Inventory Overview")

    inventory_summary = dashboard.get_inventory_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Items", inventory_summary["total_items"])

    with col2:
        st.metric("Inventory Value", f"${inventory_summary['total_value']:,.2f}")

    with col3:
        low_stock = inventory_summary["low_stock_items"]
        st.metric(
            "Low Stock Items",
            low_stock,
            delta=None,
            delta_color="inverse" if low_stock > 0 else "normal",
        )

    with col4:
        out_of_stock = inventory_summary["out_of_stock_items"]
        st.metric(
            "Out of Stock",
            out_of_stock,
            delta=None,
            delta_color="inverse" if out_of_stock > 0 else "normal",
        )

    # Inventory chart
    inventory_chart = dashboard.get_inventory_levels_chart()
    st.plotly_chart(inventory_chart, use_container_width=True)

    # Recent stock movements
    st.subheader("📦 Recent Stock Movements")
    recent_movements = dashboard.get_recent_stock_movements(15)
    if not recent_movements.empty:
        st.dataframe(recent_movements, use_container_width=True, height=300)
    else:
        st.info("No recent stock movements found.")

    # HR section
    st.header("👥 Human Resources")

    hr_summary = dashboard.get_hr_summary()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Employees", hr_summary["total_employees"])

    with col2:
        st.metric("Active Employees", hr_summary["active_employees"])

    with col3:
        st.metric("Recent Time Entries", hr_summary["recent_time_entries"])

    # Recent transactions
    st.header("💰 Recent Transactions")

    recent_transactions = dashboard.get_recent_transactions(20)
    if not recent_transactions.empty:
        st.dataframe(recent_transactions, use_container_width=True, height=400)
    else:
        st.info("No recent transactions found.")

    # Agent Decisions Section - Historical View
    st.header("🤖 Agent Decisions & Recommendations")

    agent_decisions = dashboard.load_agent_decisions()

    if agent_decisions:
        # Display decisions in expandable cards
        for decision in agent_decisions[:10]:  # Show last 10 decisions
            agent_name = decision["agent_id"].replace("_agent", "").title()

            # Color code by agent type
            agent_colors = {"accounting_agent": "🔴", "inventory_agent": "🟢", "hr_agent": "🔵"}

            color_indicator = agent_colors.get(decision["agent_id"], "⚪")
            time_ago = datetime.now() - decision["timestamp"]

            if time_ago.days > 0:
                time_str = f"{time_ago.days} days ago"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds // 3600} hours ago"
            else:
                time_str = f"{time_ago.seconds // 60} minutes ago"

            with st.expander(
                f"{color_indicator} {agent_name} - {decision['decision_type'].replace('_', ' ').title()} ({time_str})"
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write("**Action Taken:**")
                    st.write(decision["action"])

                    st.write("**Reasoning:**")
                    reasoning_text = str(decision["reasoning"])
                    # Use st.text for literal text rendering
                    st.text(reasoning_text)

                    if decision.get("context"):
                        st.write("**Context:**")
                        for key, value in decision["context"].items():
                            st.write(f"- {key.replace('_', ' ').title()}: {value}")

                with col2:
                    confidence_pct = int(decision["confidence"] * 100)
                    st.metric("Confidence", f"{confidence_pct}%")

                    if confidence_pct >= 90:
                        st.success("High confidence")
                    elif confidence_pct >= 70:
                        st.warning("Medium confidence")
                    else:
                        st.error("Low confidence")
    else:
        st.info(
            "No agent decisions available. Agents may not be running or no decisions have been made yet."
        )

    # System status
    st.header("⚙️ System Status")

    col1, col2 = st.columns(2)

    with col1:
        business_name = dashboard.config["business"]["name"]
        business_type = dashboard.config["business"]["type"].title()
        st.info(f"Business: {business_name}")
        st.info(f"Type: {business_type}")
        st.info(f"Database: {dashboard.db_url}")
        st.info(f"Analysis Period: {time_period} days")

    with col2:
        st.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if st.button("Refresh Data"):
            st.rerun()

    # Agent Status
    st.subheader("🤖 Agent Status")

    agent_status = dashboard.get_agent_status()

    if agent_status:
        cols = st.columns(len(agent_status))

        for idx, (agent_name, status) in enumerate(agent_status.items()):
            with cols[idx]:
                agent_display_name = agent_name.replace("_", " ").title()

                if status["enabled"]:
                    if status["status"] == "running":
                        st.success(f"✅ {agent_display_name}")
                        st.write(f"Last check: {status['last_check'].strftime('%H:%M:%S')}")
                        st.write(f"Decisions: {status['decision_count']}")
                        st.write(f"Interval: {status['check_interval']}s")
                    else:
                        st.warning(f"⚠️ {agent_display_name}")
                        st.write("Status: Not running")
                else:
                    st.error(f"❌ {agent_display_name}")
                    st.write("Status: Disabled")
    else:
        st.info("No agent configuration found.")


if __name__ == "__main__":
    main()
