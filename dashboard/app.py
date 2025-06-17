import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import sys
import os
import time
import yaml
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.financial import Transaction, Account, AccountsReceivable, AccountsPayable, TransactionType
from models.inventory import Item, StockMovement, StockMovementType
from models.employee import Employee, TimeRecord, Schedule


class BusinessDashboard:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.db_url = self.config["database"]["url"]
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_financial_summary(self, days: int = 30) -> dict:
        session = self.SessionLocal()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get transactions for the period
            transactions = session.query(Transaction).filter(
                Transaction.transaction_date >= start_date
            ).all()
            
            total_revenue = sum(
                float(t.amount) for t in transactions 
                if t.transaction_type == TransactionType.INCOME
            )
            total_expenses = sum(
                float(t.amount) for t in transactions 
                if t.transaction_type == TransactionType.EXPENSE
            )
            
            # Get cash balance
            cash_accounts = session.query(Account).filter(
                Account.account_type.in_(["checking", "savings"])
            ).all()
            cash_balance = sum(float(acc.balance) for acc in cash_accounts)
            
            # Get receivables and payables
            receivables = session.query(AccountsReceivable).filter(
                AccountsReceivable.status == "unpaid"
            ).all()
            total_receivables = sum(float(ar.amount) for ar in receivables)
            
            payables = session.query(AccountsPayable).filter(
                AccountsPayable.status == "unpaid"
            ).all()
            total_payables = sum(float(ap.amount) for ap in payables)
            
            return {
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "net_income": total_revenue - total_expenses,
                "cash_balance": cash_balance,
                "accounts_receivable": total_receivables,
                "accounts_payable": total_payables,
                "transaction_count": len(transactions)
            }
        finally:
            session.close()
    
    def get_daily_revenue_chart(self, days: int = 30):
        session = self.SessionLocal()
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get daily revenue
            daily_revenue = session.query(
                func.date(Transaction.transaction_date).label('date'),
                func.sum(Transaction.amount).label('revenue')
            ).filter(
                Transaction.transaction_type == TransactionType.INCOME,
                func.date(Transaction.transaction_date) >= start_date
            ).group_by(func.date(Transaction.transaction_date)).all()
            
            df = pd.DataFrame(daily_revenue, columns=['date', 'revenue'])
            df['date'] = pd.to_datetime(df['date'])
            
            fig = px.line(df, x='date', y='revenue', 
                         title='Daily Revenue Trend',
                         labels={'revenue': 'Revenue ($)', 'date': 'Date'})
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
            expense_by_category = session.query(
                Transaction.category,
                func.sum(Transaction.amount).label('total')
            ).filter(
                Transaction.transaction_type == TransactionType.EXPENSE,
                Transaction.transaction_date >= start_date
            ).group_by(Transaction.category).all()
            
            df = pd.DataFrame(expense_by_category, columns=['category', 'total'])
            
            fig = px.pie(df, values='total', names='category',
                        title='Expense Breakdown by Category')
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
            low_stock_items = len([item for item in items if item.current_stock <= item.reorder_point])
            out_of_stock_items = len([item for item in items if item.current_stock <= 0])
            
            return {
                "total_items": total_items,
                "total_value": total_value,
                "low_stock_items": low_stock_items,
                "out_of_stock_items": out_of_stock_items
            }
        finally:
            session.close()
    
    def get_inventory_levels_chart(self):
        session = self.SessionLocal()
        try:
            items = session.query(Item).filter(Item.current_stock > 0).all()
            
            item_data = []
            for item in items:
                item_data.append({
                    'name': item.name,
                    'current_stock': item.current_stock,
                    'reorder_point': item.reorder_point,
                    'status': 'Low Stock' if item.current_stock <= item.reorder_point else 'Normal'
                })
            
            df = pd.DataFrame(item_data)
            
            if len(df) > 0:
                fig = px.bar(df.head(20), x='name', y='current_stock',
                           color='status',
                           title='Current Stock Levels (Top 20 Items)',
                           labels={'current_stock': 'Stock Level', 'name': 'Item'})
                fig.add_scatter(x=df['name'].head(20), y=df['reorder_point'].head(20),
                              mode='markers', name='Reorder Point',
                              marker=dict(color='red', symbol='diamond', size=8))
                fig.update_layout(height=400, xaxis_tickangle=-45)
                return fig
            else:
                return px.bar(title='No inventory data available')
        finally:
            session.close()
    
    def get_hr_summary(self) -> dict:
        session = self.SessionLocal()
        try:
            employees = session.query(Employee).all()
            active_employees = len([emp for emp in employees if emp.status == "active"])
            
            # Get recent time records (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_records = session.query(TimeRecord).filter(
                TimeRecord.timestamp >= week_ago
            ).all()
            
            return {
                "total_employees": len(employees),
                "active_employees": active_employees,
                "recent_time_entries": len(recent_records)
            }
        finally:
            session.close()
    
    def get_recent_transactions(self, limit: int = 10):
        session = self.SessionLocal()
        try:
            recent_transactions = session.query(Transaction).order_by(
                Transaction.created_at.desc()
            ).limit(limit).all()
            
            transaction_data = []
            for tx in recent_transactions:
                transaction_data.append({
                    'Date': tx.transaction_date.strftime('%Y-%m-%d %H:%M'),
                    'Description': tx.description,
                    'Type': tx.transaction_type.title(),
                    'Amount': f"${tx.amount:,.2f}",
                    'Category': tx.category or 'N/A'
                })
            
            return pd.DataFrame(transaction_data)
        finally:
            session.close()


def main():
    st.set_page_config(
        page_title="Business Management Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏢 Business Management Dashboard")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Config file selection
    config_files = [
        "config/restaurant_config.yaml",
        "config/retail_config.yaml"
    ]
    
    # Check which config files exist
    available_configs = [f for f in config_files if os.path.exists(f)]
    
    if not available_configs:
        st.error("No configuration files found. Please ensure config files exist in the config/ directory.")
        return
    
    selected_config = st.sidebar.selectbox(
        "Select Business Configuration",
        available_configs,
        format_func=lambda x: x.split('/')[-1].replace('_config.yaml', '').title()
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Analysis Period",
        [7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"Last {x} days"
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Initialize dashboard
    try:
        dashboard = BusinessDashboard(selected_config)
        business_name = dashboard.config["business"]["name"]
        business_type = dashboard.config["business"]["type"].title()
        
        st.sidebar.success(f"Connected to {business_name} ({business_type})")
        
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get financial summary
    financial_summary = dashboard.get_financial_summary(time_period)
    
    with col1:
        st.metric(
            "Revenue", 
            f"${financial_summary['total_revenue']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Expenses", 
            f"${financial_summary['total_expenses']:,.2f}",
            delta=None
        )
    
    with col3:
        net_income = financial_summary['net_income']
        st.metric(
            "Net Income", 
            f"${net_income:,.2f}",
            delta=None,
            delta_color="normal" if net_income >= 0 else "inverse"
        )
    
    with col4:
        st.metric(
            "Cash Balance", 
            f"${financial_summary['cash_balance']:,.2f}",
            delta=None
        )
    
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
        st.metric("Total Items", inventory_summary['total_items'])
    
    with col2:
        st.metric("Inventory Value", f"${inventory_summary['total_value']:,.2f}")
    
    with col3:
        low_stock = inventory_summary['low_stock_items']
        st.metric(
            "Low Stock Items", 
            low_stock,
            delta=None,
            delta_color="inverse" if low_stock > 0 else "normal"
        )
    
    with col4:
        out_of_stock = inventory_summary['out_of_stock_items']
        st.metric(
            "Out of Stock", 
            out_of_stock,
            delta=None,
            delta_color="inverse" if out_of_stock > 0 else "normal"
        )
    
    # Inventory chart
    inventory_chart = dashboard.get_inventory_levels_chart()
    st.plotly_chart(inventory_chart, use_container_width=True)
    
    # HR section
    st.header("👥 Human Resources")
    
    hr_summary = dashboard.get_hr_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Employees", hr_summary['total_employees'])
    
    with col2:
        st.metric("Active Employees", hr_summary['active_employees'])
    
    with col3:
        st.metric("Recent Time Entries", hr_summary['recent_time_entries'])
    
    # Recent transactions
    st.header("💰 Recent Transactions")
    
    recent_transactions = dashboard.get_recent_transactions(20)
    if not recent_transactions.empty:
        st.dataframe(recent_transactions, use_container_width=True, height=400)
    else:
        st.info("No recent transactions found.")
    
    # System status
    st.header("⚙️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Business: {business_name}")
        st.info(f"Type: {business_type}")
        st.info(f"Database: {dashboard.db_url}")
    
    with col2:
        st.info(f"Analysis Period: {time_period} days")
        st.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("Refresh Data"):
            st.experimental_rerun()


if __name__ == "__main__":
    main()