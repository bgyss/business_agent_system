#!/usr/bin/env python3
"""
Demonstration of Enhanced AccountingAgent capabilities
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

from agents.accounting_agent_enhanced import EnhancedAccountingAgent
from models.financial import TransactionModel, TransactionType


async def demo_enhanced_accounting():
    """Demonstrate the enhanced accounting agent capabilities."""

    print("=== Enhanced AccountingAgent Demonstration ===\n")

    # Initialize the enhanced agent
    config = {
        "anomaly_threshold": 0.25,
        "alert_thresholds": {
            "cash_low": 1000,
            "receivables_overdue": 30,
            "payables_overdue": 7
        },
        "forecasting": {
            "prediction_days": 30,
            "seasonal_analysis_days": 365,
            "trend_analysis_periods": 7,
            "confidence_factors": {
                "data_volume": 0.3,
                "historical_accuracy": 0.25,
                "trend_stability": 0.25,
                "seasonal_consistency": 0.2
            }
        }
    }

    agent = EnhancedAccountingAgent(
        agent_id="demo_enhanced_agent",
        api_key="demo_key",
        config=config,
        db_url="sqlite:///:memory:"
    )

    print("✓ Enhanced AccountingAgent initialized")
    print(f"  - Anomaly threshold: {agent.anomaly_threshold}")
    print(f"  - Cash low alert: ${agent.alert_thresholds['cash_low']}")
    print(f"  - Forecasting window: {agent.forecasting_config['prediction_days']} days\n")

    # Demo 1: Advanced Anomaly Detection
    print("1. ADVANCED ANOMALY DETECTION")
    print("-" * 40)

    # Create a suspicious transaction
    suspicious_transaction = TransactionModel(
        id="suspicious_001",
        description="Large unusual payment",
        amount=Decimal("15000.00"),  # Much larger than typical
        transaction_type=TransactionType.EXPENSE,
        category="office_supplies",
        transaction_date=datetime.now().replace(hour=3, minute=30)  # Unusual time
    )

    # Create similar historical transactions
    similar_transactions = [
        Mock(
            amount=Decimal("150.00"),
            transaction_date=datetime.now().replace(hour=10)
        ),
        Mock(
            amount=Decimal("200.00"),
            transaction_date=datetime.now().replace(hour=14)
        ),
        Mock(
            amount=Decimal("175.00"),
            transaction_date=datetime.now().replace(hour=11)
        ),
        Mock(
            amount=Decimal("225.00"),
            transaction_date=datetime.now().replace(hour=15)
        ),
        Mock(
            amount=Decimal("190.00"),
            transaction_date=datetime.now().replace(hour=9)
        )
    ]

    anomaly_results = await agent._detect_transaction_anomalies(
        None, suspicious_transaction, similar_transactions
    )

    print(f"Transaction: ${suspicious_transaction.amount} at {suspicious_transaction.transaction_date.hour}:00")
    print(f"Similar transactions: ${[float(t.amount) for t in similar_transactions]}")
    print(f"Anomaly detected: {anomaly_results['is_anomaly']}")
    print(f"Detection methods triggered: {anomaly_results['anomaly_count']}/4")
    print(f"  - Statistical outlier (Z-score): {anomaly_results['statistical_outlier']} (Z={anomaly_results['z_score']:.2f})")
    print(f"  - IQR outlier: {anomaly_results['iqr_outlier']}")
    print(f"  - Variance outlier: {anomaly_results['variance_outlier']} ({anomaly_results['median_variance']:.2%})")
    print(f"  - Time anomaly: {anomaly_results['time_anomaly']}")
    print()

    # Demo 2: Cash Flow Forecasting
    print("2. CASH FLOW FORECASTING")
    print("-" * 40)

    # Create sample daily cash flows (trending downward)
    daily_flows = {}
    base_date = datetime.now().date() - timedelta(days=30)

    for i in range(30):
        date_key = (base_date + timedelta(days=i)).isoformat()
        # Simulate declining cash flow trend
        daily_flow = 500 - (i * 10) + (i % 7) * 50  # Weekly pattern with decline
        daily_flows[date_key] = daily_flow

    print(f"Historical data: {len(daily_flows)} days")
    print(f"Cash flow trend: ${list(daily_flows.values())[:5]} ... ${list(daily_flows.values())[-5:]}")

    forecasts = await agent._generate_cash_flow_forecasts(daily_flows, 30)
    confidence = await agent._calculate_forecast_confidence(daily_flows, forecasts)

    print("\nForecast methods:")
    print(f"  - Simple moving average: ${forecasts['simple_moving_average']:.2f}/day")
    print(f"  - Weighted moving average: ${forecasts['weighted_moving_average']:.2f}/day")
    print(f"  - Trend-based: ${forecasts['trend_based']:.2f}/day")
    print(f"  - Ensemble forecast: ${forecasts['ensemble']:.2f}/day")
    print(f"  - 30-day total: ${forecasts['total_forecast']:.2f}")
    print(f"Forecast confidence: {confidence:.2%}")
    print()

    # Demo 3: Trend Analysis
    print("3. TREND ANALYSIS")
    print("-" * 40)

    # Test trend calculation with different patterns
    upward_trend = agent._calculate_trend([100, 110, 120, 130, 140, 150])
    downward_trend = agent._calculate_trend([150, 140, 130, 120, 110, 100])
    flat_trend = agent._calculate_trend([100, 100, 100, 100, 100, 100])
    volatile_trend = agent._calculate_trend([100, 150, 90, 140, 80, 130])

    print("Trend calculations:")
    print(f"  - Upward trend: {upward_trend:.2f} (should be ~10)")
    print(f"  - Downward trend: {downward_trend:.2f} (should be ~-10)")
    print(f"  - Flat trend: {flat_trend:.2f} (should be ~0)")
    print(f"  - Volatile trend: {volatile_trend:.2f}")
    print()

    # Demo 4: Dynamic Confidence Scoring
    print("4. DYNAMIC CONFIDENCE SCORING")
    print("-" * 40)

    # Simulate decision outcomes for learning
    agent.decision_outcomes = {
        "decision_1": {"decision_type": "transaction_anomaly", "was_correct": True, "timestamp": datetime.now()},
        "decision_2": {"decision_type": "transaction_anomaly", "was_correct": True, "timestamp": datetime.now()},
        "decision_3": {"decision_type": "transaction_anomaly", "was_correct": False, "timestamp": datetime.now()},
        "decision_4": {"decision_type": "cash_flow_forecast", "was_correct": True, "timestamp": datetime.now()},
        "decision_5": {"decision_type": "cash_flow_forecast", "was_correct": True, "timestamp": datetime.now()},
    }

    analysis_data = {
        "anomaly_count": 3,
        "statistics": {"sample_size": 25}
    }

    confidence_result = await agent._calculate_dynamic_confidence(
        None, "transaction_anomaly", analysis_data
    )

    print("Confidence calculation for transaction anomaly:")
    print(f"  - Overall confidence: {confidence_result['score']:.2%}")
    print("  - Factors:")
    for factor, value in confidence_result['factors'].items():
        print(f"    - {factor}: {value:.2%}")

    historical_accuracy = await agent._get_historical_accuracy(None, "transaction_anomaly")
    print(f"  - Historical accuracy: {historical_accuracy:.2%}")
    print()

    # Demo 5: System Prompt
    print("5. ENHANCED SYSTEM PROMPT")
    print("-" * 40)
    print("Key capabilities mentioned in system prompt:")
    prompt_lines = agent.system_prompt.split('\n')
    start_printing = False
    for line in prompt_lines:
        if 'Advanced capabilities:' in line:
            start_printing = True
        if start_printing and line.strip().startswith('-'):
            print(f"  {line.strip()}")
        if 'When analyzing financial data' in line:
            break

    print("\n=== Demonstration Complete ===")
    print(f"Enhanced AccountingAgent successfully demonstrated {5} major improvements:")
    print("1. ✓ Multi-algorithm anomaly detection")
    print("2. ✓ Advanced cash flow forecasting")
    print("3. ✓ Financial trend analysis")
    print("4. ✓ Dynamic confidence scoring")
    print("5. ✓ Decision outcome tracking and learning")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_accounting())
