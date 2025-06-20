from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func

from agents.base_agent import AgentDecision, BaseAgent
from models.employee import (
    Employee,
    EmployeeStatus,
    HRSummary,
    LeaveRequest,
    LeaveType,
    Schedule,
    TimeRecord,
    TimeRecordType,
)


class HRAgent(BaseAgent):
    def __init__(self, agent_id: str, api_key: str, config: Dict[str, Any], db_url: str):
        super().__init__(agent_id, api_key, config, db_url)
        self.overtime_threshold = config.get("overtime_threshold", 40)  # hours per week
        self.max_labor_cost_percentage = config.get("max_labor_cost_percentage", 0.30)  # 30% of revenue
        self.scheduling_buffer_hours = config.get("scheduling_buffer_hours", 2)

    @property
    def system_prompt(self) -> str:
        return """You are an AI HR Management Agent responsible for monitoring employee schedules, labor costs, and workforce optimization.
        
        Your responsibilities include:
        1. Tracking employee time records and identifying attendance issues
        2. Monitoring labor costs vs. revenue and alerting on budget overruns
        3. Analyzing staffing levels and making scheduling recommendations
        4. Managing leave requests and identifying scheduling conflicts
        5. Calculating overtime costs and suggesting ways to minimize them
        6. Ensuring compliance with labor laws and company policies
        7. Identifying productivity trends and staffing inefficiencies
        
        When making staffing recommendations, consider:
        - Historical sales/revenue patterns by time of day and day of week
        - Employee availability and skills
        - Labor cost targets and budget constraints
        - Customer service levels and wait times
        - Overtime costs vs. hiring additional staff
        - Seasonal variations in business volume
        - Employee satisfaction and work-life balance
        
        Always provide data-driven recommendations with clear cost-benefit analysis.
        Consider both financial impact and employee welfare in your decisions.
        """

    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        session = self.SessionLocal()
        try:
            if data.get("type") == "time_record":
                return await self._analyze_time_record(session, data["record"])
            elif data.get("type") == "daily_labor_analysis":
                return await self._perform_daily_labor_analysis(session)
            elif data.get("type") == "overtime_check":
                return await self._check_overtime_patterns(session)
            elif data.get("type") == "staffing_analysis":
                return await self._analyze_staffing_needs(session, data.get("revenue_data"))
            elif data.get("type") == "leave_request":
                return await self._analyze_leave_request(session, data["request"])
        except Exception as e:
            self.logger.error(f"Error in process_data: {e}")
            return None
        finally:
            try:
                session.close()
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")

        return None

    async def _analyze_time_record(self, session, record_data: Dict[str, Any]) -> Optional[AgentDecision]:
        employee_id = record_data.get("employee_id")
        record_type = record_data.get("record_type")
        timestamp = datetime.fromisoformat(record_data.get("timestamp"))

        employee = session.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return None

        context = {
            "employee": {
                "id": employee.id,
                "name": f"{employee.first_name} {employee.last_name}",
                "position": employee.position,
                "hourly_rate": float(employee.hourly_rate)
            },
            "record": record_data,
            "timestamp_hour": timestamp.hour
        }

        # Check for unusual clock-in/out times
        if record_type == TimeRecordType.CLOCK_IN:
            if timestamp.hour < 5 or timestamp.hour > 23:  # Very early or very late
                reasoning = await self.analyze_with_claude(
                    f"Employee {employee.first_name} {employee.last_name} clocked in at "
                    f"{timestamp.strftime('%H:%M')} on {timestamp.strftime('%A, %B %d')}. "
                    f"This is outside normal business hours. Is this authorized?",
                    context
                )

                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="unusual_time_record",
                    context=context,
                    reasoning=reasoning,
                    action=f"Review unusual clock-in time for {employee.first_name} {employee.last_name}",
                    confidence=0.75
                )

        # Check for potential overtime
        elif record_type == TimeRecordType.CLOCK_OUT:
            # Get today's time records for this employee
            today = timestamp.date()
            today_records = session.query(TimeRecord).filter(
                and_(
                    TimeRecord.employee_id == employee_id,
                    func.date(TimeRecord.timestamp) == today
                )
            ).order_by(TimeRecord.timestamp).all()

            # Calculate hours worked today (simplified)
            hours_worked = self._calculate_daily_hours(today_records + [record_data])

            if hours_worked > 8:  # More than 8 hours
                context["hours_worked_today"] = hours_worked
                context["potential_overtime"] = hours_worked - 8

                reasoning = await self.analyze_with_claude(
                    f"Employee {employee.first_name} {employee.last_name} worked "
                    f"{hours_worked:.1f} hours today, which exceeds 8 hours. "
                    f"Potential overtime: {hours_worked - 8:.1f} hours. "
                    f"Should this be approved?",
                    context
                )

                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="potential_overtime",
                    context=context,
                    reasoning=reasoning,
                    action=f"Review overtime for {employee.first_name} {employee.last_name}",
                    confidence=0.8
                )

        return None

    def _calculate_daily_hours(self, time_records: List[Any]) -> float:
        """Calculate total hours worked from time records (simplified)"""
        clock_in_time = None
        total_hours = 0

        def get_timestamp(record):
            # Handle dictionary records (new records from API)
            if isinstance(record, dict) and "timestamp" in record:
                return datetime.fromisoformat(record["timestamp"]) if isinstance(record["timestamp"], str) else record["timestamp"]
            # Handle database objects with timestamp attribute
            elif hasattr(record, 'timestamp'):
                return record.timestamp
            else:
                return datetime.min

        for record in sorted(time_records, key=get_timestamp):
            # Handle dictionary vs object record types
            if isinstance(record, dict):
                record_type = record["record_type"]
            else:
                record_type = record.record_type

            if record_type == TimeRecordType.CLOCK_IN:
                if isinstance(record, dict):
                    clock_in_time = datetime.fromisoformat(record["timestamp"]) if isinstance(record["timestamp"], str) else record["timestamp"]
                else:
                    clock_in_time = record.timestamp
            elif record_type == TimeRecordType.CLOCK_OUT and clock_in_time:
                if isinstance(record, dict):
                    clock_out_time = datetime.fromisoformat(record["timestamp"]) if isinstance(record["timestamp"], str) else record["timestamp"]
                else:
                    clock_out_time = record.timestamp

                hours = (clock_out_time - clock_in_time).total_seconds() / 3600
                total_hours += hours
                clock_in_time = None

        return total_hours

    async def _perform_daily_labor_analysis(self, session) -> Optional[AgentDecision]:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Get yesterday's time records
        yesterday_records = session.query(TimeRecord).filter(
            func.date(TimeRecord.timestamp) == yesterday
        ).all()

        if not yesterday_records:
            return None

        # Calculate total labor hours and costs
        employee_hours = {}
        total_labor_cost = 0

        # Group records by employee
        for record in yesterday_records:
            employee_id = record.employee_id
            if employee_id not in employee_hours:
                employee_hours[employee_id] = []
            employee_hours[employee_id].append(record)

        # Calculate hours and costs for each employee
        employee_data = []
        for employee_id, records in employee_hours.items():
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                continue

            daily_hours = self._calculate_daily_hours(records)
            regular_hours = min(daily_hours, 8)
            overtime_hours = max(0, daily_hours - 8)

            regular_pay = regular_hours * float(employee.hourly_rate)
            overtime_pay = overtime_hours * float(employee.hourly_rate) * 1.5  # Time and a half
            total_pay = regular_pay + overtime_pay

            total_labor_cost += total_pay

            employee_data.append({
                "name": f"{employee.first_name} {employee.last_name}",
                "position": employee.position,
                "hours_worked": daily_hours,
                "overtime_hours": overtime_hours,
                "total_pay": total_pay
            })

        context = {
            "date": str(yesterday),
            "total_employees_worked": len(employee_data),
            "total_labor_hours": sum(emp["hours_worked"] for emp in employee_data),
            "total_overtime_hours": sum(emp["overtime_hours"] for emp in employee_data),
            "total_labor_cost": total_labor_cost,
            "employee_data": employee_data[:10]  # Limit to top 10
        }

        # Check if labor cost is high relative to typical revenue
        # This would normally use actual revenue data
        estimated_daily_revenue = 2500  # Placeholder - would come from revenue data
        labor_cost_percentage = total_labor_cost / estimated_daily_revenue if estimated_daily_revenue > 0 else 0

        context["estimated_revenue"] = estimated_daily_revenue
        context["labor_cost_percentage"] = labor_cost_percentage

        if labor_cost_percentage > self.max_labor_cost_percentage:
            reasoning = await self.analyze_with_claude(
                f"Yesterday's labor cost was ${total_labor_cost:.2f} "
                f"({labor_cost_percentage:.1%} of estimated revenue). "
                f"This exceeds the target of {self.max_labor_cost_percentage:.1%}. "
                f"Total overtime: {context['total_overtime_hours']:.1f} hours. "
                f"What actions should be taken to optimize labor costs?",
                context
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="high_labor_cost",
                context=context,
                reasoning=reasoning,
                action="Review staffing levels and overtime policies",
                confidence=0.85
            )

        analysis = await self.analyze_with_claude(
            f"Daily labor analysis for {yesterday}: {len(employee_data)} employees worked "
            f"{context['total_labor_hours']:.1f} total hours at a cost of ${total_labor_cost:.2f} "
            f"({labor_cost_percentage:.1%} of revenue). Provide insights and recommendations.",
            context
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="daily_labor_analysis",
            context=context,
            reasoning=analysis,
            action="Generate daily labor report",
            confidence=0.8
        )

    async def _check_overtime_patterns(self, session) -> Optional[AgentDecision]:
        # Look at the last 7 days for overtime patterns
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)

        # Get all time records for the week
        week_records = session.query(TimeRecord).filter(
            func.date(TimeRecord.timestamp) >= start_date
        ).all()

        if not week_records:
            return None

        # Group by employee and day
        employee_daily_hours = {}
        for record in week_records:
            employee_id = record.employee_id
            record_date = record.timestamp.date()

            if employee_id not in employee_daily_hours:
                employee_daily_hours[employee_id] = {}

            if record_date not in employee_daily_hours[employee_id]:
                employee_daily_hours[employee_id][record_date] = []

            employee_daily_hours[employee_id][record_date].append(record)

        # Analyze overtime patterns
        employees_with_overtime = []
        total_overtime_cost = 0

        for employee_id, daily_records in employee_daily_hours.items():
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                continue

            weekly_overtime = 0
            overtime_days = 0

            for date, records in daily_records.items():
                daily_hours = self._calculate_daily_hours(records)
                if daily_hours > 8:
                    overtime_hours = daily_hours - 8
                    weekly_overtime += overtime_hours
                    overtime_days += 1

            if weekly_overtime > 0:
                overtime_cost = weekly_overtime * float(employee.hourly_rate) * 1.5
                total_overtime_cost += overtime_cost

                employees_with_overtime.append({
                    "name": f"{employee.first_name} {employee.last_name}",
                    "position": employee.position,
                    "weekly_overtime": weekly_overtime,
                    "overtime_days": overtime_days,
                    "overtime_cost": overtime_cost
                })

        if not employees_with_overtime:
            return None

        # Sort by overtime hours
        employees_with_overtime.sort(key=lambda x: x["weekly_overtime"], reverse=True)

        context = {
            "analysis_period": f"{start_date} to {end_date}",
            "employees_with_overtime": len(employees_with_overtime),
            "total_overtime_cost": total_overtime_cost,
            "top_overtime_employees": employees_with_overtime[:5],
            "average_overtime_per_employee": total_overtime_cost / len(employees_with_overtime)
        }

        analysis = await self.analyze_with_claude(
            f"Weekly overtime analysis shows {len(employees_with_overtime)} employees "
            f"worked overtime at a total cost of ${total_overtime_cost:.2f}. "
            f"Top overtime employee worked {employees_with_overtime[0]['weekly_overtime']:.1f} "
            f"overtime hours. Recommend strategies to reduce overtime costs.",
            context
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="overtime_analysis",
            context=context,
            reasoning=analysis,
            action="Implement overtime reduction strategies",
            confidence=0.9
        )

    async def _analyze_staffing_needs(self, session, revenue_data: Optional[Dict[str, Any]]) -> Optional[AgentDecision]:
        # Get current schedule for the next 7 days
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=7)

        upcoming_schedules = session.query(Schedule).filter(
            and_(
                Schedule.work_date >= start_date,
                Schedule.work_date <= end_date
            )
        ).all()

        # Group schedules by date
        daily_schedules = {}
        for schedule in upcoming_schedules:
            date = schedule.work_date
            if date not in daily_schedules:
                daily_schedules[date] = []
            daily_schedules[date].append(schedule)

        staffing_recommendations = []

        for date in [start_date + timedelta(days=i) for i in range(7)]:
            day_schedules = daily_schedules.get(date, [])

            # Calculate scheduled hours
            total_scheduled_hours = 0
            for schedule in day_schedules:
                # Convert time objects to datetime for subtraction
                if hasattr(schedule.start_time, 'hour'):
                    start_dt = datetime.combine(date.today(), schedule.start_time)
                    end_dt = datetime.combine(date.today(), schedule.end_time)
                    hours = (end_dt - start_dt).total_seconds() / 3600
                else:
                    # Already datetime objects
                    hours = (schedule.end_time - schedule.start_time).total_seconds() / 3600
                total_scheduled_hours += hours

            # Estimate needed hours based on day of week and historical patterns
            day_of_week = date.strftime('%A').lower()
            business_multipliers = {
                'monday': 0.7, 'tuesday': 0.8, 'wednesday': 0.9,
                'thursday': 1.0, 'friday': 1.3, 'saturday': 1.4, 'sunday': 1.1
            }

            base_hours_needed = 24  # Base hours for the day
            adjusted_hours_needed = base_hours_needed * business_multipliers.get(day_of_week, 1.0)

            gap = adjusted_hours_needed - total_scheduled_hours

            if abs(gap) > self.scheduling_buffer_hours:
                staffing_recommendations.append({
                    "date": str(date),
                    "day_of_week": day_of_week.title(),
                    "scheduled_hours": total_scheduled_hours,
                    "recommended_hours": adjusted_hours_needed,
                    "gap": gap,
                    "status": "understaffed" if gap > 0 else "overstaffed",
                    "employees_scheduled": len(day_schedules)
                })

        if not staffing_recommendations:
            return None

        context = {
            "analysis_period": f"{start_date} to {end_date}",
            "total_scheduling_issues": len(staffing_recommendations),
            "understaffed_days": len([r for r in staffing_recommendations if r["gap"] > 0]),
            "overstaffed_days": len([r for r in staffing_recommendations if r["gap"] < 0]),
            "recommendations": staffing_recommendations
        }

        analysis = await self.analyze_with_claude(
            f"Staffing analysis for next 7 days shows scheduling issues on "
            f"{len(staffing_recommendations)} days. "
            f"{context['understaffed_days']} days are understaffed, "
            f"{context['overstaffed_days']} days are overstaffed. "
            f"Provide specific scheduling recommendations.",
            context
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="staffing_analysis",
            context=context,
            reasoning=analysis,
            action="Adjust upcoming schedules to optimize staffing levels",
            confidence=0.82
        )

    async def _analyze_leave_request(self, session, request_data: Dict[str, Any]) -> Optional[AgentDecision]:
        employee_id = request_data.get("employee_id")
        start_date = datetime.fromisoformat(request_data.get("start_date")).date()
        end_date = datetime.fromisoformat(request_data.get("end_date")).date()
        leave_type = request_data.get("leave_type")

        employee = session.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return None

        # Check for scheduling conflicts
        conflicting_schedules = session.query(Schedule).filter(
            and_(
                Schedule.employee_id == employee_id,
                Schedule.work_date >= start_date,
                Schedule.work_date <= end_date
            )
        ).all()

        # Check staffing levels during requested period
        affected_dates = []
        current_date = start_date
        while current_date <= end_date:
            # Get other employees scheduled for this date
            other_schedules = session.query(Schedule).filter(
                and_(
                    Schedule.work_date == current_date,
                    Schedule.employee_id != employee_id
                )
            ).count()

            affected_dates.append({
                "date": str(current_date),
                "other_employees_scheduled": other_schedules,
                "has_conflicts": len([s for s in conflicting_schedules if s.work_date == current_date]) > 0
            })

            current_date += timedelta(days=1)

        context = {
            "employee": {
                "name": f"{employee.first_name} {employee.last_name}",
                "position": employee.position,
                "department": employee.department
            },
            "leave_request": request_data,
            "days_requested": (end_date - start_date).days + 1,
            "conflicting_schedules": len(conflicting_schedules),
            "affected_dates": affected_dates,
            "minimum_staffing_concern": any(date["other_employees_scheduled"] < 2 for date in affected_dates)
        }

        # Determine recommendation
        if leave_type == LeaveType.SICK or leave_type == LeaveType.EMERGENCY:
            approval_recommendation = "approve"
            reasoning_prompt = f"Emergency/sick leave request from {employee.first_name} {employee.last_name} " \
                            f"for {context['days_requested']} days. This should typically be approved. " \
                            f"What coverage arrangements are needed?"
        else:
            if context["minimum_staffing_concern"] or len(conflicting_schedules) > 2:
                approval_recommendation = "review"
                reasoning_prompt = f"Leave request from {employee.first_name} {employee.last_name} " \
                                f"creates significant staffing challenges. " \
                                f"Should this be approved or modified?"
            else:
                approval_recommendation = "approve"
                reasoning_prompt = f"Leave request from {employee.first_name} {employee.last_name} " \
                                f"appears manageable. Confirm approval and coverage plans."

        reasoning = await self.analyze_with_claude(reasoning_prompt, context)

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="leave_request_analysis",
            context=context,
            reasoning=reasoning,
            action=f"Process leave request - recommendation: {approval_recommendation}",
            confidence=0.85
        )

    async def generate_report(self) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            # Generate comprehensive HR summary
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)

            # Get employee counts
            total_employees = session.query(Employee).count()
            active_employees = session.query(Employee).filter(
                Employee.status == EmployeeStatus.ACTIVE
            ).count()

            # Get time records for the period
            time_records = session.query(TimeRecord).filter(
                func.date(TimeRecord.timestamp) >= start_date
            ).all()

            # Calculate total hours and labor costs (simplified)
            total_hours = 0
            total_labor_cost = 0
            overtime_hours = 0

            # This is a simplified calculation - in practice, you'd need more complex logic
            employee_hours = {}
            for record in time_records:
                employee_id = record.employee_id
                if employee_id not in employee_hours:
                    employee_hours[employee_id] = []
                employee_hours[employee_id].append(record)

            for employee_id, records in employee_hours.items():
                employee = session.query(Employee).filter(Employee.id == employee_id).first()
                if not employee:
                    continue

                # Group by date and calculate daily hours
                daily_records = {}
                for record in records:
                    date = record.timestamp.date()
                    if date not in daily_records:
                        daily_records[date] = []
                    daily_records[date].append(record)

                for date, day_records in daily_records.items():
                    daily_hours = self._calculate_daily_hours(day_records)
                    total_hours += daily_hours

                    regular_hours = min(daily_hours, 8)
                    daily_overtime = max(0, daily_hours - 8)
                    overtime_hours += daily_overtime

                    regular_pay = regular_hours * float(employee.hourly_rate)
                    overtime_pay = daily_overtime * float(employee.hourly_rate) * 1.5
                    total_labor_cost += regular_pay + overtime_pay

            # Get pending leave requests
            pending_leave_requests = session.query(LeaveRequest).filter(
                LeaveRequest.status == "pending"
            ).count()

            summary = HRSummary(
                total_employees=total_employees,
                active_employees=active_employees,
                total_hours_worked=Decimal(str(total_hours)),
                total_labor_cost=Decimal(str(total_labor_cost)),
                overtime_hours=Decimal(str(overtime_hours)),
                pending_leave_requests=pending_leave_requests,
                schedule_conflicts=0,  # Would implement conflict detection
                period_start=start_date,
                period_end=end_date
            )

            return {
                "summary": summary.model_dump(),
                "recent_decisions": [d.to_dict() for d in self.get_decision_history(10)],
                "alerts": await self._get_current_alerts(session)
            }
        finally:
            session.close()

    async def _get_current_alerts(self, session) -> List[Dict[str, Any]]:
        alerts = []

        # Check for pending leave requests
        pending_requests = session.query(LeaveRequest).filter(
            LeaveRequest.status == "pending"
        ).count()

        if pending_requests > 0:
            alerts.append({
                "type": "pending_leave_requests",
                "severity": "medium",
                "message": f"{pending_requests} leave requests need review",
                "action_required": True
            })

        # Check for employees with excessive overtime this week
        week_start = datetime.now().date() - timedelta(days=7)
        week_records = session.query(TimeRecord).filter(
            func.date(TimeRecord.timestamp) >= week_start
        ).all()

        # Simplified overtime check
        try:
            has_records = len(week_records) > 0
        except TypeError:
            # Mock object without __len__, assume has records for testing
            has_records = bool(week_records)

        if has_records:
            alerts.append({
                "type": "overtime_review",
                "severity": "low",
                "message": "Weekly overtime review needed",
                "action_required": False
            })

        return alerts
