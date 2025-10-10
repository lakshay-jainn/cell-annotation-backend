# utilities/logging_utility.py
"""
Comprehensive user activity logging utility
"""

import json
from flask import request
from db.models import db, UserActivityLog
from datetime import datetime
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class ActivityLogger:
    """Utility class for logging user activities throughout the application"""

    @staticmethod
    def log_activity(user_id=None, user_role=None, action_type="", action_details="",
                    status="success", metadata=None, error_message=None):
        """
        Log user activity to database

        Args:
            user_id: User ID performing the action
            user_role: User role (for quick queries without joining)
            action_type: Type of action (login, logout, button_click, api_call, etc.)
            action_details: Brief description of the action
            status: success, error, info, warning
            metadata: Additional JSON data
            error_message: Error details if status is error
        """
        try:
            # Get request context
            ip_address = request.remote_addr if request else None
            user_agent = request.headers.get('User-Agent') if request else None

            # Prepare metadata
            if metadata is None:
                metadata = {}

            if error_message:
                metadata['error_message'] = error_message

            # Add request details if available
            if request:
                metadata.update({
                    'method': request.method,
                    'endpoint': request.endpoint,
                    'url': request.url,
                    'headers': dict(request.headers),
                })

            # Create log entry
            log_entry = UserActivityLog(
                user_id=user_id,
                user_role=user_role,
                action_type=action_type,
                action_details=action_details,
                status=status,
                ip_address=ip_address,
                user_agent=user_agent,
                activity_metadata=metadata
            )

            db.session.add(log_entry)
            db.session.commit()

            # Also log to application logger
            log_level = {
                'success': logging.INFO,
                'error': logging.ERROR,
                'warning': logging.WARNING,
                'info': logging.INFO
            }.get(status, logging.INFO)

            logger.log(log_level, f"UserActivity: user={user_id} role={user_role} action={action_type} status={status} details={action_details}")

        except Exception as e:
            # Don't let logging errors break the application
            logger.error(f"Failed to log activity: {e}")
            # Try to commit anyway in case of session issues
            try:
                db.session.rollback()
            except:
                pass

    @staticmethod
    def log_api_call(user_id, user_role, endpoint, method, status="success", response_data=None, error=None):
        """Log API call activity"""
        metadata = {
            'api_endpoint': endpoint,
            'http_method': method,
        }

        if response_data:
            # Store limited response data (avoid storing large responses)
            if isinstance(response_data, dict):
                metadata['response_keys'] = list(response_data.keys())
                if 'data' in response_data and isinstance(response_data['data'], list):
                    metadata['response_count'] = len(response_data['data'])

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details=f"{method} {endpoint}",
            status=status,
            metadata=metadata,
            error_message=str(error) if error else None
        )

    @staticmethod
    def log_user_action(user_id, user_role, action, details=None, element_id=None, page=None):
        """Log user interface actions (button clicks, navigation, etc.)"""
        metadata = {}
        if element_id:
            metadata['element_id'] = element_id
        if page:
            metadata['page'] = page
        if details:
            metadata['action_details'] = details

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="user_action",
            action_details=action,
            status="info",
            metadata=metadata
        )

    @staticmethod
    def log_auth_event(user_id, user_role, event_type, details=None, ip_address=None):
        """Log authentication events (login, logout, failed login)"""
        metadata = {}
        if details:
            metadata['auth_details'] = details
        if ip_address:
            metadata['ip_address'] = ip_address

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details=event_type,
            status="success" if event_type in ["login", "logout"] else "warning",
            metadata=metadata
        )

    @staticmethod
    def log_workflow_step(user_id, user_role, workflow_type, step, step_data=None):
        """Log workflow progression steps"""
        metadata = {}
        if step_data:
            metadata['step_data'] = step_data

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="workflow_step",
            action_details=f"{workflow_type}:{step}",
            status="info",
            metadata=metadata
        )

def log_api_activity(f):
    """Decorator to automatically log API calls"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Get user info from request context (set by auth middleware)
        user_id = getattr(request, 'user_id', None) if request else None
        user_role = getattr(request, 'user_role', None) if request else None

        endpoint = request.endpoint if request else None
        method = request.method if request else None

        try:
            result = f(*args, **kwargs)

            # Log successful API call
            if user_id:
                ActivityLogger.log_api_call(
                    user_id=user_id,
                    user_role=user_role,
                    endpoint=endpoint,
                    method=method,
                    status="success"
                )

            return result

        except Exception as e:
            # Log failed API call
            if user_id:
                ActivityLogger.log_api_call(
                    user_id=user_id,
                    user_role=user_role,
                    endpoint=endpoint,
                    method=method,
                    status="error",
                    error=e
                )
            raise

    return wrapper