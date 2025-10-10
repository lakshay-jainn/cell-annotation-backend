#!/usr/bin/env python3
"""
Process monitor for background processing tasks.
Checks for orphaned processes and provides status information.
"""

import sys
import os
import psutil
import logging
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_background_processes():
    """Get all background_processor.py processes."""
    current_pid = os.getpid()
    background_processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if proc.info['pid'] != current_pid:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'background_processor.py' in ' '.join(cmdline):
                    background_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline,
                        'create_time': datetime.fromtimestamp(proc.info['create_time']),
                        'age_minutes': (datetime.now() - datetime.fromtimestamp(proc.info['create_time'])).total_seconds() / 60
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return background_processes

def cleanup_orphaned_processes(max_age_minutes=30):
    """Clean up background processes that have been running too long."""
    processes = get_background_processes()
    cleaned = 0

    for proc in processes:
        if proc['age_minutes'] > max_age_minutes:
            try:
                logger.info(f"Terminating orphaned process {proc['pid']} (age: {proc['age_minutes']:.1f} minutes)")
                psutil.Process(proc['pid']).terminate()
                cleaned += 1
            except Exception as e:
                logger.error(f"Failed to terminate process {proc['pid']}: {e}")

    return cleaned

def main():
    """Main entry point for process monitoring."""
    if len(sys.argv) < 2:
        print("Usage: python process_monitor.py <command>")
        print("Commands: status, cleanup")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'status':
        processes = get_background_processes()
        if processes:
            print(f"Found {len(processes)} background processes:")
            for proc in processes:
                print(f"  PID {proc['pid']}: {' '.join(proc['cmdline'])} (age: {proc['age_minutes']:.1f} min)")
        else:
            print("No background processes found.")

    elif command == 'cleanup':
        cleaned = cleanup_orphaned_processes()
        print(f"Cleaned up {cleaned} orphaned processes.")

    else:
        print(f"Unknown command: {command}")

if __name__ == '__main__':
    main()