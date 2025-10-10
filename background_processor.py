#!/usr/bin/env python3
"""
Background processor for handling model processing tasks.
This runs as a separate process to avoid blocking the admin interface.
"""

import sys
import os
import logging
import signal
import atexit

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import db at module level
from db.models import db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/background_processor.log'),  # Log to file in production
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown handling
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def setup_database():
    """Set up database connection for background processing using SQLAlchemy directly."""
    try:
        import os
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker, scoped_session
        
        # Get database URI from environment
        SQL_URI = os.environ.get("SQL_URI", "sqlite:///test.db")
        
        # Configure SQLAlchemy directly
        engine = create_engine(SQL_URI)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Note: Tables are assumed to be created by the main Flask app
        # We don't create them here to avoid conflicts
        
        # Return the session
        return SessionLocal()
        
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def process_sample(sample_id):
    """Process a single sample in the background."""
    session = None
    try:
        logger.info(f"Starting background processing for sample: {sample_id}")

        # Set up Flask app context
        from server import create_app
        app = create_app()
        
        with app.app_context():
            # Setup database connection
            session = setup_database()
            if not session:
                logger.error("Failed to setup database connection")
                return False

            # Import processing function
            from worker_manual import process_sample_manually

            # Update sample status to processing
            from db.models import Sample
            from datetime import datetime
            sample = session.query(Sample).filter_by(job_id=sample_id).first()
            if not sample:
                logger.error(f"Sample {sample_id} not found")
                return False

            sample.inference_status = 'processing'
            sample.updated_at = datetime.utcnow()
            session.commit()

            # Process the sample
            result = process_sample_manually(sample_id)

            # Update status based on result
            if result.get('success', False):
                sample.inference_status = 'completed'
                logger.info(f"Sample {sample_id} processed successfully")
                
                # Delete the SQS message for this completed job
                try:
                    from sqs_processor import delete_message_for_job_id
                    delete_message_for_job_id(sample.job_id)
                except Exception as e:
                    logger.warning(f"Failed to delete SQS message for completed job {sample.job_id}: {e}")
                
                success = True
            else:
                sample.inference_status = 'failed'
                logger.error(f"Sample {sample_id} processing failed: {result.get('error', 'Unknown error')}")
                success = False

            sample.updated_at = datetime.utcnow()
            session.commit()

            return success

    except Exception as e:
        logger.error(f"Background processing failed for {sample_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Try to update status to failed
        try:
            if session:
                from db.models import Sample
                from datetime import datetime
                sample = session.query(Sample).filter_by(job_id=sample_id).first()
                if sample:
                    sample.inference_status = 'failed'
                    sample.updated_at = datetime.utcnow()
                    session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update sample status: {db_error}")
        return False
    finally:
        if session:
            session.close()

def process_sqs_messages(max_messages=1):
    """Process SQS messages in the background."""
    try:
        logger.info(f"Starting background SQS processing (max {max_messages} messages)")

        # Set up Flask app context
        from server import create_app
        app = create_app()
        
        with app.app_context():
            # Import here to avoid circular imports
            from sqs_processor import process_sqs_messages

            result = process_sqs_messages(max_messages)

            processed = result.get('processed', 0)
            errors = result.get('errors', 0)

            logger.info(f"SQS processing completed: {processed} processed, {errors} errors")
            return True

    except Exception as e:
        logger.error(f"Background SQS processing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main entry point for background processing."""
    if len(sys.argv) < 2:
        logger.error("Usage: python background_processor.py <task_type> [args...]")
        sys.exit(1)

    task_type = sys.argv[1]

    if task_type == 'sample' and len(sys.argv) >= 3:
        sample_id = sys.argv[2]
        success = process_sample(sample_id)
        sys.exit(0 if success else 1)

    elif task_type == 'sqs':
        max_messages = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
        success = process_sqs_messages(max_messages)
        sys.exit(0 if success else 1)

    elif task_type == 'cleanup':
        reset_count = cleanup_stuck_processing_samples()
        print(f"Reset {reset_count} stuck processing samples")
        sys.exit(0)

    else:
        logger.error(f"Unknown task type: {task_type}")
        logger.error("Supported task types: 'sample <sample_id>', 'sqs [max_messages]', or 'cleanup'")
        sys.exit(1)

def cleanup_stuck_processing_samples():
    """Reset all samples that are currently in 'processing' state back to 'pending'."""
    session = None
    try:
        logger.info("Resetting all processing samples to pending...")
        
        # Set up Flask app context
        from server import create_app
        app = create_app()
        
        with app.app_context():
            # Setup database connection
            session = setup_database()
            if not session:
                logger.error("Failed to setup database connection for cleanup")
                return 0
            
            from db.models import Sample
            from datetime import datetime
            
            # Find all samples currently in processing state
            processing_samples = session.query(Sample).filter(
                Sample.inference_status == 'processing'
            ).all()
            
            reset_count = 0
            for sample in processing_samples:
                logger.info(f"Resetting sample {sample.job_id} from processing to pending")
                sample.inference_status = 'pending'
                sample.updated_at = datetime.utcnow()
                reset_count += 1
            
            if reset_count > 0:
                session.commit()
                logger.info(f"Reset {reset_count} processing samples back to pending")
            else:
                logger.info("No samples were in processing state")
            
            return reset_count
        
    except Exception as e:
        logger.error(f"Error during sample cleanup: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 0
    finally:
        if session:
            session.close()

if __name__ == '__main__':
    main()