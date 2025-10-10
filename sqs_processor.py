# sqs_processor.py
"""
Simple SQS message processor without continuous polling.
"""

import json
import logging
import os
import boto3
from datetime import datetime
from db.models import Sample
from worker_manual import process_sample_manually

logger = logging.getLogger(__name__)

def get_sqs_client():
    """Get SQS client configured for LocalStack."""
    return boto3.client(
        'sqs',
        region_name=os.environ.get('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'test'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'test'),
        endpoint_url=os.environ.get('AWS_ENDPOINT_URL', 'http://localstack:4566')
    )

def ensure_sqs_queue():
    """
    Ensure SQS queue exists, create it if it doesn't.
    
    Returns:
        str: Queue URL if successful, None if failed
    """
    sqs = get_sqs_client()
    queue_name = os.environ.get('SQS_QUEUE_NAME', 'my-queue')
    queue_url_env = os.environ.get('SQS_QUEUE_URL')
    
    # If queue URL is explicitly set, try to use it
    if queue_url_env:
        try:
            # Check if queue exists by getting its attributes
            sqs.get_queue_attributes(QueueUrl=queue_url_env, AttributeNames=['QueueArn'])
            logger.info(f"SQS queue exists: {queue_url_env}")
            return queue_url_env
        except sqs.exceptions.QueueDoesNotExist:
            logger.warning(f"SQS queue does not exist at URL: {queue_url_env}")
        except Exception as e:
            logger.error(f"Error checking queue at URL {queue_url_env}: {e}")
    
    # Try to create queue or get existing one
    try:
        # First, try to get queue URL by name (works if queue exists)
        response = sqs.get_queue_url(QueueName=queue_name)
        queue_url = response['QueueUrl']
        logger.info(f"SQS queue exists: {queue_url}")
        return queue_url
    except sqs.exceptions.QueueDoesNotExist:
        # Queue doesn't exist, create it
        try:
            # Create queue with minimal attributes for LocalStack compatibility
            response = sqs.create_queue(QueueName=queue_name)
            queue_url = response['QueueUrl']
            logger.info(f"Created SQS queue: {queue_url}")
            return queue_url
        except Exception as e:
            logger.error(f"Failed to create SQS queue {queue_name}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error getting queue URL for {queue_name}: {e}")
        return None

def get_queue_url():
    """
    Get the SQS queue URL, ensuring the queue exists.
    Handles both Docker (localstack hostname) and host (localhost) contexts.
    
    Returns:
        str: Queue URL or None if failed
    """
    queue_url = ensure_sqs_queue()
    if queue_url:
        return queue_url
    else:
        # Fallback: use localhost for host machine access
        return 'http://localhost:4566/000000000000/my-queue'

def process_upload_pipeline_message(message_data):
    """
    Process upload pipeline message format:
    {"jobId": "...", "s3ObjectKey": "...", "s3BucketName": "...", "status": "pending"}
    """
    logger.info(f"Processing upload pipeline message: {message_data}")
    
    try:
        # Ensure we have Flask application context
        from flask import current_app
        if not current_app:
            # Create app context if we don't have one
            from server import create_app
            app = create_app()
            with app.app_context():
                return _process_message_with_context(message_data)
        else:
            return _process_message_with_context(message_data)
            
    except Exception as exc:
        logger.error(f"Upload pipeline message processing failed: {exc}")
        return False

def _process_message_with_context(message_data):
    """Helper function that processes message within Flask context."""
    try:
        # Extract information from the message
        job_id = message_data.get('jobId')
        s3_object_key = message_data.get('s3ObjectKey')
        s3_bucket_name = message_data.get('s3BucketName')
        status = message_data.get('status')
        
        if not all([job_id, s3_object_key, s3_bucket_name]):
            logger.error(f"Invalid message format - missing required fields: {message_data}")
            return False
        
        # Find the sample in the database
        sample = Sample.query.filter_by(job_id=job_id).first()
        
        # If not found, try to find by filename matching s3_object_key
        if not sample:
            filename = s3_object_key.split('/')[-1]  # Extract filename from S3 key
            sample = Sample.query.filter(Sample.original_filename.ilike(f'%{filename}%')).first()
        
        if not sample:
            logger.warning(f"No sample found for job_id: {job_id}, s3_object_key: {s3_object_key}")
            return False
        
        logger.info(f"Found sample {sample.job_id} for upload pipeline message")
        
        # Note: process_sample_manually will handle setting status to 'processing'
        # Process the sample using the existing manual processing function
        result = process_sample_manually(sample.job_id)
        
        if result['success']:
            logger.info(f"Successfully processed upload pipeline message for sample {sample.job_id}")
            return True
        else:
            logger.error(f"Failed to process upload pipeline message for sample {sample.job_id}: {result.get('error')}")
            return False
            
    except Exception as exc:
        logger.error(f"Upload pipeline message processing failed: {exc}")
        return False

def queue_samples_for_processing(sample_ids):
    """
    Add sample IDs to SQS queue for sequential processing.
    
    Args:
        sample_ids (list): List of sample job_ids to queue for processing
        
    Returns:
        dict: Result with success status and queued count
    """
    sqs = get_sqs_client()
    queue_url = get_queue_url()
    
    if not queue_url:
        return {
            'success': False,
            'error': 'Could not get or create SQS queue',
            'queued': 0,
            'errors': len(sample_ids)
        }
    
    logger.info(f"Adding {len(sample_ids)} samples to SQS queue: {queue_url}")
    
    queued_count = 0
    error_count = 0
    errors = []
    
    try:
        for sample_id in sample_ids:
            try:
                # Get sample info from database
                sample = Sample.query.filter_by(job_id=sample_id).first()
                if not sample:
                    logger.warning(f"Sample {sample_id} not found, skipping")
                    error_count += 1
                    errors.append(f"Sample {sample_id} not found")
                    continue
                
                # Create message in upload pipeline format
                message_body = {
                    "jobId": sample.job_id,
                    "s3ObjectKey": sample.s3_object_key,
                    "s3BucketName": os.environ.get("S3_BUCKET_NAME", "my-test-bucket"),
                    "status": "pending",
                    "queuedAt": datetime.now().isoformat(),
                    "source": "admin_batch_queue"
                }
                
                # Send message to SQS
                response = sqs.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(message_body),
                    MessageAttributes={
                        'source': {
                            'StringValue': 'admin_batch_queue',
                            'DataType': 'String'
                        },
                        'jobId': {
                            'StringValue': sample.job_id,
                            'DataType': 'String'
                        }
                    }
                )
                
                logger.info(f"Queued sample {sample_id} for processing (MessageId: {response.get('MessageId', 'unknown')})")
                queued_count += 1
                
            except Exception as e:
                logger.error(f"Failed to queue sample {sample_id}: {e}")
                error_count += 1
                errors.append(f"Sample {sample_id}: {str(e)}")
        
        return {
            'success': True,
            'queued': queued_count,
            'errors': error_count,
            'error_details': errors[:5] if errors else []  # Limit to first 5 errors
        }
        
    except Exception as e:
        logger.error(f"Error in queue_samples_for_processing: {e}")
        return {
            'success': False,
            'error': str(e),
            'queued': queued_count,
            'errors': error_count
        }

def get_queue_status():
    """
    Get the current status of the SQS queue.
    
    Returns:
        dict: Queue status information
    """
    try:
        sqs = get_sqs_client()
        queue_url = get_queue_url()
        
        if not queue_url:
            return {
                'success': False,
                'error': 'Could not get or create SQS queue',
                'messages_available': 0,
                'messages_in_flight': 0,
                'total_messages': 0
            }
        
        # Get queue attributes
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
        )
        
        attributes = response.get('Attributes', {})
        
        return {
            'success': True,
            'queue_url': queue_url,
            'messages_available': int(attributes.get('ApproximateNumberOfMessages', 0)),
            'messages_in_flight': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
            'total_messages': int(attributes.get('ApproximateNumberOfMessages', 0)) + 
                            int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0))
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {
            'success': False,
            'error': str(e),
            'messages_available': 0,
            'messages_in_flight': 0,
            'total_messages': 0
        }

def process_sqs_messages(max_messages=10, process_all=True):
    """
    Process messages from SQS queue on-demand (not continuous polling).
    
    Args:
        max_messages (int): Maximum messages to receive per batch
        process_all (bool): If True, process all messages in queue until empty
    """
    logger.info("Starting process_sqs_messages function")
    sqs = get_sqs_client()
    queue_url = get_queue_url()
    
    if not queue_url:
        logger.error("Could not get or create SQS queue")
        return {
            'success': False,
            'error': 'Could not get or create SQS queue',
            'processed': 0,
            'errors': 0
        }
    
    logger.info(f"Processing SQS messages from queue: {queue_url}")
    
    total_processed = 0
    total_errors = 0
    
    try:
        # STEP 1: Collect ALL messages first
        all_messages = []
        batch_count = 0
        
        logger.info("Step 1: Collecting all messages from queue...")
        while True:
            batch_count += 1
            logger.info(f"Collecting batch {batch_count}: Polling for up to {max_messages} messages")
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=0,  # No waiting - return immediately with available messages
                MessageAttributeNames=['All'],
                AttributeNames=['ApproximateReceiveCount'],
                VisibilityTimeout=300  # 5 minutes to process all
            )
            
            messages = response.get('Messages', [])
            logger.info(f"Collecting batch {batch_count}: Retrieved {len(messages)} messages")
            
            if not messages:
                logger.info(f"No more messages found after {batch_count-1} batches")
                break
                
            all_messages.extend(messages)
            
            if not process_all:
                break
        
        if not all_messages:
            logger.info("No messages found in queue")
            return {
                'success': True,
                'processed': 0,
                'errors': 0,
                'total_messages': 0,
                'message': 'No messages in queue'
            }
        
        logger.info(f"Collected {len(all_messages)} total messages")
        
        # STEP 2: Set ALL corresponding samples to 'processing' status
        logger.info("Step 2: Setting all samples to 'processing' status...")
        from db.models import db
        
        processing_job_ids = []
        for message in all_messages:
            try:
                message_data = json.loads(message['Body'])
                job_id = message_data.get('jobId')
                if job_id:
                    # Find the sample
                    sample = Sample.query.filter_by(job_id=job_id).first()
                    if sample and sample.inference_status == 'pending':
                        sample.inference_status = 'processing'
                        sample.updated_at = datetime.utcnow()
                        processing_job_ids.append(job_id)
                        logger.info(f"Set sample {job_id} to 'processing' status")
            except Exception as e:
                logger.warning(f"Failed to parse message for status update: {e}")
        
        # Commit all status updates at once
        try:
            db.session.commit()
            logger.info(f"Updated {len(processing_job_ids)} samples to 'processing' status")
        except Exception as e:
            logger.error(f"Failed to commit status updates: {e}")
            db.session.rollback()
        
        # STEP 3: Now process each message
        logger.info("Step 3: Processing all messages...")
        for i, message in enumerate(all_messages):
            message_body = message['Body']
            receipt_handle = message['ReceiptHandle']
            
            logger.info(f"Processing message {i+1}/{len(all_messages)}: {message_body[:100]}...")
            
            try:
                message_data = json.loads(message_body)
                
                # Check if it's an upload pipeline message
                required_fields = ['jobId', 's3ObjectKey', 's3BucketName', 'status']
                if all(field in message_data for field in required_fields):
                    logger.info(f"Processing upload pipeline message {i+1}/{len(all_messages)}")
                    
                    if process_upload_pipeline_message(message_data):
                        # Delete message after successful processing
                        sqs.delete_message(
                            QueueUrl=queue_url,
                            ReceiptHandle=receipt_handle
                        )
                        total_processed += 1
                        logger.info(f"Message {i+1}/{len(all_messages)} processed and deleted")
                    else:
                        # Delete message even on failure to prevent infinite reprocessing
                        # The sample status is already set to 'failed' in the database
                        sqs.delete_message(
                            QueueUrl=queue_url,
                            ReceiptHandle=receipt_handle
                        )
                        total_errors += 1
                        logger.warning(f"Message {i+1}/{len(all_messages)} processing failed, message deleted to prevent reprocessing")
                
                else:
                    logger.warning(f"Unknown message format: {message_data}")
                    # Delete unknown messages
                    sqs.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=receipt_handle
                    )
                
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON message received: {message_body[:100]}")
                # Delete non-JSON messages
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
            
            except Exception as e:
                logger.error(f"Error processing message {i+1}/{len(all_messages)}: {e}")
                import traceback
                traceback.print_exc()
                total_errors += 1
        
        logger.info(f"SQS processing complete: Total processed {total_processed}, Total errors {total_errors}")
        
        return {
            'success': True,
            'processed': total_processed,
            'errors': total_errors,
            'total_messages': len(all_messages),
            'message': f'Processed {total_processed} messages, {total_errors} errors'
        }
        
    except Exception as e:
        logger.error(f"Error in SQS message processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'processed': total_processed,
            'errors': total_errors
        }

def delete_message_for_job_id(job_id):
    """
    Delete SQS message for a specific job_id when sample processing is completed.
    This prevents completed jobs from staying in the queue.
    
    Args:
        job_id (str): The job_id of the completed sample
        
    Returns:
        bool: True if message was found and deleted, False otherwise
    """
    try:
        sqs = get_sqs_client()
        queue_url = get_queue_url()
        
        if not queue_url:
            logger.error("Could not get SQS queue URL for message deletion")
            return False
        
        logger.info(f"Searching for SQS message to delete for completed job: {job_id}")
        
        # Search for the message with matching job_id
        # We need to receive messages and check their content
        max_attempts = 3
        for attempt in range(max_attempts):
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=2,
                MessageAttributeNames=['All'],
                VisibilityTimeout=60  # Give us time to process
            )
            
            messages = response.get('Messages', [])
            if not messages:
                logger.info(f"No messages found in queue (attempt {attempt+1}/{max_attempts})")
                break
            
            for message in messages:
                receipt_handle = message['ReceiptHandle']
                try:
                    body = json.loads(message['Body'])
                    message_job_id = body.get('jobId')
                    
                    if message_job_id == job_id:
                        # Found the message for this job_id, delete it
                        sqs.delete_message(
                            QueueUrl=queue_url,
                            ReceiptHandle=receipt_handle
                        )
                        logger.info(f"Successfully deleted SQS message for completed job: {job_id}")
                        return True
                    else:
                        # Not the message we're looking for, put it back by doing nothing
                        # (it will become visible again after visibility timeout)
                        pass
                        
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message found in queue: {message['Body'][:100]}")
                except Exception as e:
                    logger.error(f"Error processing message during deletion search: {e}")
        
        logger.warning(f"SQS message not found for job_id: {job_id}")
        return False
        
    except Exception as e:
        logger.error(f"Error deleting SQS message for job_id {job_id}: {e}")
        return False