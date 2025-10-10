# utilities/enqueue_utility.py
import os
import json
import boto3
import botocore
import logging
import time
from sqs_processor import get_queue_url

log = logging.getLogger(__name__)

# Cache for queue status to avoid repeated expensive checks
_queue_status_cache = {}
_cache_expiry_seconds = 5  # Cache for 5 seconds (increased from 3 for better performance)

def invalidate_queue_cache():
    """Invalidate the queue status cache to force a fresh check."""
    global _queue_status_cache
    _queue_status_cache.clear()
    log.debug("Queue status cache invalidated")

REGION = os.environ.get("AWS_REGION")
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")

if ENDPOINT:
    sqs = boto3.client("sqs", region_name=REGION, endpoint_url=ENDPOINT)
else:
    sqs = boto3.client("sqs", region_name=REGION)

def get_all_queued_job_ids():
    """
    Get all job IDs currently in the queue.
    Returns a set of job_ids.
    Uses caching to avoid repeated expensive queue scans.
    """
    global _queue_status_cache
    
    # Check if we have a valid cache
    cache_key = 'all_job_ids'
    current_time = time.time()
    
    if cache_key in _queue_status_cache:
        cached_data, cached_time = _queue_status_cache[cache_key]
        if current_time - cached_time < _cache_expiry_seconds:
            log.debug(f"Using cached queue status ({current_time - cached_time:.1f}s old)")
            return cached_data
    
    # Cache miss or expired - fetch from queue
    try:
        queue_url = get_queue_url()
        if not queue_url:
            log.warning("Could not get queue URL")
            return set()

        job_ids = set()
        max_iterations = 30  # Reduced from 100 to scan up to 300 messages max
        iterations = 0
        
        while iterations < max_iterations:
            # Use ReceiveMessage to peek at messages
            # Note: Even with VisibilityTimeout=0, messages may temporarily disappear
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                VisibilityTimeout=0,
                WaitTimeSeconds=0,  # No waiting, return immediately
                AttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            
            if not messages:
                break
                
            for message in messages:
                try:
                    body = json.loads(message.get('Body', '{}'))
                    job_id = body.get('jobId')
                    if job_id:
                        job_ids.add(job_id)
                except json.JSONDecodeError:
                    continue
            
            iterations += 1
        
        # Cache the result
        _queue_status_cache[cache_key] = (job_ids, current_time)
        log.info(f"Fetched {len(job_ids)} job IDs from queue (cached for {_cache_expiry_seconds}s)")
        
        return job_ids
        
    except Exception as e:
        log.error(f"Error getting queued job IDs: {e}")
        return set()

def is_job_in_queue(job_id):
    """
    Check if a job_id already exists in the SQS queue.
    Returns True if found, False otherwise.
    
    Uses cached queue status to provide consistent results.
    """
    try:
        queued_job_ids = get_all_queued_job_ids()
        return job_id in queued_job_ids
    except Exception as e:
        log.error(f"Error checking for duplicate job_id {job_id}: {e}")
        # On error, allow queuing to proceed (fail open)
        return False

def enqueue(payload, check_duplicates=True):
    """
    Enqueue a payload to SQS.
    
    Args:
        payload: Dictionary containing job information (must include 'jobId')
        check_duplicates: If True, check if job_id already exists in queue before adding
    
    Returns:
        Dictionary with enqueue status and details
    """
    try:
        # 1. Validate the payload
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary.")

        queue_url = get_queue_url()
        if not queue_url:
            raise ValueError("Could not get or create SQS queue.")

        # 2. Check for duplicates if requested
        job_id = payload.get('jobId')
        if check_duplicates and job_id:
            if is_job_in_queue(job_id):
                log.info(f"Job {job_id} already in queue, skipping")
                return {
                    "enqueued": False,
                    "already_queued": True,
                    "job_id": job_id,
                    "message": "Job already in queue"
                }

        # 3. Send the message to SQS
        resp = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(payload)
        )

        # 4. Invalidate cache since queue state changed
        invalidate_queue_cache()

        # 5. Return a successful result
        return {
            "enqueued": True,
            "already_queued": False,
            "message_id": resp.get("MessageId"),
            "queue_url": queue_url
        }

    except botocore.exceptions.ClientError as e:
        log.exception("AWS ClientError while enqueueing")
        return {
            "error": "aws_client_error",
            "details": e.response,
        }
    except ValueError as e:
        log.error(f"Validation error: {e}")
        return {
            "error": "validation_error",
            "details": str(e)
        }
    except Exception as e:
        log.exception("Unexpected error while enqueueing")
        return {
            "error": "internal_server_error",
            "details": str(e)
        }

def clear_queue():
    """
    Purge all messages from the SQS queue.
    Returns dict with success status and message count if available.
    """
    try:
        queue_url = get_queue_url()
        if not queue_url:
            raise ValueError("Could not get or create SQS queue.")

        # Purge the queue (deletes all messages)
        sqs.purge_queue(QueueUrl=queue_url)
        log.info(f"Successfully purged queue: {queue_url}")
        
        # Invalidate cache since queue is now empty
        invalidate_queue_cache()
        
        return {
            "cleared": True,
            "queue_url": queue_url
        }

    except botocore.exceptions.ClientError as e:
        log.exception("AWS ClientError while clearing queue")
        return {
            "error": "aws_client_error",
            "details": e.response,
        }
    except ValueError as e:
        log.error(f"Validation error: {e}")
        return {
            "error": "validation_error",
            "details": str(e)
        }
    except Exception as e:
        log.exception("Unexpected error while clearing queue")
        return {
            "error": "internal_server_error",
            "details": str(e)
        }