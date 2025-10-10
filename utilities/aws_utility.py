import boto3
import os
import botocore
import logging
from flask import current_app,jsonify
log = logging.getLogger(__name__)
REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
FRONTEND_URL = os.environ.get("FRONTEND_URL")
# Initialize s3 client once
if ENDPOINT:
    s3_client = boto3.client("s3", region_name=REGION, endpoint_url=ENDPOINT)
else:
    s3_client = boto3.client("s3", region_name=REGION)

cors = {
    "CORSRules": [
        {
            "AllowedOrigins": ["*"],  # Allow all origins for development
            "AllowedMethods": ["GET", "HEAD", "PUT", "POST", "DELETE"],
            "AllowedHeaders": ["*"],
            "ExposeHeaders": ["ETag", "x-amz-meta-custom-header"],
            "MaxAgeSeconds": 3000
        }
    ]
}

def initialize_s3_bucket():
    """Initialize S3 bucket and configure CORS"""
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"S3 bucket {S3_BUCKET_NAME} already exists")
        bucket_exists = True
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            # Bucket doesn't exist, create it
            try:
                s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
                print(f"Created S3 bucket: {S3_BUCKET_NAME}")
                bucket_exists = True
            except Exception as create_error:
                print(f"Failed to create S3 bucket: {create_error}")
                return False
        else:
            print(f"Error checking S3 bucket: {e}")
            return False
    
    # Configure CORS regardless of whether bucket was just created or already existed
    if bucket_exists:
        try:
            s3_client.put_bucket_cors(Bucket=S3_BUCKET_NAME, CORSConfiguration=cors)
            print(f"Configured CORS for S3 bucket: {S3_BUCKET_NAME}")
            return True
        except Exception as cors_error:
            print(f"Failed to configure CORS: {cors_error}")
            return False
    
    return False

def ensure_bucket_cors():
    """Ensure S3 bucket exists and has CORS configured - can be called multiple times"""
    if not S3_BUCKET_NAME:
        log.error("S3_BUCKET_NAME not configured, skipping bucket setup")
        return False
    
    log.info(f"Starting S3 bucket and CORS setup for bucket: {S3_BUCKET_NAME}")
    try:
        # Always try to configure CORS, even if bucket already exists
        log.debug(f"Configuring CORS for bucket {S3_BUCKET_NAME}")
        s3_client.put_bucket_cors(Bucket=S3_BUCKET_NAME, CORSConfiguration=cors)
        log.info(f"CORS configured/updated for S3 bucket: {S3_BUCKET_NAME}")
        return True
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        log.warning(f"S3 ClientError with code {error_code} for bucket {S3_BUCKET_NAME}")
        if error_code == 404:
            # Bucket doesn't exist, create it first
            try:
                log.info(f"Creating S3 bucket: {S3_BUCKET_NAME}")
                s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
                log.info(f"Created S3 bucket: {S3_BUCKET_NAME}")
                # Now configure CORS
                s3_client.put_bucket_cors(Bucket=S3_BUCKET_NAME, CORSConfiguration=cors)
                log.info(f"CORS configured for newly created S3 bucket: {S3_BUCKET_NAME}")
                return True
            except Exception as create_error:
                log.error(f"Failed to create S3 bucket or configure CORS: {create_error}")
                return False
        else:
            log.error(f"Error configuring CORS for S3 bucket: {e}")
            return False
    except Exception as e:
        log.error(f"Unexpected error configuring CORS: {e}")
        return False

# Initialize bucket on import (but handle errors gracefully)
if S3_BUCKET_NAME:
    initialize_s3_bucket()

def upload_to_s3(bucket, Fileobj, key):

    try:
        # ensure stream at start
        try:
            Fileobj.stream.seek(0)
        except Exception:
            pass
        s3_client.upload_fileobj(Fileobj.stream, bucket, key)
        current_app.logger.info(f"Uploaded to s3://{bucket}/{key}")
    except botocore.exceptions.ClientError as e:
        current_app.logger.exception("S3 upload failed")
        return jsonify({"message": "S3 upload failed", "details": str(e)}), 500
    except Exception as e:
        current_app.logger.exception("S3 upload failed (unknown)")
        return jsonify({"message": "S3 upload failed", "details": str(e)}), 500
    

def get_presigned_url(bucket, key, expires=300):
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )
    # Replace localstack hostname with localhost for external access
    if url and 'localstack' in url:
        url = url.replace('localstack', 'localhost')
    return url