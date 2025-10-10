import os
import time
import json
import logging
import boto3
import botocore
import requests
import tempfile
from botocore.config import Config
from db.models import db,Sample
from sqlalchemy import create_engine,update, not_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
QUEUE_NAME = os.environ.get("SQS_QUEUE_NAME", "my-queue")
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "5"))
WAIT_TIME = int(os.environ.get("WAIT_TIME", "10"))
VISIBILITY_TIMEOUT = int(os.environ.get("VISIBILITY_TIMEOUT", "60"))
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_API_URL = os.environ.get("MODEL_API_URL")
HF_API_KEY = os.environ.get("HF_API_KEY")

boto_conf = Config(retries={"max_attempts": 4, "mode": "standard"})

log.info("Worker config: REGION=%s ENDPOINT=%s SQS_QUEUE_URL=%s S3_BUCKET=%s",
         REGION, ENDPOINT, bool(SQS_QUEUE_URL), S3_BUCKET_NAME)

if not S3_BUCKET_NAME or not MODEL_API_URL:
    log.error("Error: S3_BUCKET_NAME or MODEL_API_URL environment variable is not set.")
    exit(1)

if ENDPOINT:
    sqs = boto3.client("sqs", region_name=REGION, endpoint_url=ENDPOINT, config=boto_conf)
    s3 = boto3.client("s3", region_name=REGION, endpoint_url=ENDPOINT, config=boto_conf)
else:
    sqs = boto3.client("sqs", region_name=REGION, config=boto_conf)
    s3 = boto3.client("s3", region_name=REGION, config=boto_conf)



def download_from_s3(bucket, key, local_path):

    log.info("Downloading file %s/%s to %s", bucket, key, local_path)
    try:
        s3.download_file(bucket, key, local_path)
        log.info("Download successful.")
    except botocore.exceptions.ClientError as e:
        log.error("Failed to download file from S3. Error: %s", e)
        raise

def upload_to_s3(bucket, local_path, key):

    log.info("Uploading file %s to %s/%s", local_path, bucket, key)
    try:
        s3.upload_file(local_path, bucket, key)
        log.info("Upload successful.")
    except botocore.exceptions.ClientError as e:
        log.error("Failed to upload file to S3. Error: %s", e)
        raise

def call_model_api(local_image_path):
    """
    Makes a POST request to the model API with the image as form data.
    Simulates a model returning a CSV file.
    """
    log.info("Sending image to model API at %s", MODEL_API_URL)
    with open(local_image_path, 'rb') as img_file:
        files = {"image": img_file}
        # Add the Authorization header with the Bearer token
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        try:
            response = requests.post(MODEL_API_URL, files=files, headers=headers, timeout=180)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            log.info("Model API call successful. Received %d bytes.", len(response.content))
            return response.content
        except requests.exceptions.RequestException as e:
            log.error("Model API call failed. Error: %s", e)
            raise

SQL_URI = os.getenv("SQL_URI")
engine = create_engine(
    SQL_URI,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def poll_loop():
    qurl = SQS_QUEUE_URL
    log.info("Begin polling queue: %s", qurl)
    backoff = 1.0

    while True:
        temp_image_path = None
        temp_csv_path = None
        
        try:
            # 1. Receive message from SQS
            # existing = db.session.execute(db.select(User).filter_by(email="lakshay6690@gmail.com")).scalar()
            # print(existing)
            resp = sqs.receive_message(
                QueueUrl=qurl,
                MaxNumberOfMessages=MAX_MESSAGES,
                WaitTimeSeconds=WAIT_TIME,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
            )

        except botocore.exceptions.ClientError as e:
            log.error("ClientError on receive_message. error.response: %s", json.dumps(e.response, default=str))
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        except Exception:
            log.exception("Unexpected exception from receive_message")
            time.sleep(2)
            continue

        # Reset backoff on successful SQS poll.
        backoff = 1.0

        messages = resp.get("Messages", [])
        if not messages:
            continue

        log.info("Received %d messages", len(messages))
        for m in messages:
            receipt_handle = m["ReceiptHandle"]
            
            # Use a try-finally block to ensure temp files are cleaned up.
            try:
                # 2. Parse SQS message body
                body = json.loads(m["Body"])
                s3_key = body.get("s3ObjectKey")
                s3_bucket = body.get("s3BucketName")
                job_id = body.get("jobId")

                if not s3_key or not s3_bucket or not job_id:
                    log.error("Message body is missing 'key' or 'bucket' or 'job_id'. Deleting message to avoid poison.")
                    sqs.delete_message(QueueUrl=qurl, ReceiptHandle=receipt_handle)
                    continue

                # Get job ID from the S3 key
                # Example: "uploads/job_123/original.png" -> "uploads/job_123"
                job_dir = os.path.dirname(s3_key)
                
                # 3. Download image from S3 to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                    temp_image_path = temp_img.name
                download_from_s3(s3_bucket, s3_key, temp_image_path)
                
                # 4. Process the image using the model API
                csv_content = call_model_api(temp_image_path)
                
                # 5. Save the CSV content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
                    temp_csv_path = temp_csv.name
                    temp_csv.write(csv_content)
                
                # 6. Upload the generated CSV to S3
                csv_key = f"{job_dir}/results.csv"
                upload_to_s3(s3_bucket, temp_csv_path, csv_key)
                
                # 7. add inferenced result to db
                with SessionLocal() as session:
                    try:
                        stmt = (
                            update(Sample)
                            .where(Sample.job_id == job_id)
                            .values(inference_status='completed', s3_inference_key = csv_key)
                        )

                        result = session.execute(stmt)
                        # result.rowcount -> number of rows updated (DB-dependent; some backends may return -1)
                        updated = result.rowcount

                        if updated == 0:
                            # no row matched — handle accordingly
                            print("No row updated (id=1 not found).")
                        else:
                            print(f"{updated} row(s) updated.")
                            log.info("Successfully updated the database with results.csv")

                        session.commit()
                    except SQLAlchemyError as e:
                        # rollback and handle/log
                        session.rollback()
                        print("DB error during update:", type(e).__name__, str(e))
                        # decide whether to re-raise or handle gracefully
                        raise


                # 8. Delete the SQS message after successful processing and upload
                sqs.delete_message(QueueUrl=qurl, ReceiptHandle=receipt_handle)
                log.info("Successfully processed and deleted message from queue.")

            except Exception:
                # Log any unexpected errors and do NOT delete the message, allowing SQS to retry
                log.exception("Processing failed for message with handle: %s", receipt_handle)
            finally:
                # 9. Cleanup temporary files, regardless of success or failure
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    log.info("Deleted temporary image file.")
                if temp_csv_path and os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
                    log.info("Deleted temporary CSV file.")

if __name__ == "__main__":
    log.info("Worker started, polling SQS...")
    try:
        poll_loop()
    except KeyboardInterrupt:
        log.info("Worker interrupted, exiting")
