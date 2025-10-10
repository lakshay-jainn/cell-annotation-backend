# Cell Annotation Backend API

A comprehensive Flask-based backend system for medical cell annotation with automated lymphocyte detection, queue-based processing, and administrative management.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CELL ANNOTATION SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React)  │  Backend (Flask)  │  Storage & Queue      │
│  ┌───────────────┐ │ ┌───────────────┐ │ ┌───────────────────┐ │
│  │ Pulmonologist │ │ │   Admin Panel │ │ │  AWS S3/LocalStack│ │
│  │ Dashboard     │ │ │   (Flask-Admin)│ │ │  Image Storage    │ │
│  ├───────────────┤ │ ├───────────────┤ │ ├───────────────────┤ │
│  │ Pathologist  │ │ │   API Routes  │ │ │  SQS Message Queue│ │
│  │ Annotation   │ │ │   /api/v1/*   │ │ │  Job Processing   │ │
│  ├───────────────┤ │ ├───────────────┤ │ ├───────────────────┤ │
│  │ Admin Login  │ │ │  Background   │ │ │  PostgreSQL DB    │ │
│  │ Interface    │ │ │  Processors   │ │ │  Data Persistence │ │
│  └───────────────┘ │ └───────────────┘ │ └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
myapp/
├── 📄 server.py                 # Main Flask application & configuration
├── 📄 api_prefix.py             # API blueprint registration (v1 endpoints)
├── 📄 admin_views.py            # Flask-Admin views & processing interface
├── 📄 background_processor.py   # Subprocess-based job execution
├── 📄 sqs_processor.py          # SQS queue management & message handling
├── 📄 worker_manual.py          # Manual processing logic
├── 📄 process_monitor.py        # Background process monitoring
├── 📄 requirements.txt          # Python dependencies
├── 📄 docker-compose.yml        # Container orchestration
├── 📄 Dockerfile               # Application containerization
│
├── 📂 db/
│   └── 📄 models.py             # SQLAlchemy ORM models & database schema
│
├── 📂 routes/                   # API endpoint definitions
│   ├── 📄 annot_route.py        # Cell & patient annotation endpoints
│   ├── 📄 cells_route.py        # Automated cell detection & analysis
│   ├── 📄 dynamic_cells_route.py# Dynamic cell detection algorithms
│   ├── 📄 health_route.py       # System health monitoring
│   ├── 📄 login_route.py        # Authentication & user management
│   ├── 📄 patient_route.py      # Patient data management
│   ├── 📄 samples_route.py      # Sample upload & management
│   └── 📄 upload_img_route.py   # Image upload handling
│
├── 📂 utilities/                # Shared utility modules
│   ├── 📄 auth_utility.py       # JWT authentication & password hashing
│   ├── 📄 aws_utility.py        # S3 storage & presigned URL generation
│   ├── 📄 enqueue_utility.py    # SQS message queuing
│   └── 📄 logging_utility.py    # Activity logging & audit trails
│
└── 📂 templates/                # Server-side templates
    ├── 📄 admin_login.html      # Admin authentication page
    └── 📂 model_processing/
        └── 📄 index.html        # Processing status dashboard
```

## 🗄️ Database Schema

### Core Models

#### **User** - Authentication & Role Management

```sql
- id (Primary Key)
- name, email, password (hashed)
- role (ADMIN, UPLOADER, ANNONATOR)
- hospital, location
- created_at, updated_at
```

#### **Patient** - Patient Information

```sql
- patient_id (Primary Key)
- name, age, gender, contact
- uploader_id (Foreign Key → User)
- created_at, updated_at
```

#### **Sample** - Medical Image Samples

```sql
- job_id (Primary Key, UUID)
- patient_id (Foreign Key → Patient)
- uploader_id (Foreign Key → User)
- original_filename
- s3_key (S3 storage path)
- inference_status (pending, processing, completed, failed)
- created_at, updated_at
```

#### **SampleAnnotation** - Cell-Level Annotations

```sql
- id (Primary Key)
- sample_id (Foreign Key → Sample, CASCADE)
- user_id (Foreign Key → User)
- cell_coordinates (JSON: {x, y, width, height})
- cell_type (WBC-Lymphocyte, WBC-Others, RBC, etc.)
- comments
- created_at, updated_at
```

#### **PatientAnnotation** - Patient-Level Diagnoses

```sql
- id (Primary Key)
- patient_id (Foreign Key → Patient, CASCADE)
- user_id (Foreign Key → User)
- adequacy (adequate, inadequate)
- diagnosis (JSON array of diagnosis codes)
- comments
- created_at, updated_at
```

#### **UserActivityLog** - Audit Trail

```sql
- id (Primary Key)
- user_id (Foreign Key → User)
- activity_type, description
- metadata (JSON)
- created_at
```

## 🚀 API Endpoints

### Authentication (`/api/v1/`)

- `POST /register` - User registration with reCAPTCHA
- `POST /login` - JWT-based authentication
- `POST /log-activity` - Activity logging

### Sample Management

- `POST /upload_img` - Image upload to S3 with SQS queuing
- `GET /samples` - List user samples with pagination
- `GET /samples/{sample_id}` - Sample details with metadata

### Cell Detection & Annotation

- `POST /cells/detect` - Automated lymphocyte detection
- `GET /cells/dynamic-detect` - Real-time cell detection
- `POST /annotate` - Submit cell annotations
- `POST /annotate/patient` - Submit patient-level diagnosis

### Patient Data

- `GET /patients` - List patients with sample counts
- `GET /patient/{patient_id}/samples` - Patient's samples

### System Health

- `GET /health` - API health check
- `GET /health/background-processes` - Background job status

## 🔧 Admin Interface (`/admin/`)

### Flask-Admin Dashboard

- **Users**: User management, role assignment, password resets
- **Patients**: Patient data overview and management
- **Samples**: Sample upload status and metadata
- **Sample Annotations**: Cell annotation review and editing
- **Patient Annotations**: Diagnosis review and validation
- **Activity Logs**: System audit trail and user activity

### Model Processing Interface (`/admin/modelprocessing/`)

- **Real-time Status**: Live sample processing dashboard
- **Queue Management**: SQS message processing controls
- **AJAX Updates**: Efficient status polling (every 5 seconds)
- **Batch Processing**: Process multiple samples simultaneously
- **Stuck Sample Cleanup**: Reset samples stuck in processing state

### Key Admin Endpoints

- `GET /admin/modelprocessing/` - Processing dashboard
- `POST /admin/modelprocessing/process-sqs` - Trigger SQS processing
- `GET /admin/modelprocessing/status-api` - AJAX status updates
- `POST /admin/cleanup-stuck-samples` - Reset stuck processing jobs

## ⚙️ Background Processing

### Queue-Based Architecture

1. **Upload Pipeline**: Images uploaded → S3 storage → SQS message queued
2. **Processing Pipeline**: SQS messages → Background processor → Model inference
3. **Status Updates**: Database status tracking → Real-time dashboard updates

### Processing Components

#### **sqs_processor.py** - Message Queue Management

- SQS client configuration and queue management
- Batch message processing (process all available messages)
- Automatic message deletion after successful processing
- Queue status monitoring and metrics

#### **background_processor.py** - Job Execution

- Subprocess-based processing for isolation
- Sample status lifecycle management
- Error handling and recovery
- Integration with external model APIs

#### **worker_manual.py** - Processing Logic

- Manual sample processing workflows
- Model API integration (Hugging Face, custom endpoints)
- Result parsing and database updates
- S3 result storage and presigned URL generation

### Processing States

```
pending → processing → completed ✅
                   ↘ failed ❌
```

## 🐳 Deployment & Configuration

### Docker Services

```yaml
# LocalStack - AWS service emulation (development)
localstack:
  - SQS message queuing
  - S3 object storage
  - Health checks and service discovery

# Flask Web Application
web:
  - Gunicorn WSGI server (2 workers)
  - Environment-based configuration
  - Volume mounting for development
  - Network isolation and service linking
```

### Environment Variables

```bash
# Database
SQL_URI=postgresql://user:pass@host:port/db

# AWS Services
AWS_REGION=us-east-1
AWS_ENDPOINT_URL=http://localstack:4566  # LocalStack
S3_BUCKET_NAME=cell-annotation-bucket
SQS_QUEUE_URL=http://localstack:4566/queue/processing

# Authentication
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRY=3600
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=secure-password

# External Services
MODEL_API_URL=https://api-inference.huggingface.co/models/...
HF_API_KEY=your-huggingface-api-key
RECAPTCHA_SECRET=your-recaptcha-secret
```

## 🧪 Key Features

### 1. **Automated Cell Detection**

- OpenCV-based lymphocyte detection algorithms
- Configurable detection parameters (downscale, thresholds)
- Real-time detection with dynamic parameters
- Integration with manual annotation workflows

### 2. **Multi-User Authentication**

- Role-based access control (Admin, Uploader, Annotator)
- JWT-based stateless authentication
- Password hashing with bcrypt
- Activity logging and audit trails

### 3. **Scalable Processing**

- Queue-based background processing
- Subprocess isolation for job safety
- Batch processing capabilities
- Real-time status monitoring

### 4. **Medical Data Management**

- HIPAA-conscious data handling
- Structured annotation schema
- Patient-sample relationship tracking
- Comprehensive audit logging

### 5. **Administrative Controls**

- Real-time processing dashboard
- Queue management and monitoring
- Sample lifecycle management
- System health monitoring

## 🚦 Getting Started

### 1. **Installation**

```bash
cd d:\cell_annotation_BE\myapp
pip install -r requirements.txt
```

### 2. **Environment Setup**

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. **Database Initialization**

```bash
python -c "from server import create_app; from db.models import db; app = create_app(); app.app_context().push(); db.create_all()"
```

### 4. **Run Development Server**

```bash
python server.py
```

### 5. **Run with Docker**

```bash
docker-compose up --build
```

## 📊 System Monitoring

### Health Endpoints

- `/health/background-processes` - Background job status
- `/admin/modelprocessing/status-api` - Processing queue status
- Built-in Flask-Admin monitoring dashboard

### Logging & Debugging

- Activity logging for all user actions
- Processing job status tracking
- Error logging and exception handling
- Real-time status updates via AJAX

## 🔒 Security Features

- JWT-based authentication with configurable expiry
- Role-based access control across all endpoints
- Password hashing with bcrypt (salt rounds: 12)
- reCAPTCHA integration for registration
- Session management with server-side storage
- CORS configuration for frontend integration
- Rate limiting for API endpoints

---

## 🏥 Medical Workflow Integration

This system is designed specifically for medical cell annotation workflows:

1. **Pulmonologists** upload BAL (Bronchoalveolar Lavage) samples
2. **Automated processing** detects lymphocytes and other cell types
3. **Pathologists** review and annotate detected cells
4. **System generates** comprehensive reports and analysis
5. **Administrators** monitor processing and manage system health

The backend provides a robust, scalable foundation for medical image analysis with enterprise-grade features for healthcare environments.
