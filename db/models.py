# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
import enum 
from flask_login import UserMixin
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import exists
from sqlalchemy.orm import declarative_base

db = SQLAlchemy()

# Create a Base class for direct SQLAlchemy usage (for background processes)
Base = declarative_base()

def gen_uuid():
    return str(uuid.uuid4())

class UserRole(enum.Enum):
    ADMIN = "ADMIN"
    UPLOADER = "UPLOADER"
    ANNONATOR = "ANNONATOR"
    
class Patient(db.Model):
    __tablename__ = "patients"
    
    patient_id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_typed_id = db.Column(db.String(100), nullable=False, unique=True, index=True)  # The ID entered by pulmonologist
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship to samples
    samples = db.relationship(
        "Sample",
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    
    # Relationship to sample annotations
    sample_annotations = db.relationship(
        "SampleAnnotation",
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    
    # Relationship to patient annotations - one-to-many but unique per pathologist
    # Each pathologist can have only one patient annotation per patient
    patient_annotations = db.relationship(
        "PatientAnnotation",
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    
    
    def get_patient_annotation_by_pathologist(self, pathologist_user_id):
        """Get the patient annotation for this patient by a specific pathologist."""
        return next((annotation for annotation in self.patient_annotations 
                    if annotation.user_id == pathologist_user_id), None)
    
    def has_patient_annotation_by_pathologist(self, pathologist_user_id):
        """Check if this patient has been annotated by a specific pathologist."""
        return any(annotation.user_id == pathologist_user_id for annotation in self.patient_annotations)
    
    def get_sample_annotations_by_pathologist(self, pathologist_user_id):
        """Get all sample annotations for this patient by a specific pathologist."""
        return [annotation for annotation in self.sample_annotations 
                if annotation.user_id == pathologist_user_id]
    
    def __repr__(self):
        return f"<Patient {self.patient_id}>"
    

class SampleAnnotation(db.Model):
    __tablename__ = "sample_annotations"
    # Unique constraint to ensure one annotation per pathologist per sample
    __table_args__ = (
        db.UniqueConstraint('user_id', 'sample_id', name='uq_user_sample'),
    )

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    sample_id = db.Column(db.String(36), db.ForeignKey("samples.job_id", ondelete="CASCADE"), nullable=False)
    patient_id = db.Column(db.String(36), db.ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    annotated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(32), default="completed", nullable=False)
    s3_annotation_key = db.Column(db.String(1024), nullable=True, unique=False)

    # Sample-level metadata fields
    image_quality = db.Column(db.Boolean, nullable=True)
    cells = db.Column(db.JSON, nullable=True)

    # relationships
    user = db.relationship("User", back_populates="sample_annotations")
    sample = db.relationship("Sample", back_populates="sample_annotations")
    patient = db.relationship("Patient", back_populates="sample_annotations")

    def __repr__(self):
        return f"<SampleAnnotation {self.id} user={self.user_id} sample={self.sample_id} status={self.status}>"
    

class PatientAnnotation(db.Model):
    __tablename__ = "patient_annotations"
    # Unique constraint to ensure one patient annotation per pathologist per patient
    __table_args__ = (
        db.UniqueConstraint('user_id', 'patient_id', name='uq_user_patient_annotation'),
    )

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    patient_id = db.Column(db.String(36), db.ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    annotated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(32), default="completed", nullable=False)

    # Patient-level metadata
    adequacy = db.Column(db.Boolean, nullable=True)
    inadequacy_reason = db.Column(db.String(200), nullable=True)
    provisional_diagnosis = db.Column(db.Boolean, nullable=True)
    provisional_diagnosis_reason = db.Column(db.String(200), nullable=True)
    
    # Overall patient annotation status
    annotation_completed = db.Column(db.Boolean, default=False, nullable=False)

    # relationships
    user = db.relationship("User", back_populates="patient_annotations")
    patient = db.relationship("Patient", back_populates="patient_annotations")

    def __repr__(self):
        return f"<PatientAnnotation {self.id} user={self.user_id} patient={self.patient_id} status={self.status}>"


class User(UserMixin,db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    name = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    role = db.Column(db.Enum(UserRole), default=UserRole.UPLOADER, nullable=False)
    location = db.Column(db.String(50), nullable = False)
    hospital = db.Column(db.String(50), nullable = False)
    password = db.Column(db.String(255), nullable=False)

    # Load sample annotations lazily
    sample_annotations = db.relationship(
        "SampleAnnotation",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    
    # Load patient annotations lazily
    patient_annotations = db.relationship(
        "PatientAnnotation",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    samples = db.relationship(
        "Sample",
        back_populates="uploader",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    def __repr__(self):
        return f"<User {self.id} {self.email}>"

class Sample(db.Model):
    __tablename__ = "samples"


    # Foreign Key to Patient
    patient_id = db.Column(db.String(36), db.ForeignKey("patients.patient_id", ondelete="SET NULL"), nullable=True)
    
    # Foreign Key to User (who uploaded the sample)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    job_id = db.Column(db.String(36), primary_key=True, default=gen_uuid)  # UUID
    original_filename = db.Column(db.String(512), nullable=True)
    s3_object_key = db.Column(db.String(1024), nullable=False, unique=False, index=True)  # index if frequently queried
    s3_inference_key = db.Column(db.String(1024), nullable=True, unique=False, index=True)  # index if frequently queried
    # Metadata fields

    node_station = db.Column(db.String(16), nullable=True)
    needle_size = db.Column(db.String(8), nullable=True)
    sample_type = db.Column(db.String(64), nullable=True)
    microscope = db.Column(db.String(128), nullable=True)
    magnification = db.Column(db.String(16), nullable=True)
    stain = db.Column(db.String(64), nullable=True)
    camera = db.Column(db.String(128), nullable=True)

    # Processing status: 'pending', 'processing', 'completed', 'failed'
    inference_status = db.Column(db.String(16), default='pending', nullable=False)

    # patient relationship
    patient = db.relationship("Patient", back_populates="samples")
    
    # user relationship (who uploaded the sample)
    uploader = db.relationship("User", back_populates="samples", lazy="select")

    # sample annotation relationship
    sample_annotations = db.relationship(
        "SampleAnnotation",
        back_populates="sample",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    @hybrid_property
    def annotation_status(self):
        """Python-side check: True if at least one SampleAnnotation is loaded."""
        return bool(self.sample_annotations)   # works for instance objects

    @annotation_status.expression
    def annotation_status(cls):
        """SQL expression for querying in filter() etc."""
        return exists().where(SampleAnnotation.sample_id == cls.job_id)
    
    def __repr__(self):
        return f"<Sample {self.job_id} ({self.s3_object_key})>"

class UserActivityLog(db.Model):
    __tablename__ = "user_activity_logs"
    
    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True, index=True)
    user_role = db.Column(db.String(20), nullable=True)  # Store role for quick queries
    
    # Action details
    action_type = db.Column(db.String(50), nullable=False, index=True)  # login, logout, button_click, api_call, etc.
    action_details = db.Column(db.String(255), nullable=True)  # Brief description
    status = db.Column(db.String(20), default="success", index=True)  # success, error, info, warning
    
    # Request context
    ip_address = db.Column(db.String(45), nullable=True)  # IPv4/IPv6
    user_agent = db.Column(db.Text, nullable=True)
    
    # Additional data (JSON for flexibility)
    activity_metadata = db.Column(db.JSON, nullable=True)  # Store additional context like button_id, api_endpoint, error_messages, etc.
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = db.relationship("User", backref="activity_logs", lazy="select")
    
    def __repr__(self):
        return f"<UserActivityLog {self.id} user={self.user_id} action={self.action_type} status={self.status}>"
