from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from db.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True)  # SHA256 of path
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False, unique=True)
    # Legacy metadata fields retained for DB backward compatibility.
    magnification = Column(String, nullable=False)
    subtype = Column(String, nullable=False)
    ground_truth = Column(String, nullable=False)  # e.g. glioma/meningioma/notumor/pituitary
    split = Column(String, nullable=False)  # train, val, test
    width = Column(Integer)
    height = Column(Integer)

    predictions = relationship("Prediction", back_populates="image", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("image_id", "model_version"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String, ForeignKey("images.id"), nullable=False)
    model_version = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)  # e.g. glioma/meningioma/notumor/pituitary
    confidence = Column(Float, nullable=False)
    class_probs_json = Column(Text)  # JSON string of class probability distribution
    subtype_predicted = Column(String)
    subtype_probs_json = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    image = relationship("Image", back_populates="predictions")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String, ForeignKey("images.id"), nullable=False)
    label_class = Column(String, nullable=False)  # e.g. tumor class or "gradcam_focus"
    geometry_type = Column(String, nullable=False)  # "polygon" or "brush"
    geometry_json = Column(Text, nullable=False)  # normalized [0,1] coords
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    image = relationship("Image", back_populates="annotations")


class Kernel(Base):
    __tablename__ = "kernels"

    id = Column(String, primary_key=True)  # "{layer_name}_{filter_index}"
    layer_name = Column(String, nullable=False)
    filter_index = Column(Integer, nullable=False)
    importance_score = Column(Float, default=0.0)
    assigned_class = Column(String)  # doctor-assigned class label
    is_deleted = Column(Boolean, default=False)
    doctor_notes = Column(Text)
    last_scored_at = Column(DateTime)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True)  # UUID
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, error, stopped
    config_json = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    final_train_loss = Column(Float)
    final_val_loss = Column(Float)
    final_train_acc = Column(Float)
    final_val_acc = Column(Float)
    model_version = Column(String)
    error_message = Column(Text)

    epochs = relationship("TrainingEpoch", back_populates="run", cascade="all, delete-orphan")


class TrainingEpoch(Base):
    __tablename__ = "training_epochs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float)
    val_loss = Column(Float)
    train_acc = Column(Float)
    val_acc = Column(Float)
    duration_sec = Column(Float)

    run = relationship("TrainingRun", back_populates="epochs")
