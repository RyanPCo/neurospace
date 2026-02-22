from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TrainingConfig(BaseModel):
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    spatial_loss_weight: float = 0.3
    # Backward-compat key used by older frontend builds.
    annotation_weight: Optional[float] = None


class TrainingRunResponse(BaseModel):
    id: str
    status: str
    config_json: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_train_acc: Optional[float] = None
    final_val_acc: Optional[float] = None
    model_version: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


class TrainingEpochResponse(BaseModel):
    id: int
    run_id: str
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    duration_sec: Optional[float] = None

    model_config = {"from_attributes": True}


class TrainingStatusResponse(BaseModel):
    run_id: Optional[str] = None
    status: str  # "idle", "running", "completed", "error"
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    latest_metrics: Optional[dict] = None


class StartTrainingResponse(BaseModel):
    run_id: str
    message: str
