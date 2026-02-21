from db.database import get_db
from services.model_manager import get_model_manager, ModelManager
from services.training_service import get_training_service, TrainingService
from sqlalchemy.orm import Session
from fastapi import Depends


def get_manager() -> ModelManager:
    return get_model_manager()


def get_trainer() -> TrainingService:
    return get_training_service()
