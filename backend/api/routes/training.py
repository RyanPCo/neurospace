"""Training endpoints: start, stop, status, history, WebSocket."""
import json
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from db.database import get_db
from db import crud
from schemas.training import (
    TrainingConfig, TrainingRunResponse, TrainingStatusResponse,
    StartTrainingResponse, TrainingEpochResponse
)
from services.training_service import get_training_service

router = APIRouter(tags=["training"])
ws_router = APIRouter(tags=["websocket"])


@router.post("/api/training/start", response_model=StartTrainingResponse)
async def start_training(config: TrainingConfig, db: Session = Depends(get_db)):
    service = get_training_service()
    if service.get_status()["status"] == "running":
        raise HTTPException(409, "Training already in progress")
    try:
        run_id = await service.start_training(config.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return StartTrainingResponse(run_id=run_id, message="Training started")


@router.post("/api/training/stop")
def stop_training():
    service = get_training_service()
    service.stop_training()
    return {"message": "Stop signal sent"}


@router.get("/api/training/status", response_model=TrainingStatusResponse)
def get_status(db: Session = Depends(get_db)):
    service = get_training_service()
    svc_status = service.get_status()
    run = None
    metrics = None

    if svc_status["run_id"]:
        run = crud.get_training_run(db, svc_status["run_id"])
        epochs = crud.get_training_epochs(db, svc_status["run_id"])
        if epochs:
            last = epochs[-1]
            metrics = {
                "epoch": last.epoch,
                "train_loss": last.train_loss,
                "val_loss": last.val_loss,
                "train_acc": last.train_acc,
                "val_acc": last.val_acc,
            }

    return TrainingStatusResponse(
        run_id=svc_status["run_id"],
        status=svc_status["status"],
        latest_metrics=metrics,
    )


@router.get("/api/training/history", response_model=list[TrainingRunResponse])
def get_history(db: Session = Depends(get_db)):
    runs = crud.get_all_training_runs(db)
    return [TrainingRunResponse.model_validate(r) for r in runs]


@router.get("/api/training/{run_id}/epochs", response_model=list[TrainingEpochResponse])
def get_run_epochs(run_id: str, db: Session = Depends(get_db)):
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(404, "Training run not found")
    epochs = crud.get_training_epochs(db, run_id)
    return [TrainingEpochResponse.model_validate(e) for e in epochs]


@ws_router.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    service = get_training_service()
    await service.ws_manager.connect(websocket)
    try:
        # Send current status immediately
        status = service.get_status()
        await websocket.send_json({"type": "status", **status})
        # Keep alive — wait for disconnect
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        service.ws_manager.disconnect(websocket)
