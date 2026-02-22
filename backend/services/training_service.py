"""TrainingService: runs trainer in ThreadPoolExecutor, broadcasts progress via WebSocket."""
import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import WebSocket

from config import settings
from db.database import SessionLocal
from db.crud import (
    create_training_run, update_training_run, add_training_epoch, get_training_run
)


class WebSocketManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class TrainingService:
    """Singleton service that manages training runs."""

    _instance: Optional["TrainingService"] = None

    def __init__(self):
        self.ws_manager = WebSocketManager()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_run_id: Optional[str] = None
        self._stop_flag: list[bool] = [False]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._status: str = "idle"

    @classmethod
    def get_instance(cls) -> "TrainingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _progress_callback(self, data: dict):
        """Called from trainer thread — schedules WS broadcast on event loop."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.ws_manager.broadcast(data), self._loop
        )

        # Persist epoch data to DB
        if data.get("type") == "epoch_end" and self._current_run_id:
            db = SessionLocal()
            try:
                add_training_epoch(
                    db,
                    run_id=self._current_run_id,
                    epoch=data["epoch"],
                    train_loss=data.get("train_loss"),
                    val_loss=data.get("val_loss"),
                    train_acc=data.get("train_acc"),
                    val_acc=data.get("val_acc"),
                )
            finally:
                db.close()

    def _run_training(self, run_id: str, config: dict):
        """Blocking training function — runs in executor thread."""
        from core.model.trainer import Trainer
        from core.dataset.brain_tumor import make_dataloaders

        db = SessionLocal()
        try:
            update_training_run(db, run_id, status="running")
        finally:
            db.close()

        self._status = "running"

        try:
            # Load current best model (or fresh if none)
            from services.model_manager import get_model_manager
            mm = get_model_manager()
            mm.ensure_loaded()

            # Apply any kernel deletions
            db = SessionLocal()
            try:
                from db.crud import get_kernels
                deleted_kernels, _ = get_kernels(db, include_deleted=True)
                deleted_ids = [k.id for k in deleted_kernels if k.is_deleted]
            finally:
                db.close()

            if deleted_ids:
                mm.model.apply_kernel_deletions(deleted_ids)

            dataloaders = make_dataloaders(
                batch_size=config.get("batch_size", settings.batch_size),
            )

            trainer = Trainer(
                model=mm.model,
                dataloaders=dataloaders,
                db_session_factory=SessionLocal,
                lr=config.get("learning_rate", settings.learning_rate),
                weight_decay=config.get("weight_decay", settings.weight_decay),
                num_epochs=config.get("num_epochs", settings.num_epochs),
                spatial_loss_weight=config.get(
                    "spatial_loss_weight",
                    config.get("annotation_weight", settings.spatial_loss_weight),
                ),
                run_id=run_id,
                stop_flag=self._stop_flag,
                progress_callback=self._progress_callback,
            )

            results = trainer.train()

            model_version = f"v_{run_id[:8]}"
            db = SessionLocal()
            try:
                update_training_run(
                    db, run_id,
                    status="completed",
                    end_time=datetime.now(timezone.utc),
                    final_train_loss=results["final"].get("train_loss"),
                    final_val_loss=results["final"].get("val_loss"),
                    final_train_acc=results["final"].get("train_acc"),
                    final_val_acc=results["final"].get("val_acc"),
                    model_version=model_version,
                )
            finally:
                db.close()

            # Reload model with new weights
            mm.reload(str(settings.best_model_path))
            self._status = "idle"

        except InterruptedError:
            self._status = "idle"
            db = SessionLocal()
            try:
                update_training_run(db, run_id, status="stopped", end_time=datetime.now(timezone.utc))
            finally:
                db.close()
            self._progress_callback({"type": "error", "message": "Training stopped by user.", "run_id": run_id})

        except Exception as e:
            self._status = "error"
            db = SessionLocal()
            try:
                update_training_run(
                    db, run_id,
                    status="error",
                    end_time=datetime.now(timezone.utc),
                    error_message=str(e),
                )
            finally:
                db.close()
            self._progress_callback({"type": "error", "message": str(e), "run_id": run_id})
            raise

        finally:
            self._current_run_id = None

    async def start_training(self, config: dict) -> str:
        if self._status == "running":
            raise RuntimeError("Training already in progress.")

        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        self._stop_flag = [False]
        self._loop = asyncio.get_event_loop()

        db = SessionLocal()
        try:
            create_training_run(db, run_id, config)
        finally:
            db.close()

        self._executor.submit(self._run_training, run_id, config)
        return run_id

    def stop_training(self):
        self._stop_flag[0] = True

    def get_status(self) -> dict:
        return {
            "status": self._status,
            "run_id": self._current_run_id,
        }


def get_training_service() -> TrainingService:
    return TrainingService.get_instance()
