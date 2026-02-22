"""Trainer: annotation-weighted loss loop with WebSocket progress callbacks."""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import settings
from core.model.cnn import BrainTumorCNN
from core.feedback.weighted_loss import AnnotationWeightedLoss, build_annotation_mask, build_gradcam_focus_mask


ProgressCallback = Callable[[dict], None]


class Trainer:
    def __init__(
        self,
        model: BrainTumorCNN,
        dataloaders: dict,
        db_session_factory,
        device: str | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        num_epochs: int | None = None,
        spatial_loss_weight: float | None = None,
        run_id: str | None = None,
        stop_flag: list[bool] | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.db_session_factory = db_session_factory
        self.device = device or settings.device
        self.lr = lr or settings.learning_rate
        self.weight_decay = weight_decay or settings.weight_decay
        self.num_epochs = num_epochs or settings.num_epochs
        self.spatial_loss_weight = (
            spatial_loss_weight
            if spatial_loss_weight is not None
            else settings.spatial_loss_weight
        )
        self.run_id = run_id
        self.stop_flag = stop_flag or [False]
        self.progress_callback = progress_callback or (lambda d: None)

        self.criterion = AnnotationWeightedLoss(spatial_weight=self.spatial_loss_weight)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

    def _emit(self, msg: dict):
        if self.run_id:
            msg.setdefault("run_id", self.run_id)
        self.progress_callback(msg)

    def _load_annotations(self) -> dict[str, list]:
        from db.crud import get_all_active_annotations
        db = self.db_session_factory()
        try:
            return get_all_active_annotations(db)
        finally:
            db.close()

    def _build_weight_maps(
        self, image_ids: list[str], annotations_map: dict
    ) -> tuple[list[Optional[torch.Tensor]], list[Optional[torch.Tensor]]]:
        regular_masks, focus_masks = [], []
        for iid in image_ids:
            anns = annotations_map.get(iid, [])
            regular_masks.append(build_annotation_mask(anns, 256, 256) if anns else None)
            focus_masks.append(build_gradcam_focus_mask(anns, 256, 256) if anns else None)
        return regular_masks, focus_masks

    def _run_epoch(self, split: str, epoch: int, annotations_map: dict, total_epochs: int) -> dict[str, float]:
        is_train = split == "train"
        self.model.train(is_train)
        loader = self.dataloaders[split]
        total_batches = len(loader)

        running_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.set_grad_enabled(is_train):
            for batch_idx, batch in enumerate(loader):
                if self.stop_flag[0]:
                    raise InterruptedError("Training stopped by user.")

                images, labels, image_ids = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                annotation_masks, focus_masks = self._build_weight_maps(list(image_ids), annotations_map)

                loss, metrics = self.criterion(
                    outputs, labels,
                    annotation_masks=annotation_masks,
                    focus_masks=focus_masks,
                )

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += metrics["total_loss"] * images.size(0)
                preds = outputs["logits"].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_samples += images.size(0)

                if is_train and (batch_idx + 1) % 10 == 0:
                    self._emit({
                        "type": "batch",
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "message": f"Epoch {epoch}/{total_epochs} batch {batch_idx+1}/{total_batches}",
                    })

        return {
            "loss": running_loss / max(total_samples, 1),
            "accuracy": correct / max(total_samples, 1),
        }

    def train(self) -> dict:
        annotations_map = self._load_annotations()
        best_val_loss = float("inf")
        best_epoch = 0
        history = []

        for epoch in range(1, self.num_epochs + 1):
            if self.stop_flag[0]:
                break

            epoch_start = time.time()
            self._emit({"type": "epoch_start", "epoch": epoch,
                        "message": f"Starting epoch {epoch}/{self.num_epochs}"})

            train_metrics = self._run_epoch("train", epoch, annotations_map, self.num_epochs)

            if "val" in self.dataloaders:
                val_metrics = self._run_epoch("val", epoch, annotations_map, self.num_epochs)
            else:
                val_metrics = {"loss": 0.0, "accuracy": 0.0}

            self.scheduler.step()
            duration = time.time() - epoch_start

            epoch_data = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc":  train_metrics["accuracy"],
                "val_loss":   val_metrics["loss"],
                "val_acc":    val_metrics["accuracy"],
                "duration_sec": duration,
            }
            history.append(epoch_data)

            self._emit({
                "type": "epoch_end",
                "epoch": epoch,
                "batch": None,
                "total_batches": None,
                **{k: epoch_data[k] for k in ("train_loss", "val_loss", "train_acc", "val_acc")},
                "message": f"Epoch {epoch}/{self.num_epochs} complete",
            })

            checkpoint_path = settings.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(), **epoch_data},
                       checkpoint_path)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), **epoch_data},
                            settings.best_model_path)

        final = history[-1] if history else {}
        self._emit({
            "type": "completed",
            "epoch": self.num_epochs,
            "message": f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}",
            **{k: final.get(k) for k in ("train_loss", "val_loss", "train_acc", "val_acc")},
        })

        return {"best_val_loss": best_val_loss, "best_epoch": best_epoch, "history": history, "final": final}
