"""Trainer: annotation-weighted loss loop with WebSocket progress callbacks."""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import settings
from core.model.resnet import CancerScopeModel, BINARY_CLASSES
from core.feedback.weighted_loss import AnnotationWeightedLoss, build_annotation_mask


ProgressCallback = Callable[[dict], None]


class Trainer:
    """
    Trains a CancerScopeModel with annotation-weighted loss.
    Designed to run in a background thread; progress is reported via callback.
    """

    def __init__(
        self,
        model: CancerScopeModel,
        dataloaders: dict,
        db_session_factory,
        device: str | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        num_epochs: int | None = None,
        run_id: str | None = None,
        stop_flag: list[bool] | None = None,  # [False] — set to [True] to stop
        progress_callback: ProgressCallback | None = None,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.db_session_factory = db_session_factory
        self.device = device or settings.device
        self.lr = lr or settings.learning_rate
        self.weight_decay = weight_decay or settings.weight_decay
        self.num_epochs = num_epochs or settings.num_epochs
        self.run_id = run_id
        self.stop_flag = stop_flag or [False]
        self.progress_callback = progress_callback or (lambda d: None)

        self.criterion = AnnotationWeightedLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

    def _emit(self, msg: dict):
        if self.run_id:
            msg.setdefault("run_id", self.run_id)
        self.progress_callback(msg)

    def _load_annotations(self) -> dict[str, list]:
        """Fetch all active annotations from DB at training start."""
        from db.crud import get_all_active_annotations
        db = self.db_session_factory()
        try:
            return get_all_active_annotations(db)
        finally:
            db.close()

    def _build_weight_maps(
        self, image_ids: list[str], annotations_map: dict
    ) -> list[Optional[torch.Tensor]]:
        masks = []
        for iid in image_ids:
            anns = annotations_map.get(iid, [])
            if not anns:
                masks.append(None)
                continue
            # Use 224x224 (training resolution)
            mask = build_annotation_mask(anns, 224, 224)
            masks.append(mask)
        return masks

    def _run_epoch(
        self, split: str, epoch: int, annotations_map: dict, total_epochs: int
    ) -> dict[str, float]:
        is_train = split == "train"
        self.model.train(is_train)
        loader = self.dataloaders[split]
        total_batches = len(loader)

        running_loss = 0.0
        correct_binary = 0
        total_samples = 0

        binary_head_weight = self.model.binary_head.weight  # (2, 2048)

        with torch.set_grad_enabled(is_train):
            for batch_idx, batch in enumerate(loader):
                if self.stop_flag[0]:
                    raise InterruptedError("Training stopped by user.")

                images, (binary_labels, subtype_labels), image_ids = batch
                images = images.to(self.device)
                binary_labels = binary_labels.to(self.device)
                subtype_labels = subtype_labels.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)

                annotation_masks = self._build_weight_maps(list(image_ids), annotations_map)

                loss, metrics = self.criterion(
                    outputs,
                    binary_labels,
                    subtype_labels,
                    annotation_masks=annotation_masks,
                    binary_head_weight=binary_head_weight,
                )

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += metrics["total_loss"] * images.size(0)
                preds = outputs["binary_logits"].argmax(dim=1)
                correct_binary += (preds == binary_labels).sum().item()
                total_samples += images.size(0)

                # Emit batch progress every 10 batches (train only)
                if is_train and (batch_idx + 1) % 10 == 0:
                    self._emit({
                        "type": "batch",
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "message": f"Epoch {epoch}/{total_epochs} batch {batch_idx+1}/{total_batches}",
                    })

        avg_loss = running_loss / max(total_samples, 1)
        accuracy = correct_binary / max(total_samples, 1)
        return {"loss": avg_loss, "accuracy": accuracy}

    def train(self) -> dict:
        """Run full training loop. Returns final metrics dict."""
        annotations_map = self._load_annotations()

        best_val_loss = float("inf")
        best_epoch = 0
        history = []

        for epoch in range(1, self.num_epochs + 1):
            if self.stop_flag[0]:
                break

            epoch_start = time.time()
            self._emit({
                "type": "epoch_start",
                "epoch": epoch,
                "message": f"Starting epoch {epoch}/{self.num_epochs}",
            })

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
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "duration_sec": duration,
            }
            history.append(epoch_data)

            self._emit({
                "type": "epoch_end",
                "epoch": epoch,
                "batch": None,
                "total_batches": None,
                "train_loss": epoch_data["train_loss"],
                "val_loss": epoch_data["val_loss"],
                "train_acc": epoch_data["train_acc"],
                "val_acc": epoch_data["val_acc"],
                "message": f"Epoch {epoch}/{self.num_epochs} complete",
            })

            # Save checkpoint
            checkpoint_path = settings.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                **epoch_data,
            }, checkpoint_path)

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    **epoch_data,
                }, settings.best_model_path)

        final = history[-1] if history else {}
        self._emit({
            "type": "completed",
            "epoch": self.num_epochs,
            "message": f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}",
            "train_loss": final.get("train_loss"),
            "val_loss": final.get("val_loss"),
            "train_acc": final.get("train_acc"),
            "val_acc": final.get("val_acc"),
        })

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "final": final,
        }
