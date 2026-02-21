"""Standalone initial training script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.database import init_db, SessionLocal
from core.model.resnet import CancerScopeModel
from core.model.trainer import Trainer
from core.dataset.breakhis import make_dataloaders


def main():
    if not (settings.splits_dir / "train.json").exists():
        print("ERROR: Split files not found. Run 'make index' first.")
        sys.exit(1)

    print(f"Device: {settings.device}")
    print(f"Loading pretrained ResNet-50...")

    init_db()
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model = CancerScopeModel(pretrained=True)
    model = model.to(settings.device)

    print("Building dataloaders...")
    dataloaders = make_dataloaders()
    print(f"  Train: {len(dataloaders['train'].dataset)} samples")
    print(f"  Val: {len(dataloaders['val'].dataset)} samples")

    def progress(data):
        t = data.get("type", "")
        if t == "epoch_end":
            print(
                f"  Epoch {data['epoch']} — "
                f"train_loss={data['train_loss']:.4f} "
                f"val_loss={data['val_loss']:.4f} "
                f"train_acc={data['train_acc']:.3f} "
                f"val_acc={data['val_acc']:.3f}"
            )
        elif t == "batch":
            print(f"    batch {data['batch']}/{data['total_batches']}", end="\r")
        elif t in ("completed", "error"):
            print(f"\n{data['message']}")

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        db_session_factory=SessionLocal,
        num_epochs=settings.num_epochs,
        run_id="initial",
        progress_callback=progress,
    )

    print(f"\nStarting training for {settings.num_epochs} epochs...")
    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Best val_loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")
    print(f"  Model saved to: {settings.best_model_path}")


if __name__ == "__main__":
    main()
