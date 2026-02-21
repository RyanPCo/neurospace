"""Pre-compute kernel filter images and importance scores."""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.database import init_db, SessionLocal
from db.crud import upsert_kernel
from core.model.resnet import load_model
from core.model.kernel_analyzer import KernelAnalyzer, ANALYZED_LAYERS
from core.dataset.breakhis import make_dataloaders


def main():
    if not settings.best_model_path.exists():
        print("ERROR: No model found. Run 'make train' first.")
        sys.exit(1)

    if not (settings.splits_dir / "val.json").exists():
        print("ERROR: Val split not found. Run 'make index' first.")
        sys.exit(1)

    print(f"Loading model from {settings.best_model_path}...")
    model = load_model(str(settings.best_model_path), settings.device)
    model.eval()

    settings.kernel_cache_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    db = SessionLocal()

    analyzer = KernelAnalyzer(model, settings.device)

    print("Building validation dataloader for importance scoring...")
    loaders = make_dataloaders(batch_size=32, num_workers=0)
    val_loader = loaders.get("val")

    for layer_name in ANALYZED_LAYERS:
        print(f"\nProcessing layer: {layer_name}")

        # Determine number of filters for this layer
        module_path = ANALYZED_LAYERS[layer_name]
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                break

        if module is None or not hasattr(module, "weight"):
            print(f"  Skipping {layer_name} (module not found)")
            continue

        n_filters = module.weight.shape[0]
        print(f"  {n_filters} filters")

        # Compute importance scores
        if val_loader:
            print("  Computing importance scores...")
            importance_scores = analyzer.compute_importance_scores(val_loader, layer_name)
        else:
            importance_scores = {}

        # Generate filter visualizations and save to DB
        for filter_idx in range(n_filters):
            kernel_id = f"{layer_name}_{filter_idx}"
            cache_path = settings.kernel_cache_dir / f"{kernel_id}.png"

            if not cache_path.exists():
                try:
                    png_bytes = analyzer.visualize_filter(layer_name, filter_idx)
                    cache_path.write_bytes(png_bytes)
                except Exception as e:
                    print(f"    Warning: filter {filter_idx} viz failed: {e}")
                    continue

            importance = importance_scores.get(filter_idx, 0.0)
            upsert_kernel(
                db,
                kernel_id=kernel_id,
                layer_name=layer_name,
                filter_index=filter_idx,
                importance_score=importance,
                last_scored_at=datetime.now(timezone.utc),
            )

            if (filter_idx + 1) % 64 == 0:
                print(f"    {filter_idx + 1}/{n_filters} done...")

        db.commit()
        print(f"  Layer {layer_name} complete.")

    db.close()
    print(f"\nKernel pre-computation complete!")
    print(f"  Cache: {settings.kernel_cache_dir}")


if __name__ == "__main__":
    main()
