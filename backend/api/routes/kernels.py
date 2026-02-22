"""Kernel endpoints: list, visualize, activations, update, delete."""
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session

from db.database import get_db
from db import crud
from schemas.kernels import (
    KernelSummary, KernelUpdate, KernelActivationsResponse, PaginatedKernels, BatchKernelUpdate
)
from services.model_manager import get_model_manager
from config import settings
from core.model.kernel_analyzer import KernelAnalyzer, ANALYZED_LAYERS

router = APIRouter(prefix="/api/kernels", tags=["kernels"])


@router.get("", response_model=PaginatedKernels)
def list_kernels(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    layer_name: Optional[str] = Query(None),
    assigned_class: Optional[str] = Query(None),
    sort_by: str = Query("importance"),
    db: Session = Depends(get_db),
):
    skip = (page - 1) * page_size
    items, total = crud.get_kernels(
        db, layer_name=layer_name, assigned_class=assigned_class,
        sort_by=sort_by, skip=skip, limit=page_size
    )
    return PaginatedKernels(
        items=[KernelSummary.model_validate(k) for k in items],
        total=total, page=page, page_size=page_size,
    )


@router.get("/{kernel_id}/image")
def get_kernel_image(kernel_id: str, db: Session = Depends(get_db)):
    """Return filter visualization PNG from cache or compute on-the-fly."""
    # Check cache first
    cache_path = settings.kernel_cache_dir / f"{kernel_id}.png"
    if cache_path.exists():
        return StreamingResponse(
            io.BytesIO(cache_path.read_bytes()),
            media_type="image/png",
        )

    # Compute on-the-fly
    kernel = crud.get_kernel(db, kernel_id)
    if not kernel:
        raise HTTPException(404, "Kernel not found")

    mm = get_model_manager()
    if not mm.is_loaded():
        raise HTTPException(503, "Model not loaded")

    analyzer = KernelAnalyzer(mm.model)
    try:
        png_bytes = analyzer.visualize_filter(kernel.layer_name, kernel.filter_index)
    except Exception as e:
        raise HTTPException(500, f"Filter visualization failed: {e}")

    # Cache it
    settings.kernel_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(png_bytes)

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/{kernel_id}/activations", response_model=KernelActivationsResponse)
def get_kernel_activations(kernel_id: str, db: Session = Depends(get_db)):
    """Return top-9 activating images for a kernel."""
    kernel = crud.get_kernel(db, kernel_id)
    if not kernel:
        raise HTTPException(404, "Kernel not found")

    mm = get_model_manager()
    if not mm.is_loaded():
        raise HTTPException(503, "Model not loaded")

    from core.dataset.brain_tumor import make_dataloaders
    loaders = make_dataloaders(batch_size=16, num_workers=0)
    if "val" not in loaders:
        raise HTTPException(503, "Validation dataset not available. Run 'make index' first.")

    analyzer = KernelAnalyzer(mm.model)
    try:
        top_images = analyzer.get_top_activating_images(
            loaders["val"], kernel.layer_name, kernel.filter_index, top_k=9
        )
    except Exception as e:
        raise HTTPException(500, f"Activation computation failed: {e}")

    return KernelActivationsResponse(
        kernel_id=kernel_id,
        top_images=top_images,
    )


@router.put("/{kernel_id}", response_model=KernelSummary)
def update_kernel(
    kernel_id: str, update: KernelUpdate, db: Session = Depends(get_db)
):
    kernel = crud.get_kernel(db, kernel_id)
    if not kernel:
        raise HTTPException(404, "Kernel not found")
    updates = update.model_dump(exclude_none=True)
    updated = crud.update_kernel(db, kernel_id, **updates)
    return KernelSummary.model_validate(updated)


@router.delete("/{kernel_id}")
def delete_kernel(kernel_id: str, db: Session = Depends(get_db)):
    kernel = crud.get_kernel(db, kernel_id)
    if not kernel:
        raise HTTPException(404, "Kernel not found")
    crud.soft_delete_kernel(db, kernel_id)
    return {"message": f"Kernel {kernel_id} soft-deleted."}


@router.post("/batch")
def batch_update_kernels(update: BatchKernelUpdate, db: Session = Depends(get_db)):
    results = []
    for kid in update.kernel_ids:
        if update.action == "delete":
            crud.soft_delete_kernel(db, kid)
            results.append({"id": kid, "action": "deleted"})
        elif update.action == "reclassify":
            crud.update_kernel(db, kid, assigned_class=update.assigned_class)
            results.append({"id": kid, "action": "reclassified", "class": update.assigned_class})
    return {"results": results}
