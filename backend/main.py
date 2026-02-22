"""CancerScope FastAPI application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.database import init_db
from api.routes.images import router as images_router
from api.routes.kernels import router as kernels_router
from api.routes.annotations import router as annotations_router
from api.routes.predictions import router as predictions_router
from api.routes.training import router as training_router, ws_router
from api.routes.volumes import router as volumes_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    init_db()
    # Ensure required directories exist
    settings.kernel_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    # Load model if checkpoint exists
    from services.model_manager import get_model_manager
    mm = get_model_manager()
    try:
        mm.load()
        print(f"Model loaded: {mm.model_version} on {mm.device}")
    except Exception as e:
        print(f"Model not loaded at startup (run 'make train' first): {e}")
    yield


app = FastAPI(
    title="CancerScope",
    description="Brain tumor MRI AI workbench",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(images_router)
app.include_router(kernels_router)
app.include_router(annotations_router)
app.include_router(predictions_router)
app.include_router(training_router)
app.include_router(ws_router)
app.include_router(volumes_router)


@app.get("/api/health")
def health():
    from services.model_manager import get_model_manager
    mm = get_model_manager()
    return {
        "status": "ok",
        "model_loaded": mm.is_loaded(),
        "model_version": mm.model_version,
        "device": mm.device,
    }
