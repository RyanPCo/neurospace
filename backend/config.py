from pydantic_settings import BaseSettings
from pathlib import Path
import torch


TUMOR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


class Settings(BaseSettings):
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw" / "brain_tumor_mri"
    processed_dir: Path = data_dir / "processed"
    splits_dir: Path = data_dir / "splits"
    models_dir: Path = base_dir / "models"
    checkpoints_dir: Path = models_dir / "checkpoints"
    kernel_cache_dir: Path = models_dir / "kernel_cache"
    best_model_path: Path = models_dir / "best_model.pt"
    db_url: str = f"sqlite:///{base_dir}/cancerscope.db"

    # Training defaults
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 30
    num_workers: int = 4
    spatial_loss_weight: float = 0.3

    # Model
    num_classes: int = 4
    image_size: int = 256
    dropout_rate: float = 0.5

    # Device
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    page_size: int = 50

    model_config = {"env_prefix": "CANCERSCOPE_"}


settings = Settings()
