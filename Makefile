.PHONY: setup index train kernels anim dev backend frontend clean

PYTHON := python
UVICORN := python -m uvicorn
NPM := npm

setup:
	@echo "==> Installing Python dependencies..."
	pip install -r backend/requirements.txt
	@echo "==> Installing Node dependencies..."
	cd frontend && $(NPM) install
	@echo "==> Creating directories..."
	mkdir -p data/raw data/processed data/splits models/checkpoints models/kernel_cache
	@echo "==> Setup complete."

index:
	@echo "==> Indexing brain tumor MRI dataset..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) backend/scripts/index_dataset.py
	@echo "==> Dataset indexing complete."

train:
	@echo "==> Starting initial training..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) backend/scripts/initial_train.py
	@echo "==> Training complete."

kernels:
	@echo "==> Pre-computing kernel visualizations and importance scores..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) backend/scripts/precompute_kernels.py
	@echo "==> Kernel pre-computation complete."

anim:
	@echo "==> Rendering Manim animation (requires: pip install manim)..."
	pip install -q manim
	cd backend/scripts && manim -ql manim_animation.py ConstrainedCAMScene
	mkdir -p frontend/public
	cp backend/scripts/media/videos/manim_animation/480p15/ConstrainedCAMScene.mp4 frontend/public/constrained_cam.mp4
	@echo "==> Animation rendered → frontend/public/constrained_cam.mp4"

backend:
	@echo "==> Starting backend on :8000..."
	cd backend && PYTORCH_ENABLE_MPS_FALLBACK=1 $(UVICORN) main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	@echo "==> Starting frontend on :5173..."
	cd frontend && $(NPM) run dev

dev:
	@echo "==> Starting CancerScope (backend :8000, frontend :5173)..."
	@trap 'kill 0' SIGINT; \
	(cd backend && PYTORCH_ENABLE_MPS_FALLBACK=1 $(UVICORN) main:app --host 0.0.0.0 --port 8000 --reload) & \
	(cd frontend && $(NPM) run dev) & \
	wait

clean:
	@echo "==> Cleaning generated files..."
	rm -f cancerscope.db
	rm -rf models/kernel_cache/*
	rm -rf data/processed/* data/splits/*
	@echo "==> Clean complete."
