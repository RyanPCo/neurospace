from cortex import CortexClient
from pathlib import Path
import torch
from PIL import Image as PILImage

# Import the exact logic from your first file to guarantee matching embeddings!
# (Change 'vectorizer' to whatever you named your first Python file)
from vectorizer import build_model, get_transform 

# ─── Model (lazy singleton) ────────────────────────────────────────────────────

_model = None
_transform = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_model_and_transform():
    global _model, _transform
    if _model is None:
        # This re-uses your exact DINOv3 HuggingFace config and remapping logic
        _model = build_model()
        _transform = get_transform()
    return _model, _transform

# ─── Preprocessing & Vectorization ─────────────────────────────────────────────

def vectorize_image(file_path: str | Path) -> list[float]:
    """Return the CLS-token embedding for the image at *file_path*."""
    model, transform = _get_model_and_transform()
    
    img = PILImage.open(file_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(_device)  # (1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(pixel_values=x)
        
    # DINOv3ViTModel outputs the sequence in last_hidden_state. 
    # Grab the CLS token at sequence index 0, exactly as you did in embed_subject.
    cls_token = outputs.last_hidden_state[:, 0, :] 
    
    return cls_token.squeeze(0).cpu().float().tolist()

# ─── Vector DB query ───────────────────────────────────────────────────────────

def query_similar_slices(
    file_path: str | Path,
    *,
    host: str = "localhost",
    port: int = 5432,
    collection_name: str = "breakhis_slices",
    top_n_subjects: int = 10,
    candidates: int = 200,
) -> list[dict]:
    """
    Vectorize the image at *file_path*, search the Actian VectorAI collection, 
    and return the payload of the best-scoring slice from each of the 
    *top_n_subjects* most similar distinct subjects.
    """
    query_vec = vectorize_image(file_path)

    with CortexClient(host=host, port=port) as client:
        client.open_collection(collection_name)
        results = client.search(
            collection_name=collection_name,
            query=query_vec,
            top_k=candidates,
            with_payload=True,
        )

    # Deduplicate by subject — keep the highest-scoring slice per subject
    seen_subjects: set[str] = set()
    payloads: list[dict] = []

    for r in results:
        payload: dict = r.payload or {}
        subject_id: str = (
            payload.get("subject_id")
            or payload.get("patient_id")
            or str(r.id)
        )
        if subject_id in seen_subjects:
            continue
        
        seen_subjects.add(subject_id)
        payloads.append({**payload, "_id": r.id, "_score": r.score})
        
        if len(payloads) >= top_n_subjects:
            break

    return payloads