import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import DINOv3ViTConfig, DINOv3ViTModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
SLICES_DIR = os.path.join(BASE_DIR, "2d_slices")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 224


def remap_checkpoint(ck: dict) -> dict:
    """Remap raw DINOv3 checkpoint keys to HuggingFace DINOv3ViTModel keys."""
    new_sd = {}

    # Embeddings
    new_sd["embeddings.cls_token"] = ck["cls_token"]
    new_sd["embeddings.mask_token"] = ck["mask_token"].unsqueeze(0) if ck["mask_token"].dim() == 2 else ck["mask_token"]
    new_sd["embeddings.register_tokens"] = ck["storage_tokens"]
    new_sd["embeddings.patch_embeddings.weight"] = ck["patch_embed.proj.weight"]
    new_sd["embeddings.patch_embeddings.bias"] = ck["patch_embed.proj.bias"]

    # Final norm
    new_sd["norm.weight"] = ck["norm.weight"]
    new_sd["norm.bias"] = ck["norm.bias"]

    num_layers = max(int(k.split(".")[1]) for k in ck if k.startswith("blocks.")) + 1

    for i in range(num_layers):
        prefix = f"blocks.{i}"
        hf = f"layer.{i}"

        # Layer norms
        new_sd[f"{hf}.norm1.weight"] = ck[f"{prefix}.norm1.weight"]
        new_sd[f"{hf}.norm1.bias"] = ck[f"{prefix}.norm1.bias"]
        new_sd[f"{hf}.norm2.weight"] = ck[f"{prefix}.norm2.weight"]
        new_sd[f"{hf}.norm2.bias"] = ck[f"{prefix}.norm2.bias"]

        # Attention: split combined qkv into q, k, v
        qkv_w = ck[f"{prefix}.attn.qkv.weight"]   # (3*dim, dim)
        qkv_b = ck[f"{prefix}.attn.qkv.bias"]      # (3*dim,)
        dim = qkv_w.shape[0] // 3
        q_w, k_w, v_w = qkv_w[:dim], qkv_w[dim:2*dim], qkv_w[2*dim:]
        q_b, _k_b, v_b = qkv_b[:dim], qkv_b[dim:2*dim], qkv_b[2*dim:]

        new_sd[f"{hf}.attention.q_proj.weight"] = q_w
        new_sd[f"{hf}.attention.q_proj.bias"] = q_b
        new_sd[f"{hf}.attention.k_proj.weight"] = k_w
        new_sd[f"{hf}.attention.v_proj.weight"] = v_w
        new_sd[f"{hf}.attention.v_proj.bias"] = v_b

        new_sd[f"{hf}.attention.o_proj.weight"] = ck[f"{prefix}.attn.proj.weight"]
        new_sd[f"{hf}.attention.o_proj.bias"] = ck[f"{prefix}.attn.proj.bias"]

        # LayerScale
        new_sd[f"{hf}.layer_scale1.lambda1"] = ck[f"{prefix}.ls1.gamma"]
        new_sd[f"{hf}.layer_scale2.lambda1"] = ck[f"{prefix}.ls2.gamma"]

        # MLP (SwiGLU): w1=gate, w2=up, w3=down
        new_sd[f"{hf}.mlp.gate_proj.weight"] = ck[f"{prefix}.mlp.w1.weight"]
        new_sd[f"{hf}.mlp.gate_proj.bias"] = ck[f"{prefix}.mlp.w1.bias"]
        new_sd[f"{hf}.mlp.up_proj.weight"] = ck[f"{prefix}.mlp.w2.weight"]
        new_sd[f"{hf}.mlp.up_proj.bias"] = ck[f"{prefix}.mlp.w2.bias"]
        new_sd[f"{hf}.mlp.down_proj.weight"] = ck[f"{prefix}.mlp.w3.weight"]
        new_sd[f"{hf}.mlp.down_proj.bias"] = ck[f"{prefix}.mlp.w3.bias"]

    return new_sd


def build_model() -> DINOv3ViTModel:
    print("Loading checkpoint...")
    ck = torch.load(CHECKPOINT, map_location="cpu")
    remapped = remap_checkpoint(ck)

    cfg = DINOv3ViTConfig(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        use_gated_mlp=True,
        num_register_tokens=4,
        patch_size=16,
        image_size=IMAGE_SIZE,
    )
    model = DINOv3ViTModel(cfg)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    model.eval()
    return model.to(DEVICE)


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def embed_subject(model, transform, slice_dir: str):
    """Return per-slice CLS embeddings and their slice indices for one subject."""
    png_files = sorted([
        os.path.join(slice_dir, f)
        for f in os.listdir(slice_dir) if f.endswith(".png")
    ])
    if not png_files:
        return None, None

    all_embeddings = []
    slice_indices = []

    for batch_start in range(0, len(png_files), BATCH_SIZE):
        batch_paths = png_files[batch_start:batch_start + BATCH_SIZE]
        imgs = torch.stack([transform(Image.open(p)) for p in batch_paths]).to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=imgs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B, 384)
            all_embeddings.append(cls_tokens.cpu().float().numpy())

        for p in batch_paths:
            # filename is slice_010.png → extract 10
            fname = os.path.basename(p)
            idx = int(fname.replace("slice_", "").replace(".png", ""))
            slice_indices.append(idx)

    return np.concatenate(all_embeddings, axis=0), slice_indices  # (N_slices, 384), [int, ...]


def process_split(model, transform, split: str):
    split_dir = os.path.join(SLICES_DIR, split)
    output_file = os.path.join(BASE_DIR, f"{split}_t1_slice_embeddings.npz")

    subjects = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])
    print(f"\n[{split}] Found {len(subjects)} subjects", flush=True)

    all_subjects = []
    all_slice_indices = []
    all_embeddings = []

    for i, subject in enumerate(subjects):
        t1_dir = os.path.join(split_dir, subject, "t1")
        if not os.path.isdir(t1_dir):
            print(f"  [{i+1}/{len(subjects)}] Skipping {subject} — no t1 folder", flush=True)
            continue

        embs, indices = embed_subject(model, transform, t1_dir)
        if embs is None:
            print(f"  [{i+1}/{len(subjects)}] Skipping {subject} — no slices", flush=True)
            continue

        all_subjects.extend([subject] * len(indices))
        all_slice_indices.extend(indices)
        all_embeddings.append(embs)
        print(f"  [{i+1}/{len(subjects)}] {subject}  slices={len(indices)}", flush=True)

    embeddings_arr = np.concatenate(all_embeddings, axis=0)  # (total_slices, 384)
    np.savez(
        output_file,
        subjects=np.array(all_subjects),
        slice_indices=np.array(all_slice_indices),
        embeddings=embeddings_arr,
        model=np.array("dinov3_vits16plus"),
        modality=np.array("t1"),
        split=np.array(split),
        dim=np.array(384),
    )
    print(f"Saved {len(all_subjects)} slice embeddings to {output_file}", flush=True)
    print(f"Embeddings shape: {embeddings_arr.shape}", flush=True)


def main():
    model = build_model()
    transform = get_transform()

    for split in ("training",):
        split_dir = os.path.join(SLICES_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split} — directory not found", flush=True)
            continue
        process_split(model, transform, split)


if __name__ == "__main__":
    main()
