"""Image transforms for Brain Tumor MRI (grayscale 256×256)."""
from torchvision import transforms

# Single-channel mean/std computed on the dataset (approx 0.2/0.3 for brain MRI).
# Using 0.5/0.5 centres the distribution simply without ImageNet values.
_MEAN = [0.5]
_STD  = [0.5]

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),               # (1, 256, 256) — PIL L-mode → 1 channel
    transforms.Normalize(mean=_MEAN, std=_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

inference_transforms = val_transforms
