import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 104
BATCH_SIZE = 4


# ============================================================
# Image Transformations
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# FoodSeg103 Dataset
# ============================================================

class FoodSegDataset(Dataset):

    def __init__(self, dataset, transform=None):

        self.dataset = dataset

        self.transform = (
            transform
            if transform is not None
            else image_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = sample["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        image = image.convert("RGB")

        # Apply image transformations.
        image = self.transform(image)

        # ----------------------------------------------------
        # Load segmentation mask
        # ----------------------------------------------------

        mask = sample["label"]

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        # Nearest-neighbor interpolation preserves class IDs.
        mask = mask.resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            Image.Resampling.NEAREST
        )

        mask_array = np.array(mask, dtype=np.int64)

        # ----------------------------------------------------
        # Validate mask
        # ----------------------------------------------------

        if mask_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D segmentation mask at index "
                f"{index}, got shape {mask_array.shape}."
            )

        if mask_array.size == 0:
            raise ValueError(
                f"Empty segmentation mask at index {index}."
            )

        min_class = int(mask_array.min())
        max_class = int(mask_array.max())

        if min_class < 0 or max_class >= NUM_CLASSES:
            raise ValueError(
                f"Invalid class ID in mask at index {index}: "
                f"minimum={min_class}, maximum={max_class}. "
                f"Expected IDs from 0 to {NUM_CLASSES - 1}."
            )

        mask = torch.from_numpy(mask_array)

        # ----------------------------------------------------
        # Validate image and mask dimensions
        # ----------------------------------------------------

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected image shape [3, H, W], "
                f"got {tuple(image.shape)} at index {index}."
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask shape [H, W], "
                f"got {tuple(mask.shape)} at index {index}."
            )

        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"Image and mask dimensions do not match "
                f"at index {index}: "
                f"image={tuple(image.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        return image, mask


# ============================================================
# Test Dataset Pipeline
# ============================================================

if __name__ == "__main__":

    print("Loading FoodSeg103...")

    dataset = load_dataset("EduardoPacheco/FoodSeg103")

    train_data = dataset["train"]
    validation_data = dataset["validation"]

    print("Training images:", len(train_data))
    print("Validation images:", len(validation_data))

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = FoodSegDataset(
        train_data
    )

    validation_dataset = FoodSegDataset(
        validation_data
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Test one batch
    # --------------------------------------------------------

    print("\nTesting DataLoader...")

    images, masks = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    print("Image data type:", images.dtype)
    print("Mask data type:", masks.dtype)

    print("Minimum mask class ID:", masks.min().item())
    print("Maximum mask class ID:", masks.max().item())

    assert images.shape[1:] == (3, 256, 256)
    assert masks.shape[1:] == (256, 256)
    assert masks.dtype == torch.int64

    print("\nDataset pipeline is working!")

    print("\nTesting validation dataset...")

    val_images, val_masks = next(iter(validation_loader))

    print("Validation image batch:", val_images.shape)
    print("Validation mask batch:", val_masks.shape)

    print("\nAll dataset checks passed!")
