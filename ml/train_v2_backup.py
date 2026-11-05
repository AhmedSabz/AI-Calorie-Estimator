
import os
import random
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from prepare_dataset import FoodSegDataset
from model import create_model


# =========================================================
# Configuration
# =========================================================

SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 40
NUM_CLASSES = 104
IMAGE_SIZE = 256

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 7
MIN_DELTA = 0.001

MAX_CLASS_WEIGHT = 4.0
EXCLUDE_BACKGROUND_FROM_MIOU = True

NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"
MODEL_DIR = "models"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# V3 uses separate files so earlier models remain untouched.
LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_checkpoint_v3.pth"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "foodseg_model_v3_best.pth"
)

# Initialize V3 from the better V1 model if available.
INITIAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "foodseg_model_best.pth"
)

CHECKPOINT_VERSION = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

USE_AMP = DEVICE.type == "cuda"


# =========================================================
# Dataset Augmentation
# =========================================================

class AugmentedDataset(Dataset):

    def __init__(self, base_dataset, augment=False):
        self.base_dataset = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        image, mask = self.base_dataset[index]

        # Synchronized horizontal flip.
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        return image, mask


# =========================================================
# Calculate Training Class Frequencies
# =========================================================

def calculate_class_weights(train_data):

    print("\nCalculating training class frequencies...")

    pixel_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    for index in range(len(train_data)):

        sample = train_data[index]
        mask = sample["label"]

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        mask_array = np.asarray(mask, dtype=np.int64)

        if mask_array.ndim != 2:
            raise ValueError(
                f"Invalid mask shape at index {index}: "
                f"{mask_array.shape}"
            )

        if mask_array.min() < 0 or mask_array.max() >= NUM_CLASSES:
            raise ValueError(
                f"Invalid class ID in mask at index {index}"
            )

        counts = np.bincount(
            mask_array.reshape(-1),
            minlength=NUM_CLASSES
        )

        pixel_counts += torch.from_numpy(
            counts.astype(np.float64)
        )

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1}/{len(train_data)} masks"
            )

    present_classes = pixel_counts > 0

    if not present_classes.any():
        raise RuntimeError("No valid training pixels found.")

    # Median frequency among classes present in training data.
    median_frequency = pixel_counts[present_classes].median()

    frequencies = pixel_counts / pixel_counts.sum()

    # Square-root inverse-frequency weighting.
    weights = torch.ones(NUM_CLASSES, dtype=torch.float32)

    weights[present_classes] = torch.sqrt(
        median_frequency / pixel_counts[present_classes]
    ).float()

    # Avoid extreme weights for very rare classes.
    weights = torch.clamp(
        weights,
        min=0.25,
        max=MAX_CLASS_WEIGHT
    )

    # Normalize average weight to approximately 1.
    weights = weights / weights.mean()

    print("\nClass weight statistics:")
    print(f"Classes present: {present_classes.sum().item()}")
    print(f"Minimum weight: {weights.min().item():.3f}")
    print(f"Maximum weight: {weights.max().item():.3f}")

    rare_classes = torch.argsort(pixel_counts)

    print("\n10 least frequent training classes:")

    for class_id in rare_classes[:10]:
        print(
            f"Class {class_id.item():3d}: "
            f"{int(pixel_counts[class_id].item()):,} pixels, "
            f"weight={weights[class_id].item():.3f}"
        )

    return weights


# =========================================================
# Load Dataset
# =========================================================

print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

train_data = dataset["train"]
val_data = dataset["validation"]

class_weights = calculate_class_weights(
    train_data
).to(DEVICE)

base_train_dataset = FoodSegDataset(
    dataset=train_data
)

base_val_dataset = FoodSegDataset(
    dataset=val_data
)

train_dataset = AugmentedDataset(
    base_train_dataset,
    augment=True
)

val_dataset = AugmentedDataset(
    base_val_dataset,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    persistent_workers=NUM_WORKERS > 0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    persistent_workers=NUM_WORKERS > 0
)

print(f"\nTraining images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {DEVICE}")
print(f"Mixed precision: {USE_AMP}")


# =========================================================
# Create Model
# =========================================================

print("\nCreating DeepLabV3 model...")

model = create_model().to(DEVICE)

# =========================================================
# Loss Function
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.classifier.parameters(),
            "lr": HEAD_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# Learning Rate Scheduler
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)


# =========================================================
# Mixed Precision
# =========================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# =========================================================
# Validation
# =========================================================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0.0
    total_images = 0

    confusion_matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.int64,
        device=DEVICE
    )

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            ).long()

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                outputs = model(images)
                predictions = outputs["out"]

                loss = criterion(predictions, masks)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

            predicted_classes = predictions.argmax(dim=1)

            valid_pixels = (
                (masks >= 0) &
                (masks < NUM_CLASSES)
            )

            encoded = (
                masks[valid_pixels] * NUM_CLASSES
                + predicted_classes[valid_pixels]
            )

            confusion_matrix += torch.bincount(
                encoded,
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    validation_loss = total_loss / max(total_images, 1)

    true_positives = confusion_matrix.diag()

    ground_truth_count = confusion_matrix.sum(dim=1)
    predicted_count = confusion_matrix.sum(dim=0)

    union = (
        ground_truth_count
        + predicted_count
        - true_positives
    )

    valid_classes = union > 0

    if EXCLUDE_BACKGROUND_FROM_MIOU:
        valid_classes[0] = False

    iou_per_class = (
        true_positives[valid_classes].float()
        / union[valid_classes].float()
    )

    mean_iou = (
        iou_per_class.mean().item()
        if iou_per_class.numel() > 0
        else 0.0
    )

    pixel_accuracy = (
        true_positives.sum().item()
        / confusion_matrix.sum().item()
        if confusion_matrix.sum().item() > 0
        else 0.0
    )

    return validation_loss, pixel_accuracy, mean_iou


# =========================================================
# Checkpoint Loading / V1 Initialization
# =========================================================

start_epoch = 0
best_iou = -1.0
epochs_without_improvement = 0

if os.path.exists(LATEST_CHECKPOINT):

    print("\nFound V3 checkpoint.")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "Incompatible V3 checkpoint version."
        )

    print("Resuming V3 training...")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = checkpoint["epoch"]
    best_iou = checkpoint["best_iou"]

    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0
    )

    print(f"Resuming at epoch {start_epoch + 1}")
    print(f"Best mIoU: {best_iou * 100:.2f}%")

else:

    print("\nNo V3 checkpoint found.")

    if os.path.exists(INITIAL_MODEL_PATH):

        print("Initializing from V1 best checkpoint...")

        v1_state = torch.load(
            INITIAL_MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(v1_state)

        print("V1 weights loaded successfully.")

    else:

        print(
            "V1 checkpoint not found. "
            "Using ImageNet-pretrained initialization."
        )


# =========================================================
# Training Loop
# =========================================================

print("\nStarting V3 training...")

for epoch in range(start_epoch, NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    total_train_images = 0

    for batch_index, (images, masks) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        ).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(images)
            predictions = outputs["out"]

            loss = criterion(predictions, masks)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_train_images += batch_size

        if (batch_index + 1) % 25 == 0:

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = (
        running_loss / max(total_train_images, 1)
    )

    val_loss, pixel_accuracy, val_iou = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_iou)

    print("\n" + "=" * 55)
    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Pixel accuracy: {pixel_accuracy * 100:.2f}%")
    print(f"Validation mIoU: {val_iou * 100:.2f}%")
    print(f"Backbone LR: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"Classifier LR: {optimizer.param_groups[1]['lr']:.8f}")
    print("=" * 55)

    improved = val_iou > best_iou + MIN_DELTA

    if improved:

        best_iou = val_iou
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"New best V3 model saved! "
            f"mIoU: {best_iou * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No significant improvement. "
            f"Patience: {epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "class_weights": class_weights.cpu(),
        "best_iou": best_iou,
        "epochs_without_improvement": epochs_without_improvement
    }

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =========================================================
# Training Complete
# =========================================================

print("\nV3 training complete!")
print(f"Best validation mIoU: {best_iou * 100:.2f}%")
print(f"Best model: {BEST_MODEL_PATH}")
print(f"Latest checkpoint: {LATEST_CHECKPOINT}")
