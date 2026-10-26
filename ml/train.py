import os
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from prepare_dataset import FoodSegDataset
from model import create_model


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 104

# Different learning rates for backbone and segmentation head
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 0.001

# Set this to True to exclude background class 0 from mIoU.
EXCLUDE_BACKGROUND_FROM_MIOU = True

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducibility
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Checkpoint Configuration
# =========================================================

DRIVE_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "AI-Calorie-Estimator/checkpoints"
)

os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

LATEST_CHECKPOINT = os.path.join(
    DRIVE_CHECKPOINT_DIR,
    "latest_checkpoint.pth"
)

BEST_MODEL_DRIVE = os.path.join(
    DRIVE_CHECKPOINT_DIR,
    "foodseg_model_best.pth"
)

BEST_MODEL_LOCAL = os.path.join(
    "models",
    "foodseg_model_best.pth"
)

CHECKPOINT_VERSION = 2


# =========================================================
# Dataset Augmentation
# =========================================================

class AugmentedDataset(Dataset):
    """
    Applies synchronized horizontal flipping to images and masks.

    Images and masks must already be returned as tensors by
    FoodSegDataset.
    """

    def __init__(self, base_dataset, augment=False):
        self.base_dataset = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, mask = self.base_dataset[index]

        # Apply the exact same geometric transformation
        # to the image and its segmentation mask.
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        return image, mask


# =========================================================
# Load Dataset
# =========================================================

print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

base_train_dataset = FoodSegDataset(
    dataset=dataset["train"]
)

base_val_dataset = FoodSegDataset(
    dataset=dataset["validation"]
)

train_dataset = AugmentedDataset(
    base_train_dataset,
    augment=True
)

# Do not augment validation data.
val_dataset = AugmentedDataset(
    base_val_dataset,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {DEVICE}")


# =========================================================
# Create Model
# =========================================================

print("\nCreating model...")

model = create_model().to(DEVICE)

print(f"Using device: {DEVICE}")


# =========================================================
# Loss Function
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# Optimizer
# =========================================================

# Standard torchvision DeepLabV3 models have backbone and
# classifier modules. Use separate learning rates for them.

if hasattr(model, "backbone") and hasattr(model, "classifier"):

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

else:
    # Fallback if create_model() returns a different architecture.
    print(
        "Warning: Could not find backbone/classifier modules. "
        "Using one learning rate for all parameters."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY
    )


# =========================================================
# Learning Rate Scheduler
# =========================================================

# Improve learning rate when validation mIoU stops improving.

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)


# =========================================================
# Validation Function
# =========================================================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0.0
    total_images = 0

    # Confusion matrix:
    # rows = ground-truth classes
    # columns = predicted classes
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

            outputs = model(images)
            predictions = outputs["out"]

            loss = criterion(predictions, masks)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

            predicted_classes = predictions.argmax(dim=1)

            # Build a confusion matrix efficiently.
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

    # Classes with no ground-truth or predicted pixels are
    # undefined for IoU and are excluded from the mean.
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

    total_correct = true_positives.sum().item()
    total_pixels = confusion_matrix.sum().item()

    pixel_accuracy = (
        total_correct / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return (
        validation_loss,
        pixel_accuracy,
        mean_iou
    )


# =========================================================
# Checkpoint Loading
# =========================================================

start_epoch = 0
best_iou = 0.0
epochs_without_improvement = 0

# Resume only checkpoints created by this revised training
# script, since the optimizer configuration has changed.

if os.path.exists(LATEST_CHECKPOINT):

    print("\nFound a previous checkpoint.")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE
    )

    if checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION:

        print("Resuming compatible training checkpoint.")

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = checkpoint["epoch"]
        best_iou = checkpoint["best_iou"]

        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement",
            0
        )

        print(f"Resuming after epoch {start_epoch}.")
        print(f"Best mIoU so far: {best_iou * 100:.2f}%")

    else:

        print(
            "Old checkpoint detected. Its optimizer settings "
            "are incompatible with this revised script."
        )

        # Reuse the previous best model weights if available.
        if os.path.exists(BEST_MODEL_DRIVE):

            print("Loading previous best model weights.")

            weights = torch.load(
                BEST_MODEL_DRIVE,
                map_location=DEVICE
            )

            model.load_state_dict(weights)

        elif os.path.exists(BEST_MODEL_LOCAL):

            print("Loading local best model weights.")

            weights = torch.load(
                BEST_MODEL_LOCAL,
                map_location=DEVICE
            )

            model.load_state_dict(weights)

        else:
            print(
                "No compatible best model found. "
                "Training will start with create_model() weights."
            )

else:

    # If the latest checkpoint is missing but a best model exists,
    # use those weights as the starting point.
    if os.path.exists(BEST_MODEL_DRIVE):

        print("\nLoading existing best model from Drive.")

        weights = torch.load(
            BEST_MODEL_DRIVE,
            map_location=DEVICE
        )

        model.load_state_dict(weights)

    elif os.path.exists(BEST_MODEL_LOCAL):

        print("\nLoading existing local best model.")

        weights = torch.load(
            BEST_MODEL_LOCAL,
            map_location=DEVICE
        )

        model.load_state_dict(weights)

    else:
        print("\nNo previous model found. Starting from create_model().")


# =========================================================
# Establish Baseline
# =========================================================

# When starting a new training run from existing weights,
# evaluate them first so the best-model checkpoint is not
# overwritten by a worse model.

if start_epoch == 0:

    print("\nEvaluating starting model...")

    baseline_loss, baseline_accuracy, baseline_iou = evaluate(
        model,
        val_loader
    )

    best_iou = baseline_iou

    print(f"Starting validation loss: {baseline_loss:.4f}")
    print(f"Starting pixel accuracy: {baseline_accuracy * 100:.2f}%")
    print(f"Starting mIoU: {baseline_iou * 100:.2f}%")


# =========================================================
# Training Loop
# =========================================================

print("\nStarting training...")

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

        outputs = model(images)
        predictions = outputs["out"]

        loss = criterion(predictions, masks)

        loss.backward()

        # Limit unusually large gradients to stabilize training.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_train_images += batch_size

        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_train_loss = (
        running_loss / max(total_train_images, 1)
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    val_loss, pixel_accuracy, mean_iou = evaluate(
        model,
        val_loader
    )

    # The scheduler now optimizes for mIoU, not validation loss.
    scheduler.step(mean_iou)

    current_lrs = [
        group["lr"]
        for group in optimizer.param_groups
    ]


    # -----------------------------------------------------
    # Print Results
    # -----------------------------------------------------

    print("\n" + "=" * 45)

    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}")
    print(f"Training Loss:   {average_train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Pixel Accuracy:  {pixel_accuracy * 100:.2f}%")
    print(f"Mean IoU:        {mean_iou * 100:.2f}%")
    print(
        "Learning Rates: "
        + ", ".join(f"{lr:.2e}" for lr in current_lrs)
    )

    print("=" * 45)


    # -----------------------------------------------------
    # Save Best Model and Apply Early Stopping
    # -----------------------------------------------------

    improved = mean_iou > best_iou + MIN_DELTA

    if improved:

        best_iou = mean_iou
        epochs_without_improvement = 0

        model_state = model.state_dict()

        torch.save(
            model_state,
            BEST_MODEL_DRIVE
        )

        torch.save(
            model_state,
            BEST_MODEL_LOCAL
        )

        print("New best model saved!")
        print(f"Best mIoU: {best_iou * 100:.2f}%")

    else:

        epochs_without_improvement += 1

        print(
            "No significant mIoU improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )


    # -----------------------------------------------------
    # Save Latest Checkpoint
    # -----------------------------------------------------

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou": best_iou,
        "val_loss": val_loss,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "epochs_without_improvement": epochs_without_improvement
    }

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    print("Latest checkpoint saved.")


    # -----------------------------------------------------
    # Early Stopping
    # -----------------------------------------------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print(
            "\nEarly stopping: validation mIoU has not "
            "improved sufficiently."
        )

        break


# =========================================================
# Training Complete
# =========================================================

print("\nTraining completed.")

print(f"Best validation mIoU: {best_iou * 100:.2f}%")

print(f"Best model: {BEST_MODEL_LOCAL}")
print(f"Drive model: {BEST_MODEL_DRIVE}")
print(f"Latest checkpoint: {LATEST_CHECKPOINT}")