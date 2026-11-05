
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from datasets import load_dataset

from prepare_dataset import FoodSegDataset
from model import create_model


# ============================================================
# Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 40
NUM_CLASSES = 104
IMAGE_SIZE = 256

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4

DICE_WEIGHT = 0.5

PATIENCE = 8
MIN_DELTA = 0.001

NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

BEST_MODEL = "models/foodseg_model_v2_best.pth"
LATEST_CHECKPOINT = "checkpoints/v2_latest_checkpoint.pth"

os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# Augmentation
# ============================================================

class AugmentedDataset(Dataset):

    def __init__(self, base_dataset, augment=False):
        self.base_dataset = base_dataset
        self.augment = augment

        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        image, mask = self.base_dataset[index]

        if not self.augment:
            return image, mask

        # Undo ImageNet normalization for color augmentation.
        image = (image * self.std + self.mean).clamp(0, 1)

        # Random square crop and resize.
        if random.random() < 0.7:

            crop_size = random.randint(
                int(IMAGE_SIZE * 0.75),
                IMAGE_SIZE
            )

            top = random.randint(
                0, IMAGE_SIZE - crop_size
            )

            left = random.randint(
                0, IMAGE_SIZE - crop_size
            )

            image = TF.resized_crop(
                image,
                top,
                left,
                crop_size,
                crop_size,
                [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR
            )

            mask = TF.resized_crop(
                mask.unsqueeze(0).float(),
                top,
                left,
                crop_size,
                crop_size,
                [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0).long()

        # Horizontal flip.
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Color augmentation only affects the image.
        image = self.color_jitter(image)

        # Restore ImageNet normalization.
        image = (image - self.mean) / self.std

        return image, mask


# ============================================================
# Loss: CrossEntropy + Multiclass Dice
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(self, dice_weight=0.5):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        probabilities = torch.softmax(logits, dim=1)

        one_hot = F.one_hot(
            targets,
            num_classes=NUM_CLASSES
        ).permute(0, 3, 1, 2).float()

        # Exclude background from Dice calculation.
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

        intersection = (
            probabilities * one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probabilities.sum(dim=(0, 2, 3))
            + one_hot.sum(dim=(0, 2, 3))
        )

        dice_score = (
            (2.0 * intersection + 1.0)
            / (denominator + 1.0)
        )

        dice_loss = 1.0 - dice_score.mean()

        return ce_loss + self.dice_weight * dice_loss


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.int64,
        device=DEVICE
    )

    total_loss = 0.0
    total_images = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                logits = model(images)["out"]
                loss = criterion(logits, masks)

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

            predictions = logits.argmax(dim=1)

            encoded = (
                masks.reshape(-1) * NUM_CLASSES
                + predictions.reshape(-1)
            )

            confusion += torch.bincount(
                encoded,
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    tp = confusion.diag()

    gt_count = confusion.sum(dim=1)
    pred_count = confusion.sum(dim=0)

    union = gt_count + pred_count - tp

    valid = union > 0
    valid[0] = False

    per_class_iou = torch.zeros(
        NUM_CLASSES,
        device=DEVICE
    )

    per_class_iou[valid] = (
        tp[valid].float() / union[valid].float()
    )

    miou = per_class_iou[valid].mean().item()

    pixel_accuracy = (
        tp.sum().float() / confusion.sum().float()
    ).item()

    validation_loss = total_loss / max(total_images, 1)

    return validation_loss, pixel_accuracy, miou, per_class_iou, union


# ============================================================
# Load Dataset
# ============================================================

print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

base_train = FoodSegDataset(dataset["train"])
base_val = FoodSegDataset(dataset["validation"])

train_dataset = AugmentedDataset(
    base_train,
    augment=True
)

val_dataset = AugmentedDataset(
    base_val,
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

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Device:", DEVICE)


# ============================================================
# Model and Optimizer
# ============================================================

print("\nCreating model...")

model = create_model().to(DEVICE)

criterion = CombinedLoss(
    dice_weight=DICE_WEIGHT
)

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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ============================================================
# Training
# ============================================================

best_iou = -1.0
epochs_without_improvement = 0

print("\nStarting V2 training...")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    total_images = 0

    for batch_index, (images, masks) in enumerate(train_loader):

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            logits = model(images)["out"]
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total_images += images.size(0)

        if (batch_index + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / max(total_images, 1)

    val_loss, pixel_acc, val_iou, class_iou, class_union = (
        evaluate(model, val_loader)
    )

    scheduler.step(val_iou)

    print("\n" + "=" * 55)
    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Pixel accuracy: {pixel_acc * 100:.2f}%")
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
            BEST_MODEL
        )

        print("Saved new best V2 model.")

    else:

        epochs_without_improvement += 1

        print(
            "No significant improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_iou": best_iou,
            "epochs_without_improvement": epochs_without_improvement
        },
        LATEST_CHECKPOINT
    )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


# ============================================================
# Final Results
# ============================================================

print("\nV2 training complete!")
print(f"Best validation mIoU: {best_iou * 100:.2f}%")
print("Best model:", BEST_MODEL)

# Print weakest evaluated classes from the best checkpoint.
best_model = create_model().to(DEVICE)
best_model.load_state_dict(
    torch.load(BEST_MODEL, map_location=DEVICE, weights_only=True)
)

_, _, _, class_iou, class_union = evaluate(best_model, val_loader)

valid_classes = (class_union > 0)
valid_classes[0] = False

valid_ids = torch.where(valid_classes)[0]
weakest = valid_ids[
    torch.argsort(class_iou[valid_ids])[:10]
]

print("\n10 weakest classes by validation IoU:")

for class_id in weakest:
    cid = class_id.item()
    print(
        f"Class {cid}: "
        f"IoU={class_iou[cid].item() * 100:.2f}% "
        f"Pixels={class_union[cid].item()}"
    )
