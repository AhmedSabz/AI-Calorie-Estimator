import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from prepare_dataset import FoodSegDataset
from model import create_model


# =========================
# Configuration
# =========================

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 15

NUM_CLASSES = 104

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Checkpoint Configuration
# =========================

# Google Drive checkpoint directory
DRIVE_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "AI-Calorie-Estimator/checkpoints"
)

os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

# Local model directory
os.makedirs("models", exist_ok=True)

LATEST_CHECKPOINT = os.path.join(
    DRIVE_CHECKPOINT_DIR,
    "latest_checkpoint.pth"
)

BEST_MODEL_DRIVE = os.path.join(
    DRIVE_CHECKPOINT_DIR,
    "foodseg_model_best.pth"
)

BEST_MODEL_LOCAL = (
    "models/foodseg_model_best.pth"
)


# =========================
# Load Dataset
# =========================

print("Loading FoodSeg103...")

dataset = load_dataset(
    "EduardoPacheco/FoodSeg103"
)

train_dataset = FoodSegDataset(
    dataset=dataset["train"]
)

val_dataset = FoodSegDataset(
    dataset=dataset["validation"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Training images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)


# =========================
# Create Model
# =========================

print("\nCreating model...")

model = create_model()

model = model.to(DEVICE)

print(f"Using device: {DEVICE}")


# =========================
# Loss Function
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Learning Rate Scheduler
# =========================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# =========================
# Validation Function
# =========================

def evaluate(model, dataloader):

    model.eval()

    total_correct = 0
    total_pixels = 0

    intersection = torch.zeros(
        NUM_CLASSES,
        device=DEVICE
    )

    union = torch.zeros(
        NUM_CLASSES,
        device=DEVICE
    )

    validation_loss = 0.0

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            predictions = outputs["out"]

            loss = criterion(
                predictions,
                masks
            )

            validation_loss += loss.item()

            predicted_classes = predictions.argmax(
                dim=1
            )

            total_correct += (
                predicted_classes == masks
            ).sum().item()

            total_pixels += masks.numel()

            for class_id in range(NUM_CLASSES):

                pred_class = (
                    predicted_classes == class_id
                )

                true_class = (
                    masks == class_id
                )

                intersection[class_id] += (
                    pred_class & true_class
                ).sum()

                union[class_id] += (
                    pred_class | true_class
                ).sum()

    pixel_accuracy = (
        total_correct / total_pixels
    )

    valid_classes = union > 0

    iou = (
        intersection[valid_classes]
        / union[valid_classes]
    )

    mean_iou = iou.mean().item()

    validation_loss /= len(dataloader)

    return (
        validation_loss,
        pixel_accuracy,
        mean_iou
    )


# =========================
# Resume From Checkpoint
# =========================

start_epoch = 0
best_iou = 0.0

if os.path.exists(LATEST_CHECKPOINT):

    print("\nFound existing checkpoint.")
    print("Loading checkpoint...")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE
    )

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

    print(
        f"Resuming from epoch "
        f"{start_epoch}/{NUM_EPOCHS}"
    )

    print(
        f"Best Mean IoU so far: "
        f"{best_iou * 100:.2f}%"
    )

else:

    print("\nNo checkpoint found.")
    print("Starting training from epoch 1.")


# =========================
# Training
# =========================

print("\nStarting training...")

for epoch in range(
    start_epoch,
    NUM_EPOCHS
):

    model.train()

    running_loss = 0.0

    for batch_index, (images, masks) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        predictions = outputs["out"]

        # Calculate loss
        loss = criterion(
            predictions,
            masks
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_index + 1}/"
                f"{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_train_loss = (
        running_loss / len(train_loader)
    )


    # =========================
    # Validation
    # =========================

    val_loss, pixel_accuracy, mean_iou = evaluate(
        model,
        val_loader
    )

    # Update learning rate
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # =========================
    # Print Results
    # =========================

    print("\n==============================")

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Training Loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Pixel Accuracy: "
        f"{pixel_accuracy * 100:.2f}%"
    )

    print(
        f"Mean IoU: "
        f"{mean_iou * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.6f}"
    )

    print("==============================")


    # =========================
    # Save Best Model
    # =========================

    if mean_iou > best_iou:

        best_iou = mean_iou

        # Save to Google Drive
        torch.save(
            model.state_dict(),
            BEST_MODEL_DRIVE
        )

        # Save locally
        torch.save(
            model.state_dict(),
            BEST_MODEL_LOCAL
        )

        print(
            f"New best model saved!"
        )

        print(
            f"Best Mean IoU: "
            f"{best_iou * 100:.2f}%"
        )


    # =========================
    # Save Latest Checkpoint
    # =========================

    checkpoint = {

        "epoch": epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_iou":
            best_iou,

        "val_loss":
            val_loss,

        "pixel_accuracy":
            pixel_accuracy,

        "mean_iou":
            mean_iou
    }

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    print(
        f"Checkpoint saved to Google Drive."
    )


# =========================
# Training Complete
# =========================

print(
    "\nTraining completed successfully!"
)

print(
    f"Best Mean IoU: "
    f"{best_iou * 100:.2f}%"
)

print(
    "\nBest model saved to:"
)

print(
    BEST_MODEL_LOCAL
)

print(
    "\nBest model also saved to:"
)

print(
    BEST_MODEL_DRIVE
)