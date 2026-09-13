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

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")


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
# Training
# =========================

print("\nStarting training...")

best_iou = 0.0

for epoch in range(NUM_EPOCHS):

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

    print("\n==============================")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
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
    print("==============================")

    # =========================
    # Save Best Model
    # =========================

    if mean_iou > best_iou:

        best_iou = mean_iou

        torch.save(
            model.state_dict(),
            "models/foodseg_model_best.pth"
        )

        print(
            f"New best model saved! "
            f"Mean IoU: "
            f"{best_iou * 100:.2f}%"
        )


# =========================
# Training Complete
# =========================

print("\nTraining completed successfully!")

print(
    f"Best Mean IoU: "
    f"{best_iou * 100:.2f}%"
)

print(
    "\nBest model saved to:"
    "\nmodels/foodseg_model_best.pth"
)