import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from prepare_dataset import FoodSegDataset
from model import create_model

# Configuration

BATCH_SIZE = 2
NUM_CLASSES = 104
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load validation dataset

print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

validation_dataset = FoodSegDataset(
    dataset=dataset["validation"]
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation images: {len(validation_dataset)}")
print(f"Batch size: {BATCH_SIZE}")

# Create model

print("\nCreating model...")

model = create_model()
model.load_state_dict(
    torch.load(
        "models/foodseg_model.pth",
        map_location=DEVICE
    )
)
model = model.to(DEVICE)

print(f"Using device: {DEVICE}")

# Evaluation

print("\nStarting evaluation...")

model.eval()

total_correct = 0
total_pixels = 0

intersection = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)
union = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)

with torch.no_grad():

    for batch_index, (images, masks) in enumerate(validation_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Run model
        outputs = model(images)

        predictions = outputs["out"]

        # Get predicted class for every pixel
        predictions = torch.argmax(
            predictions,
            dim=1
        )

        # Pixel accuracy
        total_correct += (
            predictions == masks
        ).sum().item()

        total_pixels += masks.numel()

        # Calculate IoU for each class
        for class_id in range(NUM_CLASSES):

            prediction_class = predictions == class_id
            mask_class = masks == class_id

            intersection[class_id] += (
                prediction_class & mask_class
            ).sum().item()

            union[class_id] += (
                prediction_class | mask_class
            ).sum().item()

        # Print progress
        if (batch_index + 1) % 100 == 0:

            print(
                f"Evaluated "
                f"{batch_index + 1}/{len(validation_loader)} "
                f"batches..."
            )

# Calculate pixel accuracy

pixel_accuracy = (
    total_correct / total_pixels
) * 100

# Calculate IoU

iou = intersection / (union + 1e-10)

# Only calculate mean IoU for classes
# that actually appear in the validation set
valid_classes = union > 0

mean_iou = (
    iou[valid_classes].mean().item()
)

# Print results

print("\nEvaluation complete!")

print(
    f"\nPixel Accuracy: "
    f"{pixel_accuracy:.2f}%"
)
print(
    f"Mean IoU: "
    f"{mean_iou:.4f}"
)
print(
    f"Mean IoU: "
    f"{mean_iou * 100:.2f}%"
)