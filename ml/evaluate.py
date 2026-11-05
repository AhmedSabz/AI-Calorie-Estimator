
import json
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset

from prepare_dataset import FoodSegDataset
from model import create_model
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_PATH = PROJECT_ROOT / "datasets" / "id2label.json"

with open(LABEL_PATH, "r") as f:
    id2label = json.load(f)

print(f"Loaded {len(id2label)} food classes")

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 2
NUM_CLASSES = 104
BACKGROUND_CLASS = 0

MODEL_PATH = "models/foodseg_model_best.pth"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load Class Names
# ============================================================

def load_class_names():

    project_root = Path(__file__).resolve().parent.parent

    possible_paths = [
        project_root / "id2label.json",
        project_root / "datasets" / "id2label.json",
        project_root / "ml" / "id2label.json",
    ]

    for path in possible_paths:

        if path.exists():

            with open(path, "r") as file:
                mapping = json.load(file)

            print(f"Loaded class mapping: {path}")

            return {
                int(key): str(value)
                for key, value in mapping.items()
            }

    print("Class mapping JSON not found.")

    return {
        i: f"Class {i}"
        for i in range(NUM_CLASSES)
    }


class_names = load_class_names()


def get_class_name(class_id):

    return class_names.get(
        class_id,
        f"Class {class_id}"
    )


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

validation_dataset = FoodSegDataset(
    dataset=dataset["validation"]
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

print("Validation images:", len(validation_dataset))
print("Batch size:", BATCH_SIZE)


# ============================================================
# Load Model
# ============================================================

print("\nCreating model...")

model = create_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(DEVICE)

model.eval()

print("Model loaded:", MODEL_PATH)
print("Using device:", DEVICE)


# ============================================================
# Evaluation Counters
# ============================================================

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

ground_truth_count = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)

predicted_count = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float64
)


# ============================================================
# Evaluation
# ============================================================

print("\nStarting evaluation...")

with torch.inference_mode():

    for batch_index, (images, masks) in enumerate(
        validation_loader
    ):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()

        outputs = model(images)

        predictions = outputs["out"].argmax(dim=1)

        # Pixel accuracy.
        total_correct += (
            predictions == masks
        ).sum().item()

        total_pixels += masks.numel()

        # Per-class statistics.
        for class_id in range(NUM_CLASSES):

            prediction_class = (
                predictions == class_id
            )

            mask_class = (
                masks == class_id
            )

            intersection[class_id] += (
                prediction_class & mask_class
            ).sum().item()

            union[class_id] += (
                prediction_class | mask_class
            ).sum().item()

            ground_truth_count[class_id] += (
                mask_class.sum().item()
            )

            predicted_count[class_id] += (
                prediction_class.sum().item()
            )

        if (batch_index + 1) % 100 == 0:

            print(
                f"Evaluated {batch_index + 1}/"
                f"{len(validation_loader)} batches..."
            )


# ============================================================
# Calculate Metrics
# ============================================================

pixel_accuracy = (
    total_correct / total_pixels
) * 100

iou = intersection / (union + 1e-10)

# Match training: exclude background.
valid_classes = union > 0
valid_classes[BACKGROUND_CLASS] = False

mean_iou = iou[valid_classes].mean().item()

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"Pixel Accuracy: {pixel_accuracy:.2f}%")
print(f"Mean IoU: {mean_iou * 100:.2f}%")

print(
    "Evaluated food classes:",
    valid_classes.sum().item()
)


# ============================================================
# Per-Class Results
# ============================================================

print("\n" + "=" * 75)
print("PER-CLASS SEGMENTATION RESULTS")
print("=" * 75)

print(
    f"{'ID':<5}"
    f"{'Food Class':<25}"
    f"{'IoU':>10}"
    f"{'GT Pixels':>15}"
    f"{'Pred Pixels':>15}"
)

print("-" * 75)

for class_id in range(NUM_CLASSES):

    if not valid_classes[class_id]:
        continue

    name = get_class_name(class_id)

    print(
        f"{class_id:<5}"
        f"{name:<25}"
        f"{iou[class_id].item() * 100:>9.2f}%"
        f"{ground_truth_count[class_id].item():>15.0f}"
        f"{predicted_count[class_id].item():>15.0f}"
    )


# ============================================================
# Weakest Classes
# ============================================================

valid_ids = torch.where(valid_classes)[0]

weakest = valid_ids[
    torch.argsort(iou[valid_ids])[:10]
]

print("\n" + "=" * 60)
print("10 WEAKEST FOOD CLASSES")
print("=" * 60)

for class_id in weakest:

    cid = class_id.item()

    gt = ground_truth_count[cid].item()
    pred = predicted_count[cid].item()

    print(f"\nClass ID: {cid}")
    print(f"Food: {get_class_name(cid)}")
    print(f"IoU: {iou[cid].item() * 100:.2f}%")
    print(f"Ground-truth pixels: {gt:.0f}")
    print(f"Predicted pixels: {pred:.0f}")

    if gt > 0 and pred == 0:
        print("Problem: Model never predicted this class.")

    elif gt == 0:
        print("Problem: Class has no ground-truth pixels.")

    elif pred > gt * 2:
        print("Problem: Possible overprediction.")

    elif pred < gt * 0.5:
        print("Problem: Possible underprediction.")

    else:
        print("Problem: Predictions overlap poorly with ground truth.")


# ============================================================
# Save Results
# ============================================================

output_path = Path("evaluation_results.csv")

with open(output_path, "w") as file:

    file.write(
        "class_id,class_name,iou,ground_truth_pixels,"
        "predicted_pixels\n"
    )

    for class_id in range(NUM_CLASSES):

        if not valid_classes[class_id]:
            continue

        file.write(
            f'{class_id},'
            f'"{get_class_name(class_id)}",'
            f'{iou[class_id].item():.6f},'
            f'{ground_truth_count[class_id].item():.0f},'
            f'{predicted_count[class_id].item():.0f}\n'
        )

print("\nResults saved to:", output_path)

print("\nEvaluation complete!")
