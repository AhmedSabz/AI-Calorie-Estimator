import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# Allow imports from the ml directory
sys.path.append(str(Path(__file__).resolve().parent))

from general_portion_estimator import estimate_food_weight
from nutrition_database import get_nutrition
from model import create_model


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/foodseg_model_best.pth"
IMAGE_PATH = "images.jpg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FoodSeg103 class labels
# ============================================================

# Important class IDs detected in your current image.
# Add the rest of your FoodSeg103 labels here later if needed.
ID_TO_LABEL = {
    0: "background",

    8: "ice cream",
    29: "banana",
    30: "strawberry",
    46: "steak",
    49: "sausage",
    58: "bread",
    70: "potato",
}


# ============================================================
# Load trained model
# ============================================================

print(f"Using device: {DEVICE}")

model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Load and preprocess image
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

original_image = image.copy()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(DEVICE)


# ============================================================
# Run inference
# ============================================================

with torch.no_grad():
    output = model(input_tensor)["out"]


# Convert logits into class probabilities
probabilities = torch.softmax(output, dim=1)


# Best predicted class for every pixel
prediction = torch.argmax(
    probabilities,
    dim=1
)[0].cpu().numpy()


# Confidence of the selected class for every pixel
confidence_map = torch.max(
    probabilities,
    dim=1
)[0][0].cpu().numpy()


# ============================================================
# Analyze detected food classes
# ============================================================

unique_classes, pixel_counts = np.unique(
    prediction,
    return_counts=True
)

total_pixels = prediction.size

detected_foods = {}

for class_id, pixel_count in zip(
    unique_classes,
    pixel_counts
):
    class_id = int(class_id)

    food_name = ID_TO_LABEL.get(
        class_id,
        f"unknown_class_{class_id}"
    )

    detected_foods[class_id] = {
        "name": food_name,
        "pixel_count": int(pixel_count)
    }


print("\nDetected food classes:")

for class_id, food_info in detected_foods.items():

    pixel_count = food_info["pixel_count"]
    food_name = food_info["name"]

    pixel_percentage = (
        pixel_count / total_pixels
    ) * 100

    print(
        f"{food_name}: "
        f"{pixel_percentage:.2f}% of image"
    )


# ============================================================
# Find largest detected food class
# ============================================================

non_background_foods = {
    class_id: food_info
    for class_id, food_info in detected_foods.items()
    if class_id != 0
}

if non_background_foods:

    largest_class_id = max(
        non_background_foods,
        key=lambda class_id: (
            non_background_foods[class_id]["pixel_count"]
        )
    )

    largest_food = non_background_foods[
        largest_class_id
    ]["name"]

    largest_pixel_count = non_background_foods[
        largest_class_id
    ]["pixel_count"]

    largest_food_percentage = (
        largest_pixel_count / total_pixels
    ) * 100

    print("\nLargest detected food:")
    print(
        f"{largest_food} "
        f"({largest_food_percentage:.2f}% of image)"
    )

else:
    print("\nNo food detected.")


# ============================================================
# Estimate calories and protein
# ============================================================

total_calories = 0
total_protein = 0.0

MIN_PIXEL_PERCENTAGE = 2.0
MIN_CONFIDENCE = 0.30


print("\nNutrition estimates:")

for class_id, food_info in detected_foods.items():

    # Ignore background
    if class_id == 0:
        continue

    food_name = food_info["name"]
    pixel_count = food_info["pixel_count"]

    pixel_percentage = (
        pixel_count / total_pixels
    ) * 100

    # Create mask using numeric class ID
    food_mask = prediction == class_id

    # Calculate average confidence for this class
    if pixel_count > 0:
        average_confidence = np.mean(
            confidence_map[food_mask]
        )
    else:
        average_confidence = 0.0

    # Ignore small or low-confidence detections
    if (
        pixel_percentage < MIN_PIXEL_PERCENTAGE
        or average_confidence < MIN_CONFIDENCE
    ):
        print(
            f"\nIgnoring weak detection: "
            f"{food_name}"
        )

        print(
            f"  Image coverage: "
            f"{pixel_percentage:.2f}%"
        )

        print(
            f"  Average confidence: "
            f"{average_confidence * 100:.2f}%"
        )

        continue

    # Skip classes that are not yet included in the label map
    if food_name.startswith("unknown_class_"):
        print(
            f"\nSkipping unknown class: "
            f"{food_name}"
        )
        continue

    # Estimate food weight
    estimated_weight = estimate_food_weight(
        food_name,
        pixel_percentage
    )

    # Get nutrition information
    nutrition = get_nutrition(food_name)

    if nutrition is None:
        print(
            f"\nNo nutrition data available "
            f"for {food_name}"
        )
        continue

    # Calculate calories
    calories = (
        estimated_weight / 100
    ) * nutrition["calories_per_100g"]

    # Calculate protein
    protein = (
        estimated_weight / 100
    ) * nutrition["protein_per_100g"]

    calories = round(calories)
    protein = round(protein, 1)

    total_calories += calories
    total_protein += protein

    print(f"\nFood: {food_name}")

    print(
        f"  Class ID: "
        f"{class_id}"
    )

    print(
        f"  Image coverage: "
        f"{pixel_percentage:.2f}%"
    )

    print(
        f"  Average confidence: "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"  Estimated weight: "
        f"{estimated_weight:.1f} g"
    )

    print(
        f"  Calories: "
        f"{calories} kcal"
    )

    print(
        f"  Protein: "
        f"{protein} g"
    )


# ============================================================
# Total nutrition
# ============================================================

print("\n" + "=" * 45)
print("TOTAL NUTRITION ESTIMATE")
print("=" * 45)

print(
    f"Total calories: "
    f"{total_calories} kcal"
)

print(
    f"Total protein: "
    f"{total_protein:.1f} g"
)

print(
    "\nNote: These are approximate estimates. "
    "Accuracy depends on segmentation quality, "
    "portion estimation, and nutrition data."
)


# ============================================================
# Steak-specific mask analysis
# ============================================================

STEAK_CLASS_ID = 46

steak_mask = (
    prediction == STEAK_CLASS_ID
).astype(np.uint8)

steak_pixel_count = np.sum(steak_mask)

steak_percentage = (
    steak_pixel_count / total_pixels
) * 100

print("\nSteak mask analysis:")

print(
    f"Steak pixels: "
    f"{steak_pixel_count}"
)

print(
    f"Steak image coverage: "
    f"{steak_percentage:.2f}%"
)


# ============================================================
# Visualization
# ============================================================

plt.figure(figsize=(16, 5))


# Original image
plt.subplot(1, 4, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")


# Predicted segmentation
plt.subplot(1, 4, 2)
plt.imshow(prediction, cmap="tab20")
plt.title("Predicted Segmentation")
plt.axis("off")


# Steak mask
plt.subplot(1, 4, 3)
plt.imshow(steak_mask, cmap="gray")
plt.title("Steak Mask")
plt.axis("off")


# Segmentation overlay
resized_original = original_image.resize(
    (256, 256)
)

plt.subplot(1, 4, 4)
plt.imshow(resized_original)
plt.imshow(
    prediction,
    cmap="tab20",
    alpha=0.45
)
plt.title("Segmentation Overlay")
plt.axis("off")


plt.tight_layout()
plt.savefig(
    "segmentation_result.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()