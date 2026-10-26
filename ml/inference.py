
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "foodseg_model_best.pth"
IMAGE_PATH = PROJECT_ROOT / "images.jpg"
OUTPUT_PATH = PROJECT_ROOT / "segmentation_result.png"

IMAGE_SIZE = (256, 256)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FoodSeg103 class labels
# Official IDs: https://huggingface.co/datasets/EduardoPacheco/FoodSeg103
# ============================================================

ID_TO_LABEL = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients",
}


# ============================================================
# Load trained model
# ============================================================

print(f"Using device: {DEVICE}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}"
    )

model = create_model()

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Load and preprocess image
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")
original_image = image.copy()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0).to(DEVICE)


# ============================================================
# Run inference
# ============================================================

with torch.no_grad():
    output = model(input_tensor)["out"]

probabilities = torch.softmax(output, dim=1)

prediction = torch.argmax(
    probabilities,
    dim=1
)[0].cpu().numpy()

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
    pixel_percentage = (
        food_info["pixel_count"] / total_pixels
    ) * 100

    print(
        f'{food_info["name"]}: '
        f'{pixel_percentage:.2f}% of image'
    )


# ============================================================
# Find largest predicted non-background class
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

    largest_percentage = (
        non_background_foods[largest_class_id]["pixel_count"]
        / total_pixels
    ) * 100

    print("\nLargest predicted food class:")
    print(f"{largest_food} ({largest_percentage:.2f}% of image)")
else:
    print("\nNo non-background class predicted.")


# ============================================================
# Estimate calories and protein
# ============================================================

total_calories = 0
total_protein = 0.0

MIN_PIXEL_PERCENTAGE = 2.0
MIN_CONFIDENCE = 0.30

print("\nNutrition estimates:")

for class_id, food_info in detected_foods.items():

    if class_id == 0:
        continue

    food_name = food_info["name"]
    pixel_count = food_info["pixel_count"]

    pixel_percentage = (
        pixel_count / total_pixels
    ) * 100

    food_mask = prediction == class_id

    average_confidence = float(
        np.mean(confidence_map[food_mask])
    )

    if (
        pixel_percentage < MIN_PIXEL_PERCENTAGE
        or average_confidence < MIN_CONFIDENCE
    ):
        print(f"\nIgnoring weak detection: {food_name}")
        print(f"  Image coverage: {pixel_percentage:.2f}%")
        print(f"  Average confidence: {average_confidence * 100:.2f}%")
        continue

    if food_name.startswith("unknown_class_"):
        print(f"\nSkipping unknown class: {food_name}")
        continue

    # Estimate weight from image coverage.
    # This is a heuristic, not a measured portion weight.
    estimated_weight = estimate_food_weight(
        food_name,
        pixel_percentage
    )

    nutrition = get_nutrition(food_name)

    if nutrition is None:
        print(f"\nNo nutrition data available for {food_name}")
        continue

    calories = round(
        (estimated_weight / 100)
        * nutrition["calories_per_100g"]
    )

    protein = round(
        (estimated_weight / 100)
        * nutrition["protein_per_100g"],
        1
    )

    total_calories += calories
    total_protein += protein

    print(f"\nFood: {food_name}")
    print(f"  Class ID: {class_id}")
    print(f"  Image coverage: {pixel_percentage:.2f}%")
    print(f"  Average confidence: {average_confidence * 100:.2f}%")
    print(f"  Estimated weight: {estimated_weight:.1f} g")
    print(f"  Calories: {calories} kcal")
    print(f"  Protein: {protein} g")


# ============================================================
# Total nutrition
# ============================================================

print("\n" + "=" * 45)
print("TOTAL NUTRITION ESTIMATE")
print("=" * 45)

print(f"Total calories: {total_calories} kcal")
print(f"Total protein: {total_protein:.1f} g")

print(
    "\nNote: Estimates depend on segmentation quality, "
    "portion estimation, and nutrition data. "
    "Image coverage alone cannot reliably measure food weight."
)


# ============================================================
# Steak-specific mask analysis
# ============================================================

STEAK_CLASS_ID = 46

steak_mask = (
    prediction == STEAK_CLASS_ID
).astype(np.uint8)

steak_pixel_count = int(np.sum(steak_mask))

steak_percentage = (
    steak_pixel_count / total_pixels
) * 100

print("\nSteak mask analysis:")
print(f"Steak pixels: {steak_pixel_count}")
print(f"Steak image coverage: {steak_percentage:.2f}%")


# ============================================================
# Visualization
# ============================================================

# Keep image and mask dimensions consistent for overlay.
resized_original = original_image.resize(
    (prediction.shape[1], prediction.shape[0])
)

plt.figure(figsize=(16, 5))

# Original image at model input dimensions
plt.subplot(1, 4, 1)
plt.imshow(resized_original)
plt.title("Original Image")
plt.axis("off")

# Predicted segmentation
plt.subplot(1, 4, 2)
plt.imshow(prediction, cmap="tab20", vmin=0, vmax=103)
plt.title("Predicted Segmentation")
plt.axis("off")

# Steak mask
plt.subplot(1, 4, 3)
plt.imshow(steak_mask, cmap="gray", vmin=0, vmax=1)
plt.title("Steak Mask (ID 46)")
plt.axis("off")

# Segmentation overlay
plt.subplot(1, 4, 4)
plt.imshow(resized_original)
plt.imshow(
    prediction,
    cmap="tab20",
    vmin=0,
    vmax=103,
    alpha=0.45
)
plt.title("Segmentation Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print(f"\nSaved visualization to: {OUTPUT_PATH}")

plt.show()
