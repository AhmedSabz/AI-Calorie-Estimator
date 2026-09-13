import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from PIL import Image
from torchvision import transforms

from general_portion_estimator import estimate_food_weight
from nutrition_database import get_nutrition
from model import create_model


# ==============================
# Configuration
# ==============================

MODEL_PATH = "models/foodseg_model_best.pth"
IMAGE_PATH = "images.jpg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# FoodSeg103 Labels
# ==============================

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


# ==============================
# Load model
# ==============================

print("Loading model...")

model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print(f"Using device: {DEVICE}")
print("Model loaded successfully!")


# ==============================
# Load image
# ==============================

print("\nLoading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"Original image size: {image.size}")


# ==============================
# Preprocess image
# ==============================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = transform(image)

# Add batch dimension
input_tensor = input_tensor.unsqueeze(0)

# Move image to CPU/GPU
input_tensor = input_tensor.to(DEVICE)

print(f"Input tensor shape: {input_tensor.shape}")


# ==============================
# Run inference
# ==============================

print("\nRunning inference...")

with torch.no_grad():

    output = model(input_tensor)["out"]

    prediction = torch.argmax(
        output,
        dim=1
    )


# Remove batch dimension
prediction = prediction.squeeze(0)

# Move prediction to CPU
prediction = prediction.cpu().numpy()


# ==============================
# Analyze prediction
# ==============================

unique_classes, pixel_counts = np.unique(
    prediction,
    return_counts=True
)

total_pixels = prediction.size

# Store detected foods and pixel counts
detected_foods = {}

print("\n==============================")
print("Detected Foods")
print("==============================")

for class_id, count in zip(
    unique_classes,
    pixel_counts
):

    percentage = (
        count / total_pixels
    ) * 100

    food_name = ID_TO_LABEL.get(
        int(class_id),
        "unknown"
    )

    detected_foods[food_name] = int(count)

    print(
        f"{food_name}: "
        f"{percentage:.2f}% "
        f"({count} pixels)"
    )


# ==============================
# Main detected food
# ==============================

non_background = [
    (class_id, count)
    for class_id, count in zip(
        unique_classes,
        pixel_counts
    )
    if class_id != 0
]

if non_background:

    main_class, main_count = max(
        non_background,
        key=lambda x: x[1]
    )

    main_food = ID_TO_LABEL.get(
        int(main_class),
        "unknown"
    )

    main_percentage = (
        main_count / total_pixels
    ) * 100

    print("\n==============================")
    print("MAIN DETECTION")
    print("==============================")

    print(f"Food: {main_food}")
    print(f"Pixel coverage: {main_percentage:.2f}%")

else:

    print("\nNo food detected.")


# ==============================
# Multi-food nutrition estimation
# ==============================

print("\n==============================")
print("MULTI-FOOD NUTRITION ESTIMATION")
print("==============================")

total_calories = 0
total_protein = 0

for food_name, pixel_count in detected_foods.items():

    # Ignore background
    if food_name == "background":
        continue

    pixel_percentage = (
        pixel_count / total_pixels
    ) * 100

    # Ignore very small detections
    # These may be false positives.
    if pixel_percentage < 2.0:

        print(
            f"\nIgnoring small detection: "
            f"{food_name} "
            f"({pixel_percentage:.2f}%)"
        )

        continue

    # Estimate portion size
    estimated_weight = estimate_food_weight(
        food_name,
        pixel_percentage
    )

    # Look up nutrition information
    nutrition = get_nutrition(food_name)

    if nutrition is None:

        print(
            f"\n{food_name}: "
            f"nutrition data unavailable"
        )

        continue

    calories = (
        estimated_weight / 100
    ) * nutrition["calories_per_100g"]

    protein = (
        estimated_weight / 100
    ) * nutrition["protein_per_100g"]

    calories = round(calories)
    protein = round(protein, 1)

    total_calories += calories
    total_protein += protein

    print(f"\nFood: {food_name}")
    print(f"Pixel coverage: {pixel_percentage:.2f}%")
    print(f"Estimated weight: {estimated_weight} g")
    print(f"Estimated calories: {calories} kcal")
    print(f"Estimated protein: {protein} g")


# ==============================
# Total nutrition
# ==============================

print("\n==============================")
print("TOTAL NUTRITION ESTIMATE")
print("==============================")

print(
    f"Total estimated calories: "
    f"{round(total_calories)} kcal"
)

print(
    f"Total estimated protein: "
    f"{round(total_protein, 1)} g"
)

print()
print("Note: Portion sizes and nutrition values")
print("are approximate heuristic estimates.")


# ==============================
# Create steak mask
# ==============================

STEAK_CLASS_ID = 46

steak_mask = (
    prediction == STEAK_CLASS_ID
).astype(np.uint8)

steak_pixels = np.sum(steak_mask)

steak_percentage = (
    steak_pixels / total_pixels
) * 100

print("\n==============================")
print("STEAK SEGMENTATION")
print("==============================")

print(f"Steak pixels: {steak_pixels}")
print(f"Total pixels: {total_pixels}")
print(f"Steak coverage: {steak_percentage:.2f}%")


# ==============================
# Display results
# ==============================

plt.figure(figsize=(20, 5))


# ------------------------------
# Original image
# ------------------------------

plt.subplot(1, 4, 1)

plt.imshow(image)

plt.title("Original Image")

plt.axis("off")


# ------------------------------
# Full segmentation
# ------------------------------

plt.subplot(1, 4, 2)

plt.imshow(prediction)

plt.title("Predicted Segmentation")

plt.axis("off")


# ------------------------------
# Steak mask
# ------------------------------

plt.subplot(1, 4, 3)

plt.imshow(
    steak_mask,
    cmap="gray"
)

plt.title("Steak Mask")

plt.axis("off")


# ------------------------------
# Steak mask overlay
# ------------------------------

plt.subplot(1, 4, 4)

resized_image = image.resize((256, 256))

plt.imshow(resized_image)

plt.imshow(
    steak_mask,
    cmap="gray",
    alpha=0.5
)

plt.title("Steak Mask Overlay")

plt.axis("off")


plt.tight_layout()

plt.show()