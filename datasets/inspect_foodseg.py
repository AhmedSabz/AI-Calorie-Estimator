from datasets import load_dataset

# Load FoodSeg103

dataset = load_dataset("EduardoPacheco/FoodSeg103")

# Get the first training example

sample = dataset["train"][0]

print("Image size:", sample["image"].size)
print("Mask size:", sample["label"].size)

print("\nClass IDs in first image:")
print(sample["classes_on_image"])

# Print all available dataset metadata

print("\nAvailable metadata:")
print(dataset["train"].info)
