import numpy as np
from datasets import load_dataset

print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

train_dataset = dataset["train"]

print(f"Training images: {len(train_dataset)}")

print("\nChecking mask labels...")

all_labels = set()

# Check every training mask
for i in range(len(train_dataset)):

    mask = train_dataset[i]["label"]

    mask_array = np.array(mask)

    unique_labels = np.unique(mask_array)

    all_labels.update(unique_labels.tolist())

    if (i + 1) % 500 == 0:
        print(f"Checked {i + 1} images...")

print("\nUnique mask labels found:")

sorted_labels = sorted(all_labels)

print(sorted_labels)

print("\nNumber of unique labels:")
print(len(sorted_labels))

print("\nMinimum label:")
print(min(sorted_labels))

print("\nMaximum label:")
print(max(sorted_labels))