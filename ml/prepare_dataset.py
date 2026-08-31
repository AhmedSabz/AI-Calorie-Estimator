from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
#1. Load FoodSeg103

print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

train_data = dataset["train"]
validation_data = dataset["validation"]

print("Training images:", len(train_data))
print("Validation images:", len(validation_data))

#2. Image transformations
image_transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor()
])

#3. Create PyTorch Dataset
class FoodSegDataset(Dataset):

   def __init__(self, hf_dataset, transform=None):
      self.dataset = hf_dataset
      self.transform = transform

   def __len__(self):
      return len(self.dataset)

   def __getitem__(self, index):

      sample = self.dataset[index]

      image = sample["image"]
      mask = sample["label"]

    # Convert image to RGB
      image = image.convert("RGB")

    # Resize image
      if self.transform:
         image = self.transform(image)

    # Resize segmentation mask
      mask = mask.resize((256, 256))

    # Convert mask to tensor
      mask = torch.tensor(
        torch.tensor(
            list(mask.getdata())
        ).reshape(mask.height, mask.width),
        dtype=torch.long
    )

      return image, mask

#4. Create training and validation datasets
train_dataset = FoodSegDataset(
train_data,
transform=image_transform
)

validation_dataset = FoodSegDataset(
validation_data,
transform=image_transform
)

#6. Create DataLoaders
train_loader = DataLoader(
train_dataset,
batch_size=4,
shuffle=True
)

validation_loader = DataLoader(
validation_dataset,
batch_size=4,
shuffle=False
)

#7 test pipeline
print("\nTesting DataLoader...")

images, masks = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Mask batch shape:", masks.shape)
print("Mask data type:", masks.dtype)

print("\nDataset pipeline is working!")