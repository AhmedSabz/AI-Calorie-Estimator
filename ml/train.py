import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from prepare_dataset import FoodSegDataset
from model import create_model


#Configuration


BATCH_SIZE = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1

#Start with a small subset while testing the pipeline

MAX_TRAIN_IMAGES = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load dataset


print("Loading FoodSeg103...")

dataset = load_dataset("EduardoPacheco/FoodSeg103")

train_dataset = FoodSegDataset(
    dataset=dataset["train"]
)
#Use only a small number of images for our first test

train_dataset = torch.utils.data.Subset(
train_dataset,
range(min(MAX_TRAIN_IMAGES, len(train_dataset)))
)

train_loader = DataLoader(
train_dataset,
batch_size=BATCH_SIZE,
shuffle=True,
num_workers=0
)

print(f"Training images used: {len(train_dataset)}")
print(f"Batch size: {BATCH_SIZE}")


#Create model


print("\nCreating model...")

model = create_model()
model = model.to(DEVICE)

print(f"Using device: {DEVICE}")


#Loss function


criterion = nn.CrossEntropyLoss()


#Optimizer


optimizer = torch.optim.Adam(
model.parameters(),
lr=LEARNING_RATE
)

#Training Loop
print("\nStarting training...")

model.train()

for epoch in range(NUM_EPOCHS):

   running_loss = 0.0

   for batch_index, (images, masks) in enumerate(train_loader):

    # Move data to CPU/GPU
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

    # Clear previous gradients
      optimizer.zero_grad()

    # Forward pass
      outputs = model(images)

      predictions = outputs["out"]

    # Calculate loss
      loss = criterion(predictions, masks)

    # Backpropagation
      loss.backward()

    # Update model weights
      optimizer.step()

      running_loss += loss.item()

    # Print progress every 10 batches
      if (batch_index + 1) % 10 == 0:

          print(
              f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Batch [{batch_index + 1}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f}"
          )

   average_loss = running_loss / len(train_loader)

   print(
    f"\nEpoch {epoch + 1} complete. "
    f"Average Loss: {average_loss:.4f}"
   )

print("\nTraining test completed successfully!")