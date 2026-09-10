# AI Calorie Estimator

A computer vision and deep learning project that analyzes food images to identify and segment food items, with the goal of estimating portion size and calories.

The project uses **PyTorch, DeepLabV3, ResNet-50, and the FoodSeg103 dataset** to perform pixel-level food segmentation.

> **Current Status:** Food segmentation and inference are implemented. Portion estimation and calorie prediction are currently under development.

---

## Project Overview

The goal of this project is to build an end-to-end food analysis pipeline:

```text
Food Image
    ↓
Food Segmentation
    ↓
Food Identification
    ↓
Portion Estimation
    ↓
Nutrition Lookup
    ↓
Calorie Estimation
```

The current implementation focuses on **food segmentation, food identification, and inference**.

---

## Current Features

- Food image preprocessing using PyTorch and torchvision
- Semantic segmentation of food images
- DeepLabV3 with ResNet-50 backbone
- Support for 104 FoodSeg103 classes
- GPU training using NVIDIA CUDA
- Validation-based model evaluation
- Best-model checkpointing
- Food class identification from unseen images
- Binary masks for individual food categories
- Segmentation visualization and overlays

---

## Model

The project uses:

**DeepLabV3 + ResNet-50**

The model was modified to output predictions for all **104 FoodSeg103 classes**, including background.

### Input

```text
256 × 256 RGB image
```

### Output

```text
104 × 256 × 256 segmentation map
```

Each pixel is assigned a food category.

---

## Dataset

This project uses the **FoodSeg103** dataset.

### Dataset Statistics

- **7,118 total images**
- **4,983 training images**
- **2,135 validation images**
- **104 food categories**

Example categories include:

- Steak
- Chicken
- Rice
- Potato
- Bread
- Pizza
- Pasta
- Banana
- Strawberry
- Apple
- Fish
- Shrimp
- Vegetables
- and many others

### Dataset Source

**FoodSeg103**

`EduardoPacheco/FoodSeg103`

---

## Data Preprocessing

Images are resized to:

```text
256 × 256
```

and converted into PyTorch tensors.

Segmentation masks contain integer class IDs corresponding to the FoodSeg103 categories.

The preprocessing pipeline is:

```text
Image
   ↓
Resize to 256 × 256
   ↓
Convert to Tensor
   ↓
PyTorch DataLoader
   ↓
DeepLabV3
```

Training uses a batch size of **2** to reduce GPU memory requirements.

---

## Training

The model was trained using an **NVIDIA Tesla T4 GPU** through Google Colab.

### Training Configuration

```text
Model: DeepLabV3 + ResNet-50
Dataset: FoodSeg103
Image Size: 256 × 256
Batch Size: 2
Learning Rate: 0.0001
Epochs: 5
Optimizer: Adam
Device: NVIDIA Tesla T4
```

A validation set is evaluated after every epoch, and the model with the best Mean IoU is saved as the best checkpoint.

---

## Training Results

The model was trained for 5 epochs.

| Epoch | Training Loss | Validation Loss | Pixel Accuracy | Mean IoU |
|------:|--------------:|----------------:|---------------:|---------:|
| 1 | 2.0612 | 1.6185 | 60.75% | 4.63% |
| 2 | 1.6191 | 1.4388 | 63.91% | 7.36% |
| 3 | 1.4321 | 1.3962 | 64.38% | 8.88% |
| 4 | 1.3008 | 1.3134 | 66.31% | 10.80% |
| 5 | 1.1886 | 1.2526 | 67.41% | 11.87% |

### Best Model

```text
Pixel Accuracy: 67.41%
Mean IoU: 11.87%
```

The current model is considered a **prototype segmentation model** rather than a production-quality system. Further training and model improvements are planned.

---

## Inference

The project includes an inference pipeline that loads a trained model and analyzes a new food image.

Run:

```bash
python ml/inference.py
```

The inference pipeline:

1. Loads the trained DeepLabV3 model
2. Loads an input food image
3. Resizes the image to 256 × 256
4. Converts the image to a PyTorch tensor
5. Runs the image through the model
6. Generates a pixel-level segmentation
7. Identifies detected food categories
8. Calculates pixel coverage
9. Generates binary masks for individual foods
10. Visualizes the segmentation results

---

## Example Inference

The model was tested on an unseen steak image.

The model identified **steak** as the primary food class.

Example output:

```text
==============================
MAIN DETECTION
==============================

Food: steak
Pixel coverage: 46.89%
```

The model also generated a binary steak segmentation mask:

```text
Steak      → White
Everything → Black
```

This binary mask provides the foundation for future portion-size estimation.

---

## Portion Estimation

The next stage of the project is estimating the physical portion size of detected food.

The current approach uses image-derived food area as a **heuristic estimate**.

The planned pipeline is:

```text
Steak Image
     ↓
Steak Segmentation
     ↓
Steak Pixel Area
     ↓
Portion Estimation
     ↓
Estimated Weight
```

### Important Limitation

Pixel coverage does **not** directly correspond to physical weight.

Factors such as:

- Camera distance
- Viewing angle
- Perspective
- Food thickness
- Image composition
- Size of the food relative to the camera

can significantly affect the relationship between pixels and actual weight.

Therefore, portion estimates are treated as approximations rather than direct measurements.

Future versions may investigate more reliable approaches such as reference objects, additional visual cues, or learned portion-estimation models.

---

## Calorie Estimation

The final goal is to connect the detected food and estimated portion size to nutritional information.

The planned pipeline is:

```text
Food Image
     ↓
Food Segmentation
     ↓
Food Identification
     ↓
Portion Estimation
     ↓
Estimated Weight
     ↓
Nutrition Information
     ↓
Calories
```

For example:

```text
Steak
  ↓
Estimated portion
  ↓
Estimated grams
  ↓
Calories per 100 g
  ↓
Estimated total calories
```

This portion of the system is currently under development.

---

## Project Structure

```text
AI-Calorie-Estimator/
│
├── datasets/
│   ├── download_foodseg.py
│   ├── inspect_foodseg.py
│   ├── get_categories.py
│   └── check_labels.py
│
├── ml/
│   ├── prepare_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── portion_estimator.py
│
├── models/
│   └── foodseg_model_best.pth
│
├── .gitignore
├── README.md
└── images.jpg
```

> Model checkpoint files are excluded from Git using `.gitignore`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AhmedSabz/AI-Calorie-Estimator.git
```

### 2. Navigate into the project

```bash
cd AI-Calorie-Estimator
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install torch torchvision datasets pillow numpy matplotlib
```

---

## Running the Project

### Test the Dataset Pipeline

```bash
python ml/prepare_dataset.py
```

This verifies that images and segmentation masks can be loaded correctly.

---

### Test the Model

```bash
python ml/model.py
```

This verifies that the DeepLabV3 model produces the expected output dimensions.

---

### Train the Model

```bash
python ml/train.py
```

For GPU training, Google Colab with an NVIDIA GPU is recommended.

---

### Evaluate the Model

```bash
python ml/evaluate.py
```

This evaluates the trained model on the validation dataset.

---

### Run Inference

```bash
python ml/inference.py
```

This loads a trained model and performs food segmentation on an input image.

---

## Hardware

### Local Development

The project was developed locally on Windows using CPU-based development and inference.

### Model Training

Model training was performed using:

```text
GPU: NVIDIA Tesla T4
VRAM: 15 GB
Platform: Google Colab
```

GPU acceleration was used to make deep learning model training significantly more practical than CPU-only training.

---

## Roadmap

### Completed

- [x] FoodSeg103 dataset integration
- [x] Dataset inspection
- [x] Image preprocessing
- [x] Segmentation mask preprocessing
- [x] DeepLabV3 model
- [x] ResNet-50 backbone
- [x] GPU training
- [x] Validation pipeline
- [x] Best-model checkpointing
- [x] Model evaluation
- [x] Food segmentation inference
- [x] Food class identification
- [x] Binary food masks
- [x] Segmentation visualization

### In Progress

- [ ] Portion-size estimation
- [ ] Nutrition database integration
- [ ] Calorie estimation
- [ ] End-to-end calorie prediction pipeline

### Future Improvements

- [ ] Train for more epochs
- [ ] Improve Mean IoU
- [ ] Add data augmentation
- [ ] Experiment with different segmentation architectures
- [ ] Improve portion-size estimation
- [ ] Support multiple foods in a single meal
- [ ] Add confidence scores
- [ ] Build a web interface
- [ ] Deploy the model as an API
- [ ] Evaluate performance on additional real-world images
- [ ] Add automated nutrition lookup

---

## Limitations

The current model is a prototype.

### Segmentation

The model currently achieves:

```text
Pixel Accuracy: 67.41%
Mean IoU: 11.87%
```

The relatively low Mean IoU indicates that segmentation quality still has significant room for improvement.

Potential improvements include:

- More training epochs
- Data augmentation
- Improved preprocessing
- Hyperparameter tuning
- Different segmentation architectures
- Larger or more specialized training datasets

### Portion Estimation

Image pixel area does not directly provide food weight.

Portion estimates can be affected by:

- Camera distance
- Perspective
- Viewing angle
- Food thickness
- Lighting
- Image composition

Therefore, calorie estimates should be considered approximate.

---

## Technologies

### Programming

- Python

### Machine Learning

- PyTorch
- Torchvision
- DeepLabV3
- ResNet-50

### Computer Vision

- Semantic segmentation
- Pixel-level classification
- Image preprocessing
- Binary segmentation masks

### Data

- FoodSeg103
- Hugging Face Datasets
- NumPy
- Pillow

### Visualization

- Matplotlib

### Hardware / Infrastructure

- NVIDIA CUDA
- NVIDIA Tesla T4
- Google Colab

---

## Learning Objectives

This project is being developed to gain practical experience with:

- Deep learning
- Computer vision
- Semantic segmentation
- PyTorch model development
- GPU-accelerated training
- Dataset preprocessing
- Model evaluation
- Model inference
- Computer vision-based measurement
- End-to-end ML pipeline development

---

## Author

**Ahmed Sabzwari**

M.S. Computer Engineering  
San Jose State University

GitHub:

`https://github.com/AhmedSabz`

---

## Disclaimer

This project is an experimental computer vision and machine learning project.

Calorie and portion estimates are not intended to replace professional nutritional measurements or dietary advice.
