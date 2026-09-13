# AI Calorie Estimator

An image-based food segmentation and nutrition estimation system built with **Python, PyTorch, and DeepLabV3**.

The project analyzes an image of plated food, identifies food categories using semantic segmentation, estimates the approximate portion size, and calculates estimated calories and protein.

> **Note:** This project is an experimental machine learning prototype. Nutrition values are approximate and should not be treated as medically accurate.

---

## Features

- Food image classification through semantic segmentation
- Pixel-level food detection
- Food category identification
- Confidence-based filtering of weak detections
- Approximate food portion estimation
- Calorie estimation
- Protein estimation
- Food segmentation visualization
- CPU inference support
- Optional GPU acceleration with CUDA

---

## Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Matplotlib**
- **Pillow**
- **Hugging Face Datasets**
- **DeepLabV3**
- **ResNet-50**
- **FoodSeg103 Dataset**

---

## Project Architecture

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
│   ├── portion_estimator.py
│   ├── general_portion_estimator.py
│   ├── calorie_estimator.py
│   └── nutrition_database.py
│
├── models/
│   └── foodseg_model_best.pth
│
├── images.jpg
├── README.md
└── .gitignore
```

---

## How It Works

The system follows this pipeline:

```text
Food Image
    |
    v
Image Preprocessing
    |
    v
DeepLabV3 Semantic Segmentation Model
    |
    v
Pixel-Level Food Predictions
    |
    v
Confidence Filtering
    |
    v
Food Portion Estimation
    |
    v
Nutrition Database Lookup
    |
    v
Estimated Calories and Protein
```

---

## Dataset

This project uses the **FoodSeg103** dataset.

FoodSeg103 contains food images with pixel-level semantic segmentation labels for 103 food categories plus background.

### Dataset Information

- Training images: 4,983
- Validation images: 2,135
- Food categories: 103
- Total segmentation classes: 104 including background
- Input image size: `256 × 256`

Dataset source:

[FoodSeg103 on Hugging Face](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103)

---

## Model

The segmentation model is based on:

```text
DeepLabV3
    |
    └── ResNet-50 Backbone
```

The model produces a segmentation map with 104 output classes:

```text
[background + 103 food categories]
```

Each pixel is assigned a predicted food category. The model output is converted into class probabilities using:

```python
torch.softmax(output, dim=1)
```

The predicted class for each pixel is selected using:

```python
torch.argmax(probabilities, dim=1)
```

---

## Training

The model was trained using:

- Optimizer: Adam
- Learning rate: `0.0001`
- Batch size: `2`
- Image size: `256 × 256`
- Loss function: Cross-Entropy Loss
- Device: CUDA GPU during training
- Architecture: DeepLabV3 with ResNet-50

### Example Training Configuration

```python
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
EPOCHS = 5
IMAGE_SIZE = 256
```

The best model checkpoint is saved as:

```text
models/foodseg_model_best.pth
```

---

## Model Evaluation

The current experimental model achieved approximately:

| Metric | Result |
|---|---:|
| Pixel Accuracy | 67.41% |
| Mean IoU | 11.87% |

These results show that the model can identify dominant food regions, but the segmentation quality still needs improvement before production use.

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

On Windows:

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```powershell
pip install torch torchvision numpy matplotlib pillow datasets
```

---

## Running Inference

Place an input food image in the project root and name it:

```text
images.jpg
```

Then run:

```powershell
python .\ml\inference.py
```

The script will:

1. Load the trained model.
2. Load and preprocess the image.
3. Predict food classes for each pixel.
4. Filter small and low-confidence detections.
5. Estimate food weight.
6. Calculate calories and protein.
7. Display segmentation visualizations.

---

## Example Output

For an image containing steak, the system produced an output similar to:

```text
Detected food classes:
background: 47.87% of image
steak: 46.89% of image
banana: 0.57% of image
bread: 0.84% of image
potato: 1.16% of image

Largest detected food:
steak (46.89% of image)

Food: steak
Class ID: 46
Image coverage: 46.89%
Average confidence: 41.54%
Estimated weight: 300.0 g
Calories: 750 kcal
Protein: 78.0 g

TOTAL NUTRITION ESTIMATE
Total calories: 750 kcal
Total protein: 78.0 g
```

---

## Confidence Filtering

The inference pipeline filters out detections that are:

- Too small in the image
- Below the minimum confidence threshold

Current configuration:

```python
MIN_PIXEL_PERCENTAGE = 2.0
MIN_CONFIDENCE = 0.30
```

This helps reduce false detections from small regions such as background noise or incorrectly predicted food categories.

---

## Portion Estimation

The current portion estimator uses the percentage of the image occupied by a detected food class to estimate approximate food weight.

For example:

```text
Detected food coverage → Estimated food weight
```

This is a heuristic approach and does not directly measure physical weight.

Accuracy can be affected by:

- Camera distance
- Camera angle
- Plate size
- Food thickness
- Lighting
- Image resolution
- Food overlap
- Segmentation accuracy

A future version could improve portion estimation using depth estimation, reference objects, or a calibrated camera setup.

---

## Nutrition Estimation

The system uses a nutrition database containing values per 100 grams.

The calculation is:

```text
Estimated Calories =
    Estimated Weight / 100 × Calories per 100g
```

```text
Estimated Protein =
    Estimated Weight / 100 × Protein per 100g
```

Example:

```text
300g steak
250 calories per 100g
26g protein per 100g
```

Produces approximately:

```text
750 calories
78g protein
```

---

## Limitations

This project is currently a research and portfolio prototype.

Known limitations include:

- Segmentation accuracy is still limited.
- The model may detect incorrect food categories.
- Confidence scores are not guaranteed to represent true probabilities.
- Portion size is estimated from 2D image coverage.
- Multiple foods may overlap in the image.
- Food preparation methods can change nutritional values.
- The nutrition database may not contain every detected food.
- Results should not be used for medical or dietary decisions.

---

## Future Improvements

Planned improvements include:

- Improve mean Intersection over Union
- Train for more epochs
- Add stronger data augmentation
- Use a learning-rate scheduler
- Experiment with alternative segmentation architectures
- Improve small-object detection
- Add more nutrition database entries
- Improve portion estimation
- Add food detection bounding boxes
- Build a web interface using FastAPI
- Build a mobile interface using Flutter
- Add image upload functionality
- Add meal history and tracking
- Support multiple food items in one image
- Deploy GPU inference
- Add model confidence visualization
- Compare predictions against manually labeled masks

---

## Example Use Cases

- Food image analysis
- Approximate calorie tracking
- Protein estimation
- Computer vision experimentation
- Semantic segmentation research
- Machine learning portfolio development
- Nutrition estimation prototypes

---

## Disclaimer

This application provides approximate estimates for educational and experimental purposes only.

It does not replace professional nutritional advice, medical guidance, or a certified food scale.

---

## Author

**Ahmed Sabzwari**

M.S. Computer Engineering  
San Jose State University

GitHub: [AhmedSabz](https://github.com/AhmedSabz)
