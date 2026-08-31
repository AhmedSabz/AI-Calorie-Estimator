# AI Calorie Estimator

An AI-powered computer vision project that analyzes an image of a food plate, identifies and segments individual food items, and estimates the total calorie content of the meal.

The project uses deep learning for food segmentation and is designed to eventually combine computer vision with nutritional data to produce an estimated calorie count from a single image.

## 🚧 Project Status

**Current stage: Deep learning training pipeline**

The current implementation successfully:

* Loads the FoodSeg103 dataset
* Processes food images and segmentation masks
* Converts images into PyTorch tensors
* Resizes images and masks to 256×256
* Loads a pretrained DeepLabV3 segmentation model
* Configures the model for 104 food categories
* Trains the model using Cross-Entropy Loss
* Uses Adam optimization
* Successfully completes training on a test subset of the dataset

The calorie estimation and user-facing application are still under development.

---

## 🎯 Project Goal

The ultimate goal is to create a system that can take an image such as:

```text
        📷 Food Plate
             ↓
     Food Segmentation
             ↓
    Identify Food Items
             ↓
    Estimate Portions
             ↓
     Nutrition Database
             ↓
      Calorie Estimate
             ↓
       ~650 Calories
```

For example, given a plate containing rice, chicken, and vegetables, the system should identify each food item and estimate the total calories.

---

## 🧠 Machine Learning Approach

The project uses **semantic segmentation** to identify which pixels in an image belong to different food categories.

### Model

The current segmentation model is:

**DeepLabV3 + ResNet-50**

The model has been modified to output predictions for:

```text
104 classes
```

including:

* Background
* Chicken
* Rice
* Vegetables
* Fruits
* Desserts
* Other food categories

---

## 📊 Dataset

The project uses **FoodSeg103**, a food image segmentation dataset containing 103 food categories plus background.

Dataset:

[FoodSeg103 on Hugging Face](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103)

Current dataset size:

| Split      | Images |
| ---------- | -----: |
| Training   |  4,983 |
| Validation |  2,135 |
| Total      |  7,118 |

Each example contains:

* Food image
* Segmentation mask
* Food class IDs
* Image ID

The dataset contains **104 total segmentation labels**, with IDs ranging from `0–103`.

---

## 🛠️ Technologies

### Programming

* Python
* PyTorch

### Machine Learning

* DeepLabV3
* ResNet-50
* Semantic Segmentation
* Cross-Entropy Loss
* Adam Optimizer

### Computer Vision

* Torchvision
* Pillow
* NumPy

### Data

* Hugging Face Datasets
* FoodSeg103

### Development

* Visual Studio Code
* Git
* GitHub
* Python Virtual Environment

---

## 📁 Project Structure

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
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Calorie-Estimator
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\activate
```

If PowerShell blocks script execution, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download the Dataset

The project uses the FoodSeg103 dataset through Hugging Face.

Run:

```bash
python datasets/download_foodseg.py
```

The dataset contains approximately 1.25 GB of downloaded data.

---

## 🔍 Inspect the Dataset

To inspect an example image and segmentation mask:

```bash
python datasets/inspect_foodseg.py
```

To verify the segmentation labels:

```bash
python datasets/check_labels.py
```

The expected label range is:

```text
0–103
```

---

## 🧹 Data Preparation

The dataset pipeline performs several preprocessing steps.

### Image preprocessing

Images are:

1. Converted to RGB
2. Resized to `256×256`
3. Converted to PyTorch tensors

### Segmentation mask preprocessing

Masks are:

1. Resized to `256×256`
2. Resized using nearest-neighbor interpolation
3. Converted to integer PyTorch tensors

Nearest-neighbor interpolation is important because segmentation masks contain categorical class IDs rather than continuous pixel values.

---

## 🧪 Testing the Dataset Pipeline

Run:

```bash
python ml/prepare_dataset.py
```

A successful pipeline should produce output similar to:

```text
Training images: 4983
Validation images: 2135

Image batch shape:
torch.Size([4, 3, 256, 256])

Mask batch shape:
torch.Size([4, 256, 256])

Mask data type:
torch.int64

Dataset pipeline is working!
```

---

## 🤖 Model

The project uses DeepLabV3 for semantic segmentation.

The model is configured to produce:

```text
[batch_size, 104, 256, 256]
```

For example:

```text
[1, 104, 256, 256]
```

means:

* `1` → one input image
* `104` → 104 possible food classes
* `256×256` → prediction for every pixel

---

## 🏋️ Training

The initial training pipeline uses a small subset of the dataset to verify that the entire training process works.

Current test configuration:

```text
Training images: 100
Batch size: 2
Epochs: 1
Learning rate: 0.0001
Device: CPU
```

Run:

```bash
python ml/train.py
```

The current pipeline successfully performs:

```text
Image
 ↓
DataLoader
 ↓
DeepLabV3
 ↓
Predictions
 ↓
Cross-Entropy Loss
 ↓
Backpropagation
 ↓
Adam Optimizer
 ↓
Updated Model
```

Example test result:

```text
Epoch [1/1] Batch [10/50] Loss: 4.6495
Epoch [1/1] Batch [20/50] Loss: 4.2496
Epoch [1/1] Batch [30/50] Loss: 4.0204
Epoch [1/1] Batch [40/50] Loss: 4.0850
Epoch [1/1] Batch [50/50] Loss: 3.6244

Epoch 1 complete.
Average Loss: 4.0967
```

---

## 💻 Hardware

The project is designed to work without a dedicated GPU.

The current training pipeline automatically selects:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Therefore:

* NVIDIA GPU available → CUDA is used
* No GPU → CPU is used

The initial development and testing has been performed using **CPU training**.

For larger training runs, a cloud GPU can be used to significantly reduce training time.

---

## 🗺️ Roadmap

### Phase 1 — Dataset & Preprocessing

* [x] Download FoodSeg103
* [x] Inspect dataset
* [x] Extract category mappings
* [x] Verify segmentation labels
* [x] Build PyTorch Dataset
* [x] Build DataLoader
* [x] Preprocess images
* [x] Preprocess segmentation masks

### Phase 2 — Deep Learning Model

* [x] Create DeepLabV3 model
* [x] Configure 104 output classes
* [x] Test model inference
* [x] Implement training loop
* [x] Verify forward pass
* [x] Implement loss calculation
* [x] Implement backpropagation
* [x] Implement optimizer updates
* [ ] Train on full dataset
* [ ] Add validation loop
* [ ] Calculate IoU
* [ ] Calculate pixel accuracy
* [ ] Save best model

### Phase 3 — Food Recognition

* [ ] Test model on new food images
* [ ] Generate segmentation masks
* [ ] Identify individual food items
* [ ] Visualize segmentation results
* [ ] Improve model accuracy

### Phase 4 — Calorie Estimation

* [ ] Estimate food portion sizes
* [ ] Create nutrition database
* [ ] Map detected foods to nutritional information
* [ ] Estimate calories per food item
* [ ] Calculate total meal calories

### Phase 5 — Application

* [ ] Build image upload interface
* [ ] Display detected foods
* [ ] Display segmentation results
* [ ] Display estimated calories
* [ ] Add nutritional breakdown
* [ ] Deploy application

---

## ⚠️ Limitations

Calorie estimation from a single image is inherently difficult.

The project will need to account for factors such as:

* Food portion size
* Food density
* Ingredients
* Cooking methods
* Hidden ingredients
* Camera angle
* Lighting
* Overlapping food items

Therefore, the final calorie value should be treated as an **estimate rather than an exact measurement**.

---

## 🚀 Future Improvements

Potential improvements include:

* Better portion-size estimation
* Depth information from images
* Object detection combined with segmentation
* More food categories
* Larger nutrition databases
* User-provided portion information
* Confidence scores
* Mobile application
* Real-time inference

---

## 📌 Current Achievement

The project has successfully progressed from raw FoodSeg103 data to a working deep-learning training pipeline.

The current pipeline can:

```text
FoodSeg103
    ↓
Preprocessing
    ↓
PyTorch DataLoader
    ↓
DeepLabV3
    ↓
104-Class Segmentation
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Model Updates
```

The next major milestone is **training and evaluating the segmentation model on the full dataset** before building the calorie-estimation component.

---

## 👨‍💻 Author

**Ahmed Sabzwari**

Computer Engineering — San José State University

This project was created as a computer vision/deep learning portfolio project focused on applying semantic segmentation to real-world food analysis.
