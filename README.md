# AI Calorie Estimator

A computer vision and deep learning project that analyzes food images to identify and segment food items, with the goal of estimating portion size and calories.

The project uses **PyTorch, DeepLabV3, and the FoodSeg103 dataset** to perform pixel-level food segmentation.

> **Current status:** Food segmentation and inference are implemented. Portion estimation and calorie prediction are currently under development.

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
