# 🧵 High-Resolution Fabric Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Model](https://img.shields.io/badge/Model-EfficientNet--B0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This project tackles the challenge of classifying **high-resolution fabric textures** (2560x1440 px) into biodegradable categories such as **Wool, Cotton, Silk, Linen, etc.**

Standard Convolutional Neural Networks (CNNs) typically expect small inputs (e.g., 224x224). Resizing a 2560px image directly to 224px destroys the fine fiber details required to distinguish textures. To solve this, I implemented a **Patch-Based Learning** strategy using **EfficientNet-B0**.

## 🧠 Methodology: The Patch-Based Approach

Instead of resizing the entire image and losing detail, the pipeline focuses on "micro-textures":

1.  **Training Phase (Random Crop):**
    * Resize original image (2560px) to a still-large intermediate size (512px).
    * Take a **Random Crop** of 224x224.
    * *Result:* The model learns from different parts of the fabric in every epoch, effectively acting as data augmentation.

2.  **Inference Phase (Center Crop):**
    * Resize to 512px and take a **Center Crop** of 224x224 to evaluate the core texture.

## 📂 Dataset
The dataset used is the **Biodegradable Fabrics Dataset** from Kaggle/IEEE.
* **Classes:** Abaca, Cotton, Hessian, Linen, Silk, Wool.
* **Resolution:** 2560x1440.
* **Source:** [Kaggle Link](https://www.kaggle.com/datasets/jhamilgutierrez/biodegradable-fabrics-dataset)

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Model Architecture:** EfficientNet-B0 (Pre-trained on ImageNet via `timm`)
* **Data Processing:** Albumentations / Torch Transforms
* **Visualization:** Matplotlib, Seaborn

## 🚀 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone https://github.com/goktani/high-res-fabric-classification.git
    cd high-res-fabric-classification
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook**
    Open `fabric_classification.ipynb` in Jupyter Lab or VS Code to see the training pipeline.

## 📊 Results

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **99.44%** |
| **Training Loss** | 0.02 |

*The model successfully differentiates between visually similar textures (e.g., Silk vs. Cotton) thanks to the patch-based strategy.*

## 📜 Citation
If you use the dataset, please cite the original authors:
> J. G. Gutierrez and J. F. Villaverde, "Classification and Identification of Natural Biodegradable Fabrics using Convolutional Neural Network," 2025.

---
*Developed by [Goktan](https://github.com/goktani)*
