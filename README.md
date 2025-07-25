# ILD-USING-CNN
# Revolutionizing Interstitial Lung Disease Diagnosis with Deep Convolutional Neural Networks

## 📘 Project Overview

This project focuses on diagnosing Interstitial Lung Diseases (ILDs) using a Deep Convolutional Neural Network (CNN). It automates the classification of six lung tissue patterns from CT scan images to assist radiologists and improve accuracy in diagnosis through Computer-Aided Diagnosis (CAD) systems.

## 👨‍🔬 project developer 
- Swarna Manjunadh [21021A0506]  
- **Supervisor**: Dr. S. Chandra Sekhar, Assistant Professor, Dept. of CSE, UCEK (A), JNTUK

---

## 🗂️ Project Structure

ILD-Diagnosis-CNN/
│
├── pickle/ # Folder containing preprocessed .pkl datasets and saved models
│ ├── X_train.pkl
│ ├── y_train.pkl
│ └── ...
│
├── src/ # Source code directory
│ ├── data_loader.py # Functions for loading and preprocessing data
│ ├── model_builder.py # CNN/LeNet/AlexNet/VGG architectures
│ ├── train.py # Model training pipeline
│ ├── evaluate.py # Evaluation and metrics
│ └── predict.py # Model prediction logic
│
├── output/ # Results (logs, predictions, confusion matrix)
│
├── README.md # This file
└── requirements.txt # Python dependencies
---

## 🧪 Dataset Details

- 12,500 image patches from 109 CT scans
- Classes:
  - Healthy
  - Ground Glass Opacity (GGO)
  - Micronodules
  - Consolidation
  - Reticulation
  - Fibrosis
- Data Source: Public CT image databases from Swiss University Hospitals

---

## 🧰 Technologies Used

- Python 3.9
- Jupyter Notebook
- Keras + TensorFlow
- OpenCV
- Scikit-learn
- NumPy, Pickle

---

## 🧠 CNN Model Summary

- CNN with 5, 7, and 9 layers tested
- Activation: LeakyReLU
- Output: Softmax classifier
- Optimizer: Adam
- Regularization: Dropout layers
- Models Compared: Custom CNN vs. LeNet, AlexNet, VGG19

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Best Accuracy:** 72% using AlexNet configuration  
**Best F1-Score:** ~0.76 (Micronodules class)

---

## 🚀 How to Run

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure `.pkl` data files are in `/pickle/` directory**

3. **Train the model**
   ```bash
   python src/train.py
   ```

4. **Evaluate the model**
   ```bash
   python src/evaluate.py
   ```

5. **Make predictions**
   ```bash
   python src/predict.py
   ```

---

## 🔬 Results Snapshot

| Layer Configuration | Accuracy | F1-Score (Macro Avg) |
|---------------------|----------|-----------------------|
| 5 Layers            | 67.6%    | 54.0%                |
| 7 Layers            | 73.7%    | 55.2%                |
| 9 Layers            | 71.2%    | 53.9%                |
| AlexNet             | 72.0%    | 60.0%                |


---
## Requirements.txt
numpy==1.24.3
opencv-python==4.8.0.76
scikit-learn==1.2.2
keras==2.11.0
tensorflow==2.11.0
matplotlib==3.7.1
pandas==1.5.3
pickle-mixin==1.0.2
jupyter==1.0.0


## 🛣️ Future Scope

- Use of 3D CNNs for full-volume lung CT scan analysis
- Improve class balance using synthetic data or augmentation
- Real-time integration into hospital diagnostic tools
- Visual explanation using Grad-CAM or similar techniques

---

## 📄 License

Developed as a final year B.Tech project under the Department of Computer Science and Engineering, UCEK (A), JNTUK. Not for commercial use without permission.


