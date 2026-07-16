# Multimodal Emotion Classification using SEED-IV Dataset

This project is a PyTorch implementation of a multimodal deep learning model for emotion classification using the **SEED-IV dataset**. It combines **EEG** and **Eye-Tracking** data to classify emotions into four categories.

I built this project to learn more about multimodal learning, Transformers, and attention mechanisms. The model processes both modalities separately and then combines them using a Cross-Attention layer for emotion prediction.

---

## Repository

```bash
git clone https://github.com/Suryakant2112/emotion-detection.git
cd emotion-detection
```

---

## Emotion Classes

The model predicts one of the following emotions:

- Neutral
- Sad
- Fear
- Happy

---

## Features

- Multimodal learning using EEG and Eye-Tracking data
- Separate feature extraction for each modality
- Cross-Attention based feature fusion
- Transformer Encoder for temporal feature learning
- Gaussian Noise and Temporal Masking for data augmentation
- Mixup augmentation during training
- Label Smoothing
- Gradient Clipping
- ReduceLROnPlateau Learning Rate Scheduler
- 5-Fold Cross Validation
- Automatic generation of learning curves, confusion matrix and evaluation reports

---

## Project Structure

```text
emotion-detection/
│
├── dataset.py         # Custom Dataset class
├── model.py           # Model architecture
├── preprocess.py      # Data preprocessing
├── train.py           # Training script
├── evaluate.py        # Model evaluation
├── summary.py         # Model summary
├── check_data.py      # Dataset validation
├── requirements.txt   # Required libraries
└── README.md
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/Suryakant2112/emotion-detection.git
cd emotion-detection
```

Install all required packages.

```bash
pip install -r requirements.txt
```

---

# Dataset

Download the **SEED-IV dataset** and extract it.

Update the dataset path inside the following files:

- `preprocess.py`
- `check_data.py`
- `train.py`
- `evaluate.py`

Example folder structure:

```text
Separated_Data/
├── Train_EEG/
├── Train_Eye_Tracking/
├── Validation_EEG/
├── Validation_Eye_Tracking/
├── Test_EEG/
└── Test_Eye_Tracking/
```

---

# Running the Project

## Step 1: Preprocess the Data

Normalize the EEG and Eye-Tracking data.

```bash
python preprocess.py
```

---

## Step 2: Validate the Dataset

Check whether every EEG sample correctly matches its corresponding Eye-Tracking sample.

```bash
python check_data.py
```

---

## Step 3: View Model Summary (Optional)

Print the model architecture and parameter count.

```bash
python summary.py
```

---

## Step 4: Train the Model

Train the model using 5-Fold Cross Validation.

```bash
python train.py
```

During training, the script automatically:

- Splits the dataset into 5 folds
- Trains for each fold
- Saves the best model
- Generates learning curves
- Stores fold-wise performance

---

## Step 5: Evaluate the Model

Evaluate the trained model.

```bash
python evaluate.py
```

The evaluation script generates:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- CSV Report

---

# Output Files

After training and evaluation, the following files are generated automatically.

```text
learning_curves_fold_1.png
learning_curves_fold_2.png
learning_curves_fold_3.png
learning_curves_fold_4.png
learning_curves_fold_5.png

kfold_variance_boxplot.png

confusion_matrix_heatmap.png

detailed_evaluation_metrics.csv

seed_iv_best_fold_1.pth
seed_iv_best_fold_2.pth
seed_iv_best_fold_3.pth
seed_iv_best_fold_4.pth
seed_iv_best_fold_5.pth
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm

---

# Model Overview

The model consists of:

- EEG Feature Encoder
- Eye-Tracking Feature Encoder
- Transformer Encoder
- Cross-Attention Fusion Layer
- Fully Connected Classification Head

The model is trained using Cross-Entropy Loss with Label Smoothing and Mixup augmentation.

---

# Future Improvements

Some ideas for improving the project:

- Experiment with different attention mechanisms
- Use larger Transformer architectures
- Try other multimodal fusion techniques
- Perform hyperparameter optimization
- Test on additional emotion recognition datasets

---

# Acknowledgements

This project uses the **SEED-IV Dataset** for multimodal emotion recognition research.

---
