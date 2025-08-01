[فارسی (Persian)](README.fa.md)

# EEG Signal Analysis for ADHD Classification

This project is dedicated to the analysis and classification of Electroencephalogram (EEG) signals to diagnose Attention-Deficit/Hyperactivity Disorder (ADHD) in children. It employs a combination of classical machine learning algorithms and deep learning models to compare their performance and achieve the highest possible classification accuracy.

---

## 📑 Table of Contents

- [Project Introduction](#-project-introduction)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Feature Extraction](#2-feature-extraction)
  - [Classification Models](#3-classification-models)
- [How to Use](#-how-to-use)
  - [Prerequisites](#prerequisites)
  - [Folder Structure](#folder-structure)
  - [Running the Code](#running-the-code)
- [Results](#-results)
- [Visualization](#-visualization)

---

## 📖 Project Introduction

The primary goal of this project is to develop an intelligent model capable of distinguishing between children with ADHD and a healthy control group based on their recorded EEG signals. The project pipeline covers various stages, including signal cleaning, extraction of features related to brain frequency bands, and finally, training and evaluating different classification models.

---

## 📊 Dataset

The dataset used in this project is available from IEEE DataPort and includes EEG signals from 60 children with ADHD and 60 healthy children (control group).

- **Dataset Link:** [EEG Data for ADHD/Control Children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

The dataset consists of 19-channel EEG signals recorded at a sampling frequency of 128 Hz.

---

## 🛠️ Methodology

The project workflow is divided into three main stages:

### 1. Data Preprocessing

- **Data Loading:** EEG signals are read from `.mat` files.
- **Artifact Removal:** **Independent Component Analysis (ICA)** is used to identify and remove artifacts caused by eye movements (EOG). This step is crucial for cleaning the signals and improving data quality.
- **Filtering:** A band-pass filter is applied to limit the signal frequencies to the 1-40 Hz range.
- **Epoching:** Continuous signals are segmented into 2-second epochs to prepare them for feature extraction and model input.

### 2. Feature Extraction

For classical machine learning models, power-based features are extracted from the signals:

- **Power Spectral Density (PSD) Estimation:** The Welch's method is used to calculate the PSD for each epoch.
- **Frequency Band Features:** The relative power is calculated for different frequency bands:
  - **Delta:** 1-4 Hz
  - **Theta:** 4-8 Hz
  - **Alpha:** 8-13 Hz
  - **Beta:** 13-30 Hz
  - **Gamma:** 30-40 Hz
- **Theta/Beta Ratio (TBR):** This ratio, a significant biomarker for ADHD, is computed for each channel.

### 3. Classification Models

Two categories of models are used and compared for classification:

#### Classical Machine Learning Algorithms

These models are trained on the features extracted in the previous step, and their performance is evaluated using 5-Fold Cross-Validation:
- **Support Vector Machine (SVM)** with an RBF kernel
- **Random Forest**
- **XGBoost**

#### Deep Learning Models

These models are trained directly on the raw EEG data (2-second epochs):
- **EEGNet:** A compact convolutional neural network specifically designed for EEG signal classification.
- **DeepConvNet:** A deep convolutional network for learning complex features from the signals.
- **LSTM:** A recurrent neural network for analyzing temporal patterns in EEG signals.

---

## ⚙️ How to Use

### Prerequisites

To run this project, the following libraries must be installed. You can install them using `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn mne tensorflow xgboost
```

### Folder Structure
Before running the code, download the dataset and place it in your desired directory. Then, set up the folder structure as follows:

```bash
/path/to/your/dataset/
├── ADHD/
│   ├── v1p.mat
│   ├── v3p.mat
│   └── ...
└── Control/
    ├── v107.mat
    ├── v108.mat
    └── ...
```
Next, update the BASE_DATA_PATH variable in the data loading cell to point to your dataset's path.

### Running the Code
Open the Final_project.ipynb file in a Jupyter Notebook or JupyterLab environment and run the cells sequentially. The code is structured in a step-by-step manner, with clear explanations for each section.

## 📈 Results
The model evaluation results, including Accuracy, Precision, Recall, and F1-Score, are displayed in the code's output. Additionally, Confusion Matrices for classical models and training history plots for deep learning models are generated to facilitate a more detailed performance analysis.

The project outputs, including plots and confusion matrices, will be saved in a directory named `project_results.`

## 🎨 Visualization
The project includes several visualizations to provide better insights into the data and results:

**PSD Plots:** The average Power Spectral Density is compared between the ADHD and control groups.

**Topomaps:** The spatial distribution of power in different frequency bands across the scalp is displayed for both groups.

**Feature Distribution:** Violin plots are used to show the distribution of the most discriminative features between the two groups.
