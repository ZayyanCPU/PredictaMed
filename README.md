<div align="center">

# 🏥 PredictaMed

### **AI Powered Multi Disease Prediction System**

*Leveraging Machine Learning to Enable Early Disease Detection and Classification*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

---

[Overview](#-overview) •
[Features](#-key-features) •
[Tech Stack](#-tech-stack) •
[Pipeline](#-analysis-pipeline) •
[Results](#-results--insights) •
[Getting Started](#-getting-started) •
[Structure](#-project-structure) •
[Skills](#-skills-demonstrated)

</div>

---

## 📋 Overview

**PredictaMed** is a comprehensive machine learning project that develops predictive models for early detection and classification of multiple diseases. By analyzing medical and demographic data from **59,068 patient records** with **18 health attributes**, this system evaluates six different ML algorithms to identify the most effective approach for each disease prediction task.

### 🎯 Diseases Analyzed
| Disease | Prevalence in Dataset | Best Model | Accuracy |
|---------|----------------------|------------|----------|
| Heart Disease | 46.34% | Random Forest | 72.98% |
| Stroke | 8.80% | Decision Tree | 91.14% |
| Diabetes | 21.29% | Random Forest | 80.39% |
| Asthma | 15.69% | Random Forest | 85.08% |
| Kidney Disease | 7.44% | Logistic Regression | 92.41% |
| Skin Cancer | 13.77% | Logistic Regression | 86.49% |

---

## ⭐ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi Disease Analysis** | Simultaneous prediction models for 6 different diseases using shared patient data |
| 📊 **Comprehensive EDA** | In depth exploratory analysis with correlation heatmaps, distribution plots, and feature importance |
| 🤖 **6 ML Algorithms** | Logistic Regression, Naive Bayes, Decision Trees, Random Forest, KNN, and SVM comparison |
| 🎛️ **Hyperparameter Tuning** | GridSearchCV optimization for each model to maximize performance |
| 📈 **Dual Metric Evaluation** | Both accuracy and AUC ROC scores for robust model assessment |
| 🔄 **Feature Selection** | Correlation based threshold filtering for each disease target |
| ⚖️ **Data Preprocessing** | Missing value imputation, feature scaling (StandardScaler/MinMaxScaler), and encoding |
| 📉 **Visualization Suite** | Feature distributions, target class balance, correlation matrices, and model comparisons |

---

## 🛠 Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python_3.8+-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Machine Learning** | ![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Environment** | ![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white) |

</div>

---

## 🔄 Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PredictaMed Analysis Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  📥 DATA INGESTION                                                              │
│  ├── Load CSV dataset (59,068 records × 18 features)                           │
│  └── Initial data inspection and type validation                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🔧 PREPROCESSING                                                               │
│  ├── Categorical encoding (Yes/No → 1/0, Age categories → ordinal)             │
│  ├── Missing value imputation (SimpleImputer with mean strategy)               │
│  └── Feature scaling (StandardScaler / MinMaxScaler)                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  📊 EXPLORATORY DATA ANALYSIS                                                   │
│  ├── Feature distribution histograms (BMI, Age, Sleep Time, etc.)              │
│  ├── Target variable class balance visualization                               │
│  └── Correlation matrix heatmap (18×18 features)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🎯 FEATURE SELECTION                                                           │
│  ├── Compute correlation with each target disease                              │
│  ├── Apply threshold filtering (0.06 → 0.20 based on disease)                  │
│  └── Select high correlation features for each model                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🔀 DATA SPLITTING                                                              │
│  └── Train (60%) │ Validation (20%) │ Test (20%)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🤖 MODEL TRAINING & OPTIMIZATION                                               │
│  ├── Logistic Regression (C, solver, max_iter tuning)                          │
│  ├── Naive Bayes (var_smoothing tuning)                                        │
│  ├── Decision Tree (max_depth, min_samples_split/leaf, criterion)              │
│  ├── Random Forest (n_estimators, max_depth, criterion, bootstrap)             │
│  ├── KNN (n_neighbors, distance metric)                                        │
│  └── SVM (kernel selection: linear/rbf)                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  📈 EVALUATION & COMPARISON                                                     │
│  ├── Accuracy scores (Train / Validation / Test)                               │
│  ├── AUC ROC scores for classification quality                                 │
│  ├── Side by side model comparison charts                                      │
│  └── Best model selection per disease                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results & Insights

### 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 59,068 |
| **Features** | 18 |
| **Target Diseases** | 6 |
| **Average BMI** | 28.57 |
| **Average Age Category** | 7.68 (55 to 59 years) |
| **Average Sleep Time** | 7.14 hours |
| **Smokers** | 48.76% |
| **Physical Activity** | 72.35% |

### 🏆 Best Performing Models

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         MODEL PERFORMANCE SUMMARY                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Disease          │  Best Model          │  Test Accuracy  │  Test AUC       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Heart Disease    │  Random Forest       │     72.98%      │    80.63%       ║
║  Stroke           │  Decision Tree       │     91.14%      │    77.60%       ║
║  Diabetes         │  Random Forest       │     80.39%      │    78.31%       ║
║  Asthma           │  Random Forest       │     85.08%      │    63.88%       ║
║  Kidney Disease   │  Logistic Regression │     92.41%      │    78.81%       ║
║  Skin Cancer      │  Logistic Regression │     86.49%      │    74.10%       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 🔍 Key Findings

- **🥇 Highest Accuracy**: Kidney Disease prediction achieved **92.41%** accuracy using Logistic Regression
- **🌳 Random Forest Dominance**: Best performer for 3 out of 6 diseases (Heart Disease, Diabetes, Asthma)
- **📐 Logistic Regression**: Optimal for Kidney Disease and Skin Cancer with strong linear separability
- **🎯 AUC Scores**: Heart Disease models show highest AUC (80.63%), indicating excellent class discrimination
- **⚠️ Class Imbalance**: Stroke (8.80%) and Kidney Disease (7.44%) show significant imbalance, affecting model performance
- **🔗 Feature Correlations**: Age Category and Physical Health show strongest correlations across multiple diseases

### 📊 Correlation Analysis Highlights

| Disease | Strongest Predictors |
|---------|---------------------|
| Heart Disease | Age Category (0.42), General Health (0.39), Difficulty Walking (0.29) |
| Stroke | Age Category (0.14), Physical Health (0.17), Difficulty Walking (0.21) |
| Diabetes | Age Category (0.21), BMI (0.25), Difficulty Walking (0.25) |
| Asthma | Physical Health (0.14), Mental Health (0.12), Difficulty Walking (0.13) |
| Kidney Disease | Age Category (0.14), Physical Health (0.18), Heart Disease (0.18) |
| Skin Cancer | Age Category (0.27), Heart Disease (0.12), Kidney Disease (0.07) |

### 📉 Model Comparison Visualization

```
Test Accuracy by Model (All Diseases Average)
══════════════════════════════════════════════════════════════

Logistic Regression  ████████████████████████████████████  84.73%
Naive Bayes          ██████████████████████████████████    81.98%
Decision Tree        ████████████████████████████████████  84.52%
Random Forest        █████████████████████████████████████ 85.08%
KNN                  █████████████████████████████████     83.44%
SVM                  ████████████████████████████████████  84.64%
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ZayyanCPU/PredictaMed.git

# Navigate to project directory
cd PredictaMed

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook code.ipynb
```

### Usage

1. **Open the notebook**: Launch `code.ipynb` in Jupyter Notebook or VS Code
2. **Run all cells**: Execute cells sequentially to reproduce the analysis
3. **View results**: Model comparison charts and best model recommendations are displayed at the end
4. **Customize**: Modify threshold values or hyperparameters to experiment with different configurations

---

## 📁 Project Structure

```
PredictaMed/
│
├── 📓 code.ipynb                    # Main analysis notebook
├── 📊 Multiple Disease Data.csv     # Dataset (59,068 records)
├── 📖 README.md                     # Project documentation
├── 📜 LICENSE                       # License file
└── 🔒 .git/                         # Git repository
```

---

## 💡 Skills Demonstrated

<div align="center">

| Category | Skills |
|----------|--------|
| **Data Science** | Exploratory Data Analysis, Feature Engineering, Statistical Analysis, Data Visualization |
| **Machine Learning** | Classification Algorithms, Model Selection, Hyperparameter Tuning, Cross Validation |
| **Programming** | Python, Pandas, NumPy, Scikit Learn API, Matplotlib/Seaborn |
| **Best Practices** | Data Preprocessing, Train/Validation/Test Split, Performance Metrics (Accuracy, AUC) |
| **Domain Knowledge** | Healthcare Analytics, Disease Prediction, Medical Data Interpretation |

</div>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

> **Note**: The dataset used in this project was provided for educational purposes and is not owned by the developer.

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**Built with ❤️ by [Zayyan](https://github.com/ZayyanCPU)**

[![GitHub](https://img.shields.io/badge/GitHub-ZayyanCPU-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ZayyanCPU)

</div>
