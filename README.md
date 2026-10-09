# Obesity Level Prediction using Machine Learning

This project aims to predict obesity levels based on eating habits and physical condition data using various machine learning algorithms. The dataset includes demographic features, eating patterns, and physical attributes to classify individuals into 7 distinct obesity categories.

---

## 📌 Project Overview

Obesity is a major global health challenge. By analyzing lifestyle, physical activity, and eating habits, this project leverages classification algorithms to automatically determine an individual's obesity classification.

### **Target Categories**
- `Insufficient_Weight`
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Obesity_Type_II`
- `Obesity_Type_III`

---

## 🛠️ Data Preprocessing & Feature Engineering

1. **Data Cleaning**: Handled missing values and removed duplicate records (24 duplicate rows dropped).
2. **Outlier Handling**: Capped numerical outliers using the Interquartile Range (IQR) method.
3. **Feature Renaming**: Standardized column names for better readability (e.g., `FAVC` $\rightarrow$ `frequent_high_calorie_food`, `NObeyesdad` $\rightarrow$ `target`).
4. **Encoding Categorical Data**:
   - **Label Encoding**: Binary categorical variables (`gender`, `family_history_overweight`, `is_smoker`, `calorie_monitoring`).
   - **One-Hot Encoding**: Multi-class categorical features (`food_between_meals_freq`, `alcohol_consumption_freq`, `transportation_type`).
5. **Class Imbalance**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance target classes across the dataset.
6. **Feature Selection**: Used `SelectKBest` with ANOVA F-value (`f_classif`) to extract top predictive features.
7. **Feature Scaling**: Applied `StandardScaler` prior to model training.

---

## 📊 Model Evaluation Results

Multiple classification models were trained and evaluated on the test dataset ($20\%$ test split):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.98** | **0.98** | **0.98** | **0.98** |
| **Random Forest** | 0.96 | 0.96 | 0.96 | 0.96 |
| **Decision Tree** | 0.93 | 0.93 | 0.93 | 0.93 |
| **Logistic Regression** | 0.88 | 0.89 | 0.88 | 0.88 |
| **AdaBoost** | 0.50 | 0.46 | 0.50 | 0.45 |

> 🏆 **Best Performing Model**: **Gradient Boosting Classifier** achieved the highest accuracy of **98%**.

---

## 🚀 Requirements & Installation

### **Dependencies**
Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
