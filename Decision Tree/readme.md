# Diabetes Prediction Using Decision Tree

## Overview
This project demonstrates the use of a **Decision Tree Regressor** to predict diabetes progression using the **Diabetes Dataset** from `sklearn.datasets`. The model is trained on medical data to predict the quantitative measure of diabetes progression based on health metrics.

---

## Key Features
- Utilizes the **Diabetes Dataset** from Scikit-learn.
- Implements a **Decision Tree Regressor** to predict disease progression.
- Visualizes the Decision Tree structure for interpretability.
- Evaluates performance using metrics like **Mean Squared Error (MSE)** and **R² Score**.

---

## Technologies Used
- Python
- Scikit-learn
- Matplotlib
- Numpy

---

## Workflow

1. **Data Loading and Preprocessing**:
   - Load the dataset using `load_diabetes()` from Scikit-learn.
   - Analyze features like age, BMI, and blood pressure.

2. **Data Splitting**:
   - Split data into training (80%) and testing (20%) sets using `train_test_split`.

3. **Model Training**:
   - Train a **Decision Tree Regressor** with parameters like `max_depth` to prevent overfitting.

4. **Prediction**:
   - Predict diabetes progression on the test dataset.

5. **Evaluation**:
   - Assess the model using:
     - **Mean Squared Error (MSE)**
     - **R² Score**

6. **Visualization**:
   - Display the structure of the Decision Tree for better understanding.

