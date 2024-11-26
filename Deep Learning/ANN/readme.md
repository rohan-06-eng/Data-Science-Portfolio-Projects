# Artificial Neural Networks (ANN) Projects

This directory contains projects demonstrating the use of **Artificial Neural Networks (ANN)** for solving various real-world problems. Each folder includes datasets and code files necessary for understanding and running the models.

---

## Project Structure

### 1. Churn Prediction using ANN
- **Files:**
  - `Churn_Modelling.csv`: The dataset containing customer information and churn labels.
  - `churn_analysis.ipynb`: A Jupyter Notebook that implements an ANN to predict customer churn.
- **Overview:**
  - This project uses an ANN to predict whether a customer is likely to churn (leave the service).
  - The model is trained on a dataset with features like customer demographics, account information, and transaction history.
  - Techniques:
    - Data preprocessing
    - Model architecture design
    - Evaluation of model performance (accuracy, precision, recall)

---

### 2. Finetuning Using Keras Tuner for ANN
- **Files:**
  - `tuner_results/diabetes_ann_tuning`: Folder containing results from Keras Tuner.
  - `diabetes.csv`: The dataset containing features and labels for diabetes prediction.
  - `diabetes.ipynb`: A Jupyter Notebook that implements an ANN with hyperparameter tuning using **Keras Tuner**.
- **Overview:**
  - This project demonstrates hyperparameter tuning to optimize an ANN for predicting diabetes.
  - **Keras Tuner** is used to find the best combination of hyperparameters such as:
    - Number of layers
    - Neurons per layer
    - Activation functions
    - Learning rate
  - The model achieves improved accuracy and generalization.

---

### 3. Number Classification using ANN
- **Files:**
  - `mnist_classification.ipynb`: A Jupyter Notebook that implements an ANN to classify handwritten digits from the MNIST dataset.
- **Overview:**
  - This project uses an ANN to classify images of handwritten digits (0-9) from the popular MNIST dataset.
  - Techniques:
    - Data normalization and preprocessing
    - Building and training a feedforward neural network
    - Evaluation using metrics such as accuracy and confusion matrix
  - The project highlights the potential of ANNs in image-based classification tasks.

---

## How to Run

### Prerequisites
- Python 3.7 or higher
- Libraries: `TensorFlow`, `Keras`, `NumPy`, `Pandas`, `Matplotlib`

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ANN