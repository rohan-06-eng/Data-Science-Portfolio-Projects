# Used Car Price Prediction using Random Forest

## Project Overview
This project demonstrates how the Random Forest Regression algorithm can be used to predict the price of used cars based on various features such as the car's age, mileage, model, and other relevant attributes. The goal is to create a regression model that accurately estimates the price of a used car given these features. The predictions can help provide price suggestions for new sellers based on current market conditions, aiding them in setting competitive prices.

## Problem Statement
This dataset consists of used cars sold on Cardekho.com in India, along with important attributes of these cars. The task is to predict the price of a used car based on these attributes. The price prediction model can be used to provide sellers with price suggestions based on current market trends, helping them price their cars more effectively.

The prediction model will take various input features (like the car's age, mileage, model, and others) and estimate its price. This information can help new sellers set competitive prices when listing their cars.

## Key Steps in the Project

### 1. Data Loading and Preprocessing
- Load the dataset and perform data cleaning, including handling missing values and encoding categorical features.
- Scale numerical features if required to standardize the data for the model.

### 2. Feature Engineering
- Select relevant features from the dataset and apply transformations such as one-hot encoding to categorical variables (like car make, model, fuel type, etc.).
- Create new features if needed to improve model performance (e.g., calculating the car's age from the year of manufacture).

### 3. Model Building
- Use the Random Forest Regressor to build a model that can predict the car prices based on the input features.
- Train the model using the prepared dataset.

### 4. Model Evaluation
- Evaluate the model using performance metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared to assess the accuracy and robustness of the predictions.

### 5. Hyperparameter Tuning
- Tune the hyperparameters of the Random Forest model (e.g., the number of trees, maximum depth) using techniques like Grid Search or Random Search to optimize model performance.

### 6. Prediction
- Use the trained model to predict the prices of used cars based on new input features.

## Dataset Information
The dataset used in this project consists of used car listings scraped from Cardekho.com, a popular used car marketplace in India. It contains relevant information about the cars that can be used for price prediction.

### Dataset Details
- **Number of Rows**: 15,411
- **Number of Columns**: 13

### Key Features:
- **Car Make**: The manufacturer of the car (e.g., Toyota, Honda).
- **Car Model**: The specific model of the car (e.g., Corolla, Civic).
- **Year**: The year the car was manufactured.
- **Mileage**: The total distance the car has traveled, usually measured in kilometers.
- **Fuel Type**: Type of fuel the car uses (e.g., Petrol, Diesel, CNG).
- **Transmission**: Type of transmission (e.g., Manual, Automatic).
- **Owner**: The number of previous owners.
- **Price**: The price of the car (target variable).

## Objective
The objective is to predict the **Price** of a used car based on the other features in the dataset. This can help new sellers get a price suggestion based on market trends, thus making their listings more competitive.

## Technologies Used
- **Python**: Primary programming language used for data analysis, model building, and evaluation.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For implementing machine learning models, including Random Forest Regressor.
- **Matplotlib/Seaborn**: For data visualization and model performance analysis.

## Requirements
The following Python libraries are required to run the project:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
