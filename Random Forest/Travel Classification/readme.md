# Holiday Package Prediction

## Project Overview
This project aims to predict customer purchases of holiday packages for a travel company, "Trips & Travel.Com". The company is planning to introduce a new offering: Wellness Tourism Package, and wants to optimize its marketing expenditure by predicting which customers are likely to purchase the new package. By analyzing historical data of existing customers, we can efficiently target customers for the new product offering, improving marketing efficiency and increasing customer acquisition.

The goal is to predict whether a customer is likely to purchase one of the existing holiday packages (Basic, Standard, Deluxe, Super Deluxe, or King) and the newly introduced Wellness Tourism Package.

## Problem Statement
"Trips & Travel.Com" wants to establish a viable business model to expand its customer base. Currently, the company offers five types of packages: Basic, Standard, Deluxe, Super Deluxe, and King. However, only 18% of customers purchased packages last year, and marketing costs have been high due to random outreach without analyzing customer data.

The company is now planning to launch a Wellness Tourism Package, which focuses on health and well-being during travel. To ensure efficient marketing and outreach, the company wants to use historical data of existing and potential customers to identify which customers are more likely to purchase the new package.

## Key Objectives
- **Customer Segmentation**: Identify customer segments that are more likely to purchase any of the holiday packages, including the newly introduced Wellness Tourism Package.
- **Marketing Efficiency**: Help the company focus marketing efforts on customers who are more likely to purchase a package, reducing marketing costs.
- **Predictive Modeling**: Build a predictive model that can classify customers based on their likelihood to purchase one of the holiday packages.

## Dataset Information
The dataset for this project is collected from Kaggle. It consists of 20 columns and 4,888 rows. The dataset contains customer-related features such as demographics, purchase history, and customer engagement metrics. The key target variable is whether the customer purchased a holiday package or not.

### Columns in the Dataset
The dataset contains the following columns:
- **Age**: Customer's age.
- **Gender**: Customer's gender.
- **Income**: Customer's income.
- **Marital Status**: Customer's marital status.
- **No. of Trips Last Year**: The number of trips taken by the customer last year.
- **Travel Frequency**: How often the customer travels.
- **Contacted Before**: Whether the customer has been contacted before.
- **Package Purchased**: The holiday package purchased (Basic, Standard, Deluxe, Super Deluxe, King, or None).
- **Package Preference**: Whether the customer prefers a package.
- **Package Type**: Type of package purchased (Basic, Standard, Deluxe, Super Deluxe, King, etc.).
- **Average Spending**: Average spending on travel and packages.
- **Email Engagement**: Level of customer engagement with marketing emails.
- **Online Behavior**: Customer’s online behavior metrics (e.g., website visits, clicks).
- **Customer Type**: Type of customer (New or Returning).
- **Social Media Engagement**: Level of customer engagement on social media.
- **Feedback Score**: Customer's feedback score (on previous trips/packages).
- **Package Interest**: Customer's interest in a particular package.
- **Region**: Geographical region where the customer is located.
- **Device Type**: Type of device used to engage with the company's website (Mobile/Desktop).
- **Customer ID**: Unique identifier for each customer.

### Size of the Dataset
- **Rows**: 4,888
- **Columns**: 20

## Project Steps

### 1. Data Loading and Preprocessing
- Load the dataset into a suitable format (e.g., DataFrame).
- Handle missing values (impute or drop).
- Encode categorical variables using techniques such as one-hot encoding or label encoding.
- Scale numerical features if necessary to prepare the data for machine learning models.

### 2. Exploratory Data Analysis (EDA)
- Perform a thorough exploratory analysis to understand the relationships between different features.
- Visualize data distributions and correlations using libraries like Matplotlib and Seaborn.
- Identify any patterns or trends in customer demographics, preferences, and purchasing behavior.

### 3. Feature Engineering
- Extract useful features or combine existing features to improve model performance.
- Create new features like age groups, income ranges, or travel frequency categories to better predict the likelihood of package purchases.

### 4. Model Building
- Split the dataset into training and testing sets (e.g., 80/20 split).
- Use machine learning algorithms (e.g., Random Forest, Logistic Regression, SVM) to build predictive models.
- Train the model on historical customer data to predict the likelihood of purchasing a package.

### 5. Model Evaluation
- Evaluate the models using appropriate metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Select the best performing model based on cross-validation and test set performance.

### 6. Model Tuning
- Perform hyperparameter tuning using techniques such as Grid Search or Random Search to improve model performance.

### 7. Deployment and Predictions
- Use the final model to make predictions on unseen data (i.e., potential customers who have not purchased packages yet).
- Identify customers who are most likely to purchase the new Wellness Tourism Package.

## Technologies Used
- **Python**: For implementing data preprocessing, model building, and evaluation.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning models and evaluation.
- **Matplotlib/Seaborn**: For visualizing the data and model results.
- **Jupyter Notebook**: For creating interactive data analysis and model development environments.

## Requirements
You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
