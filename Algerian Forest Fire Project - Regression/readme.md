# Fire Prediction in Algeria Using Weather Data

## Project Overview

This project focuses on predicting the occurrence of fire or no-fire events in the Bejaia and Sidi Bel-Abbes regions of Algeria, based on weather data recorded from June 2012 to September 2012. The dataset includes weather attributes such as temperature, humidity, wind speed, and other fire-related indices. The goal is to predict the likelihood of fire using various regression models.

The models used for training and prediction are:
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Elastic Net**

Each of these models was tested to evaluate their performance on the dataset to predict whether a fire event occurred (fire class) or not (no-fire class).

## Dataset Information

The dataset consists of 244 instances, with 122 instances from each of the two regions (Bejaia and Sidi Bel-Abbes). These instances are classified into two categories:
- **Fire (138 instances)**
- **Not Fire (106 instances)**

### Dataset Period
- **Time Frame**: June 2012 to September 2012
- **Regions**: Bejaia (Northwest of Algeria) and Sidi Bel-Abbes

### Data Columns
The dataset includes 11 weather-related attributes and 1 output attribute (Class). The attributes provide critical information for predicting fire events. The attributes are:

1. **Date**: (DD/MM/YYYY) The date of weather observation, including the day, month (June to September), and year (2012).
2. **Temp**: The noon temperature (maximum temperature) in Celsius degrees, ranging from 22°C to 42°C.
3. **RH**: Relative Humidity in percentage, ranging from 21% to 90%.
4. **Ws**: Wind speed in km/h, ranging from 6 km/h to 29 km/h.
5. **Rain**: Total rainfall in mm, ranging from 0 mm to 16.8 mm.
6. **Fine Fuel Moisture Code (FFMC)**: Index from the Fire Weather Index (FWI) system, ranging from 28.6 to 92.5.
7. **Duff Moisture Code (DMC)**: Index from the FWI system, ranging from 1.1 to 65.9.
8. **Drought Code (DC)**: Index from the FWI system, ranging from 7 to 220.4.
9. **Initial Spread Index (ISI)**: Index from the FWI system, ranging from 0 to 18.5.
10. **Buildup Index (BUI)**: Index from the FWI system, ranging from 1.1 to 68.
11. **Fire Weather Index (FWI)**: The FWI index, ranging from 0 to 31.1.

### Output Attribute (Class)
- **Classes**: The output class is either "Fire" or "Not Fire", which indicates whether a fire event occurred based on the weather data.

## Methodology

The models implemented in this project are used to predict fire occurrence (Fire vs. Not Fire) based on the weather-related features in the dataset.

### 1. **Linear Regression**  
Linear regression is a fundamental model used for regression tasks, and it was applied to predict the class (fire or no fire) based on the weather data. The continuous nature of the model was adapted to classify fire and non-fire events.

### 2. **Lasso Regression**  
Lasso regression is a regularization technique that adds an L1 penalty to the linear regression. It helps in feature selection by shrinking some coefficients to zero, which may improve model performance when handling many predictors.

### 3. **Ridge Regression**  
Ridge regression is another regularization technique that adds an L2 penalty to the linear regression. It helps prevent overfitting by penalizing large coefficients and provides better generalization.

### 4. **Elastic Net**  
Elastic Net is a combination of both Lasso and Ridge regression. It uses both L1 and L2 penalties, allowing for a more flexible model that can handle correlated features better than Lasso or Ridge alone.

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Accuracy**: The proportion of correct predictions (fire or no-fire).
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1-Score**: The harmonic mean of precision and recall, providing a single metric for the model’s performance.

## Requirements

To run this project, ensure that the following libraries are installed:
- **Python** (3.x or higher)
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning models and evaluation metrics.
- **Matplotlib**: For plotting and visualizing data.
- **Seaborn**: For statistical data visualization.

### Installation
To install the required libraries, run the following:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
