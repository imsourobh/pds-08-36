# Predicting Customer Churn Using Naïve Bayes

## Dataset
- **Name:** Telecom Churn Dataset
- **Description:** This dataset contains customer-related information to help predict whether a customer will churn or not.

## Objective
- Compare the performance of the **Naïve Bayes** model for predicting customer churn.

## Preprocessing
- Normalize numerical variables to ensure fair comparisons and improve model performance.

## Visualization
- Generate a **heatmap** to visualize correlations between churn and numerical features.
- Display model accuracy using visual representations.

## Modeling
- Train a **Naïve Bayes** model on the dataset.
- Evaluate model performance using the **R² (R-squared) metric** to assess predictive accuracy.

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Usage
Run the following script to execute the model:
```bash
python main.py
```

## Results
- The heatmap visualization provides insights into feature correlations.
- The Naïve Bayes model performance is evaluated based on R² scores.
- The Naïve Bayes model achieved an **accuracy of 85%**.
