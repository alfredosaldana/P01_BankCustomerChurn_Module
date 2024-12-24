## P01_BankCustomerChurn_Module

I used a dataset from Kaggle.com to predict whether a customer will "exit" the bank, a classification task complicated by a highly imbalanced target variable. To address this, I applied SMOTE (Synthetic Minority Oversampling Technique) to balance the data, ensuring the model could effectively learn from both classes.

This type of analysis has broad applications in the financial industry, such as predicting credit card fraud, identifying loan default risks, assessing customer retention in mortgage portfolios, and optimizing credit card marketing strategies.

#### Data source : https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

## Customer Churn Prediction
This project is designed to predict customer churn for a banking institution using a pre-trained machine learning model. The CustomerChurn class manages all aspects of data preprocessing, model integration, and predictions, making it easy to use and integrate into various workflows.

## Features
### 1. Data Preprocessing
Automatically cleans and preprocesses the input dataset:
Drops unnecessary columns like RowNumber, CustomerId, Surname, and Complain.
Standardizes numerical columns (e.g., CreditScore, Age, Balance) using a pre-saved scaler (CustomScaler).
Encodes categorical columns (Geography, Gender, Card Type) using one-hot encoding.
Ensures the input data format matches the training data for accurate predictions.

### 2. Model Integration
Loads a pre-trained machine learning model and scaler from serialized .pkl files.
Uses the loaded model to make predictions on new, unseen data.

### 3. Prediction Output
Predicts customer churn and appends a new column, Predicted_Exited, to the original dataset.
Returns a DataFrame with the original data and the churn predictions for easy analysis.

### 4. Error Handling
Verifies the length of predictions matches the number of rows in the dataset.
Safely handles missing or unnecessary columns, ensuring robustness.

## How It Works
Load the CustomerChurn class by specifying the paths to the serialized model and scaler files.
Use the load_and_clean_data() method to preprocess the input data.
Call the predict_churn() method to generate predictions and return the updated dataset.

from your_module import CustomerChurn , this is the class that contains all the data cleaning, engineering, model training, testing and predition

## Initialize with model and scaler file paths
churn_predictor = CustomerChurn(model_file='model.pkl', scaler_file='scaler.pkl')

## Load and preprocess the new data
data = churn_predictor.load_and_clean_data(data_file='new_customers.csv')

## Generate predictions
predictions_df = churn_predictor.predict_churn()

## Display the DataFrame with predictions
print(predictions_df.head())

### Requirements
Python 3.8+
pandas
numpy
scikit-learn

### Files
model.pkl: Serialized pre-trained machine learning model.
scaler.pkl: Serialized pre-trained CustomScaler object.
new_customers.csv: Input CSV file with customer data, this new data will not contain the target variable, which will be predicted.

***********
### Additional notes on Addressing Imbalanced Data:
In this classification project, the target variable (Exited) was highly imbalanced, with a significantly smaller number of positive cases compared to negative cases. This imbalance posed a challenge for training the machine learning model, as it could lead to biased predictions favoring the majority class.

To address this issue, I used SMOTE (Synthetic Minority Oversampling Technique), a powerful technique that generates synthetic samples for the minority class to achieve a balanced dataset. By balancing the dataset, I ensured that the model could learn effectively from both classes, leading to improved prediction performance and fairness.

### Why SMOTE?
- Enhances Model Learning: Prevents the model from being biased toward the majority class.
- Data Augmentation: Creates new, realistic examples for the minority class rather than simply duplicating existing ones.
- Improves Metrics: Helps improve recall, precision, and F1-score for the minority class, resulting in a more robust model.
### Implementation Highlights
- Applied SMOTE after splitting the dataset into training and testing sets to avoid data leakage.
- Verified that the synthetic samples were created only within the training dataset.
- Ensured the testing set remained untouched for fair evaluation of model performance.
This approach not only balanced the dataset but also contributed to a more accurate and reliable churn prediction model.
