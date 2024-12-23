## P01_BankCustomerChurn_Module
I used a data set from Kaggle.com
Data source : https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

##**Customer Churn Prediction**
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

