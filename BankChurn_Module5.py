import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator,TransformerMixin):
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy=copy,with_mean=with_mean,with_std=with_std)
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_ = np.mean(X[self.columns])
        self.std_ = np.std(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns= self.columns)
        X_notscaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_notscaled,X_scaled], axis = 1)[init_col_order]


# CustomerChurn class for managing training, prediction, and preprocessing
class CustomerChurn:

    def __init__(self, model_file, scaler_file):
        # Load the trained model and scaler
        with open(model_file, 'rb') as model_file_obj:
            self.model_selected = pickle.load(model_file_obj)
        with open(scaler_file, 'rb') as scaler_file_obj:
            self.scaler_selected = pickle.load(scaler_file_obj)
        self.data = None

    def load_and_clean_data(self, data_file):
        # Import the new data ** (which does not contain the target column)
        df = pd.read_csv(data_file, delimiter=',')

        # Store data in a new instance variable (to keep original data intact)
        self.df_original = df.copy()

        # Remove columns not necessary for the model (similar to training preprocessing)
        df = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Complain'], axis=1)
        
        # Identify numerical columns to scale ( same selection as the data cleaning stage)
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                          'EstimatedSalary', 'Satisfaction Score', 'Point Earned']

        # Apply the saved scaler to standardize the numerical columns using the pre-trained scaler
        df[numerical_cols] = self.scaler_selected.transform(df[numerical_cols])

        # Create dummy variables for categorical columns / notice that here we code that the variable type 'objects' will be selected
        categorical_cols = ['Geography', 'Gender', 'Card Type']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype='int')
        
        df = df[['HasCrCard', 'IsActiveMember', 'CreditScore', 'Age', 'Tenure',
       'Balance', 'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score',
       'Point Earned', 'Geography_Germany', 'Geography_Spain', 'Gender_Male',
       'Card Type_GOLD', 'Card Type_PLATINUM', 'Card Type_SILVER']]
        
        # Store the preprocessed data for later use
        self.data = df.copy()
        return self.data

    def predict_churn(self):
        # Make predictions using the trained model
        predictions = self.model_selected.predict(self.data)
        predictions_series = pd.Series(predictions, name = 'Predicted_Exited')   # converting a list to a pd series
        self.df_original['Predicted_Exited'] = predictions_series    # creating a column with the new output
        # naming consistent
   
        
        return self.df_original # Return self.df_original that now contain a new column 'Predicted_Exited
        
