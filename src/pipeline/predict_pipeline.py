import os
import sys
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

from src.exception import CustomException
from src.utils import load_object

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.h5")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logger.info("Loading model and preprocessor")
            model = tf.keras.models.load_model(model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logger.info("Transforming input features")
            data_scaled = preprocessor.transform(features)
            logger.info(f"Scaled Data: {data_scaled}")

            logger.info("Performing prediction")
            preds = model.predict(data_scaled)
            logger.info(f"Raw Prediction: {preds}")

            return preds
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

    def get_data_as_data_frame(self):
        try:
            logger.info("Converting input data to DataFrame")
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "NumOfProducts": [self.NumOfProducts],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logger.info(f"Created DataFrame: {df}")
            return df

        except Exception as e:
            logger.error(f"Error converting input data to DataFrame: {str(e)}")
            raise CustomException(e, sys)
