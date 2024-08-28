# import os
# import sys
# import logging
# from src.exception import CustomException
# from src.logger import logging

# import pandas as pd

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransforamtionConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass

# @dataclass
# class DataIngestionConfig:
#     train_data_path: str=os.path.join('artifacts',"train.csv")
#     test_data_path: str=os.path.join('artifacts',"test.csv")
#     raw_data_path: str=os.path.join('artifacts',"data.csv")

# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config=DataIngestionConfig()



#     def initiate_data_ingestion(self):
#         logging.info("Enter the data ingestion method or component")
#         try:
#             df=pd.read_csv('Notebook\Data\Churn_Modelling.csv')
#             logging.info('Read the dataset as dataframe')

#             df=df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
#             logging.info('Dropped unwanted columns: customer_id, surname and row_number')


#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

#             df.to_csv(self.ingestion_config.raw_data_path)

#             logging.info("Train test split initiated")
#             train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

#             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

#             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
#             logging.info("Ingestion of the data is completed")
#             return(
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )
#         except Exception as e:
#             raise CustomException(e,sys)

# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()    

#     data_tranformation=DataTransformation()
#     train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data,test_data)

#     modeltrainer=ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging  # Ensure logging is configured correctly in src/logger.py

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv('Notebook/Data/Churn_Modelling.csv')  # Make sure this path is correct
            logging.info('Read the dataset as dataframe')

            df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
            logging.info('Dropped unwanted columns: customer_id, surname and row_number')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved raw data to CSV")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved to CSV")

            logging.info("Ingestion of the data is completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()  # Ensure the initialization matches your implementation
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Model Training
        model_trainer = ModelTrainer()
        best_model, history, evaluation_metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Print evaluation metrics
        print(f"Evaluation Metrics: {evaluation_metrics}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise
