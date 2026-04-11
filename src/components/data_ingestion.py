import os, sys
from src.exceptions.exception import CustomException
from src.loggers.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'data_ingestion', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'data_ingestion', 'test.csv')
    raw_file_path: str=os.path.join('artifacts', 'data_ingestion', 'data.csv')


class DataIngestion:
    def __init__(self):
        try:
            logging.info("Reading the configurations for data ingestion")
            self.data_ingestion_config=DataIngestionConfig()

        except Exception as e:
            logging.info("Exception occurred in Data Ingestion component")
            raise CustomException(e, sys)
        
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Reading data from source")
            df=pd.read_csv("data\stud.csv")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_file_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Saving train and test data to respective paths")
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed successfully")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred in Data Ingestion component")
            raise CustomException(e, sys)
    
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_df, test_df,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    best_model_name, r2_square = model_trainer.initialte_model_trainer(train_array=train_df, test_array=test_df)
    print(f"Best model found on both training and testing dataset is {best_model_name} with r2 score of {r2_square}")