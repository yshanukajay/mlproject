import os, sys
from src.exceptions.exception import CustomException
from src.loggers.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_file_path: str=os.path.join('artifacts', 'data.csv')


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
    obj.initiate_data_ingestion()