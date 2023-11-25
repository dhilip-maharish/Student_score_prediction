import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.data_transformation import DataTransformation

from src.component.model_trainer import ModelTrainer



@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    
    def __init__(self):
        self.injection_config = DataIngestionconfig()
        
    def initiate_data_injection(self):
        logging.info("Enter the data injection method")
        try:
            data = pd.read_csv("data\stud.csv")
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.injection_config.train_data_path),exist_ok=True)
            
            data.to_csv(self.injection_config.raw_data_path, index= False, header = True)
            
            logging.info("Train and test data split")
            train_set, test_set = train_test_split(data, test_size= 0.20, random_state=42)
            
            train_set.to_csv(self.injection_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.injection_config.test_data_path, index = False, header = True)
            
            logging.info("Injection data completed")
            return (
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_injection()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path,test_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
    
    
            
            
