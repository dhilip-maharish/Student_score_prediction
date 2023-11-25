import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.datatrans_config = DataTransformationconfig()
        
    def creating_datatransfomation_object(self):
        try:
            numerical_featues = ["reading_score","writing_score"]
            categorical_featues = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
            numerical_pipelines = Pipeline(steps = [("impute", SimpleImputer(strategy = "median")),("scaler", StandardScaler())])
            
            categorical_pipelines = Pipeline(steps = [("impute", SimpleImputer(strategy = "most_frequent")),("one_hot_encoder", OneHotEncoder()),("standard_scaler", StandardScaler(with_mean=False))])
            
            logging.info("Numerical column standard scaling completed")
            logging.info("categroical column encoding completed")
            
            preprocessor = ColumnTransformer([("num_pipline", numerical_pipelines, numerical_featues),
                                             ("cat_pipline", categorical_pipelines, categorical_featues )])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
           train_data = pd.read_csv(train_path) 
           test_data = pd.read_csv(test_path)
           
           preprocessing_obj = self.creating_datatransfomation_object()
           
           target_column_name = "math_score"
           
           input_features_train_data = train_data.drop(columns = [target_column_name], axis=1)
           print(input_features_train_data)
           target_features_train_data = train_data[target_column_name]
           
           input_features_test_data = test_data.drop(columns = [target_column_name], axis=1)
           print(input_features_test_data)
           target_features_test_data = test_data[target_column_name]
           
           input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_data)
           input_featue_test_arr = preprocessing_obj.transform(input_features_test_data)
           
           train_arr = np.c_[input_feature_train_arr , np.array(target_features_train_data)]
           test_arr = np.c_[input_featue_test_arr, np.array(target_features_test_data)]
           
           save_object(file_path = self.datatrans_config.preprocessor_obj_file_path, obj = preprocessing_obj)
           
           return(
               train_arr,
               test_arr,
               self.datatrans_config.preprocessor_obj_file_path
           )
           
        except Exception as e:
            raise CustomException(e,sys)
            
