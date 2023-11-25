import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import model_training 

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("aritifacts", "model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(len(X_train))
            print(len(y_train))
            
            models = {
                "Random forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBREgressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose = True),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            train_data_prediction, test_data_prediction = model_training(X_train,y_train,X_test,models=models)
            
            model_evaluation = evaluate_model(models,y_train, train_data_prediction,y_test, test_data_prediction)
            
            best_model_score = max(sorted(model_evaluation.values()))
            logging.info("best_model_score")
            
            best_model_name = list(model_evaluation.keys())[list(model_evaluation.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            print(best_model)
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
                
        except Exception as e:
            raise CustomException(e, sys)
        