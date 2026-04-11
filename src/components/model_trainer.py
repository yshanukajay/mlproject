import os, sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor, 
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from src.exceptions.exception import CustomException
from src.loggers.logger import logging

from src.utils.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join('artifacts', 'model_trainer', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        try:
            logging.info("Reading the configurations for model trainer")
            self.model_trainer_config=ModelTrainerConfig()
        except Exception as e:
            logging.info("Exception occurred in Model Trainer component")
            raise CustomException(e, sys)
        
    
    def initialte_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            logging.info("Evaluating the models")
            model_report: dict=evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
            )   

            logging.info("Evaluation is completed. Extracting the best model from the report")
            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No best model found with score greater than 0.6")
                raise CustomException("No best model found", sys)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test, predicted)
    
            logging.info(f"Best model: {best_model_name}, R2 Score: {r2_square}")

            return best_model_name, r2_square

        except Exception as e:
            logging.info("Exception occurred in Model Trainer component")
            raise CustomException(e, sys)
