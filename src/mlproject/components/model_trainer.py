#modelling 
import os
import sys
from src.mlproject.app_logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import warnings
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.mlproject.utils import save_object,evaluate_models


@dataclass
class ModelTrainConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init_(self):
        self.model_trainer_config=ModelTrainConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spit training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5],
                },

                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'max_features': ['sqrt'],
                },

                "Gradient Boosting": {
                    'learning_rate': [0.1],
                    'n_estimators': [100],
                    'max_depth': [3],
                },

                "Linear Regression": {
                    'fit_intercept': [True],
                },

                "XGBRegressor": {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [3],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'n_jobs': [4],  # use parallel threads for speed
                },

                "Lasso": {
                    'alpha': [0.1, 1.0],
                    'max_iter': [1000],
                },

                "Ridge": {
                    'alpha': [0.1, 1.0],
                    'max_iter': [1000],
                },

                "AdaBoost Regressor": {
                    'n_estimators': [50],
                    'learning_rate': [0.1, 1.0],
                },
            }

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found.")
            logging.info(f"Best found model on both training and testing dataset: {best_model} at {best_model_score*100}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,y_predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)