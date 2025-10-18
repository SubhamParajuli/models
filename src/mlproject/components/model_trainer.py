#modelling 
import os
import sys
from src.mlproject.app_logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,r2_score
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
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import dagshub

# Try to initialize DagsHub, but don't block if it fails
try:
    dagshub.init(repo_owner='SubhamParajuli', repo_name='models', mlflow=True)
    DAGSHUB_ENABLED = True
except Exception as e:
    logging.warning(f"DagsHub initialization failed: {str(e)}. Continuing with local model training.")
    DAGSHUB_ENABLED = False

def evaluate_model(true,predicted):
    mae=mean_absolute_error(true,predicted)
    rmse=root_mean_squared_error(true,predicted)
    r2_square=r2_score(true,predicted)
    return mae,rmse,r2_square

@dataclass
class ModelTrainConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

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
            print(f"This is the best model: {best_model_name}")

            # Only attempt MLflow logging if DagsHub connection was successful
            if DAGSHUB_ENABLED:
                try:
                    model_names = list(params.keys())
                    actual_model = next((model for model in model_names if best_model_name == model), "")

                    mlflow.set_registry_uri("https://dagshub.com/SubhamParajuli/models.mlflow")
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    best_params = params.get(actual_model, {})

                    with mlflow.start_run():
                        predicted_qualities = best_model.predict(X_test)
                        mae = mean_absolute_error(y_test, predicted_qualities)
                        rmse = root_mean_squared_error(y_test, predicted_qualities)
                        r2 = r2_score(y_test, predicted_qualities)

                        mlflow.log_params(best_params)
                        mlflow.log_metric("rmse", round(rmse * 100, 4))
                        mlflow.log_metric("r2", round(r2 * 100, 4))
                        mlflow.log_metric("mae", round(mae * 100, 4))

                        # DagsHub does not support model registry endpoint, so only log the model artifact
                        mlflow.sklearn.log_model(best_model, name="model")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    logging.error(f"MLflow logging failed: {str(e)}\nTraceback:\n{tb}\nContinuing with local model save only.")
                    print(f"MLflow logging failed: {str(e)}\nTraceback:\n{tb}\nContinuing with local model save only.")


      

            

            if best_model_score<0.6:
                raise CustomException("No best model found.")
            logging.info(f"Best found model on both training and testing dataset: {best_model} at {best_model_score*100}")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            y_predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,y_predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)