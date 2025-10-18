from src.mlproject.app_logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from src.mlproject.components.model_trainer import ModelTrainConfig,ModelTrainer
import sys

if __name__== "__main__":
    logging.info('exectution has started')

    try:
      #  data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        ##Model trainig
        model_trainer_config=ModelTrainConfig()
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    except Exception as e:
        raise CustomException(e,sys)