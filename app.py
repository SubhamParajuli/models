from src.mlproject.app_logger import logging
from src.mlproject.exception import CustomException
import sys

if __name__== "__main__":
    logging.info('exectution has started')

    try:
        a=1/0
    except Exception as e:
        raise CustomException(e,sys)