from cnnClassifier import logger
from cnnClassifier.exception import CustomException
import sys

def divide_numbers(a, b):
    try:
        logger.info("Attempting division...")
        result = a / b
        return result
    except Exception as e:
        # Capture the exception with sys to extract exact file line numbers
        logger.error("An error occurred during division computation")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        divide_numbers(10, 0)
    except Exception as e:
        print(e)