from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.Data_Transformation import DataTransformation
from cnnClassifier import logger
from cnnClassifier.exception import CustomException
import sys

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_and_save_data()
        

if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
    


