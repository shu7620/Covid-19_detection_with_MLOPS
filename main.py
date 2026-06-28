from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from cnnClassifier.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline 
from cnnClassifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from cnnClassifier.exception import CustomException
import sys



STAGE_NAME= 'Data Ingestion stage'
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)

STAGE_NAME= 'Data Validation stage'

if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
    
    
STAGE_NAME = "Data Transformation Stage"
if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
    
    
STAGE_NAME = "Model Training Stage"
if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)


STAGE_NAME = "Model Evaluation Stage"    
if __name__=='__main__':   
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)