from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.Model_Evaluation import ModelEvaluation
from cnnClassifier import logger
from cnnClassifier.exception import CustomException
import sys

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_evaluation_config()
        model_evaluation= ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()
        

if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
    


