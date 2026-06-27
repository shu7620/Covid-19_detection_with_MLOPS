from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.Model_Trainer import ModelTrainer
from cnnClassifier import logger
from cnnClassifier.exception import CustomException
import sys

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_training_config()
        model_trainer= ModelTrainer(config=model_training_config)
        model_trainer.train()
        

if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
    


