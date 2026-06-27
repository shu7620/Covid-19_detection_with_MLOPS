import os
import sys
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataValidationConfig
from cnnClassifier.exception import CustomException

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_classes_exist(self) -> bool:
        """
        Validates if all expected class directories exist in the ingested dataset.
        Writes 'Validation status: True/False' to the status file.
        """
        try:
            validation_status = True
            
            # List all directories in the ingested dataset path
            available_classes = os.listdir(self.config.unzip_data_dir)

            for expected_cls in self.config.all_classes:
                if expected_cls not in available_classes:
                    validation_status = False
                    logger.error(f"Missing expected class directory: {expected_cls}")
                    break
            
            # Write the final status to the status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            if validation_status:
                logger.info("All expected classes found. Data validation successful.")
            else:
                logger.warning("Data validation failed. Expected classes are missing.")

            return validation_status
            
        except Exception as e:
            raise CustomException(e,sys)