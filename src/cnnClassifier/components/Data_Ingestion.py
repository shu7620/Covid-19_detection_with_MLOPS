import os
import shutil
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


# Load environment variables from the .env file BEFORE importing kaggle
load_dotenv()
import kaggle

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset using the Kaggle API securely via .env credentials.
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info("Authenticating with Kaggle API...")
            
            # Parse the dataset slug from the URL
            # E.g., 'https://.../datasets/tawsifurrahman/covid19-radiography-database' 
            # becomes 'tawsifurrahman/covid19-radiography-database'
            url_parts = self.config.source_URL.split('/')
            dataset_slug = f"{url_parts[-2]}/{url_parts[-1]}"
            
            logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
            
            # Authenticate using the environment variables
            kaggle.api.authenticate()
            
            # Download the zip file to the root directory
            kaggle.api.dataset_download_files(
                dataset_slug, 
                path=self.config.root_dir, 
                unzip=False
            )
            
            # Kaggle saves the file using the dataset name (e.g., covid19-radiography-database.zip)
            # We rename it to our standard 'data.zip' so the rest of the pipeline works seamlessly
            downloaded_zip_name = f"{url_parts[-1]}.zip"
            downloaded_zip_path = os.path.join(self.config.root_dir, downloaded_zip_name)
            
            if os.path.exists(downloaded_zip_path):
                os.rename(downloaded_zip_path, self.config.local_data_file)
            
            logger.info(f"Dataset downloaded successfully to {self.config.local_data_file}")
        else:
            logger.info(f"File already exists. Size: {get_size(Path(self.config.local_data_file))}")  

    def extract_zip_file(self):
        """
        Extracts the raw zip archive into a temporary extraction directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file into: {unzip_path}")

    def clean_and_restructure_dataset(self):
        """
        Parses the Kaggle structure, filtering out masks and metadata.
        Moves only valid X-ray images into standard class directories.
        """
        logger.info("Restructuring dataset to standard format...")
        target_classes = ["COVID", "Normal"]
        
        raw_root = None
        for path in Path(self.config.unzip_dir).rglob("COVID"):
            if path.is_dir():
                raw_root = path.parent
                break
                
        if not raw_root:
            raise FileNotFoundError("Could not find the extracted dataset structure. Check zip contents.")

        for cls in target_classes:
            dest_dir = Path(self.config.final_dataset_dir) / cls
            os.makedirs(dest_dir, exist_ok=True)
            
            source_img_dir = raw_root / cls / "images"
            
            if source_img_dir.exists():
                logger.info(f"Moving images from {source_img_dir} to {dest_dir}...")
                for img_file in source_img_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        shutil.move(str(img_file), str(dest_dir / img_file.name))
            else:
                logger.warning(f"Expected source folder {source_img_dir} not found.")

        # Cleanup raw files
        if os.path.exists(self.config.unzip_dir):
            shutil.rmtree(self.config.unzip_dir)
            logger.info("Cleaned up raw temporary extraction files.")