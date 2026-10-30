
import os
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import read_mongodb_data


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts", "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts", "raw.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Read data from MongoDB
            logging.info("Reading data from MongoDB")

            df = read_mongodb_data()

            logging.info(
                f"MongoDB data loaded successfully. Shape: {df.shape}"
            )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            # Split into train and test sets
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            # Save test data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
