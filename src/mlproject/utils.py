
import os
import sys
import pickle
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging


# Load environment variables from .env
load_dotenv()

# Read configuration from .env
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def read_mongodb_data():

    # Validate environment variables
    if not MONGODB_URL:
        raise ValueError("MONGODB_URL is missing from the .env file")

    if not DATABASE_NAME:
        raise ValueError("DATABASE_NAME is missing from the .env file")

    if not COLLECTION_NAME:
        raise ValueError("COLLECTION_NAME is missing from the .env file")

    client = MongoClient(MONGODB_URL)

    try:
        # Access database and collection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Read data without MongoDB's internal _id field
        data = list(collection.find({}, {"_id": 0}))

        if not data:
            raise ValueError(
                f"No data found in {DATABASE_NAME}.{COLLECTION_NAME}"
            )

        # Convert MongoDB data into a Pandas DataFrame
        df = pd.DataFrame(data)

        logging.info(
            f"Successfully read {len(df)} records from "
            f"{DATABASE_NAME}.{COLLECTION_NAME}"
        )

        return df

    except Exception as e:
        logging.error(f"Error reading MongoDB data: {e}")
        raise

    finally:
        client.close()


