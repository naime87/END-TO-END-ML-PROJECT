
import os
import sys

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
    logging.info("Reading MongoDB database started")

    client = MongoClient(MONGODB_URL)

    try:
        # Access database and collection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Read data without MongoDB's internal _id field
        data = list(collection.find({}, {"_id": 0}))


        logging.info("Connection Established")


        # Convert MongoDB data into a Pandas DataFrame
        df = pd.DataFrame(data)
        print(df.head())

         
        return df

    except Exception as e:
       raise CustomException(e, sys)



