
import os
import sys

import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pickle
import numpy as np

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


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

            
    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():

            logging.info(f"Training model: {model_name}")

            # Get parameters for the current model
            model_params = params[model_name]

            # Hyperparameter Tuning

            gs = GridSearchCV(
                estimator=model,
                param_grid=model_params,
                cv=3
            )
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)



