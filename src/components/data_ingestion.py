import os
import sys
import warnings
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from src.utils.exception import CustomException

warnings.filterwarnings("ignore", category=FutureWarning)

class DataIngestion:
    """
    Data Ingestion class for preparing train and test datasets

    This class handles data ingestion from 2 sources: CSV and Postgres Database
    The Data Ingestion process includes Reading CSV, Reading Postgres, Merging, Preprocessing, Splitting into Train and Test and saving the Intermediate files as csv

    Arguments:
        config: Config.yaml file read as dictionary from Airflow Pipeline
    """
    def __init__(self, config, logger):
        """
        Initializes the DataIngestion class with the provided configuration.

        Arguments:
            config (dict): Configuration dictionary from the Airflow pipeline.
        """
        self.df = pd.DataFrame()
        self.df_postgres = pd.DataFrame()
        self.config = config["data_ingestion"]
        self.logger = logger

    def read_data_csv(self):
        """
        Reads data from a CSV file specified in the configuration.
        The data is stored in a DataFrame and saved as a raw CSV file.
        """
        try:
            self.logger.warning("Data Ingestion: Read Data from CSV: Start")
            self.df = pd.read_csv(self.config["csv"]["data_path"])
            os.makedirs(os.path.dirname(self.config["csv"]["raw_data_path"]), exist_ok=True)
            self.df.to_csv(self.config["csv"]["raw_data_path"], index=False, header=True)
            self.logger.warning("Data Ingestion: Read Data from CSV: End")
        except Exception as e:
            self.logger.error("Data Ingestion: Error in Read Data from CSV")
            raise CustomException(e, sys)
    
    def read_data_postgres(self):
        """
        Reads data from a Postgres database using SQLAlchemy.
        
        The data is fetched using credentials and query details provided in the configuration.
        If an error occurs during fetching, it prints an error message.
        """
        try:
            self.logger.warning("Data Ingestion: Read Data from Postgres: Start")
            db_url = f'postgresql+psycopg2://{self.config["postgres"]["POSTGRES_USER"]}:{self.config["postgres"]["POSTGRES_PASSWORD"]}@{self.config["postgres"]["POSTGRES_HOST"]}:{self.config["postgres"]["POSTGRES_PORT"]}/{self.config["postgres"]["POSTGRES_DB"]}'
            engine = create_engine(db_url)
            query = f'SELECT * FROM {self.config["postgres"]["POSTGRES_TABLE_NAME"]}'
            with engine.connect() as connection:
                self.df_postgres = pd.read_sql(query, connection)
            self.logger.warning("Data Ingestion: Read Data from Postgres: End")
        except Exception as e:
            self.logger.error("Data Ingestion: Error in Read Data from Postgres")
            raise CustomException(e, sys)
    
    def preproc_data(self):
        """
        Preprocesses the data by renaming columns, merging datasets, and encoding categorical variables.
        
        - Renames the "clientnum" column to "CLIENTNUM" for consistency.
        - Merges the CSV data and Postgres data on "CLIENTNUM".
        - Converts column names to lowercase.
        - Renames the "attrition_flag" column to "churn_flag" and encodes values:
          - "Attrited Customer" -> 1
          - "Existing Customer" -> 0
        - Saves the preprocessed data as a CSV file.
        """
        try:
            self.logger.warning("Data Ingestion: Preprocessing : Start")
            self.df_postgres.rename({"clientnum": "CLIENTNUM"}, axis=1, inplace=True)

            #Join Dataframes
            self.df = pd.merge(self.df, self.df_postgres, on="CLIENTNUM", how="left")

            self.df.columns = [x.lower() for x in self.df.columns]
            self.df.rename(columns={'attrition_flag': 'churn_flag'}, inplace=True)
            self.df['churn_flag'] = self.df['churn_flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
            os.makedirs(os.path.dirname(self.config["csv"]["preproc_data_path"]), exist_ok=True)
            self.df.to_csv(self.config["csv"]["preproc_data_path"], index=False, header=True)
            self.logger.warning("Data Ingestion: Preprocessing : End")
        except Exception as e:
            self.logger.error("Data Ingestion: Error in Preprocessing")
            raise CustomException(e, sys)
        
    def split_data(self):
        """
        Splits the preprocessed data into training and testing datasets.
        
        - Drops the "churn_flag" column from features (X) and assigns it as the target variable (y).
        - Performs stratified sampling to maintain class distribution.
        - Saves the training and testing datasets as CSV files.
        """
        try:
            self.logger.warning("Data Ingestion: Split Data : Start")
            X = self.df.drop(columns=['churn_flag'])
            y = self.df['churn_flag'].copy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

            train.to_csv(self.config["csv"]["train_data_path"], index=False, header=True)
            test.to_csv(self.config["csv"]["test_data_path"], index=False, header=True)
            self.logger.warning("Data Ingestion: Split Data : End")
        except Exception as e:
            self.logger.error("Data Ingestion: Error in Split Data")
            raise CustomException(e, sys)
    
    def prepare_data(self):
        """
        Executes the full data ingestion pipeline.
        
        - Reads data from CSV
        - Reads data from Postgres
        - Preprocesses the data
        - Splits it into training and testing datasets
        """
        try:
            self.logger.warning("Data Ingestion: Prepare Data: Start")
            self.read_data_csv()
            self.read_data_postgres()
            self.preproc_data()
            self.split_data()
            self.logger.warning("Data Ingestion: Prepare Data: End")
        except Exception as e:
            self.logger.error("Data Ingestion: Error in Prepare Data")
            raise CustomException(e, sys)
        


