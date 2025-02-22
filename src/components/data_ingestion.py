import os
import sys
import warnings
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

warnings.filterwarnings("ignore", category=FutureWarning)

class DataIngestion:
    def __init__(self, config):
        self.df = pd.DataFrame()
        self.df_postgres = pd.DataFrame()
        self.config = config["data_ingestion"]

    def read_data_csv(self):
        self.df = pd.read_csv(self.config["csv"]["data_path"])
        os.makedirs(os.path.dirname(self.config["csv"]["raw_data_path"]), exist_ok=True)
        self.df.to_csv(self.config["csv"]["raw_data_path"], index=False, header=True)
    
    def read_data_postgres(self):
        try:
            db_url = f'postgresql+psycopg2://{self.config["postgres"]["POSTGRES_USER"]}:{self.config["postgres"]["POSTGRES_PASSWORD"]}@{self.config["postgres"]["POSTGRES_HOST"]}:{self.config["postgres"]["POSTGRES_PORT"]}/{self.config["postgres"]["POSTGRES_DB"]}'
            engine = create_engine(db_url)
            query = f'SELECT * FROM {self.config["postgres"]["POSTGRES_TABLE_NAME"]}'
            with engine.connect() as connection:
                self.df_postgres = pd.read_sql(query, connection)
            
        except Exception as e:
            print("Error fetching data:", e)
    
    def preproc_data(self):
        self.df_postgres.rename({"clientnum": "CLIENTNUM"}, axis=1, inplace=True)

        #Join Dataframes
        self.df = pd.merge(self.df, self.df_postgres, on="CLIENTNUM", how="left")

        self.df.columns = [x.lower() for x in self.df.columns]
        self.df.rename(columns={'attrition_flag': 'churn_flag'}, inplace=True)
        self.df['churn_flag'] = self.df['churn_flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        os.makedirs(os.path.dirname(self.config["csv"]["preproc_data_path"]), exist_ok=True)
        self.df.to_csv(self.config["csv"]["preproc_data_path"], index=False, header=True)
        
    def split_data(self):
        X = self.df.drop(columns=['churn_flag'])
        y = self.df['churn_flag'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(self.config["csv"]["train_data_path"], index=False, header=True)
        test.to_csv(self.config["csv"]["test_data_path"], index=False, header=True)
    
    def prepare_data(self):
        self.read_data_csv()
        self.read_data_postgres()
        self.preproc_data()
        self.split_data()
        


