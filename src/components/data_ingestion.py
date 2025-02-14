import os
import sys
import warnings
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

class DataIngestion:
    def __init__(self, config):
        self.df = pd.DataFrame()
        self.config = config["data_ingestion"]

    def read_data(self):
        self.df = pd.read_csv(self.config["data_path"])
        os.makedirs(os.path.dirname(self.config["raw_data_path"]), exist_ok=True)
        self.df.to_csv(self.config["raw_data_path"], index=False, header=True)
    
    def preproc_data(self):
        self.df.columns = [x.lower() for x in self.df.columns]
        self.df.rename(columns={'attrition_flag': 'churn_flag'}, inplace=True)
        self.df['churn_flag'] = self.df['churn_flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        os.makedirs(os.path.dirname(self.config["preproc_data_path"]), exist_ok=True)
        self.df.to_csv(self.config["preproc_data_path"], index=False, header=True)
        
    def split_data(self):
        X = self.df.drop(columns=['churn_flag'])
        y = self.df['churn_flag'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(self.config["train_data_path"], index=False, header=True)
        test.to_csv(self.config["test_data_path"], index=False, header=True)
    
    def prepare_data(self):
        self.read_data()
        self.preproc_data()
        self.split_data()
        


