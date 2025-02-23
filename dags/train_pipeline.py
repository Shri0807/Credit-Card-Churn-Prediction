from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import yaml

import sys
import os

# Get the absolute path of the src directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import Logger


# Function to read config
def read_config():
    print("Reading Config")
    with open('/opt/airflow/src/config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def ingest_data(config):
    logger = Logger("data_ingestion").get_logger()
    ingestion = DataIngestion(config, logger)
    ingestion.prepare_data()

def transform_data(config):
    logger = Logger("data_transformation").get_logger()
    transformer = DataTransformation(config, logger)
    transformer.apply_transformations()

def train_model(config):
    logger = Logger("train_model").get_logger()
    trainer = ModelTrainer(config, logger)
    report, auc = trainer.train_model()
    print(f"Model Training Report:\n{report}\nAUC Score: {auc}")

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 14),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='An ML pipeline with data ingestion, transformation, and model training',
    schedule_interval='@daily',
)

task_read_config = PythonOperator(
    task_id='read_config',
    python_callable=read_config,
    dag=dag,
)

task_ingest = PythonOperator(
    task_id='data_ingestion',
    python_callable=lambda **kwargs: ingest_data(kwargs['ti'].xcom_pull(task_ids='read_config')),
    dag=dag,
)

task_transform = PythonOperator(
    task_id='data_transformation',
    python_callable=lambda **kwargs: transform_data(kwargs['ti'].xcom_pull(task_ids='read_config')),
    dag=dag,
)

task_train = PythonOperator(
    task_id='model_training',
    python_callable=lambda **kwargs: train_model(kwargs['ti'].xcom_pull(task_ids='read_config')),
    dag=dag,
)

task_read_config >> task_ingest >> task_transform >> task_train