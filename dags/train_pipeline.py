from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
import yaml

import sys
import os

# Get the absolute path of the src directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidator

# Function to read config
def read_config():
    print("Reading Config")
    with open('/opt/airflow/src/config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def ingest_data(config):
    print("Data Ingestion")
    ingestion = DataIngestion(config)
    ingestion.prepare_data()

def validate_data(config):
    print("Data Validation")
    validation = DataValidator(config)
    output = validation.run_validation()

    if output == True:
        return "data_transformation"
    else:
        return "stop_pipeline"

def transform_data(config):
    transformer = DataTransformation(config)
    transformer.apply_transformations()

def train_model(config):
    trainer = ModelTrainer(config)
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

task_validation = BranchPythonOperator(
    task_id='data_validation',
    python_callable=lambda **kwargs: validate_data(kwargs['ti'].xcom_pull(task_ids='read_config')),
    dag=dag
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

task_stop = PythonOperator(
    task_id="stop_pipeline",
    python_callable=lambda: print("Validation failed. Stopping pipeline."),
    dag=dag
)

# DAG flow
task_read_config >> task_ingest >> task_validation
task_validation >> task_transform >> task_train
task_validation >> task_stop