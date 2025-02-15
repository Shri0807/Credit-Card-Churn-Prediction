FROM apache/airflow:2.6.0-python3.9

USER root
RUN apt-get update && apt-get install -y libgomp1

USER airflow