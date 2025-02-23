import logging
import os
from datetime import datetime

class Logger:
    """
    Logger class to handle logging for different stages of the pipeline.

    Logs are saved in the `logs` directory with filenames based on the stage name.
    """

    def __init__(self, stage_name):
        """
        Initialize the logger with a specific stage name.

        Arguments:
            stage_name (str): The stage name for logging (e.g., "data_ingestion").
        """
        log_dir = "/opt/airflow/logs"
        os.makedirs(log_dir, exist_ok=True)

        log_filename = os.path.join(log_dir, f"{stage_name}_{datetime.now().strftime('%Y-%m-%d')}.log")

        logging.basicConfig(
            filename=log_filename,
            level=logging.WARN,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        self.logger = logging.getLogger("airflow.task")

    def get_logger(self):
        """
        Returns the logger instance.

        Returns:
            logging.Logger: Configured logger for the given stage.
        """
        return self.logger