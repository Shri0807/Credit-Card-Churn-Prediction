import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.utils.utils import save_object
from src.utils.exception import CustomException
import sys

class ModelTrainer:
    """
    Model Trainer class for training and evaluating a LightGBM classifier.

    This class handles loading preprocessed training and test datasets, training the model
    with the best hyperparameters, and evaluating its performance.

    Arguments:
        config (dict): Configuration dictionary containing model and preprocessing settings.
    """
    def __init__(self, config):
        """
        Initializes the ModelTrainer class with the provided configuration.

        Arguments:
            config (dict): Configuration dictionary with paths and model parameters.
        """
        self.config = config

    def train_model(self):
        """
        Trains a LightGBM model using the best hyperparameters and evaluates its performance.

        - Loads preprocessed training and testing datasets.
        - Extracts features (X) and target variable (y).
        - Trains a LightGBM classifier using specified hyperparameters.
        - Saves the trained model.
        - Generates predictions and evaluates model performance using classification report and AUC score.

        Returns:
            tuple: Classification report (str) and AUC score (float).
        """
        try:
            train_preprocessed = pd.read_csv(self.config["preprocessing"]["paths"]["train_preprocessed_path"])
            test_preprocessed = pd.read_csv(self.config["preprocessing"]["paths"]["test_preprocessed_path"])

            X_train, y_train = train_preprocessed.drop(columns=["churn_flag"]), train_preprocessed['churn_flag'].copy()
            X_test, y_test = test_preprocessed.drop(columns=["churn_flag"]), test_preprocessed['churn_flag'].copy()
            
            best_params = self.config["model"]["best_params"]
            
            best_model = LGBMClassifier(**best_params)

            best_model.fit(X_train, y_train)

            save_object(file_path=self.config["model"]["save_path"], object=best_model)

            y_pred = best_model.predict(X_test)
            churn_probas = best_model.predict_proba(X_test)[:, 1]
            
            class_report = classification_report(y_test, y_pred)
            auc_score = roc_auc_score(y_test, churn_probas)

            return class_report, auc_score
        except Exception as e:
            raise CustomException(e, sys)