import pandas as pd
from sklearn.pipeline import Pipeline 
from src.utils.modelling_utils import FeatureEngineer, OneHotFeatureEncoder, OrdinalFeatureEncoder, TargetFeatureEncoder, RecursiveFeatureEliminator, ColumnDropper
from src.utils.utils import save_object
from src.utils.exception import CustomException
from lightgbm import LGBMClassifier

import sys

class DataTransformation():
    """
    Data Transformation class for preprocessing datasets.

    This class applies feature engineering, encoding techniques, and feature selection
    to prepare the data for machine learning models.

    Arguments:
        config (dict): Configuration dictionary read from the pipeline.
    """
    def __init__(self, config):
        """
        Initializes the DataTransformation class with the provided configuration.

        Arguments:
            config (dict): Configuration dictionary containing preprocessing settings.
        """
        self.config = config

    def get_preprocessor(self):
        """
        Creates a preprocessing pipeline with feature engineering, encoding, and selection.

        The pipeline includes:
        - Feature Engineering
        - One-Hot Encoding
        - Ordinal Encoding
        - Target Encoding
        - Column Dropping
        - Recursive Feature Elimination

        Returns:
            Pipeline: A scikit-learn pipeline for preprocessing.
        """
        try:
            one_hot_encoding_features = self.config["preprocessing"]["one_hot_encoding"]["columns"]
            ordinal_encoding_orders = self.config["preprocessing"]["ordinal_encoding"]["columns"]
            target_encoding_features = self.config["preprocessing"]["target_encoding"]["columns"]
            to_drop_features = self.config["preprocessing"]["to_drop"]["columns"]

            preprocessor = Pipeline(
                    steps=[
                        ('feature_engineer', FeatureEngineer()),
                        ('one_hot_encoder', OneHotFeatureEncoder(to_encode=one_hot_encoding_features)),
                        ('ordinal_encoder', OrdinalFeatureEncoder(to_encode=ordinal_encoding_orders)),
                        ('target_encoder', TargetFeatureEncoder(to_encode=target_encoding_features)),
                        ('col_dropper', ColumnDropper(to_drop=to_drop_features)),
                        ('rfe_selector', RecursiveFeatureEliminator(n_folds=5, 
                                                                    scoring='roc_auc',
                                                                    estimator=LGBMClassifier()))
                    ]
                )
        except Exception as e:
            raise CustomException(e, sys)
        
        return preprocessor

    def apply_transformations(self):
        """
        Applies preprocessing transformations to train and test datasets.

        - Reads raw training and testing data.
        - Extracts features (X) and target variable (y).
        - Fits and applies the preprocessing pipeline.
        - Saves the transformed datasets and the preprocessor object.

        Returns:
            tuple: Preprocessed training and testing datasets.
        """
        try:
            train = pd.read_csv(self.config["data_ingestion"]["csv"]["train_data_path"])
            test = pd.read_csv(self.config["data_ingestion"]["csv"]["test_data_path"])

            X_train, y_train = train.drop(columns=['churn_flag']), train['churn_flag'].copy()
            X_test, y_test = test.drop(columns=['churn_flag']), test['churn_flag'].copy()

            preprocessor = self.get_preprocessor()

            X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
            X_test_preprocessed = preprocessor.transform(X_test)

            train_preprocessed = pd.concat([X_train_preprocessed, y_train.reset_index(drop=True)], axis=1)
            test_preprocessed = pd.concat([X_test_preprocessed, y_test.reset_index(drop=True)], axis=1)

            train_preprocessed.to_csv(self.config["preprocessing"]["paths"]["train_preprocessed_path"], index=False, header=True)
            test_preprocessed.to_csv(self.config["preprocessing"]["paths"]["test_preprocessed_path"], index=False, header=True)

            save_object(file_path=self.config["preprocessing"]["paths"]["preprocessor_path"], object=preprocessor)

            return train_preprocessed, test_preprocessed
        except Exception as e:
            raise CustomException(e, sys)

