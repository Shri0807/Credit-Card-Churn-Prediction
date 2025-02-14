import pandas as pd
from sklearn.pipeline import Pipeline
from utils.utils import FeatureEngineer, OneHotFeatureEncoder, OrdinalFeatureEncoder, TargetFeatureEncoder, RecursiveFeatureEliminator, ColumnDropper
from lightgbm import LGBMClassifier

class DataTransformation():
    def __init__(self, config):
        self.config = config

    def get_preprocessor(self):
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
        
        return preprocessor

    def apply_transformations(self):

        train = pd.read_csv(self.config["data_ingestion"]["train_data_path"])
        test = pd.read_csv(self.config["data_ingestion"]["test_data_path"])

        X_train, y_train = train.drop(columns=['churn_flag']), train['churn_flag'].copy()
        X_test, y_test = test.drop(columns=['churn_flag']), test['churn_flag'].copy()

        preprocessor = self.get_preprocessor()

        X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        train_preprocessed = pd.concat([X_train_preprocessed, y_train.reset_index(drop=True)], axis=1)
        test_preprocessed = pd.concat([X_test_preprocessed, y_test.reset_index(drop=True)], axis=1)

        train_preprocessed.to_csv(self.config["preprocessing"]["paths"]["train_preprocessed_path"], index=False, header=True)
        test_preprocessed.to_csv(self.config["preprocessing"]["paths"]["test_preprocessed_path"], index=False, header=True)

        return train_preprocessed, test_preprocessed

