import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from utils.utils import save_object

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train_model(self):
        train_preprocessed = pd.read_csv(self.config["preprocessing"]["paths"]["train_preprocessed_path"])
        test_preprocessed = pd.read_csv(self.config["preprocessing"]["paths"]["test_preprocessed_path"])

        X_train, y_train = train_preprocessed.drop(columns=["churn_flag"]), train_preprocessed['churn_flag'].copy()
        X_test, y_test = test_preprocessed.drop(columns=["churn_flag"]), test_preprocessed['churn_flag'].copy()
        
        best_params = { 'objective': 'binary',
                            'metric': 'roc_auc',
                            'n_estimators': 1000,
                            'verbosity': -1,
                            'bagging_freq': 1,
                            'class_weight': 'balanced', 
                            'learning_rate': 0.017535345166904838,
                            'num_leaves': 942,
                            'subsample': 0.8490611533540497,
                            'colsample_bytree': 0.3775159533799494,
                            'min_data_in_leaf': 90}
        
        best_model = LGBMClassifier(**best_params)

        best_model.fit(X_train, y_train)

        save_object(file_path=self.config["model"]["save_path"], object=best_model)

        y_pred = best_model.predict(X_test)
        churn_probas = best_model.predict_proba(X_test)[:, 1]
        
        class_report = classification_report(y_test, y_pred)
        auc_score = roc_auc_score(y_test, churn_probas)

        return class_report, auc_score