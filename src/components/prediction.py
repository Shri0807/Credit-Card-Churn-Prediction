import sys
import pandas as pd
from src.utils.exception import CustomException
import os


class InputData:
    """
    Represents customer input data for churn prediction.

    This class initializes and stores customer-related features and provides
    a method to return the data as a DataFrame for model inference.
    """
    def __init__(self, customer_age: int,
                 gender: str,
                 dependent_count: int,
                 education_level: str,
                 marital_status: str,
                 income_category: str,
                 card_category: str,
                 months_on_book: int,
                 total_relationship_count: int,
                 months_inactive_12_mon: int,
                 contacts_count_12_mon: int,
                 credit_limit: float,
                 total_revolving_bal: int,
                 total_amt_chng_q4_q1: float,
                 total_trans_amt: int,
                 total_trans_ct: int,
                 total_ct_chng_q4_q1: float,
                 avg_utilization_ratio: float):
        
        """
        Initializes the customer input data object.

        Arguments:
            Various customer-related features including demographics, account
            activity, and transaction details.
        """
        self.customer_age = customer_age
        self.gender = gender
        self.dependent_count = dependent_count
        self.education_level = education_level
        self.marital_status = marital_status
        self.income_category = income_category
        self.card_category = card_category
        self.months_on_book = months_on_book
        self.total_relationship_count = total_relationship_count
        self.months_inactive_12_mon = months_inactive_12_mon
        self.contacts_count_12_mon = contacts_count_12_mon
        self.credit_limit = credit_limit
        self.total_revolving_bal = total_revolving_bal
        self.total_amt_chng_q4_q1 = total_amt_chng_q4_q1
        self.total_trans_amt = total_trans_amt
        self.total_trans_ct = total_trans_ct
        self.total_ct_chng_q4_q1 = total_ct_chng_q4_q1
        self.avg_utilization_ratio = avg_utilization_ratio
    
    def get_input_data(self):
        """
        Converts the stored input data into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the input data for prediction.
        """
        try:
            input_dict = dict()

            input_dict['customer_age'] = [self.customer_age]
            input_dict['gender'] = [self.gender]
            input_dict['dependent_count'] = [self.dependent_count]
            input_dict['education_level'] = [self.education_level]
            input_dict['marital_status'] = [self.marital_status]
            input_dict['income_category'] = [self.income_category]
            input_dict['card_category'] = [self.card_category]
            input_dict['months_on_book'] = [self.months_on_book]
            input_dict['total_relationship_count'] = [self.total_relationship_count]
            input_dict['months_inactive_12_mon'] = [self.months_inactive_12_mon]
            input_dict['contacts_count_12_mon'] = [self.contacts_count_12_mon]
            input_dict['credit_limit'] = [self.credit_limit]
            input_dict['total_revolving_bal'] = [self.total_revolving_bal]
            input_dict['total_amt_chng_q4_q1'] = [self.total_amt_chng_q4_q1]
            input_dict['total_trans_amt'] = [self.total_trans_amt]
            input_dict['total_trans_ct'] = [self.total_trans_ct]
            input_dict['total_ct_chng_q4_q1'] = [self.total_ct_chng_q4_q1]
            input_dict['avg_utilization_ratio'] = [self.avg_utilization_ratio]

            input_data_df = pd.DataFrame(input_dict)

            return input_data_df
        except Exception as e:
            raise CustomException(e, sys)

class ModelPredictor:
    """
    Model Predictor class for generating churn probability predictions.

    This class preprocesses input features, applies the trained model,
    and returns a probability-based churn prediction.
    """
    def __init__(self, features, model, preprocessor):
        """
        Initializes the ModelPredictor class with the provided components.

        Arguments:
            features (pd.DataFrame): Input features for prediction.
            model: Trained classification model.
            preprocessor: Preprocessing pipeline to transform features.
        """
        self.features = features
        self.model = model
        self.preprocessor = preprocessor

    def predict(self):
        """
        Processes input data and predicts customer churn probability.

        Returns:
            str: Formatted churn probability prediction.
        """
        try:
            preprocessed_data = self.preprocessor.transform(self.features)
            preprocessed_data = preprocessed_data.astype('float64')

            predicted_proba = self.model.predict_proba(preprocessed_data)[:, 1][0]

            prediction = f"""Customer's probability of churning: {round(predicted_proba * 100, 3)}%"""

            return prediction
        except Exception as e:
            raise CustomException(e, sys)