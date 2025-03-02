from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
import sys
import pickle
import joblib
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.components.prediction import InputData, ModelPredictor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["GET", "POST"])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            input_data = InputData(
                customer_age=request.form.get('customer_age'),
                gender=request.form.get('gender'),
                dependent_count=request.form.get('dependent_count'),
                education_level=request.form.get('education_level'),
                marital_status=request.form.get('marital_status'),
                income_category=request.form.get('income_category'),
                card_category=request.form.get('card_category'),
                months_on_book=request.form.get('months_on_book'),
                total_relationship_count=request.form.get('total_relationship_count'),
                months_inactive_12_mon=request.form.get('months_inactive_12_mon'),
                contacts_count_12_mon=request.form.get('contacts_count_12_mon'),
                credit_limit=request.form.get('credit_limit'),
                total_revolving_bal=request.form.get('total_revolving_bal'),
                total_amt_chng_q4_q1=request.form.get('total_amt_chng_q4_q1'),
                total_trans_amt=request.form.get('total_trans_amt'),
                total_trans_ct=request.form.get('total_trans_ct'),
                total_ct_chng_q4_q1=request.form.get('total_ct_chng_q4_q1'),
                avg_utilization_ratio=request.form.get('avg_utilization_ratio')
            )

            input_df = input_data.get_input_data()
            print(input_df)

            with open("/home/model/predict_model.pkl", 'rb') as file_object:
                model = pickle.load(file_object)
            
            preprocessor = joblib.load('/home/model/preprocessor.gz')
            
            predictor = ModelPredictor(features=input_df, model=model, preprocessor=preprocessor)
            prediction = predictor.predict()
        except Exception as e:
            print(traceback.print_exc(e))

        return prediction

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == 'GET':
        # with open("/home/model/preprocessor.", 'rb') as file_object:
        #         preprocessor = pickle.load(file_object)
        preprocessor = joblib.load('/home/model/preprocessor.gz')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


