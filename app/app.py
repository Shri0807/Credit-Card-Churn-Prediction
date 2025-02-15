from flask import Flask, request, render_template
# from src.components.prediction import InputData
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        # input_data = InputData(
        #     customer_age=request.form.get('customer_age'),
        #     gender=request.form.get('gender'),
        #     dependent_count=request.form.get('dependent_count'),
        #     education_level=request.form.get('education_level'),
        #     marital_status=request.form.get('marital_status'),
        #     income_category=request.form.get('income_category'),
        #     card_category=request.form.get('card_category'),
        #     months_on_book=request.form.get('months_on_book'),
        #     total_relationship_count=request.form.get('total_relationship_count'),
        #     months_inactive_12_mon=request.form.get('months_inactive_12_mon'),
        #     contacts_count_12_mon=request.form.get('contacts_count_12_mon'),
        #     credit_limit=request.form.get('credit_limit'),
        #     total_revolving_bal=request.form.get('total_revolving_bal'),
        #     total_amt_chng_q4_q1=request.form.get('total_amt_chng_q4_q1'),
        #     total_trans_amt=request.form.get('total_trans_amt'),
        #     total_trans_ct=request.form.get('total_trans_ct'),
        #     total_ct_chng_q4_q1=request.form.get('total_ct_chng_q4_q1'),
        #     avg_utilization_ratio=request.form.get('avg_utilization_ratio')
        # )

        # input_df = input_data.get_input_data_df()
        # print(input_df)
        customer_age=request.form.get('customer_age')
        gender=request.form.get('gender')
        dependent_count=request.form.get('dependent_count')
        education_level=request.form.get('education_level')
        marital_status=request.form.get('marital_status')
        income_category=request.form.get('income_category')
        card_category=request.form.get('card_category')
        months_on_book=request.form.get('months_on_book')
        total_relationship_count=request.form.get('total_relationship_count')
        months_inactive_12_mon=request.form.get('months_inactive_12_mon')
        contacts_count_12_mon=request.form.get('contacts_count_12_mon')
        credit_limit=request.form.get('credit_limit')
        total_revolving_bal=request.form.get('total_revolving_bal')
        total_amt_chng_q4_q1=request.form.get('total_amt_chng_q4_q1')
        total_trans_amt=request.form.get('total_trans_amt')
        total_trans_ct=request.form.get('total_trans_ct')
        total_ct_chng_q4_q1=request.form.get('total_ct_chng_q4_q1')
        avg_utilization_ratio=request.form.get('avg_utilization_ratio')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


