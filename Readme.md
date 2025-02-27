# Credit Card Churn Prediction

# 1. Description

- This project is an end-to-end machine learning pipeline using LightGBM to predict the probability of customer churn in a bank's credit card service. It is a supervised classification problem, where the target variable is 1 if the customer has churned and 0 otherwise.
- To ensure a structured and maintainable approach, I followed CI/CD principles and modular coding practices. Initially, I conducted exploratory data analysis (EDA) and model development in Jupyter notebooks. Then, I modularized the workflow by creating separate components for data ingestion, transformation, and model training, mirroring the notebook-based approach.
- To automate the process, I built an Apache Airflow pipeline, deployed using Docker. The training pipeline orchestrates data processing and model training, generating machine learning model artifacts. Best practices such as exception handling, logging, and thorough documentation (including function and class definitions) were implemented throughout.
- Finally, I developed a Flask-based web API to integrate the entire pipeline, providing a structured and deployable solution.

# 2. Business problem and project objective

**3.1 What is the business problem?**
- A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them how likely is a customer to churn so they can proactively go to the customers to provide them better services and turn their decisions in the opposite direction.

**3.2 What is the context?**
- When a bank acquires a customer for its credit card service, three essential Key Performance Indicators (KPIs) to consider include:
    1. Customer Acquisition Cost (CAC): This measures the expenses associated with acquiring each credit card customer, encompassing marketing, sales, and related costs. Lower CAC reflects efficient customer acquisition.
    2. Customer Lifetime Value (CLV): CLV estimates the total revenue the bank can expect to generate from a credit card customer over their relationship. A higher CLV indicates that the customer's value surpasses the acquisition cost, ensuring long-term profitability.
    3. **Churn Rate:** Churn rate is typically expressed as a percentage and represents the number of credit card customers who have left during a specific period divided by the total number of customers at the beginning of that period.
- These KPIs help the bank assess the effectiveness of its strategies in acquiring credit card customers and gauge the potential long-term financial benefit of these acquisitions.
- In order to maximize profitability, the bank aims to minimize CAC and Churn while maximizing CLV.

**3.3 Which are the project objectives?**
1. Identify the factors associated with customer churn.
2. Construct a model capable of accurately predicting the probability of a customer to churn.
3. Offer action plans for the bank to reduce credit card customer churn.

**3.4 Which are the project benefits?**
1. Cost Savings.
2. Improved Customer Retention.
3. Enhanced Customer Experience.
4. Targeted Marketing.
5. Revenue Protection.
- And as a result, the mentioned business problem will be resolved.

**3.5 Conclusion**
- When deploying the model so that the bank can make predictions, the primary objective is to generate probability scores for each customer. This is typically more valuable for businesses when compared to making binary predictions (1/0), as it enables better decision-making and more effective customer retention strategies.
- For instance, predicting the probability of churn provides more actionable insights. Instead of simply determining whether a customer will churn or not, you gain an understanding of how likely it is to happen. This information enables the bank to allocate its efforts and resources more effectively. For example, it can concentrate its retention efforts on customers with a high probability of churning.

# 4. Solution pipeline
The following pipeline was used, based on CRISP-DM framework:

1. Define the business problem.
2. Collect the data and get a general overview of it.
3. Split the data into train and test sets.
4. Explore the data (exploratory data analysis)
5. Feature engineering, data cleaning and preprocessing.
6. Model training, comparison, feature selection and tuning.
7. Final production model testing and evaluation.
8. Conclude and interpret the model results.
9. Deploy.

Each step is explained in detail inside the notebooks, where I provide the rationale for the decisions made

# 5. Data Sources
2 Main Data Sources

1. CSV File containing Numeric features
2. Postgres Database containing Categorical Features

# 6. Model Preprocessing and Selection  

We developed two preprocessing pipelines tailored for different model types:  

- **Linear Models:**  
  - Applied **one-hot encoding** to categorical variables to preserve the linearity assumption.  
  - Used **standard scaling** for numerical features, as linear models rely on distance-based calculations and gradient descent optimization.  

- **Tree-Based Models:**  
  - Used **ordinal encoding** for ordinal categorical features to retain their natural order.  
  - Applied **target encoding** to other categorical variables, avoiding the sparsity and dimensionality issues of one-hot encoding.  
  - Numerical features were included **without scaling**, as tree-based models are scale-invariant.  

## 6.1 Feature Engineering  

Before preprocessing, we performed **extensive feature engineering** to enhance churn prediction, generating key attributes such as:  

- **Average transaction amount**  
- **Proportion of inactive months relative to customer tenure**  
- **Total spending**  

All preprocessing steps were implemented as **transformer classes** and integrated into an **scikit-learn pipeline**, ensuring seamless deployment in a production environment.  

## 6.2 Model Selection  

- We evaluated a set of **linear and tree-based models** using **stratified k-fold cross-validation**, comparing them based on **ROC-AUC scores** (as accuracy is unsuitable due to class imbalance).  
- **LightGBM** achieved the highest average validation score, making it the optimal choice for:  
  - Feature selection  
  - Hyperparameter tuning  
  - Final model evaluation  

While LightGBM exhibited some overfitting, its strong validation performance was driven by the dataset’s quality rather than data leakage or modeling issues. The independent variables effectively differentiate churners from non-churners.

## 6.3 Feature Selection and Hyperparameter Tuning  

- **Feature Selection:**  
  - We applied **Recursive Feature Elimination (RFE)** to iteratively select the most important features based on feature importance scores.  
  - The process continued until the desired number of features was reached, resulting in **25 out of 40** variables being selected.  
  - Many of these selected features were derived from the **feature engineering step**, highlighting its significance in improving model performance.  

- **Hyperparameter Tuning:**  
  - We fine-tuned the **LightGBM model** using **Bayesian Search**, which leverages probabilistic models to efficiently explore the hyperparameter space.  
  - This method balances **exploration and exploitation**, leading to more optimal parameter selection.  
  - A key tuning aspect was defining the **class_weight** hyperparameter, ensuring the model better learned patterns in the **minority target class (churn customers)**.  

# 7. Final Model Performance  

The **LightGBM model** demonstrated excellent performance, effectively identifying churners while maintaining high precision:  

- **Recall (0.89):** The model correctly identifies **89% of churners**, meaning:  
  - Out of **325 actual churners**, it correctly predicted **290**.  

- **Precision (0.90):** Of all customers predicted as churners, **90% were correctly classified**, meaning:  
  - Out of **324 predicted churners**, **297 were actual churners**.  

- **Probability Scores:** The predicted probabilities align well with actual outcomes, with churners assigned higher probabilities, reinforcing the model’s reliability.  

| Model    | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  | KS       | Gini     | PR-AUC   | Brier    |
|----------|----------|-----------|----------|----------|----------|----------|----------|----------|----------|
| LightGBM | 0.965943 | 0.895062  | 0.892308 | 0.893683 | 0.991279 | 0.898897 | 0.982559 | 0.964932 | 0.025852 |