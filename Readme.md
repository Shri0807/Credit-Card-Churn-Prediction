# Credit Card Churn Prediction

# 1. Description  

- This project is an end-to-end machine learning pipeline using **LightGBM** to predict the probability of customer churn in a bank’s credit card service. It is a **supervised classification problem**, where the target variable is **1 if the customer has churned and 0 otherwise**.  
- To ensure a **structured and maintainable** approach, we followed **CI/CD principles** and **modular coding** practices. Initially, we conducted **exploratory data analysis (EDA)** and model development in **Jupyter notebooks**. We then modularized the workflow by creating separate components for **data ingestion, transformation, and model training**, mirroring the notebook-based approach.  
- To **automate the process**, we built an **Apache Airflow pipeline**, deployed using **Docker**. The training pipeline orchestrates **data processing and model training**, generating machine learning model artifacts. **Best practices** such as **exception handling, logging, and thorough documentation** (including function and class definitions) were implemented throughout.  
- Finally, we developed a **Flask-based web API** to integrate the entire pipeline, ensuring a **structured and deployable** solution.  

# 2. Business Problem and Project Objective  

## 2.1 What is the Business Problem?  
- A **bank manager** is concerned about the increasing number of customers leaving their **credit card services**.  
- They need a solution to **predict the likelihood of customer churn**, allowing them to **proactively engage customers**, improve services, and retain them.  

## 2.2 What is the Context?  
When acquiring a credit card customer, three key **performance indicators (KPIs)** are essential:  

1. **Customer Acquisition Cost (CAC):**  
   - Measures the cost of acquiring a credit card customer, including **marketing, sales, and related expenses**.  
   - Lower CAC indicates **efficient acquisition**.  

2. **Customer Lifetime Value (CLV):**  
   - Estimates the **total revenue** a customer is expected to generate over their relationship with the bank.  
   - A **higher CLV** ensures that the **customer’s value outweighs acquisition costs**, leading to long-term profitability.  

3. **Churn Rate:**  
   - Expressed as a **percentage**, representing the proportion of customers who leave within a given period.  
   - High churn negatively impacts **revenue and profitability**.  

These KPIs help the bank assess the effectiveness of its **customer acquisition** and **retention strategies**.  
To **maximize profitability**, the goal is to **minimize CAC and churn while maximizing CLV**.  

## 2.3 What are the Project Objectives?  
1. **Identify** the key factors contributing to customer churn.  
2. **Build** a predictive model to estimate a customer’s churn probability.  
3. **Provide** actionable strategies to reduce churn and improve customer retention.  

## 2.4 What are the Project Benefits?  
1. **Cost Savings** – Reducing churn minimizes lost revenue and acquisition expenses.  
2. **Improved Customer Retention** – Proactive interventions help retain valuable customers.  
3. **Enhanced Customer Experience** – Personalized services improve satisfaction and loyalty.  
4. **Targeted Marketing** – Focused retention efforts optimize marketing strategies.  
5. **Revenue Protection** – Lower churn directly impacts long-term profitability.  

By addressing these factors, the **bank can mitigate the churn issue effectively**.  

## 2.5 Conclusion  
- The primary objective of deploying the model is to **generate probability scores** for each customer, rather than making simple **binary (1/0) predictions**.  
- Probability-based insights allow the bank to:  
  - **Prioritize retention efforts** on high-risk customers.  
  - **Allocate resources efficiently** to maximize retention impact.  
- For example, instead of treating all customers equally, the bank can **focus on those with a high churn probability**, improving retention strategies and overall business performance.  


# 3. Solution pipeline
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

# 4. Data Sources
2 Main Data Sources

1. CSV File containing Numeric features
2. Postgres Database containing Categorical Features

# 5. Model Preprocessing and Selection  

We developed two preprocessing pipelines tailored for different model types:  

- **Linear Models:**  
  - Applied **one-hot encoding** to categorical variables to preserve the linearity assumption.  
  - Used **standard scaling** for numerical features, as linear models rely on distance-based calculations and gradient descent optimization.  

- **Tree-Based Models:**  
  - Used **ordinal encoding** for ordinal categorical features to retain their natural order.  
  - Applied **target encoding** to other categorical variables, avoiding the sparsity and dimensionality issues of one-hot encoding.  
  - Numerical features were included **without scaling**, as tree-based models are scale-invariant.  

## 5.1 Feature Engineering  

Before preprocessing, we performed **extensive feature engineering** to enhance churn prediction, generating key attributes such as:  

- **Average transaction amount**  
- **Proportion of inactive months relative to customer tenure**  
- **Total spending**  

All preprocessing steps were implemented as **transformer classes** and integrated into an **scikit-learn pipeline**, ensuring seamless deployment in a production environment.  

## 5.2 Model Selection  

- We evaluated a set of **linear and tree-based models** using **stratified k-fold cross-validation**, comparing them based on **ROC-AUC scores** (as accuracy is unsuitable due to class imbalance).  
- **LightGBM** achieved the highest average validation score, making it the optimal choice for:  
  - Feature selection  
  - Hyperparameter tuning  
  - Final model evaluation  

While LightGBM exhibited some overfitting, its strong validation performance was driven by the dataset’s quality rather than data leakage or modeling issues. The independent variables effectively differentiate churners from non-churners.

## 5.3 Feature Selection and Hyperparameter Tuning  

- **Feature Selection:**  
  - We applied **Recursive Feature Elimination (RFE)** to iteratively select the most important features based on feature importance scores.  
  - The process continued until the desired number of features was reached, resulting in **25 out of 40** variables being selected.  
  - Many of these selected features were derived from the **feature engineering step**, highlighting its significance in improving model performance.  

- **Hyperparameter Tuning:**  
  - We fine-tuned the **LightGBM model** using **Bayesian Search**, which leverages probabilistic models to efficiently explore the hyperparameter space.  
  - This method balances **exploration and exploitation**, leading to more optimal parameter selection.  
  - A key tuning aspect was defining the **class_weight** hyperparameter, ensuring the model better learned patterns in the **minority target class (churn customers)**.  

# 6. Final Model Performance  

The **LightGBM model** demonstrated excellent performance, effectively identifying churners while maintaining high precision:  

- **Recall (0.89):** The model correctly identifies **89% of churners**, meaning:  
  - Out of **325 actual churners**, it correctly predicted **290**.  

- **Precision (0.90):** Of all customers predicted as churners, **90% were correctly classified**, meaning:  
  - Out of **324 predicted churners**, **297 were actual churners**.  

- **Probability Scores:** The predicted probabilities align well with actual outcomes, with churners assigned higher probabilities, reinforcing the model’s reliability.  

| Model    | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  | KS       | Gini     | PR-AUC   | Brier    |
|----------|----------|-----------|----------|----------|----------|----------|----------|----------|----------|
| LightGBM | 0.965449 | 0.915309  | 0.864615 | 0.889241 | 0.989830 | 0.890217 | 0.979661 | 0.960793 | 0.028718 |

# 6. How to Run the Code Using Docker

To run the project using Docker, follow these steps:

1. Ensure you have **Docker** and **Docker Compose** installed on your system.
2. Navigate to the project directory where the `docker-compose.yml` file is located.
3. Run the following command:
   ```sh
   docker compose up --build -d
   ```
4. This command will build the required Docker images and start the services in detached mode.
5. To check the running containers, use:
  ```sh
  docker ps
  ```
6. To stop the containers, run:
  ```sh
  docker compose down
  ```