CREATE TABLE IF NOT EXISTS bank_churners (
    CLIENTNUM BIGINT PRIMARY KEY,
    Attrition_Flag TEXT,
    Gender TEXT,
    Education_Level TEXT,
    Marital_Status TEXT,
    Income_Category TEXT,
    Card_Category TEXT
);