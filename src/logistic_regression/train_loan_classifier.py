import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from logistic_regression.loan_data_eda import  numeric_columns  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# Loading Loan Dataset
loan_data = pd.read_csv('dataset/loan_data.csv')

# X features 
features_df =  loan_data[numeric_columns] # After loan_data EDA , Picked the numeric columns of Dataset for training features

# Y target feature
loan_status  = loan_data['loan_status']

print(f" features for training are : {numeric_columns}")


X_train, X_test, y_train, y_test = train_test_split(
    features_df,
    loan_status,
    train_size=0.8,
    random_state=42,
)

print(f' X_train length is: {len(X_train)} \n y_train length is: {len(y_train)}')
print(f' X_test  length is: {len(X_test)} \n y_test length is: {len(y_test)}')

print(f'X_train  Shape is: {(X_train.shape)} \n  y_train Shape is: {y_train.shape}')
print(f'X_test  Shape is: {X_test.shape} \n y_test Shape is: {y_test.shape}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Create model instance

logistic_model = LogisticRegression(max_iter=100) 

# Train model

logistic_model.fit(X_train_scaled,y_train)

# test model
y_pred = logistic_model.predict(X_test_scaled)

y_prob = logistic_model.predict_proba(X_test_scaled) 

print(f"this is  ypred : {y_prob[0]}")   

y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

# model prediction shape (checking if their shape  align)
print(f" y_pred shape is: {y_pred.shape}")
print(f" y_prob shape is: {y_prob.shape}")


# Logistic Regression Metrics
print("-------- LOGISTIC REGRESSION METRICS  ---------- ")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Log Loss:", log_loss(y_test, y_prob))


#  Logistic Regression Metrics  on training dataset after scaling features
# Accuracy: 0.8243333333333334
# Precision: 0.6780082987551868
# Recall: 0.4064676616915423
# F1: 0.5082426127527216
# ROC-AUC: 0.8283293119523982
# Log Loss: 0.39774428419572205