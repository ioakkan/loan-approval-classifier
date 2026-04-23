import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from logistic_regression.loan_data_eda import  numeric_columns  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    average_precision_score, # PR AUC or # import auc
    recall_score,
    f1_score,
    roc_auc_score, # ROC-AUC
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import StandardScaler #, MinMaxScaler, RobustScaler commenting other scaling options


# Loading Loan Dataset
loan_data = pd.read_csv('dataset/loan_data.csv')

# Creating Folder to Save Confusion matrix of classification
os.makedirs("visualization/confusion_matrix",exist_ok=True)
os.makedirs("visualization/roc_curve",exist_ok=True)
os.makedirs("visualization/PR_curve",exist_ok=True)

# X training features 
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

# model prediction labels
y_pred_train = logistic_model.predict(X_train_scaled) # labels (n_samples,)

# model prediction probabilities
y_prob_train = logistic_model.predict_proba(X_train_scaled)   # probabilities (n_samples,n_classes) 

print(f"checking y_pred_train shape:{y_pred_train.shape} checking y_prob_train shape:{y_prob_train.shape}")

#  # Grabbing the predicted probabilities of class 1   ->  1:'approved loan'  (less common class - minority ) 
y_prob_train = logistic_model.predict_proba(X_train_scaled)[:,1]     # (class 1  samples are considered positive for predictions)

print(f" the classes-order shape is: {logistic_model.classes_}") # checking the order of classes that matches the order of probabilities 

# Logistic Regression Metrics Training metrics
print("-------- LOGISTIC REGRESSION TRAINING METRICS  ---------- ")
print("Accuracy:", accuracy_score(y_train,y_pred_train)) 
print("Precision:", precision_score(y_train,y_pred_train,pos_label=1)) # pos_label by default is 1 , arguement can be removed specified  its used just to make it clear  samples of class 1 are considered positive
print("Recall:", recall_score(y_train,y_pred_train,pos_label=1))
print("F1:", f1_score(y_train,y_pred_train,pos_label=1))
print("ROC-AUC:", roc_auc_score(y_train,y_prob_train)) 
print("Log Loss:", log_loss(y_train,y_prob_train))
print("PR_Curve-AUC", average_precision_score(y_train,y_prob_train))

# test model

# model prediction labels
y_pred = logistic_model.predict(X_test_scaled)

# model prediction probabilities
y_prob = logistic_model.predict_proba(X_test_scaled) 
 
# Grabbing the predicted probabilities of class 1
y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1] #   (class 1  samples are considered positive for predictions)


print(f"checking y_pred shape:{y_pred.shape} checking y_prob shape:{y_prob.shape}")


# Logistic Regression Metrics Test Evaluation
test_accuracy_score = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_prob)
test_log_loss = log_loss(y_test, y_prob)
test_pr_auc = average_precision_score(y_test,y_prob)

print("-------- LOGISTIC REGRESSION EVALUATION METRICS  ---------- ")
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Precision:", test_precision ) 
print("Recall:", test_recall ) 
print("F1:", test_f1) # 
print("ROC-AUC:", test_roc_auc )
print("Log Loss:", test_log_loss)
print("PR_Curve-AUC", test_pr_auc)

# Metrics after scaling and checking  correct arguements for each metric
# Accuracy: 0.8243333333333334     # Indicates overall predictions of model (can be misleading in our case data is imbalanced 35000 samples loans rejected 10000 approved )
# Precision: 0.6780082987551868   ( Indicates How many approved loans the model predicted are correct(legit approved loans) )
# Recall: 0.4064676616915423 ( Indicates How many approved loans the model predicted (How  many approved loans it missed )
# F1: 0.5082426127527216 # Indicates the balance between precision-recall
# ROC-AUC: 0.8283293119523982 # Indicates how well  the model separates the two classes (distinct difference in probabilities of two classes )
# Log Loss: 0.39774428419572205 # Log loss indicates the score the adjusted final weights result to. (the lesser the better)
# PR_Curve-AUC 0.6199420262971839 # Indicates  how much the change in precision or recall affect each other


# Visualization Of metrics
labels = ["Approved", "Rejected"]
cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize= (8,10))
sns.heatmap(
    cm,
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("visualization/confusion_matrix/confusion_matrix.png")

error_cm = cm.copy()
np.fill_diagonal(error_cm, 0)
plt.figure(figsize= (8,10))
sns.heatmap(
    error_cm, 
    annot=True, 
    fmt='d', 
    cmap='Reds',
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Misclassification Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("visualization/confusion_matrix/Misclassification_Matrix.png")


fpr, tpr,thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize= (8,10))
plt.plot(fpr, tpr, label=f"AUC = {test_roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("visualization/roc_curve/roc_curve.png")


precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize= (8,10))
plt.plot(recall, precision, label=f"AP = {test_pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.savefig("visualization/pr_curve/pr_curve.png")
