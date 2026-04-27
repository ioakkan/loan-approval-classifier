import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Loading Loan Dataset
loan_data = pd.read_csv('dataset/loan_data.csv')


# Picking numerical columns of loan dataframe 
numerical_loan_data = loan_data.select_dtypes('number')
numeric_columns = numerical_loan_data.columns.to_list()
numeric_columns.remove('loan_status') # Removing Target feature of logistic regression from training features


status_map = {1: 'Approved', 0: 'Rejected'} 
loan_data['loan_decision'] = loan_data['loan_status'].map(status_map)

# Picking categorical columns of loan dataframe 
categorical_loan_data = loan_data.select_dtypes('str')
categorical_columns = categorical_loan_data.columns.to_list()
loan_data_corr_matrix = numerical_loan_data.corr()

# Checking  outliers that affect person_income histogram 
print(f" persons max income : {loan_data['person_income'].max()}")
print(f" persons mean income : {loan_data['person_income'].mean()}")
print(f" number of persons income  bigger than usual :{(loan_data['person_income']>250000).sum()}")
print(f" top 20 largest people income : \n {(loan_data['person_income']).nlargest(20)}")

# Checking  outliers that affect person_age histogram 
print(f"  max person_age  : {loan_data['person_age'].max()}")
print(f"  mean person_age : {loan_data['person_age'].mean()}")
print(f" number of person_age  bigger than 85 :{(loan_data['person_age']>85).sum()}")
print(f" top 20 oldest people income :\n {(loan_data['person_age']).nlargest(20)}")


# Checking  outliers that affect person_emp_exp histogram 
print(f" person_emp_exp max income : {loan_data['person_emp_exp'].max()}")
print(f"person_emp_exp mean income : {loan_data['person_emp_exp'].mean()}")
print(f" number of person_emp_exp bigger than usual :{(loan_data['person_emp_exp']>40).sum()}")
print(f" top 20 longest person_emp_exp  :\n {(loan_data['person_emp_exp']).nlargest(20)}")

print(f"checking {loan_data['cb_person_cred_hist_length'].value_counts()}")
print(f"checking {loan_data['person_emp_exp'].value_counts()}")

# cb_person_cred_hist_length_bin_tickers =loan_data['cb_person_cred_hist_length'].value_counts().sort_values(ascending=True).values

if __name__ == "__main__":
    # Quick review of Dataset
    loan_data.info()

    # Checking dataset for missing values
    print(loan_data.isna().sum()) #.sum() 

    #  Create folders to save Visualization figures  
    os.makedirs('visualization/histograms', exist_ok=True)
    os.makedirs('visualization/barcharts', exist_ok=True)
    os.makedirs('visualization/heatmaps', exist_ok=True)

    # Visualize dataset columns in different ways(histograms,barcharts,heatmaps) to gain insight
    
    # Heatmaps
    plt.figure(figsize = (8,10))
    sns.heatmap(
        data = loan_data_corr_matrix,
        cmap ='coolwarm',
        annot= True,
        cbar=False
    )
    plt.title('features_corr_heatmap')
    plt.xticks(rotation=55)
    plt.savefig('visualization/heatmaps/features_corr_heatmap.png')
    plt.close()

    # Histograms
    for column in numeric_columns:
     plt.figure(figsize = (8,10))
     if column == "loan_status":
       continue
     elif column == "cb_person_cred_hist_length_histogram":
        sns.histplot(
            data=loan_data,
            x=column,
            kde = True,
            binwidth=1,
            edgecolor='black'   
     )
     plt.title(f'Distribution of {column}')
     sns.histplot(
        data=loan_data,
        x=column,
        kde = True,
        edgecolor='black'
     )
     if column == "person_income":
      plt.xlim(0,250000)
     elif column == "person_age":
      plt.xlim(0,85)
     elif column == "person_emp_exp":
      plt.xlim(0,30)  
     plt.ticklabel_format(style='plain', axis='x')
     plt.tight_layout()
     plt.savefig(f'visualization/histograms/{column}_histogram.png')
     plt.close()
    
    # Barcharts
    for column in categorical_columns:
     plt.figure(figsize = (8,10))
     
     ax = sns.countplot(
        data=loan_data,
        x=column 
     )
     ax.bar_label(
        ax.containers[0],
        color='darkorange',
        fontsize=10,
        fontweight='bold'
    ) 
     if column == "loan_intent":
        plt.xticks(rotation=45)
    
     plt.tight_layout()
     plt.savefig(f'visualization/barcharts/{column}_barchart.png')
     plt.close()




