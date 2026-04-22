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

# Picking categorical columns of loan dataframe 
categorical_loan_data = loan_data.select_dtypes('object')
categorical_columns = categorical_loan_data.columns.to_list()
loan_data_corr_matrix = numerical_loan_data.corr()



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

    for column in numeric_columns:
     plt.figure(figsize = (8,10))
     plt.title(f'Distribution of {column}')
     sns.histplot(
     data = loan_data,
     x =  column,
     kde = True
     )
     plt.savefig(f'visualization/histograms/{column}_histogram.png')
     plt.clf()
    
    for column in categorical_columns:
     plt.figure(figsize = (8,10))
     sns.countplot(
     data= loan_data,
     x = column 
     )
     plt.savefig(f'visualization/barcharts/{column}_barchart.png')
     plt.clf()

     plt.figure(figsize = (8,10))
    sns.heatmap(
    data = loan_data_corr_matrix,
    cmap ='coolwarm',
    annot= True
    )
    plt.savefig('visualization/heatmaps/features_corr_heatmap.png')

 


