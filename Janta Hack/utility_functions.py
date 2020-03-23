#Utility Functions
'''
This file contains frequently used functions for data cleaning and data analysis.
This file can be imported as a package by another programs or a Jupyter Notebook.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from collections import Counter

#Function for outlies analysis
def outlier_analysis(data, variable, capping = False):
    #Plot the boxplot
    plt.boxplot(data[variable])

    #Get the quartile, the 25th and the 75th percentiles.
	#One can define his/her own percentiles. 
    q_25, q_75 = np.percentile(data[variable], [25, 75])
    
    #IQR
    IQR = q_75 - q_25
    
    #Obtaining the lower and upper bounds
    lower_bound, upper_bound = (q_25 - (1.5*IQR), q_75 + (1.5*IQR))
    lower = data.loc[data[variable]<= lower_bound, variable]
    upper = data.loc[data[variable]>= upper_bound, variable]

    #Capping the Outliers
    if(capping):
        data.loc[data[variable]<= lower_bound, variable] = lower_bound
        data.loc[data[variable]>= upper_bound, variable] = upper_bound

#Function for One Hot Encoding categorical features
def OHE(data, categorical_features):
    
    dummy_data = pd.get_dummies(data[categorical_features])
    
    #Get the encoded variables
    encoded_variables = dummy_data.columns.tolist()
    
    #Append the dummy_data to the orignal data frame
    data = pd.concat([data, dummy_data], axis = 1)
    
    return (data, encoded_variables)

#Missing Value Analysis
def find_missing(data):
	#List of all the features present in the data
    features = data.columns.tolist()
    missing_count = data.isna().sum() #Obtain the count of missing values for each column
    missing_percentage = data.isna().sum()*100/data.shape[0] #Obtain the percentage of missing values for each column

	#Create a data frame containing the above information
    missing_data = pd.DataFrame({'Features': features,\
                                 'Missing Count': missing_count, \
                                 'Missing Percentage': missing_percentage},
                               columns = ['Features', 'Missing Count',\
                                          'Missing Percentage'])

    missing_data = missing_data.set_index('Features')
    
    missing_data = missing_data.sort_values(by = 'Missing Count',\
                                            ascending = False)
    
    return (missing_data)

#Seggregate the columns of a data frame based on their data types
def seggregate_columns_basis_types(data):
    integer_cols, float_cols = [], []
    categorical_cols, other_cols = [], []
    
    for col in data.columns:
        if(data[col].dtype == 'int64'):
            integer_cols.append(col)
        elif(data[col].dtype == 'float64'):
            float_cols.append(col)
        elif(data[col].dtype == 'O'):
            categorical_cols.append(col)
        else:
            other_cols.append(col)
            
    return (integer_cols, float_cols, categorical_cols, other_cols)

#Correlation Analysis
def karl_pearson(data, numeric_variables):
	#Set the dimensions of the correlation matrix
	f, ax = plt.subplots(figsize = (7, 7))
	
	#The correlation matrix	
	corr_mat = data[numeric_variables].corr()
	
	sbn.heatmap(corr_mat,
				mask = np.zeros_like(corr_mat, dtype = np.bool),
				cmap = sbn.diverging_palette(220, 20, as_cmap = True),
				square = True,
				annot = True,
 				ax = ax)


