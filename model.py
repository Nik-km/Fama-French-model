#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

# Data import
df = pd.read_csv("full_data.csv", index_col='Date')

#%% Build Model(s) --------------------------------------------------------------------------------
# Define the independent variables (Fama-French factors)
X = df[['Mkt-RF', 'SMB', 'HML']]
# Add a constant (intercept) to the model
X = sm.add_constant(X)
# Define the dependent variable (Excess Return)
y = df['excess_return']

# Perform the regression
model = sm.OLS(y, X).fit()
reg_summary = model.summary()
reg_summary

# Uncomment to get LaTeX output
output = model.summary().as_latex()
print(output)


#%% Notes -----------------------------------------------------------------------------------------
# References:
    # https://mortada.net/python-api-for-fred.html

# Mkt_RF (Rm - Rf) is the return spread between the capitalization-weighted stock market and cash
# SMB is the return spread of small minus large stocks (i.e. the size effect)
# HML is the return spread of cheap minus expensive stocks (i.e. the value effect)
# RMW is the return spread of the most profitable firms minus the least profitable
# CMA is the return spread of firms that invest conservatively minus aggressively (AQR, 2014)
