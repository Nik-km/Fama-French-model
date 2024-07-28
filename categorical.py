#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

# Data import

df = pd.read_csv("categorical_stock_data.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
# %%

#%% Categorical Charts 

sectors = df['sector'].unique()

for sector in sectors: 
    top = df[df['sector'] == sector].nlargest(5, 'market capitalization')
    print(top)



# %%
