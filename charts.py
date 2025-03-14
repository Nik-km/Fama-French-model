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

df = pd.read_csv("data\\categorical_stock_data.csv")
df = df.drop(['Unnamed: 0'], axis = 1)


#%% Categorical Charts 

sectors = df['sector'].unique()

top5bysector = df[df['sector'] == 'Industrials']

print('Industrial', '\n', top5bysector.nlargest(5, 'market capitalization'), '\n')
print('Number of Tickers: ', len(top5bysector['ticker']))

for sector in sectors[1:]:
    df_sub = df[df['sector'] == sector] 
    top = df_sub.nlargest(5, 'market capitalization')
    #top.to_csv('Top5\\'+str(sector)+'-top 5.csv')
    print(sector, '\n', top, '\n')
    print('Number of Tickers: ', len(df_sub['ticker']))

    top5bysector = pd.concat([top5bysector, top], ignore_index=True)

top5bysector.to_csv('top5bysector.csv')
