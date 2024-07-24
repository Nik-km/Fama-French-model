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
# output = model.summary().as_latex()
# print(output)


#%% Diagnostic Tests ------------------------------------------------------------------------------

#>> Testing Multicollinearity using VIF
def calc_VIF(df_exog):
    '''
    Parameters
    ----------
    df_exog : dataframe, (n_obs, k_vars)
        Design matrix with all explanatory variables used in a regression model.

    Returns
    -------
    VIF : Series
        Variance inflation factors
    '''
    # Compute the correlation matrix for the covariates
    corr_matrix = df_exog.corr().to_numpy()
    # Compute the inverse of the correlation matrix & extract only the diagonal elements
    inv_corr = np.linalg.inv(corr_matrix).diagonal()
    vifs = pd.Series(inv_corr, index=df_exog.columns, name='VIF')
    return vifs

X = df[['Mkt-RF', 'SMB', 'HML']]
calc_VIF(X)

#>> Other normality tests
tests = sm.stats.stattools.jarque_bera(model.resid)
print(pd.Series(tests, index=['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']))


#%% Diagnostic Plots ------------------------------------------------------------------------------

#>> Q-Q Plot
def gen_QQ_plot(mod_res):
    df_res = pd.DataFrame(sorted(mod_res), columns=['residual'])
    # Calculate the Z-score for the residuals
    df_res['z_actual'] = (df_res['residual'].map(lambda x: (x - df_res['residual'].mean()) / df_res['residual'].std()))
    # Calculate the theoretical Z-scores
    df_res['rank'] = df_res.index + 1
    df_res['percentile'] = df_res['rank'].map(lambda x: x/len(df_res.residual))
    df_res['theoretical'] = stats.norm.ppf(df_res['percentile'])
    # Construct QQ plot
    with plt.style.context('ggplot'):
        plt.figure(figsize=(9,9))
        plt.scatter(df_res['theoretical'], df_res['z_actual'], color='blue')
        plt.xlabel('Theoretical Quantile')
        plt.ylabel('Sample Quantile')
        plt.title('Normal QQ Plot')
        plt.plot(df_res['theoretical'], df_res['theoretical'])
        plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.savefig(path_file + "\\output\\QQ_plot.png")
        plt.show()
    return(df_res)

gen_QQ_plot(model.resid)


#%% Notes -----------------------------------------------------------------------------------------
# References:
    # https://mortada.net/python-api-for-fred.html

# Mkt_RF (Rm - Rf) is the return spread between the capitalization-weighted stock market and cash
# SMB is the return spread of small minus large stocks (i.e. the size effect)
# HML is the return spread of cheap minus expensive stocks (i.e. the value effect)
# RMW is the return spread of the most profitable firms minus the least profitable
# CMA is the return spread of firms that invest conservatively minus aggressively (AQR, 2014)
