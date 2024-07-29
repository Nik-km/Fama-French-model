#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.stattools as smt
import statsmodels.stats.descriptivestats as smd
import scipy.stats as stats
from stargazer.stargazer import Stargazer, LineLocation

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

# Data import
df = pd.read_csv("full_data.csv", index_col='Date')

df_ind = pd.read_csv('industry_portfolios.csv', index_col= 'Date')


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

#>> Fama-French 5 Factor Model
# Define the independent variables (Fama-French factors)
X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
X = sm.add_constant(X)
y = df['excess_return']

# Perform the regression
model = sm.OLS(y, X).fit()
model.summary()

# Uncomment to get LaTeX output
print(model.summary().as_latex())

#>> Sector Regressions
industry_results = {}

# Run industry models
for industry in df_ind.columns: 
    model = sm.OLS(df_ind[industry],X).fit()
    industry_results[industry] = model
    print('-'*10, industry, '-'*10)
    # print(model.summary().as_latex())

# Printout LaTeX table of sector regressions
mod_sectors = [industry_results['Technology'], industry_results['Energy'], industry_results['Financial Services']]
stargazer = Stargazer(mod_sectors)
stargazer.add_line('AIC', [mod_sectors[0].aic.round(2), mod_sectors[1].aic.round(2), mod_sectors[2].aic.round(2)], LineLocation.FOOTER_TOP)
stargazer.add_line('Skew', [smt.jarque_bera(mod_sectors[0].resid)[2].round(4), smt.jarque_bera(mod_sectors[1].resid)[2].round(4), smt.jarque_bera(mod_sectors[2].resid)[2].round(4)], LineLocation.FOOTER_TOP)
stargazer.add_line('Kurtosis', [smt.jarque_bera(mod_sectors[0].resid)[3].round(2), smt.jarque_bera(mod_sectors[1].resid)[3].round(2), smt.jarque_bera(mod_sectors[2].resid)[3].round(2)], LineLocation.FOOTER_TOP)
stargazer.add_line('Durbin-Watson', [smt.durbin_watson(mod_sectors[0].resid).round(3), smt.durbin_watson(mod_sectors[1].resid).round(3), smt.durbin_watson(mod_sectors[2].resid).round(3)], LineLocation.FOOTER_TOP)
print(stargazer.render_latex())


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

#>> Partial Regression Plots
# Residual plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
fig.savefig(path_file + "\\output\\partial_reg_plots.png")
fig

#>> Residuals vs. Fitted Plot
with plt.style.context('ggplot'):
    plt.figure(figsize=(9,9))
    plt.scatter(model.fittedvalues, model.resid, color='orange')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Residual by Predicted')
    plt.axhline(y = 0, color = 'black', linestyle = '-') 
    plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.savefig(path_file + "\\output\\fitted_res_plot.png")
    plt.show()

#>> Scale-Location Plot
with plt.style.context('ggplot'):
    plt.figure(figsize=(9,9))
    plt.scatter(model.fittedvalues, np.sqrt(model.resid), color='orange')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Scale-Location')
    plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.savefig(path_file + "\\output\\scale_location.png")
    plt.show()




#%% Notes -----------------------------------------------------------------------------------------
# References:
    # https://mortada.net/python-api-for-fred.html

# Mkt_RF (Rm - Rf) is the return spread between the capitalization-weighted stock market and cash
# SMB is the return spread of small minus large stocks (i.e. the size effect)
# HML is the return spread of cheap minus expensive stocks (i.e. the value effect)
# RMW is the return spread of the most profitable firms minus the least profitable
# CMA is the return spread of firms that invest conservatively minus aggressively (AQR, 2014)

#>> Partial Regression Plots
# Ideally, this plot should show a linear trend, but from the graph it shows that there doesn't seem to be
# divergence of the trends for each covariate. This also further implies that the expected value of the
# residuals does equal zero.
# The partial regression plots do not show any departures from the assumptions on the random error for
# Mkt-RF, SMB, & HML

#>> Residuals vs. Fitted Plot
# Checking whether the residuals and fitted values are correlated, ideally, this plot should show a band
# of random scattering about when the residuals equal zero (ie. about the horizontal axis). Since there
# seems to be a exponential pattern of the residuals in this plot, this implies that the variance of the
# residuals is not constant, which means that one or more of our model's assumptions is violated as well.

