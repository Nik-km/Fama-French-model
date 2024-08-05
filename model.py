#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats as stats
import statsmodels.stats.stattools as smt
import scipy.stats
from stargazer.stargazer import Stargazer, LineLocation

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

# Data import
df = pd.read_csv("data\\full_data.csv", index_col='Date')
df_ind = pd.read_csv('data\\industry_portfolios.csv', index_col= 'Date')


#%% Build Model(s) --------------------------------------------------------------------------------
#>> Fama-French 5 Factor Model
# Define the independent variables (Fama-French factors)
X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
# Add a constant (intercept) to the model
X = sm.add_constant(X)
# Define the dependent variable (Excess Return)
y = df['excess_return']

# Perform the regression
model_FF5 = sm.OLS(y, X).fit()
model_FF5.summary()

# Uncomment to get LaTeX output
print(model_FF5.summary().as_latex())

#>> Fama-French 3 Factor Model
# Define the independent variables (Fama-French factors)
X = df[['Mkt-RF', 'SMB', 'HML']]
X = sm.add_constant(X)
y = df['excess_return']

# Perform the regression
model_FF3 = sm.OLS(y, X).fit()
model_FF3.summary()

# Uncomment to get LaTeX output
print(model_FF3.summary().as_latex())

#>> Sector Regressions
industry_results = {}

# Run industry models
for industry in df_ind.columns:
    # Subset industry
    y = df_ind[industry]
    # Find the common indices
    common_indices = X.index.intersection(y.index)
    # Filter both DataFrames to only include the common indices
    X_common = X.loc[common_indices]
    y_common = y.loc[common_indices]
    # Build regression model
    model = sm.OLS(y_common, X_common).fit()
    industry_results[industry] = model
    print('-'*10, industry, '-'*10)
    print(model.summary())

# Printout LaTeX table of sector regressions
mod_sectors = [industry_results['Technology'], industry_results['Energy'], industry_results['Financial Services']]
stargazer = Stargazer(mod_sectors)
stargazer.add_line('AIC', [mod_sectors[0].aic.round(2), mod_sectors[1].aic.round(2), mod_sectors[2].aic.round(2)], LineLocation.FOOTER_TOP)
stargazer.add_line('Skew', [smt.jarque_bera(mod_sectors[0].resid)[2].round(4), smt.jarque_bera(mod_sectors[1].resid)[2].round(4), smt.jarque_bera(mod_sectors[2].resid)[2].round(4)], LineLocation.FOOTER_TOP)
stargazer.add_line('Kurtosis', [smt.jarque_bera(mod_sectors[0].resid)[3].round(2), smt.jarque_bera(mod_sectors[1].resid)[3].round(2), smt.jarque_bera(mod_sectors[2].resid)[3].round(2)], LineLocation.FOOTER_TOP)
stargazer.add_line('Durbin-Watson', [smt.durbin_watson(mod_sectors[0].resid).round(3), smt.durbin_watson(mod_sectors[1].resid).round(3), smt.durbin_watson(mod_sectors[2].resid).round(3)], LineLocation.FOOTER_TOP)
print(stargazer.render_latex())

# Printout the remaining models
mod_sectors = [
    industry_results['Healthcare'], 
    industry_results['Industrials'], 
    industry_results['Consumer Cyclical'], 
    industry_results['Consumer Defensive'], 
    industry_results['Utilities'], 
    industry_results['Basic Materials'], 
    industry_results['Real Estate'], 
    industry_results['Communication Services'], 
]
stargazer = Stargazer(mod_sectors)
stargazer.add_line('AIC', [
        mod_sectors[0].aic.round(2), 
        mod_sectors[1].aic.round(2), 
        mod_sectors[2].aic.round(2),
        mod_sectors[3].aic.round(2),
        mod_sectors[4].aic.round(2),
        mod_sectors[5].aic.round(2),
        mod_sectors[6].aic.round(2),
        mod_sectors[7].aic.round(2)
    ], LineLocation.FOOTER_TOP)
stargazer.add_line('Skew', [
        smt.jarque_bera(mod_sectors[0].resid)[2].round(4), 
        smt.jarque_bera(mod_sectors[1].resid)[2].round(4), 
        smt.jarque_bera(mod_sectors[2].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[3].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[4].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[5].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[6].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[7].resid)[2].round(4),
    ], LineLocation.FOOTER_TOP)
stargazer.add_line('Kurtosis', [
        smt.jarque_bera(mod_sectors[0].resid)[3].round(2), 
        smt.jarque_bera(mod_sectors[1].resid)[3].round(2), 
        smt.jarque_bera(mod_sectors[2].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[3].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[4].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[5].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[6].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[7].resid)[3].round(2),
    ], LineLocation.FOOTER_TOP)
stargazer.add_line('Durbin-Watson', [
        smt.durbin_watson(mod_sectors[0].resid).round(3), 
        smt.durbin_watson(mod_sectors[1].resid).round(3), 
        smt.durbin_watson(mod_sectors[2].resid).round(3),
        smt.durbin_watson(mod_sectors[3].resid).round(3),
        smt.durbin_watson(mod_sectors[4].resid).round(3),
        smt.durbin_watson(mod_sectors[5].resid).round(3),
        smt.durbin_watson(mod_sectors[6].resid).round(3),
        smt.durbin_watson(mod_sectors[7].resid).round(3),
    ], LineLocation.FOOTER_TOP)
print(stargazer.render_latex())

del(common_indices, industry, X_common, y_common)


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

#>> Breusch-Pagan (1979) & White (1980) Tests
def run_diag_test(type, resid, exog):
    if type == "BP":
        test_results = stats.diagnostic.het_breuschpagan(resid, exog)
        test_labels = ['Lagrange multiplier statistic', 'p-value', 'F-Statistic', 'F-Statistic p-value']
        title = "Breusch-Pagan Test Results"
    elif type == "White":
        test_results = stats.diagnostic.het_white(resid, exog)
        test_labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Statistic p-value']
        title = "White Test Results"
    test_results = [round(i, 4) for i in list(test_results)]
    output = dict(zip(test_labels, test_results))
    print(title)
    for key, value in output.items():
        print(f"{key}: {value}")

run_diag_test("BP", model_FF3.resid, model_FF3.model.exog)
run_diag_test("White", model_FF3.resid, model_FF3.model.exog)

stats.diagnostic.het_white(model_FF3.resid, model_FF3.model.exog)

#>> Other normality tests
tests = sm.stats.stattools.jarque_bera(model_FF3.resid)
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
    df_res['theoretical'] = scipy.stats.norm.ppf(df_res['percentile'])
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

