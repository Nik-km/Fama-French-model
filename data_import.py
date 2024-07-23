#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yahooFinance
from fredapi import Fred

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)


#%% Import Data -----------------------------------------------------------------------------------
#>> Load stock returns (Date, Ticker, and Return columns)
df_SP = yahooFinance.Ticker("^GSPC").history(start='1990-01-01', interval='1mo', actions=True)
# Compute monthly log returns
df_SP["Returns"] = np.log(df_SP["Close"]/df_SP["Close"].shift(1))
df_SP = df_SP[["Close", "Volume", "Returns"]]
print(df_SP.head())
# Extract the date of the first observation
start_date = datetime.strftime(df_SP.index[0], '%m/%d/%Y'); start_date

#>> Associate Ticker with Sector  
SPX_companys = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
SPX_tickers = SPX_companys['Symbol'].str.replace('.', '-').tolist()
 
ticker_list = []
sector_list = []

for ticker in SPX_tickers:
    ticker_list.append(ticker)
    sector = yahooFinance.Ticker(ticker).info['sector']
    sector_list.append(sector)
    
stock_industry = pd.DataFrame({'ticker': ticker_list, 'sector': sector_list})
print(stock_industry)

#%% 
#>> Download All S&P 500 stocks
# Run if update needed 
#data = yahooFinance.download(SPX_tickers, interval = '1mo', start = '1990-01-01')
#data['Close'].to_csv('individual_stocks.csv') # used Close but also have Open, Adjust Close, High, Low /month 

# Run above if data is not downloaded
SPX_constituents = pd.read_csv('individual_stocks.csv')
SPX_constituents = pd.melt(SPX_constituents,id_vars = ['Date'])
SPX_constituents = SPX_constituents.rename(columns={'variable':'ticker', 'value':'closing-price'})

#Merge Into Single Dataframe & Save

SPX_constituents_clean = pd.merge(SPX_constituents,stock_industry, on='ticker')
SPX_constituents_clean.to_csv('individual_stocks_clean.csv')
#%%


#>> Import Ken French's data directly
url_FF5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
df_FF5 = pd.read_csv(url_FF5, compression="zip", skiprows=3)
print(df_FF5.head())

#>> Import FRED data series
API_key = open(path_file + "/FRED_api_key_file.txt", "r").read()
fred = Fred(api_key=API_key)

def get_FRED_data(tickers):
    df = pd.DataFrame()
    for k,v in tickers.items():
        series = fred.get_series(k, observation_start=start_date, frequency='m', aggregation_method='eop')
        series.name = v
        df = df.join(series, how='outer')
    return(df)

tickers = {
    'DGS10':'rf',
    'EXPINF10YR':'expected_cpi',
    # 'GDPC1':'real_gdp',
}
df_FRED = get_FRED_data(tickers)
df_FRED.head()


#%% Data Cleaning ---------------------------------------------------------------------------------
#>> Clean FF data
df_FF5.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
string_location = df_FF5[df_FF5['Date'].str.contains("Annual Factors: January-December", na=False)].index[0]
df_FF5 = df_FF5[:string_location]
df_FF5['Date'] = pd.to_datetime(df_FF5['Date'], format='%Y%m')
df_FF5.set_index('Date', inplace=True)
df_FF5 = df_FF5.apply(pd.to_numeric, errors='coerce')

# Normalize timezone in datetype indexes
df_FF5 = df_FF5.tz_localize(None)
df_SP = df_SP.tz_localize(None)
df_FRED

#>> Combine data series & compute excess returns
df = df_FF5.join(df_SP, how='inner')
df["excess_return"] = df["Returns"] - df["RF"] / 100
# Drop any rows with missing values
df = df.dropna()

df.to_csv('full_data.csv')


#%% Notes -----------------------------------------------------------------------------------------
# References:
    # https://mortada.net/python-api-for-fred.html

# TODO: Adjust get_FRED_data() fn. to normalize different series frequencies by introducing NA values
# TODO: Add FRED data to the final concatenated dataframe 'df' before exporting
# TODO: Pickle dictionary to move between files in "Associate Ticker with Sector"
