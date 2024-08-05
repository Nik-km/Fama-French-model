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
df_SP = yahooFinance.Ticker("^SPX").history(start='1990-01-01', interval='1mo', actions=True)

# Compute monthly log returns
df_SP["Returns"] = np.log(df_SP["Close"]/df_SP["Close"].shift(1)) * 100
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
market_capitalization_list = []
name_list = []

for ticker in SPX_tickers:
    ticker_list.append(ticker)
    yFdata = yahooFinance.Ticker(ticker)
    
    sector = yFdata.info['sector']
    sector_list.append(sector)
    
    market_cap = yFdata.info['marketCap']
    market_capitalization_list.append(market_cap/1000000)

    name = yFdata.info['longName']
    name_list.append(name)

stock_industry = pd.DataFrame({'ticker': ticker_list, 'sector': sector_list})

stock_categorical = pd.DataFrame({'ticker': ticker_list, 'name':name_list, 'sector': sector_list, 'market capitalization': market_capitalization_list})
stock_categorical.to_csv('data\\categorical_stock_data.csv')


#%%  Download All S&P 500 stocks
# Run if update needed 
data = yahooFinance.download(SPX_tickers, interval = '1mo', start = '1990-01-01', end = '2024-06-01')
close_data = data['Close']

for ticker in SPX_tickers: 
    close_data[ticker] = np.log(close_data[ticker]/close_data[ticker].shift(1)) * 100

close_data.to_csv('data\\individual_stocks.csv') # used Close but also have Open, Adjust Close, High, Low /month 

# Run above if data is not downloaded
SPX_constituents = pd.read_csv('data\\individual_stocks.csv')

SPX_constituents = pd.melt(SPX_constituents,id_vars = ['Date'])
SPX_constituents = SPX_constituents.rename(columns={'variable':'ticker', 'value':'return'})

# Merge Into Single Dataframe & Save
SPX_constituents_clean = pd.merge(SPX_constituents, stock_industry, on='ticker')

SPX_constituents_clean.to_csv('data\\individual_stocks_clean.csv')

#%% Create Sector Portfolios 
SPX_constituents_clean = SPX_constituents_clean.set_index('Date')
sectors = SPX_constituents_clean['sector'].unique().tolist()
portfolios = pd.DataFrame()

for sector in sectors: 
    temp_sector_dataframe = SPX_constituents_clean[SPX_constituents_clean['sector'] == sector]
    tickers_in_sector = temp_sector_dataframe['ticker'].unique().tolist()
    sector_portfolio = pd.DataFrame() 

    for tk in tickers_in_sector: 
        ticker_data = temp_sector_dataframe[temp_sector_dataframe['ticker'] == tk]
        sector_portfolio[tk] = ticker_data['return']
        
    sector_portfolio.to_csv('PortfolioValidation\\'+str(sector)+'-sector_portfolio.csv')
    portfolios[sector] = sector_portfolio.mean(axis=1)

portfolios = portfolios.drop(['1990-01-01'])
portfolios.to_csv('data\\industry_portfolios.csv')


#%% Import Ken French's Data Directly
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
df["excess_return"] = df["Returns"] - df["RF"]
# Drop any rows with missing values
df = df.dropna()

df.to_csv(path_file + '\\data\\full_data.csv')
print("Completed.")


#%% Notes -----------------------------------------------------------------------------------------
# References:
    # https://mortada.net/python-api-for-fred.html

# TODO: Adjust get_FRED_data() fn. to normalize different series frequencies by introducing NA values
# TODO: Add FRED data to the final concatenated dataframe 'df' before exporting
