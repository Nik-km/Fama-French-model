### ECO 1500 TERM PROJECT DATA ANALYSIS

### IMPORT STATEMENTS 
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import yfinance as yf

def get_SPX_data(start, end=None, save = False, fetch = True):
    # start is the beginning date of the data download 
    # end is the ending data of the data download 
    # save is a binary variable indicating if the data is saved locally 
    # fetch is a binary variable indicating if the data is pulled locally

    if fetch == True:
        # get tickers 
        SPX_tickers = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        SPX_tickers_list = SPX_tickers['Symbol'].str.replace('.', '-').tolist()
        SPX_tickers_list.append("^SPX")

        # try downloading data 
        try:
            # download monthly data 
            data = yf.download(SPX_tickers_list, start=start, end=end, group_by='ticker', interval = "1mo")
            # TO DO: Need to better organize data for panel regression & CSV local save 
                # CONSIDER: pkl instead of CSV?

            if save == True: 
                data.to_csv('spxdata.csv')

        except: 
            print("Error data not downloaded!")

        # return data 
        return data
    else: 
        return pd.read_csv('spxdata.csv')

def main(): 
    ### DATA ACQUISTION

    ## SET DATA RANGE
    start_date = "1960-01-01"
    end_date = "2024-07-01"

    ## YAHOO FINANCE DATA - SPX Components
    spx = get_SPX_data(start_date)

    print("SPX Components Tail \n",
            "Head \n", 
            spx.head(),
            "\n",
            "Tail \n",
            spx.tail())

    ## FAMA & FRENCH DATA 
    factors_ff3_monthly = pdr.DataReader(
    name="F-F_Research_Data_Factors",
    data_source="famafrench", 
    start=start_date, 
    end=end_date)[0]

    print("FF3 Factors Tail \n",
          "Head \n",
          factors_ff3_monthly.head(),
          "\n",
          "Tail \n",
          factors_ff3_monthly.tail())

main()