# Fama-French-model
Replicating the seminal 1993 paper "*Common risk factors in the returns on stocks and bonds*" published in JFE

## Sourcing Data
You must have the following python packages installed:
- `yfinance`
- `fredapi`

You must register/create a FRED account and then request an API key to be created, the link to request a key can be [found here](https://fred.stlouisfed.org/docs/api/api_key.html).
Once you've received an API key for your account (should be instantaneous), create a file in the main folder of this repository titled "`FRED_api_key_file.txt`". Afterwards, copy-&-paste your API key in this file and then save. You should now be able to run the script `data_import.py`.

