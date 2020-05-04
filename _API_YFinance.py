import numpy as np
import yfinance as yf


"""Very Very slow and requires VPN"""
def convert_date_tushare_to_yfinance(date):
    result = date[0:4] + "_" + date[4:6] + "_" + date[6:8]
    return result

def download_asset(ticker="AAPL"):
    msft = yf.Ticker(ticker)

    # get stock info
    try: #sometimes can cause error
        info=msft.info
    except:
        info={"sector":np.nan,"industry":np.nan}

    sector=info["sector"] if "sector" in info else np.nan
    industry=info["industry"] if "industry" in info else np.nan
    # get historical market data
    df = msft.history(period="max")
    return [df,sector,industry]



