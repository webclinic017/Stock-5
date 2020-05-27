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



if __name__ == '__main__':
    import LB
    #df=my_stock_historical_data("AAPL", country="united states", from_date="01/01/2000", to_date=LB.trade_date_to_investpy(LB.today()))
    df=download_asset(ticker="AAPL")
    print(df)