import tushare as ts
import pandas as pd
import time
import LB
import traceback
from jqdatasdk import *
import DB

for i in range(10):
    try:
        auth('13817373362', '373362')
    except:
        pass

def break_jq_limit_helper_finance(code, limit=5000):
    """for some reason tushare only allows fund，forex to be given at max 1000 entries per request"""
    #first call
    df=finance.run_query(query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == code ))
    len_df_this = len(df)
    df_last = df

    while len_df_this == limit:  # TODO if this is fixed or add another way to loop
        day = df_last.at[len(df_last) - 1, "day"]
        df_this=finance.run_query(query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == code, finance.GLOBAL_IDX_DAILY.day > day ))
        if (df_this.equals(df_last)):
            break
        df = df.append(df_this, sort=False, ignore_index=True).drop_duplicates(subset="day")
        len_df_this = len(df_this)
        df_last = df_this
    return df




def break_jq_limit_helper_xueqiu(code, limit=3000):
    """for some reason tushare only allows fund，forex to be given at max 1000 entries per request"""
    #first call
    df=finance.run_query(query(finance.STK_XUEQIU_PUBLIC).filter(finance.STK_XUEQIU_PUBLIC.code == code ))
    len_df_this = len(df)
    df_last = df

    while len_df_this == limit:  # TODO if this is fixed or add another way to loop
        day = df_last.at[len(df_last) - 1, "day"]
        df_this=finance.run_query(query(finance.STK_XUEQIU_PUBLIC).filter(finance.STK_XUEQIU_PUBLIC.code == code, finance.STK_XUEQIU_PUBLIC.day > day ))
        if (df_this.equals(df_last)):
            break
        df = df.append(df_this, sort=False, ignore_index=True).drop_duplicates(subset="day")
        len_df_this = len(df_this)
        df_last = df_this
    return df

def my_cctv_news(day):
    return finance.run_query(query(finance.CCTV_NEWS).filter(finance.CCTV_NEWS.day == day))


def my_margin(date=""):
    date=LB.trade_date_switcher(str(date))
    df= finance.run_query(query(finance.STK_MT_TOTAL).filter(finance.STK_MT_TOTAL.date == date).limit(10))
    #df["date"]=df["date"].apply(LB.trade_date_switcher)
    return df

def my_get_bars(jq_code,freq):
    df=get_bars(security=jq_code,count=5000000,unit=freq,fields=["date","open","high","low","close"])
    return df

if __name__ == '__main__':
    pass

    df=my_get_bars(LB.ts_code_switcher("600519.SH"))
    print(df)

    # df=break_jq_limit_helper_xueqiu(code="002501.XSHE")
    # df["trade_date"]=df["day"].apply(LB.trade_date_switcher)
    # df["trade_date"]=df["trade_date"].astype(int)
    # df_asset=DB.get_asset("002501.SZ")
    # df_asset=LB.ohlc(df_asset)
    # df=pd.merge(df_asset,df,how="left",on="trade_date")
    # df.to_csv("xueqiu.csv")
    # df = get_concept_stocks(concept_code="GN185")
    # print(df)
    # df.to_csv("concept.csv",encoding='utf-8_sig')
    # a_global_index = ["800000.XHKG", "INX", "KS11", "FTSE", "RTS", "MIB", "GDAXI", "N225", "IBEX", "FCHI", "IBOV", "MXX", "GSPTSE"]
    # for code in a_global_index:
    #     df= break_jq_limit_helper_finance(code=code, limit=5000)
    #     print(len(df))

    # a_global_index = ["800000.XHKG", "INX", "KS11", "FTSE", "RTS", "MIB", "GDAXI", "N225", "IBEX", "FCHI", "IBOV", "MXX", "GSPTSE"]
    # for code in a_global_index:
    #     df = finance.run_query(query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == code, finance.GLOBAL_IDX_DAILY.day > "2015-01-01" ))
    #     print(df["day"])
