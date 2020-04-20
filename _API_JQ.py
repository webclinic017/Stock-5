import tushare as ts
import pandas as pd
import time

from jqdatasdk import macro, query

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

    while len_df_this >= limit:  # TODO if this is fixed or add another way to loop
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
    date=LB.switch_trade_date(str(date))
    df= finance.run_query(query(finance.STK_MT_TOTAL).filter(finance.STK_MT_TOTAL.date == date).limit(10))
    #df["date"]=df["date"].apply(LB.trade_date_switcher)
    return df

def my_get_bars(jq_code,freq):
    df=get_bars(security=jq_code,count=5000000,unit=freq,fields=["date","open","high","low","close"])
    return df

if __name__ == '__main__':
    pass

    df=my_get_bars(LB.switch_ts_code("600519.SH"))
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


def my_macro_run(query_content):
    macro.run_query(query(query_content))


def my_macro(macro_query):
    query_result = query(macro_query)
    df = macro.run_query(query_result)

    query_from = str(query_result).split("FROM")[1]
    query_from = query_from.replace(" ", "")
    query_from = query_from.replace('"', '')

    index_label = "stat_quarter"
    if "QUARTER" in query_from:
        index_label = "stat_quarter"
    elif "MONTH" in query_from:
        index_label = "stat_month"
    elif "YEAR" in query_from:
        index_label = "stat_year"
    else:
        columns = df.columns
        if "stat_quarter" in columns:
            index_label = "stat_quarter"
        elif "stat_month" in columns:
            index_label = "stat_month"
        elif "stat_year" in columns:
            index_label = "stat_year"
        elif "stat_date" in columns:
            index_label = "stat_date"
        else:
            index_label = "day"

    if "MAC_CPI_MONTH" == query_from:
        df = df[df["area_name"] == "全国"]

    print(f"index: {index_label}. query_:{query_from}")
    convert_index(df, index_label)
    df.sort_values(index_label, inplace=True)
    return [query_from, index_label, df]
    # df.to_csv(f"jq/{query_from}.csv", index=False,encoding="utf-8_sig")


def convert_index(df,column):
    """converts string or interger index to date format"""
    import DB
    df_sh=pd.DataFrame()
    df_sh["index_helper"]=DB.Global.d_assets["000001.SH"].index
    df_sh["index_helper"]=df_sh["index_helper"].astype(str)

    def my_converter(val):
        val=str(val).replace("-","")
        if len(val)==6:#yyyymm
            true_index=df_sh["index_helper"].str.slice(0,6)==(str(val))
            df=df_sh[true_index]

            df=df.tail(1)
            if df.empty:
                return val+"01"
            else:
                last_day_of_month=df["index_helper"].iat[-1]
                val=last_day_of_month
        elif len(val)==4:#yyyy
            return my_converter(val+"12")
        return val
    df[column]=df[column].apply(my_converter)