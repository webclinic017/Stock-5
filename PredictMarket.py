import cProfile
# set global variable flag
import Alpha
from Alpha import *
import numpy as np
import UI
from scipy.stats.mstats import gmean
from scipy.stats import gmean
import sys
import os
import matplotlib
import itertools
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import DB
import LB




def run(debug=0):
    """
    1. update index
    2. update all stocks
    3. (todo industry) check if index volume is consistent with past gain for index
    4. (todo idustry) calculate how many stocks are overma
    5. (done) check if top n stocks are doing well or not
    6. check if best 3 industry are doing well or not
    7. (todo maybe unessesary) use index to calculate time since last peak
    8. (todo not useful atm) overlay of the new year period month
    9. calculate how many institution are holding stocks
    10. us market for potential crash
    11. Never buy at market all time high (90%) and no volume. Check together with all 3 indexes
    11. Steepness. If the price gained or falled too steep, it is mostlikely in sinoid transition phase
    12. Fundamental values: sales, return data

    divide and conquer: choose most simple case for all variables
    1. Long period instead of short period
    2. Group of stocks(index, industry ETF) instead of individual stocks
    3. Only Buy Signals or Sell signals instead of both
    4.

    overlay technique: if there is a variable like freq or threshhold, instead of using one, use ALL of them and then combine them into one result

    NOTE:
    DEBUG level 0 = only show combined final result like r4:buy
    DEBUG level 1 = level 0 + index based like r4:sh_buy
    DEBUG level 2 = level 1 + different freq like r4:sh_close120.buy
    DEBUG level 3 = level 2 + junk
    """


    #0 INITIALIZATION
    a_path=LB.a_path(f"Market/CN/PredictMarket/PredictMarket")

    df_sh=DB.get_asset(ts_code="000001.SH",asset="I")
    df_sz=DB.get_asset(ts_code="399001.SZ",asset="I")
    df_cy=DB.get_asset(ts_code="399006.SZ",asset="I")

    df_sh=df_sh[["close","pct_chg","vol"]].rename(columns={"close":"close_sh","pct_chg":"pct_chg_sh","vol":"vol_sh"})
    df_sz=df_sz[["close","pct_chg","vol"]].rename(columns={"close":"close_sz","pct_chg":"pct_chg_sz","vol":"vol_sz"})
    df_cy=df_cy[["close","pct_chg","vol"]].rename(columns={"close":"close_cy","pct_chg":"pct_chg_cy","vol":"vol_cy"})

    df_result= pd.merge(df_sh, df_sz, how='left', on="trade_date", suffixes=["", ""], sort=False)
    df_result= pd.merge(df_result, df_cy, how='left', on="trade_date", suffixes=["", ""], sort=False)

    df_result["r:buy_sell"]=0.0
    df_result["r:buy"]=0.0
    df_result["r:sell"]=0.0

    #d_preload = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")

    # 0 CREATE INDEX VOLUME
    #step_zero()


    # 3 VOLUME
    #step3(df_result=df_result,debug=debug)

    # 4 OVERMA
    #step4(df_result=df_result, d_preload=d_preload,debug=debug)

    # 5 TOP N Stock
    #step5(df_result=df_result, d_preload=d_preload,debug=debug)

    # 6 TOP INDUSTRY
    #step6(df_result=df_result)

    # 8 SEASONAL
    step8(df_result=df_result)


    # 11 NEVER BUY AT MARKET at 90% HIGH
    #step_eleven()

    #12 combine r1..rn buy and sell signal into one
    buy_count_help=0
    sell_count_help=0
    for counter in range(1,12):
        if f"r{counter}:buy" in df_result.columns:
            if df_result[f"r{counter}:buy"].sum()>0:
                buy_count_help+=1
                df_result["r:buy"]=df_result["r:buy"].add(df_result[f"r{counter}:buy"],fill_value=0)

        if f"r{counter}:sell" in df_result.columns:
            if df_result[f"r{counter}:sell"].sum() > 0:
                sell_count_help += 1
                df_result["r:sell"] = df_result["r:sell"].add(df_result[f"r{counter}:sell"], fill_value=0)

    df_result["r:buy"]=df_result["r:buy"]/buy_count_help
    df_result["r:sell"]=df_result["r:sell"]/sell_count_help

    print(f"buy sell count helper",buy_count_help,sell_count_help)


    #13 combine buy and sell signal into one
    df_result["r:buy_sell"]=df_result["r:buy"]+( df_result["r:sell"] * (-1))

    # Save excel
    LB.to_csv_feather(df=df_result,a_path=a_path)


def step_zero():
    """
    create a sumarized time series to see the overall vol of the complete market

    RESULT: The summarized volume is Exactly same as the sh index volume
    """

    #load in all the stocks
    d_preload= DB.preload(asset="E",freq="D",on_asset=True,step=1,market="CN")

    df_market = DB.get_asset(ts_code="000001.SH",asset="I",freq="D",market="CN")
    df_market=df_market[["close","vol","amount"]]
    df_market["agg_abs_amount"]=0.0
    df_market["agg_rel_amount"]=0.0
    df_market["agg_stocks"]=0.0

    print("lengh of preload",len(d_preload))

    for ts_code, df_asset in d_preload.items():
        print(ts_code)

        df_asset["count_helper"]=1.0
        df_market["agg_stocks"]=df_market["agg_stocks"].add(df_asset["count_helper"],fill_value=0)
        df_market["agg_abs_amount"] = df_market["agg_abs_amount"].add(df_asset["amount"],fill_value=0)

    df_market["agg_rel_amount"] = df_market["agg_abs_amount"] /df_market["agg_stocks"]
    a_path = LB.a_path(f"Market/CN/PredictMarket/Market")
    LB.to_csv_feather(df=df_market, a_path=a_path)


def step_one_two():
    DB.update_all_in_one_cn_v2()


def step3(df_result, debug=0):
    df_result[f"r3:buy"] = 0.0
    df_result[f"r3:sell"] = 0.0

    for on_index in ["sh", "sz", "cy"]:
        result_list = []
        counter = 0

        df_result[f"r3:{on_index}.buy"] = 0.0
        df_result[f"r3:{on_index}.sell"] = 0.0

        # loop over all frequency
        for freq_close in [240, 500]:
            for freq_vol in [120, 500]:
                print(f"step3 close{freq_close} vol{freq_vol}...")
                counter += 1
                buy_sell_label = step3_single(df_result=df_result, freq_close=freq_close, freq_vol=freq_vol, on_index=on_index, debug=debug)
                result_list = result_list + [buy_sell_label]

        # combine all frequecies into one result for ONE index
        for buy_freq_signal, sell_freq_signal in result_list:
            df_result[f"r3:{on_index}.buy"] = df_result[f"r3:{on_index}.buy"].add(df_result[buy_freq_signal], fill_value=0)
            df_result[f"r3:{on_index}.sell"] = df_result[f"r3:{on_index}.sell"].add(df_result[sell_freq_signal], fill_value=0)
            if debug<2:
                del df_result[buy_freq_signal]
                del df_result[sell_freq_signal]

        # normalize the result
        df_result[f"r3:{on_index}.buy"] = df_result[f"r3:{on_index}.buy"] / (counter * 2)
        df_result[f"r3:{on_index}.sell"] = df_result[f"r3:{on_index}.sell"] / (counter * 2)

        # add the INDEX RESULT to OVERALL step 4 result
        df_result[f"r3:buy"] = df_result[f"r3:buy"].add(df_result[f"r3:{on_index}.buy"], fill_value=0)
        df_result[f"r3:sell"] = df_result[f"r3:sell"].add(df_result[f"r3:{on_index}.sell"], fill_value=0)

        if debug < 1:
            del df_result[f"r3:{on_index}.buy"]
            del df_result[f"r3:{on_index}.sell"]

    # combine 3 INDEX together into one:divide the step.4 result by the index (from the day the are present)
    df_result["sz_counter"] = (df_result["close_sz"].isna()).astype(int)
    df_result["cy_counter"] = (df_result["close_cy"].isna()).astype(int)
    df_result["sz_counter"] = df_result["sz_counter"].replace({0: 1, 1: 0})
    df_result["cy_counter"] = df_result["cy_counter"].replace({0: 1, 1: 0})

    df_result["index_counter"] = 1 + df_result["sz_counter"] + df_result["cy_counter"]
    df_result[f"r3:buy"] = df_result[f"r3:buy"] / df_result["index_counter"]
    df_result[f"r3:sell"] = df_result[f"r3:sell"] / df_result["index_counter"]

    if debug < 3:
        del df_result["sz_counter"]
        del df_result["cy_counter"]
        del df_result["index_counter"]



def step3_single(df_result, on_index, freq_close=240, freq_vol=360, debug=0):
    """
    This can detect 3 signals:
    1. high volume and high gain -> likely to reverse to bear
    2. low volume and high gain -> even more likely to reverse to bear
    3. high volume and low gain -> ikely to reverse to bull
    """

    vol_name=f"vol_{on_index}"
    close_name=f"close_{on_index}"
    result_name=f"r3:{on_index}_vol{freq_vol}_close{freq_close}"

    # normalize volume and close first with rolling 240 days
    norm_vol_name = Alpha.rollingnorm(df=df_result, abase=vol_name, freq=freq_vol, inplace=True)
    norm_close_name = Alpha.rollingnorm(df=df_result, abase=close_name, freq=freq_close, inplace=True)

    # 1. Sell Signal: filter only days where vol > 0.7 and close > 0.6
    df_helper = df_result.loc[(df_result[norm_vol_name] > 0.7) & (df_result[norm_close_name] > 0.6)]
    sell_signal1 = df_helper[norm_vol_name] + df_helper[norm_close_name]  # higher price, higher volume the more clear the signal

    # 2. Sell Signal: filter only days where vol < 0.5 and close > 0.8
    df_helper = df_result.loc[(df_result[norm_vol_name] < 0.4) & (df_result[norm_close_name] > 0.85)]
    sell_signal2 = (1 - df_helper[norm_vol_name]) + df_helper[norm_close_name]  # higher price, lower volume the more clear the signal

    # combine both type of sell signals
    df_result[f"{result_name}_sell"] = sell_signal1.add(sell_signal2,fill_value=0)


    # 3. Buy Signal: filter only days where vol > 0.6 and close < 0.4
    df_helper = df_result.loc[(df_result[norm_vol_name] > 0.7) & (df_result[norm_close_name] < 0.4)]
    buy_signal = df_helper[norm_vol_name] + (1 - df_helper[norm_close_name])  # higher volume, lower price the more clear the signal
    df_result[f"{result_name}_buy"] = buy_signal

    # 4. Delete unessesary columns produced
    if debug<3:
        del df_result[norm_vol_name]
        del df_result[norm_close_name]

    return [f"{result_name}_buy",f"{result_name}_sell"]



def step4(df_result, d_preload, debug=0):
    df_result[f"r4:buy"] = 0.0
    df_result[f"r4:sell"] = 0.0

    for on_index in ["sh", "sz","cy"]:
        result_list = []
        counter = 0

        df_result[f"r4:{on_index}.buy"] = 0.0
        df_result[f"r4:{on_index}.sell"] = 0.0

        # loop over all frequency
        for freq_close in [240, 500]:
            for freq_overma in [120, 500]:
                print(f"step4 {on_index} close{freq_close} overma{freq_overma}...")
                counter += 1
                buy_sell_label = step4_single(df_result=df_result, d_preload=d_preload, freq_close=freq_close, freq_overma=freq_overma, on_index=on_index, debug=debug)
                result_list = result_list + [buy_sell_label]

        # combine all frequecies into one result for ONE index
        for buy_signal, sell_signal in result_list:
            df_result[f"r4:{on_index}.buy"] = df_result[f"r4:{on_index}.buy"].add(df_result[buy_signal], fill_value=0)
            df_result[f"r4:{on_index}.sell"] = df_result[f"r4:{on_index}.sell"].add(df_result[sell_signal], fill_value=0)
            if debug<2:
                del df_result[buy_signal]
                del df_result[sell_signal]

        # normalize the result
        df_result[f"r4:{on_index}.buy"] = df_result[f"r4:{on_index}.buy"] / (counter * 2) # why times 2 actually
        df_result[f"r4:{on_index}.sell"] = df_result[f"r4:{on_index}.sell"] / (counter * 2)

        # add the INDEX RESULT to OVERALL step 4 result
        df_result[f"r4:buy"] = df_result[f"r4:buy"].add(df_result[f"r4:{on_index}.buy"], fill_value=0)
        df_result[f"r4:sell"] = df_result[f"r4:sell"].add(df_result[f"r4:{on_index}.sell"], fill_value=0)

        if debug < 1:
            del df_result[f"r4:{on_index}.buy"]
            del df_result[f"r4:{on_index}.sell"]

    # combine 3 INDEX together into one:divide the step.4 result by the index (from the day the are present)
    df_result["sz_counter"] = (df_result["close_sz"].isna()).astype(int)
    df_result["cy_counter"] = (df_result["close_cy"].isna()).astype(int)
    df_result["sz_counter"] = df_result["sz_counter"].replace({0: 1, 1: 0})
    df_result["cy_counter"] = df_result["cy_counter"].replace({0: 1, 1: 0})

    df_result["index_counter"] = 1 + df_result["sz_counter"] + df_result["cy_counter"]
    df_result[f"r4:buy"] = df_result[f"r4:buy"] / df_result["index_counter"]
    df_result[f"r4:sell"] = df_result[f"r4:sell"] / df_result["index_counter"]

    if debug<3:
        del df_result["sz_counter"]
        del df_result["cy_counter"]
        del df_result["index_counter"]



def step4_single(df_result, d_preload, freq_close=240, freq_overma=240, on_index="sh", debug=0):
    """calculate how many stocks are overma generally very useful

    for period in [500,240,120]:
        1. General overma
        2. Index overma
        3. Industry  overma
        4. Size overma
    """

    #1. General ALL STOCK overma
    #1.1 normalize overma series
    if f"overma{freq_overma}" not in df_result.columns:
        df_result[f"overma{freq_overma}"]=0.0
        df_result[f"counter{freq_overma}"]=0.0

        for ts_code, df_asset in d_preload.items():
            #calculate if stocks is over its ma
            df_asset[f"ma{freq_overma}"]=df_asset["close"].rolling(freq_overma).mean()
            df_asset[f"overma{freq_overma}"]= (df_asset["close"]>=df_asset[f"ma{freq_overma}"]).astype(int)
            df_asset[f"counter{freq_overma}"] = 1

            df_result[f"overma{freq_overma}"] = df_result[f"overma{freq_overma}"].add(df_asset[f"overma{freq_overma}"], fill_value=0)
            #counter to see how many stocks are available
            df_result[f"counter{freq_overma}"] = df_result[f"counter{freq_overma}"].add(df_asset[f"counter{freq_overma}"], fill_value=0)

        #finally: calculate the percentage of stocks overma
        df_result[f"overma{freq_overma}"]= df_result[f"overma{freq_overma}"] / df_result[f"counter{freq_overma}"]


    # 1.2 normalize close series
    norm_close_name = Alpha.rollingnorm(df=df_result,  freq=freq_close, abase=f"close_{on_index}", inplace=True)


    #1.3 generate  Buy Signal: price < 0.25 and overma < 0.25
    df_helper = df_result.loc[(df_result[f"overma{freq_overma}"] < 0.25) & (df_result[norm_close_name] < 0.25)]
    df_result[f"r4:overma{freq_overma}_close{freq_close}_{on_index}_buy"] = (1 - df_helper[f"overma{freq_overma}"]) + (1 - df_helper[norm_close_name])  # the lower the price, the lower overma, the better

    # 1.4 generate  Sell Signal: price > 0.75 and overma > 0.75
    df_helper = df_result.loc[(df_result[f"overma{freq_overma}"] > 0.75) & (df_result[norm_close_name] > 0.75)]
    df_result[f"r4:overma{freq_overma}_close{freq_close}_{on_index}_sell"] = df_helper[f"overma{freq_overma}"] + df_helper[norm_close_name]  # the lower the price, the lower overma, the better

    #1.5 delete unessary columns
    if debug<3:
        del df_result[f"overma{freq_overma}"]
        del df_result[f"counter{freq_overma}"]
        del df_result[norm_close_name]

    # TODO, for industry ,index and size

    return [f"r4:overma{freq_overma}_close{freq_close}_{on_index}_buy", f"r4:overma{freq_overma}_close{freq_close}_{on_index}_sell"]


def step5(df_result,d_preload, debug=0):
    """check if top n stocks (low beta stocks stocks) are doing well or not
    If even they are bad, then the whole stock market is just bad for sure

    algorith:
    1. define top n stocks using fundamentals and technicals
    2. check if they are doing well in last freq D: 5, 20, 60

    （1. cheat, use shortage to manually define these 50 stocks)
    """


    #1. Generate top n index
    """
    贵州茅台
    泸州老窖
    伊利股份
    招商银行
    海螺水泥
    恒瑞医药
    云南白药
    苏泊尔
    格力电器
    """

    a_ts_codes=["600519.SH","000568.SZ","600887.SH","600036.SH","600585.SH","600272.SH","000538.SZ","002032.SZ","000651.SZ"]

    df_result["step5_counter"]=0.0
    df_result["step5_topn_pct_chg"]=0.0

    for ts_code, df_asset in d_preload.items():
        if ts_code in a_ts_codes:
            #add counter together
            df_asset["step5_counter"]=1
            df_result["step5_counter"]=df_result["step5_counter"].add(df_asset["step5_counter"],fill_value=0)

            #add gain together
            df_result["step5_topn_pct_chg"]=df_result["step5_topn_pct_chg"].add(df_asset["pct_chg"],fill_value=0)

    df_result["step5_topn_pct_chg"]=df_result["step5_topn_pct_chg"]/df_result["step5_counter"]
    df_result["r5:topn_index"]=Alpha.comp_chg(df=df_result,abase="step5_topn_pct_chg",inplace=False, start=100)

    if debug<2:
        del df_result["step5_counter"]
        del df_result["step5_topn_pct_chg"]

    step5_single(df_result=df_result,debug=debug)


def step5_single(df_result, debug=False):
    # 2. Generate step 5 buy sell signal
    r5_freq_result=[]
    df_result["r5:buy"]=0.0
    df_result["r5:sell"]=0.0 # step5 does not produce any sell signal

    for freq in [120,180,240,300,360]:
        print(f"step5 close{freq}...")
        topn_close_name = Alpha.rollingnorm(df=df_result, freq=freq, abase="r5:topn_index", inplace=True)

        # 2.1 Buy if past normalized return is < 0.2
        df_helper = df_result.loc[(df_result[topn_close_name] < 0.25)]
        df_result[f"r5:topn_close{freq}_buy"] = 1 - df_helper[topn_close_name]

        r5_freq_result+=[f"r5:topn_close{freq}_buy"]
        if debug<2:
            del df_result[topn_close_name]

    # combine all freq into one
    counter=0
    for freq_result in r5_freq_result:
        df_result["r5:buy"] = df_result["r5:buy"].add(df_result[freq_result],fill_value=0)
        counter+=1

        if debug<1:
            del df_result[freq_result]

    df_result["r5:buy"]=df_result["r5:buy"]/counter

def step6(df_result):
    """Check how the 3 best industries are doing
    If even they are bad, then the market is bad for sure
    Best 4 industry: Biotech, consume defensive, electronics
    Step 5 and 6 are a bit correlated
    """



    return

def step8(df_result, debug=0):
    """
    currently no use of seasonal effect because they are too periodic.
    Seasonal effect are interesting, but deviation are too big.
    Hence it makes the stats useless

    1. overlay of chinese month of year effect
    2. overlay of first month prediction effect
    3. overlay of day of month effect
    """

    #init
    df_trade_date=DB.get_trade_date()

    df_result["year"] = df_trade_date["year"]
    df_result["month"] = df_trade_date["month"]
    df_result["day"] = df_trade_date["day"]
    df_result["weekofyear"] = df_trade_date["weekofyear"]
    df_result["dayofweek"] = df_trade_date["dayofweek"]
    df_result["r8:buy_sell"]=0.0

    #overlay of all divisions are NOT IN USE
    for division in ["month","weekofyear"]:
        # overlay of seasonal effect
        df_division = DB.get(a_path = LB.a_path(f"Market/CN/ATest/seasonal_stock/{division}"),set_index=division)
        df_result[division]=df_result[division].astype(int)
        df_result[division]=df_result[division].replace(df_division["pct_chg"].to_dict())
        df_result[division] = df_result[division].astype(float)
        df_result["r8:buy_sell"]+=df_result[division]

    #overlay of chinese new year effect

    #overlay first and last week of year



def step_eleven():
    """check if all three index and US index is all their all time high
    """

    def max_expanding_statistic(df, min_periods=1000):
        name=f"expanding_close_max"
        name2=f"near_expanding_close_max"

        df[name]=df["close"].expanding(min_periods=min_periods).max()
        df[name2]= (df["close"]/df[name]) > 0.85
        df[name2] =df[name2].astype(int)
        result= df[name2].mean()
        print(result)
        return result

    def max_rolling_statistic(df,periods=1000,min_periods=240):
        name=f"rolling_close_max" if periods=="max" else f"rolling_close_{periods}"
        name2=f"near_rolling_close_max" if periods=="max" else f"near_rolling_close_{periods}"

        df[name]=df["close"].rolling(periods, min_periods=min_periods).max()
        df[name2]= (df["close"]/df[name]) > 0.85
        df[name2] =df[name2].astype(int)



    #init
    df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sz = DB.get_asset(ts_code="399001.SZ", asset="I")
    df_cy = DB.get_asset(ts_code="399006.SZ", asset="I")


    # 1. check a index how long since the past they are at all time high using expanding
    #max = expanding
    max_expanding_statistic(df=df_sh, min_periods=1111)
    max_expanding_statistic(df=df_sz, min_periods=1111)
    max_expanding_statistic(df=df_cy, min_periods=1111)
    # result: 0.11, 0.05, 0.08 for sh, sz, cy

    #non-max : rolling
    for period in [2000, 1500, 1000, 500,240]:
        print(f"{period}")
        max_rolling_statistic(df=df_sh, periods=period,min_periods=240)
        max_rolling_statistic(df=df_sz, periods=period,min_periods=240)
        max_rolling_statistic(df=df_cy, periods=period,min_periods=240)

    #df_sh.to_csv("df_sh.csv")
    #df_sh.to_csv("df_sz.csv")
    #df_sh.to_csv("df_cy.csv")

    """
    2. check the time since the last all time high to estimate roughly the next high and low
    """
    df_sh["day_since_max"]=LB.consequtive_counter(df_sh["near_expanding_close_max"],count=0)


    """
    3. check if major indicies (sh,sz,cy,nasdaq) are at their high
    """

    #max period: expanding_close_max
    sh_last=df_sh["near_expanding_close_max"].iat[-1]
    sz_last=df_sz["near_expanding_close_max"].iat[-1]
    cy_last=df_cy["near_expanding_close_max"].iat[-1]

    print("expanding max check result is:", (sh_last+sz_last+cy_last)/3)


    df_result = df_sh[["close"]]
    for period in [1000]:
        df_result[f"sh_{period}"]=df_sh[f"near_rolling_close_{period}"]
        df_result[f"sz_{period}"]=df_sz[f"near_rolling_close_{period}"]
        df_result[f"cy_{period}"]=df_cy[f"near_rolling_close_{period}"]
    df_result.to_excel("result.xlsx")


    for period in [2000, 1500, 1000, 500, 240]:
        sh_last = df_sh[f"near_rolling_close_{period}"].iat[-1]
        sz_last = df_sz[f"near_rolling_close_{period}"].iat[-1]
        cy_last = df_cy[f"near_rolling_close_{period}"].iat[-1]

        print(f"rolling {period} result is:", (sh_last + sz_last + cy_last) / 3)



    #1000
    #500
    #240









if __name__ == '__main__':


    run()



