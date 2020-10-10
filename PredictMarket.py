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
from functools import partial




def run(debug=0):
    """
    1. update index
    2. update all stocks

    3. (todo industry) check if index volume is consistent with past gain for index
    4. (todo idustry) calculate how many stocks are overma
    5. (done) check if top n stocks are doing well or not
    6. (todo) check if best 3 industry are doing well or not
    8. (todo relatively not included atm) overlay of the new year period month
    9. (todo currently no way access) calculate how many institution are holding stocks  http://datapc.eastmoney.com/emdatacenter/JGCC/Index?color=w&type=
    10. (done but not that useful) rsi freq combination.
    11. (todo finished partly) Price and vol volatility

    12. (todo with all stocks and industry) check volatiliy adj macd
    13. support and resistance for index and all stocks
    16. baidu xueqiu weibo google trend index
    17. Fundamental values: sales, return data


    divide and conquer: choose most simple case for all variables
    1. Long period instead of short period
    2. Group of stocks(index, industry ETF) instead of individual stocks
    3. Only Buy Signals or Sell signals instead of both
    4. Overlay technique: Multiple freq results instead of one. If there is a variable like freq or threshhold, instead of using one, use ALL of them and then combine them into one result

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

    #df_result["year"]=((df_result.index).astype(str).str.slice(0,4)).astype(int)
    df_result["r:buy_sell"]=0.0
    df_result["ra:buy_sell"]=0.0

    d_preload = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")

    # 0 CREATE INDEX VOLUME
    #step_zero()
    # todo Normal function needs to be volatility adjusted. they dont work in crazy time. but MACD works in crazy time.
    # distinguish index vs d_preload
    # manage debu
    # manage names
    # manage efficiency
    # macd buys on bull, sell on bear, all other signal is inverse. MACD is good on crazy time. all other are bad on crazy time.
    #idea: use indicator that are slower to change. overma is hard to change, you can not simply change from abv ma to under ma for all stocks. volume is easier to change.
    #after knowing when to buy, what time to sell then?


    # Distinguish crazy and normal time
    stepi1(df_result=df_result,debug=debug)

    # 0 MACD on index
    step0(df_result=df_result, debug=debug)

    # 3 VOLUME
    step3(df_result=df_result,debug=debug)

    # 4 OVERMA
    step4(df_result=df_result, d_preload=d_preload,debug=debug)

    # 5 TOP N Stock
    step5(df_result=df_result, d_preload=d_preload,debug=debug)

    # 6 TOP INDUSTRY
    #step6(df_result=df_result)

    # 8 SEASONAL
    #step8(df_result=df_result)

    # 10 RSI
    #step10(df_result=df_result,debug=debug)

    #  NEVER BUY AT MARKET at 90% HIGH
    #step_eleven()

    # 11 VOLATILITY (maybe only good to distinguish crazy and normal time)
    #step11(df_result=df_result,debug=debug)



    # 13 combine buy and sell signal into one
    # df_result["r:buy_sell"] = df_result["r:buy"] + (df_result["r:sell"] * (-1))
    divideby=0
    for counter in range(-50, 50):
        if counter not in [0]:#because thats macd
            if f"r{counter}:buy" in df_result.columns and f"r{counter}:sell" in df_result.columns:
                df_result[f"r{counter}:buy_sell"] = df_result[f"r{counter}:buy"].add(df_result[f"r{counter}:sell"] * (-1), fill_value=0)
                df_result["r:buy_sell"] =df_result["r:buy_sell"].add(df_result[f"r{counter}:buy_sell"], fill_value=0)
                divideby+=1
            elif f"r{counter}:buy_sell" in df_result.columns:
                df_result["r:buy_sell"] = df_result["r:buy_sell"].add(df_result[f"r{counter}:buy_sell"], fill_value=0)
                divideby += 1


    df_result["r:buy_sell"]=df_result["r:buy_sell"]/divideby

    #adjust by volatility with macd freq overlay method
    df_result["ra:buy_sell"]=0.0
    for divideby, thresh in enumerate([0.2,0,22,0,24,0.26,0,28,0.3]):
        df_result[f"ra:buy_sell{thresh}"]=0.0
        df_result[f"ra:buy_sell{thresh}"].loc[df_result["volatility"] <= thresh]= df_result["r:buy_sell"]
        df_result[f"ra:buy_sell{thresh}"].loc[df_result["volatility"] > thresh]= df_result["r0:buy_sell"]
        df_result["ra:buy_sell"] +=df_result[f"ra:buy_sell{thresh}"]

        if debug<2:
            del df_result[f"ra:buy_sell{thresh}"]

    df_result["ra:buy_sell"]=df_result["ra:buy_sell"]/divideby


    #14 evaluation different result against each other


    # Save excel
    LB.to_csv_feather(df=df_result,a_path=a_path)

def stepi1(df_result, debug=0,index="sh"):
    """calculate volatility to distinguish between normal time and crazy time"""

    df_result["volatility"] = 0.0
    a_volatility_freq = []
    counter = 0
    a_freqs = [120, 240, 360, 500]
    for freq in a_freqs:
        df_result[f"volatility{freq}"] = df_result[f"close_{index}"].rolling(freq).std()
        a_volatility_freq += [f"volatility{freq}"]
        counter += 1

    for freq_vola in a_volatility_freq:
        df_result["volatility"] = df_result["volatility"].add(df_result[freq_vola], fill_value=0)

    df_result["volatility"] = df_result["volatility"] / counter

    # normalize result in dirty way
    df_result["volatility"] = Alpha.norm(df=df_result, abase="volatility", inplace=False)

    for freq in a_freqs:
        del df_result[f"volatility{freq}"]




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

    #combine buy and sell
    df_result["r3:buy_sell"] = df_result[f"r3:buy"].add(df_result[f"r3:sell"] * (-1), fill_value=0)


    #adjust with volatility


    if debug < 3:
        del df_result["sz_counter"]
        del df_result["cy_counter"]
        del df_result["index_counter"]
        del df_result[f"r3:buy"]
        del df_result[f"r3:sell"]



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

    # combine buy and sell
    df_result["r4:buy_sell"] = df_result[f"r4:buy"].add(df_result[f"r4:sell"]*(-1),fill_value=0)

    # adjust with volatility. they r4 useful when not crazy time

    if debug<3:
        del df_result["sz_counter"]
        del df_result["cy_counter"]
        del df_result["index_counter"]
        del df_result[f"r4:buy"]
        del df_result[f"r4:sell"]



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


def step5_single(df_result, debug=0):
    # 2. Generate step 5 buy sell signal
    r5_freq_buy_result=[]
    r5_freq_sell_result=[]
    df_result["r5:buy"]=0.0
    df_result["r5:sell"]=0.0 # step5 does not produce any sell signal

    for freq in [120,240,500]:
        print(f"step5 close{freq}...")
        #rolling norm
        topn_close_name = Alpha.rollingnorm(df=df_result, freq=freq, abase="r5:topn_index", inplace=True)

        #is max
        df_result["topn_emax"]=df_result["r5:topn_index"].expanding().max()
        is_top_pct = Alpha.ismax(df=df_result, abase="r5:topn_index", emax="topn_emax",inplace=True, q=0.85, score=1)

        # 2.1 Buy if past normalized return is < 0.2
        df_helper = df_result.loc[(df_result[topn_close_name] < 0.20)]
        df_result[f"r5:topn_close{freq}_buy"] = 1 - df_helper[topn_close_name]
        r5_freq_buy_result+=[f"r5:topn_close{freq}_buy"]

        # 2.2 Sell if they are not at top 15% and there is no buy signal = bear but not bear enough
        df_helper = df_result[(df_result[is_top_pct]==-1)&(df_result[f"r5:topn_close{freq}_buy"].isna())]
        df_helper["sell_helper"]=1
        df_result[f"r5:topn_close{freq}_sell"] =df_helper["sell_helper"]
        r5_freq_sell_result += [f"r5:topn_close{freq}_sell"]

        if debug<2:
            del df_result[topn_close_name]
            del df_result["topn_emax"]
            del df_result[is_top_pct]

    # combine all freq into one
    counter=0
    for freq_result in r5_freq_buy_result:
        df_result["r5:buy"] = df_result["r5:buy"].add(df_result[freq_result],fill_value=0)
        counter+=1
        if debug<1:del df_result[freq_result]

    counter = 0
    for freq_result in r5_freq_sell_result:
        df_result["r5:sell"] = df_result["r5:sell"].add(df_result[freq_result], fill_value=0)
        counter += 1
        if debug < 1: del df_result[freq_result]

    df_result["r5:buy"]=df_result["r5:buy"]/counter
    df_result["r5:sell"]=df_result["r5:sell"]/counter

    #for now exclude sell result
    df_result["r5:sell"]=0.0

    # combine buy and sell
    df_result["r5:buy_sell"] = df_result[f"r5:buy"].add(df_result[f"r5:sell"]*(-1),fill_value=0)

    # adjust with volatility

    if debug < 2:
        del df_result["r5:buy"]
        del df_result["r5:sell"]

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
    #PART 1
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
        #df_result["r8:buy_sell"]+=df_result[division]


    #PART 2
    df_sh=DB.get_asset(ts_code="000001.SH",asset="I")
    df_sh=LB.trade_date_to_calender(df_sh)
    #overlay of chinese new year effect(compare ny gain against others. If strong then the whole year is strong)
    # in order to give a more real prediction, we conduct the prediction step by step from the past

    df_sh_helper=df_sh[df_sh["month"]==2]
    df_result=df_sh_helper.groupby("year").mean()
    df_result.to_csv("test.csv")
    #todo unfinished because I feel it will not be better than other existing signals
    #overlay of first month (compare first month gain against others. If strong then the whole year is strong)



    #overlay first and last week of year







def step10(df_result, debug=0):
    """
    rsi freq: this step is to check if different freq combination of rsi would make a better rsi signal
    """

    df_result["r10:buy"] =0
    df_result["r10:sell"] =0
    for counter, freq in enumerate([20,40,60,80,100,120,180,240,300,360]):
        rsi_name=Alpha.rsi(df=df_result,abase="close_sh",freq=freq, inplace=True)


        # create buy signal
        df_helper = df_result.loc[(df_result[rsi_name] < 50)]
        df_result[f"r10:close_sh{freq}_buy"] = df_helper[rsi_name]


        # create sell signal
        df_helper = df_result.loc[(df_result[rsi_name] > 50)]
        df_result[f"r10:close_sh{freq}_sell"] = df_helper[rsi_name]


        df_result["r10:buy"] = df_result["r10:buy"].add(df_result[f"r10:close_sh{freq}_buy"], fill_value=0)
        df_result["r10:sell"] = df_result["r10:sell"].add(df_result[f"r10:close_sh{freq}_sell"], fill_value=0)

        if debug<1:
            del df_result[rsi_name]

    df_result["r10:buy"]= df_result["r10:buy"]/(counter+1)
    df_result["r10:sell"]= df_result["r10:sell"]/(counter+1)



def step11(df_result, debug=0, index="sh"):
    """
    Different kind of volatility: against past, intraday, against other stock

    1. check time with volatility against itself in the past = rolling
    2. check time with volatility against others now
    3. check time with intraday volatility
    4. check time with low volatility and uptrend (This does not exist in A Stock)

    method 1: 1. calculate price_std with freq overlay. 2. calculate together with close rolling norm
    method 2: 1. calculate price_std and rolling norm overlay together in one step
    Note: this method tried using BOTH method and the result is okish, all signals have almost the same threshhold which is bad. Therefore I conclude that this method is not that much useful.
    Note: all other steps like 3,4,5 are using method 2 and got good result
    """
    #1. Check volatility AGAINST PAST
    #1.1 check time with PRICE volatility AGAINST PAST
    #result -> can predict crazy and normal time


    #normalize price
    df_result[f"r11:buy"]=0.0
    df_result[f"r11:sell"]=0.0
    divideby=1
    for freq in [120,240,500]:
        print(f"step11 {index} close{freq}", )
        #normalize close
        norm_close_name = Alpha.rollingnorm(df=df_result, abase=f"close_{index}", freq=freq, inplace=True)

        #calcualte close std
        df_result[f"close_std{freq}"]=df_result[f"close_{index}"].rolling(freq).std()

        # normalize result(dirty way, should not be like that because it knows future)
        norm_close_std_name = Alpha.norm(df=df_result, abase=f"close_std{freq}", inplace=True)

        # generate buy signal: volatiliy is low and past price is low = Buy
        # volatility < 0.2, past gain < 0.4: buy. indicates turning point
        df_helper=df_result[ (df_result[norm_close_name]<0.3) & (df_result[norm_close_std_name]< 0.2)]
        df_result[f"r11:norm{freq}_buy"] =  (1- df_helper[norm_close_name]) +(1 - df_helper[norm_close_std_name])
        df_result[f"r11:buy"] = df_result[f"r11:buy"].add(df_result[f"r11:norm{freq}_buy"], fill_value=0)

        # generate Sell signal: volatiliy is low and past price is high = Sell
        # volatility < 0.2, past gain > 0.8: buy. indicates turning point
        df_helper = df_result[(df_result[norm_close_name] > 0.7) & (df_result[norm_close_std_name] < 0.2)]
        df_result[f"r11:norm{freq}_sell"] =  (df_helper[norm_close_name]) + (1 - df_helper[norm_close_std_name])
        df_result[f"r11:sell"] = df_result[f"r11:sell"].add(df_result[f"r11:norm{freq}_sell"], fill_value=0)


        #increment divideby
        divideby += 1

        #debug
        if debug < 2:
            del df_result[norm_close_name]
            del df_result[f"close_std{freq}"]
            del df_result[norm_close_std_name]
            del df_result[f"r11:norm{freq}_buy"]
            del df_result[f"r11:norm{freq}_sell"]


    #normalize
    df_result[f"r11:buy"]=df_result[f"r11:buy"] / (divideby*2)
    df_result[f"r11:sell"]=df_result[f"r11:sell"] / (divideby*2)



    #generate sell signal: volatiliy is low and past price is high = Sell


    #1.2 check time with VOL volatility AGAINST PAST
    #Result: ok but not as good as price std (because volume is not mean normalized?)
    """
    func_vol_partial = partial(func, abase="vol_sh")
    LB.frequency_ovelay(df=df_result, func=func_vol_partial, a_freqs=[[20, 40, 60, 120, 240]], a_names=["sh_vol_std", "vol_sh"], debug=debug)
    df_result.loc[df_result["year"] < 2000, "sh_vol_std"] = 0.0
    """

    #2. Volatility against other stock

    #3. Volatility intraday

def step0(df_result, debug=0, index="sh"):

    """MACD"""

    #create all macd
    a_results_col=[]
    for sfreq in [120,180,240]:
        for bfreq in [180,240,300,360]:
            if sfreq<bfreq:
                a_cols=macd(df=df_result, abase=f"close_{index}", freq=sfreq, freq2=bfreq, inplace=True, type=4, score=1)
                print(a_cols)

                a_results_col+=[a_cols[0]]
                if debug < 2:
                    for counter in range(1,len(a_cols)):
                        del df_result[a_cols[counter]]

    #add all macd results together
    df_result["r0:buy_sell"]=0.0
    for counter, result_col in enumerate(a_results_col):
        df_result["r0:buy_sell"]=df_result["r0:buy_sell"].add(df_result[result_col],fill_value=0)
        if debug <2:
            del df_result[result_col]
    df_result["r0:buy_sell"]=df_result["r0:buy_sell"]/counter

    #calculate overlay freq volatility: adjust the result with volatility (because macd works best on high volatile time)
    df_result["r0:buy_sell"] = df_result["r0:buy_sell"] * df_result["volatility"]


if __name__ == '__main__':


    #run()
    df=DB.get_asset()
    df=support_resistance_horizontal_expansive(df_asset=df)
    df.to_csv("support.csv")



