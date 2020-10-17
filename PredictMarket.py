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
import builtins
import Atest
from functools import partial

def a_ts_code_helper(index):
    if index == "sh":
        a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="主板"))
    elif index == "sz":
        a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="中小板"))
    elif index == "cy":
        a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="创业板"))
    else:
        group=f"{index.split('_')[0]}_{index.split('_')[1]}"
        instance = index.split("_")[-1]

        df_group_lookup=DB.get_ts_code(a_asset=[group])
        a_ts_code=df_group_lookup[df_group_lookup[group]==instance]

    return a_ts_code


def predict_industry(df_result_copy):
    # Step 2: Industry Score
    df_ts_code_G = DB.get_ts_code(a_asset=["G"], d_queries=LB.c_G_queries_small_groups())

    # Step 2.1: Industry Long Time Score
    """df_longtime_score = Atest.asset_bullishness(df_ts_code=df_ts_code_G)
    df_longtime_score =df_longtime_score [df_longtime_score["period"]>2000]
    df_longtime_score.sort_values(by="final_position",ascending=True,inplace=True)
    """
    df_longtime_score = pd.read_csv("temp.csv")
    df_longtime_score.to_csv("temp.csv", encoding="utf-8_sig")
    df_longtime_score = df_longtime_score.set_index("ts_code")

    # Step 2.2: Industry Short Time Score
    fields = ["close", "vol"]
    result_col = []
    for ts_code in df_ts_code_G.index:
        print("calculate short time score", ts_code)
        # create asset aligned with sh and cy index
        try:
            df_asset = DB.get_asset(ts_code=ts_code, asset="G", a_columns=fields)
        except:
            continue

        # check for duplicated axis
        duplicate = df_asset[df_asset.index.duplicated()]
        if not duplicate.empty:
            print(ts_code, " has duplicated bug, check out G creation")
            continue

        # calculate trendmode pquota
        df_asset = df_asset.rename(columns={key: f"{key}_{ts_code}" for key in fields})
        df_asset = pd.merge(df_result_copy, df_asset, how='left', on="trade_date", suffixes=["", ""], sort=False)
        predict_trendmode(df_result=df_asset, index=ts_code, d_preload=d_preload)

        # add asset result to result table
        df_result[f"{ts_code}_trendmode_pquota"] = df_asset[f"{ts_code}_trendmode_pquota"]
        result_col += [f"{ts_code}_trendmode_pquota"]

    # general market condition
    for column in result_col:
        df_result["market_trend"] = df_result["market_trend"].add(df_result[column])
    df_result["market_trend"] = df_result["market_trend"] / len(result_col)

    # rank the industry short score. Rank the bigger the better
    d_score_short = {}
    for column in result_col:
        d_score_short[column] = df_result[column].iat[-1]

    df_final_industry_rank = pd.Series(d_score_short, name="ts_code_trendmode_pquota")
    df_final_industry_rank = df_final_industry_rank.to_frame()
    df_final_industry_rank["short_score"] = df_final_industry_rank["ts_code_trendmode_pquota"].rank(ascending=False)

    # rank the industry long score
    d_score_long = {}
    for column in result_col:
        ts_code = column[:-17]  # 17 because the name is "_trendmode_pquota"
        if ts_code in df_longtime_score.index:
            d_score_long[column] = df_longtime_score.at[ts_code, "final_position"]
            print("correct", column)
        else:
            print("oopse ts_code wrong or something. or substring removal wrong?", ts_code)
    s_long_score = pd.Series(d_score_long, name="ts_code_trendmode_pquota")

    # Step 2.3: Industry Current Final Score
    df_final_industry_rank["long_score"] = s_long_score
    df_final_industry_rank["final_score"] = df_final_industry_rank["long_score"] * 0.4 + df_final_industry_rank["short_score"] * 0.6

    return df_final_industry_rank


def predict_trendmode(df_result, d_preload, debug=0, index="cy"):

    """1. RULE BASED:
     base pquota is used by counting the days since bull or bear. The longer it goes on the more crazy it becomes"""
    df_result[f"{index}_trendmode_pquota"] = 0.0
    df_result[f"{index}_trendmode_pquota_days_counter"] = 0

    trend_duration=0
    last_trend=0
    for trade_date,today_trend in zip(df_result.index,df_result["cy_trend"]):

        if today_trend==1 and last_trend in [0,1]: # continue bull
            trend_duration+=1
        elif today_trend==1 and last_trend in [0,-1]: # bear becomes bull
            trend_duration=0
            last_trend=1
        elif today_trend==-1 and last_trend in [0,1]: # bull becomes bear
            trend_duration=0
            last_trend = -1
        elif today_trend==-1 and last_trend in [0,-1]: # continue bear
            trend_duration+=1
        else:#not initialized
            pass


        df_result.at[trade_date,f"{index}_trendmode_pquota_days_counter"]=trend_duration

    #assign base portfolio size: bottom time 0% max time 80%
    #note one year is 240, but minus -20 days to start because the signal detects turning point with delay

    #bull
    df_result.loc[(df_result["cy_trend"]==1)&(df_result[f"{index}_trendmode_pquota_days_counter"]<220), f"{index}_trendmode_pquota"] = 0.6
    df_result.loc[(df_result["cy_trend"]==1)&(df_result[f"{index}_trendmode_pquota_days_counter"].between(220,460)), f"{index}_trendmode_pquota"] = 0.7
    df_result.loc[(df_result["cy_trend"]==1)&(df_result[f"{index}_trendmode_pquota_days_counter"].between(460,700)),f"{index}_trendmode_pquota"] = 0.8
    df_result.loc[(df_result["cy_trend"]==1)&(df_result[f"{index}_trendmode_pquota_days_counter"].between(700,1000)),f"{index}_trendmode_pquota"] = 0.9

    #bear
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"] < 220),f"{index}_trendmode_pquota"] = 0.2
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(220, 460)),f"{index}_trendmode_pquota"] = 0.1
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(460, 700)),f"{index}_trendmode_pquota"] = 0.0
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(700, 1000)),f"{index}_trendmode_pquota"] = 0.0

    del df_result[f"{index}_trendmode_pquota_days_counter"]


    """2. EVENT BASED: OVERMA
    Ruse Overma to define buy and sell signals"""
    overma_name=step4(df_result=df_result, d_preload=d_preload, a_freq_close=[ 60,80,100, 120], a_freq_overma=[ 60, 80,100,120], index=index)



    #step 3 detect overbought or oversold in bull and bear using high pass filter
    """3. EVENT BASED ROLLING NORM
    to see if all stocks are too high or not to generate trade signals
    """
    a_ts_code = a_ts_code_helper(index=index)
    a_ts_code = list(a_ts_code.index)

    df_result[f"{index}_trendmode_pquota_fol"]=0.0
    df_result[f"{index}_trendmode_pquota_fol_counter"]=0
    for ts_code, df_asset in d_preload.items():
        if ts_code in a_ts_code:
            print(f"predict_trendmode calculate fol ",ts_code)
            fol_name=Alpha.fol_rolling_norm(df=df_asset,abase="close",inplace=True, freq_obj=[60,80,100,120])
            df_asset["counter_helper"]=1
            df_result[f"{index}_trendmode_pquota_fol"]=df_result[f"{index}_trendmode_pquota_fol"].add(df_asset[fol_name],fill_value=0)
            df_result[f"{index}_trendmode_pquota_fol_counter"]=df_result[f"{index}_trendmode_pquota_fol_counter"].add(df_asset["counter_helper"],fill_value=0)

    df_result[f"{index}_trendmode_pquota_fol"]=df_result[f"{index}_trendmode_pquota_fol"]/df_result[f"{index}_trendmode_pquota_fol_counter"]
    df_result[f"{index}_trendmode_pquota_fol"]=Alpha.norm(df=df_result,abase=f"{index}_trendmode_pquota_fol",inplace=False,min=-1,max=1)#normalize to range [-1:1]
    del df_result[f"{index}_trendmode_pquota_fol_counter"]


    #add all steps together
    df_result[f"{index}_trendmode_pquota"] = df_result[f"{index}_trendmode_pquota"].add(df_result[overma_name] * 0.10, fill_value=0)
    df_result[f"{index}_trendmode_pquota"] = df_result[f"{index}_trendmode_pquota"].subtract(df_result[f"{index}_trendmode_pquota_fol"] * 0.10, fill_value=0)
    df_result.loc[(df_result["sh_volatility"]>0.35)&(df_result["cy_trend"]==1),f"{index}_trendmode_pquota"] = 1
    df_result[f"{index}_trendmode_pquota"].clip(0,1,inplace=True)






def predict_cyclemode(df_result, d_preload, debug=0, index="sh"):
    def normaltime_signal(df_result,index="sh"):
        divideby = 0
        for counter in range(-50, 50):
            if counter not in [0]:  # because thats macd
                if f"{index}_r{counter}:buy_sell" in df_result.columns:
                    df_result[f"{index}_r:buy_sell"] = df_result[f"{index}_r:buy_sell"].add(df_result[f"{index}_r{counter}:buy_sell"], fill_value=0)
                    divideby += 1
        df_result[f"{index}_r:buy_sell"] = df_result[f"{index}_r:buy_sell"] / divideby

    def alltime_signal(df_result,index="sh"):
        # when switching between normal time strategy and crazy time strategy, there is no way to gradually switch. You either choose one or the other because crazy time is very volatile. In this time. I choose macd for crazy time.

        df_result[f"{index}_ra:buy_sell"] = 0.0
        for divideby, thresh in enumerate([0.35]):
            df_result[f"{index}_ra:buy_sell{thresh}"] = 0.0
            df_result.loc[df_result["sh_volatility"] <= thresh, f"{index}_ra:buy_sell{thresh}"] = df_result[f"{index}_r:buy_sell"]  # normal time
            df_result.loc[df_result["sh_volatility"] > thresh, f"{index}_ra:buy_sell{thresh}"] = df_result[f"{index}_r0:buy_sell"]  # crazy time
            df_result[f"{index}_ra:buy_sell"] += df_result[f"{index}_ra:buy_sell{thresh}"]
            del df_result[f"{index}_ra:buy_sell{thresh}"]
        df_result[f"{index}_ra:buy_sell"] = df_result[f"{index}_ra:buy_sell"] / (divideby + 1)

    """
    APPROACH: divide and conquer: choose most simple case for all variables
    1. Long period instead of short period
    2. Group of stocks(index, industry ETF) instead of individual stocks
    3. Only Buy Signals or Sell signals instead of both
    4. Overlay technique: Multiple freq results instead of one. If there is a variable like freq or threshhold, instead of using one, use ALL of them and then combine them into one result


    STEPS:
    3. (todo industry) check if index volume is consistent with past gain for index
    4. (todo idustry) calculate how many stocks are overma
    5. (done) check if top n stocks are doing well or not
    9. (todo currently no way access) calculate how many institution are holding stocks  http://datapc.eastmoney.com/emdatacenter/JGCC/Index?color=w&type=
    11. (todo finished partly) Price and vol volatility
    12. (todo with all stocks and industry) check volatiliy adj macd
    
    not working well 
    17. Fundamental values not to predict shorterm market. Only useable to select stocks




    RESULT INTERPRETATION:
    1. magnitude is 1 to -1. The magnitude represents chance/likelihood/confidence that the market will rise or fall in future.
    2. the magnitude can be mapped to portfolio size. The higher the chance, the more money to bet in.
    3. the results mostly shows BEST buy can sell moments. If a buy moment goes back to 0 couple days after a 0.3 buy moment signal, this means the best buy moment is not anymore.
    4. the moments where signal shows 0, there are many ways to interpret:
        -last buy_sell signal counts.
        -don't do anything
        -take a smaller freq result
    5. One good way to interpret is buy when signal shows above 0.2., buy 20% portfolio. Hold until signal shows -0.2 or more and sell Everything.
    6. Whenever sell occurs, always sell everthing. In crazy time, this must happen very quickly. In normal times, the signal is not that accurate, you can have a bit more time to sell at once or bit by bit.
    7. Whenever buy occurs, the amplitude means the confidence. This can be 1:1 mapped to portfolio size.
    8. Future bull market can be detected if signals show 0.2 or more. Future bear market is vice versa. The bull or bear market goes as long as the opposing signal appears. Then the trend switches


    KNOWLEDGE SUMMARY - normal vs crazy time:
    occurence: crazy time only occurs 10% of the time. 3 years in 30 years.
    volatility: normal time low volatiliy. Crazy time high volatility.
    strategy: normal time buy low, sell high = against trend. crazy time you buy with trend.
    freq: normal time use 60-240 as freq. crazy time use 20-60 as freq.
    MACD crazy time: MACD good on crazy time because it buys with trend AND it is able to detect trend swap very good.
    MACD normal time: bad because there is not significant trend to detect. too much whipsaw. AND you should buy against the trend.
    Turning point: normal time anticipate BUT with tiny confirmation. Crazy time wait for turning point, also with Tiny confirmation.
    Volume: crazy time high, normal time low
    overma: crazy time high, normal time low
    => in crazy time, use different strategy and higher freq
    => if UP trend is SURE, then buy low sell high. if trend is not sure, wait for confirmation to buy
    
    The mistake in my previous research was to seek and define bull and bear market. When in reality. One must first define crazy and normal time.

    PORTFOLIO:
    Crazy and normal time can both have 100% portfolio. You can not choose how market gives 60% return or 600%. You can only choose your portfolio size, buy or not buy. Don't miss even if market returns 20%. 
    => The final signal is craziness adjusted portfolio size. This means that crazy time and normal time signals CAN BE COMPARED AGAINST. They are both on scala -1 to 1 to make comparison consistent.
    => This also makes portfolio decisions easier. You can directly convert the confidence into portfolio size.
    => sh stocks most time in cycle mode. cy Stocks most time in trend mode.

    DEBUG:
    level 0 = only show combined final result like r4:buy
    level 1 = level 0 + index based like r4:sh_buy
    level 2 = level 1 + different freq like r4:sh_close120.buy
    level 3 = level 2 + junk
    
    
    Design TODO
    MANUAL CORRECTION
    combination of two theories
    a pquote tester that varifies the result
    tweak accuracy
    strength of a trend, to check how strong the turning point must be to turn over the trend
    variable frequency. At high volatile time, use smaller freq
    find a better and more reliable time classifier to replace the hard coded sh version
    idea to use cummulative signals during normal time. e.g. signals from last 5 to 10 days together combined show me how strong sell should be instead of one single.
    add the idea that in bear market, holing period is short and in bull, holding period is long
    
    Technical TODO
    naming conventions
    manage efficiency of the code, less redundant
    maybe interface to see more clear the final result
    """

    # START PREDICT
    print(f"START PREDICT ---> {index} <---")
    print()

    df_result[f"{index}_r:buy_sell"] = 0.0
    df_result[f"{index}_ra:buy_sell"] = 0.0

    # 0 MACD  (on single index) CRAZY
    step0(df_result=df_result, index=index, debug=debug)

    # 3 VOLUME (on single index) NORMAL
    step3(df_result=df_result, index=index, debug=debug)

    # 4 OVERMA (on all stocks) NORMAL
    step4(df_result=df_result, index=index, d_preload=d_preload, debug=debug)

    # Combine NORMAL TIME buy and sell signal into one.
    normaltime_signal(df_result,index=index)

    # Add CRAZY TIME signal into the normal time signal = > all time signal.
    alltime_signal(df_result,index=index)

    # OPTIONAL: smooth the result to have less whipsaw
    # df_result["ra:buy_sell"]=Alpha.zlema(df=df_result, abase="ra:buy_sell", freq=5, inplace=False ,gain=0)

    # portfolio strategies
    to_cyclemode_pquota(df_result=df_result, abase=f"{index}_ra:buy_sell", index=index)

    # check only after year 2000
    df_result = LB.trade_date_to_calender(df=df_result, add=["year"])
    df_result = df_result[df_result["year"] >= 2000]
    del df_result["year"]
    return


def to_cyclemode_pquota(df_result, abase, index="sh"):
    """
    This portfolio strategy is simple: buy when > 0.2. Sell when <0.2
    buy until sell signal occurs


    """
    df_result[f"{index}_cyclemode_pquota"] = 0.0
    portfolio = 0.0

    for trade_date in df_result.index:
        # loop over each day
        signal = df_result.at[trade_date, abase]
        if signal > 0:
            portfolio = builtins.max(portfolio, signal)
        elif signal < 0:
            portfolio = 0.0  # reset portfolio to 0
        elif signal == 0:
            # variation 1: no nothing and use previous high as guideline
            # variation 2: interpret it as sell signal if previous signal was buy. interpret as buy if previous signal was sell.
            # variation 3: use a low freq strategy to take a deeper look into it
            pass

        # assign value at end of day
        df_result.at[trade_date, f"{index}_cyclemode_pquota"] = portfolio






def step0(df_result, debug=0, index="sh"):
    """MACD"""

    result_name=f"{index}_r0:buy_sell"

    # create all macd
    a_results_col = []
    for sfreq in [60, 120, 180,240]:
        for bfreq in [ 180, 240, 300,360]:
            if sfreq < bfreq:
                print(f"{index}: step0 sfreq{sfreq} bfreq{bfreq}")
                a_cols = macd(df=df_result, abase=f"close_{index}", freq=sfreq, freq2=bfreq, inplace=True, type=4, score=1)
                a_results_col += [a_cols[0]]

                # delete unessesary columns such as macd dea, diff
                if debug < 2:
                    for counter in range(1, len(a_cols)):  # start from 1 because 0 is reserved for result col
                        del df_result[a_cols[counter]]

    # add all macd results together
    df_result[result_name] = 0.0
    for counter, result_col in enumerate(a_results_col):
        df_result[result_name] = df_result[result_name].add(df_result[result_col], fill_value=0)
        if debug < 2:
            del df_result[result_col]

    # normalize
    df_result[result_name] = df_result[result_name] / (counter + 1)

    # calculate overlay freq volatility: adjust the result with volatility (because macd works best on high volatile time)
    # df_result["r0:buy_sell"] = df_result["r0:buy_sell"] * df_result["volatility"]


def step3(df_result, index="sh", debug=0 ):
    """volume

    volume is best used to predict start of crazy time. in normal time, there is not so much information in volume.
    """


    def step3_single(df_result, index, freq_close=240, freq_vol=360, debug=0):
        """
        This can detect 3 signals:
        1. high volume and high gain -> likely to reverse to bear
        2. low volume and high gain -> even more likely to reverse to bear
        3. high volume and low gain -> likely to reverse to bull
        """

        vol_name = f"vol_{index}"
        close_name = f"close_{index}"
        result_name = f"r3:{index}_vol{freq_vol}_close{freq_close}"

        # normalize volume and close first with rolling 240 days
        norm_vol_name = Alpha.rollingnorm(df=df_result, abase=vol_name, freq=freq_vol, inplace=True)
        norm_close_name = Alpha.rollingnorm(df=df_result, abase=close_name, freq=freq_close, inplace=True)

        # 1. Sell Signal: filter only days where vol > 0.7 and close > 0.6
        df_helper = df_result.loc[(df_result[norm_vol_name] > 0.7) & (df_result[norm_close_name] > 0.6)]
        sell_signal1 = df_helper[norm_vol_name] + df_helper[norm_close_name]  # higher price, higher volume the more clear the signal

        # 2. Sell Signal: filter only days where vol < 0.5 and close > 0.8
        df_helper = df_result.loc[(df_result[norm_vol_name] < 0.4) & (df_result[norm_close_name] > 0.80)]
        sell_signal2 = (1 - df_helper[norm_vol_name]) + df_helper[norm_close_name]  # higher price, lower volume the more clear the signal

        # combine both type of sell signals
        df_result[f"{result_name}_sell"] = sell_signal1.add(sell_signal2, fill_value=0)

        # 3. Buy Signal: filter only days where vol > 0.6 and close < 0.4
        df_helper = df_result.loc[(df_result[norm_vol_name] > 0.7) & (df_result[norm_close_name] < 0.4)]
        buy_signal = df_helper[norm_vol_name] + (1 - df_helper[norm_close_name])  # higher volume, lower price the more clear the signal
        df_result[f"{result_name}_buy"] = buy_signal

        # 4. Delete unessesary columns produced
        if debug < 3:
            del df_result[norm_vol_name]
            del df_result[norm_close_name]

        return [f"{result_name}_buy", f"{result_name}_sell"]



    result_name = f"{index}_r3:buy_sell"

    # loop over all frequency
    df_result[f"{index}_r3:buy"] = 0.0
    df_result[f"{index}_r3:sell"] = 0.0
    result_list = []
    counter = 0
    for freq_close in [240, 500]:
        for freq_vol in [120, 500]:
            print(f"{index}: step3 close{freq_close} vol{freq_vol}...")
            counter += 1
            buy_sell_label = step3_single(df_result=df_result, freq_close=freq_close, freq_vol=freq_vol, index=index, debug=debug)
            result_list = result_list + [buy_sell_label]

    # combine all frequecies into one result for ONE index
    for buy_freq_signal, sell_freq_signal in result_list:
        df_result[f"{index}_r3:buy"] = df_result[f"{index}_r3:buy"].add(df_result[buy_freq_signal], fill_value=0)
        df_result[f"{index}_r3:sell"] = df_result[f"{index}_r3:sell"].add(df_result[sell_freq_signal], fill_value=0)
        if debug < 2:
            del df_result[buy_freq_signal]
            del df_result[sell_freq_signal]

    # normalize the result
    df_result[f"{index}_r3:buy"] = df_result[f"{index}_r3:buy"] / (counter * 2)
    df_result[f"{index}_r3:sell"] = df_result[f"{index}_r3:sell"] / (counter * 2)

    # combine buy and sell
    df_result[result_name] = df_result[f"{index}_r3:buy"].add(df_result[f"{index}_r3:sell"] * (-1), fill_value=0)

    if debug < 3:
        del df_result[f"{index}_r3:buy"]
        del df_result[f"{index}_r3:sell"]
    return





def step4(df_result, d_preload, index="sh", a_ts_code=[],  a_freq_close=[240, 500],a_freq_overma=[120, 500],debug=0,):
    """Overma"""


    def step4_single(df_result, d_preload, a_ts_code, freq_close=240, freq_overma=240, index="sh", debug=0):
        """calculate how many stocks are overma generally very useful

        for period in [500,240,120]:
            1. General overma
            2. Index overma
            3. Industry  overma
            4. Size overma
        """

        # 1. General ALL STOCK overma
        # 1.1 normalize overma series
        if f"{index}_overma{freq_overma}" not in df_result.columns:
            df_result[f"{index}_overma{freq_overma}"] = 0.0
            df_result[f"{index}_counter{freq_overma}"] = 0.0

            for ts_code, df_asset in d_preload.items():
                if ts_code in a_ts_code:
                    # calculate if stocks is over its ma
                    df_asset[f"{index}_ma{freq_overma}"] = df_asset["close"].rolling(freq_overma).mean()
                    df_asset[f"{index}_overma{freq_overma}"] = (df_asset["close"] >= df_asset[f"{index}_ma{freq_overma}"]).astype(int)
                    df_asset[f"{index}_counter{freq_overma}"] = 1

                    df_result[f"{index}_overma{freq_overma}"] = df_result[f"{index}_overma{freq_overma}"].add(df_asset[f"{index}_overma{freq_overma}"], fill_value=0)
                    # counter to see how many stocks are available
                    df_result[f"{index}_counter{freq_overma}"] = df_result[f"{index}_counter{freq_overma}"].add(df_asset[f"{index}_counter{freq_overma}"], fill_value=0)

            # finally: calculate the percentage of stocks overma
            df_result[f"{index}_overma{freq_overma}"] = df_result[f"{index}_overma{freq_overma}"] / df_result[f"{index}_counter{freq_overma}"]


        # 1.2 normalize close series
        norm_close_name = Alpha.rollingnorm(df=df_result, freq=freq_close, abase=f"close_{index}", inplace=True)

        # 1.3 generate  Buy Signal: price < 0.25 and overma < 0.25
        df_helper = df_result.loc[(df_result[f"{index}_overma{freq_overma}"] < 0.25) & (df_result[norm_close_name] < 0.25)]
        df_result[f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_buy"] = (1 - df_helper[f"{index}_overma{freq_overma}"]) + (1 - df_helper[norm_close_name])  # the lower the price, the lower overma, the better

        # 1.4 generate  Sell Signal: price > 0.75 and overma > 0.75
        df_helper = df_result.loc[(df_result[f"{index}_overma{freq_overma}"] > 0.75) & (df_result[norm_close_name] > 0.75)]
        df_result[f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_sell"] = df_helper[f"{index}_overma{freq_overma}"] + df_helper[norm_close_name]  # the lower the price, the lower overma, the better

        # 1.5 delete unessary columns
        if debug < 3:
            del df_result[f"{index}_overma{freq_overma}"]
            del df_result[f"{index}_counter{freq_overma}"]
            del df_result[norm_close_name]

        return [f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_buy", f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_sell"]


    # generate matching list of ts_code for index to be used for overma later
    a_ts_code=a_ts_code_helper(index=index)
    a_ts_code = list(a_ts_code.index)



    df_result[f"{index}_r4:buy"] = 0.0
    df_result[f"{index}_r4:sell"] = 0.0

    # loop over all frequency
    result_list = []
    counter = 0
    for freq_close in a_freq_close:
        for freq_overma in a_freq_overma:
            print(f"{index}: step4 close{freq_close} overma{freq_overma}...")
            buy_sell_label = step4_single(df_result=df_result, d_preload=d_preload, a_ts_code=a_ts_code, freq_close=freq_close, freq_overma=freq_overma, index=index, debug=debug)
            result_list = result_list + [buy_sell_label]
            counter += 1

    # combine all frequecies into one result for ONE index
    for buy_signal, sell_signal in result_list:
        df_result[f"{index}_r4:buy"] = df_result[f"{index}_r4:buy"].add(df_result[buy_signal], fill_value=0)
        df_result[f"{index}_r4:sell"] = df_result[f"{index}_r4:sell"].add(df_result[sell_signal], fill_value=0)
        if debug < 2:
            del df_result[buy_signal]
            del df_result[sell_signal]

    # normalize the result
    df_result[f"{index}_r4:buy"] = df_result[f"{index}_r4:buy"] / (counter * 2)  # why times 2 actually
    df_result[f"{index}_r4:sell"] = df_result[f"{index}_r4:sell"] / (counter * 2)

    # combine buy and sell
    df_result[f"{index}_r4:buy_sell"] = df_result[f"{index}_r4:buy"].add(df_result[f"{index}_r4:sell"] * (-1), fill_value=0)

    # debug
    if debug < 3:
        del df_result[f"{index}_r4:buy"]
        del df_result[f"{index}_r4:sell"]

    return f"{index}_r4:buy_sell"

def step5(df_result, d_preload, debug=0):
    """check if top n stocks (low beta stocks stocks) are doing well or not
    If even they are bad, then the whole stock market is just bad for sure

    algorith:
    1. define top n stocks using fundamentals and technicals
    2. check if they are doing well in last freq D: 5, 20, 60

    （1. cheat, use shortage to manually define these 50 stocks)
    """

    def step5_single(df_result, debug=0):
        # 2. Generate step 5 buy sell signal using custom defined rules
        # works worse than v2 with macd
        r5_freq_buy_result = []
        r5_freq_sell_result = []
        df_result["r5:buy"] = 0.0
        df_result["r5:sell"] = 0.0  # step5 does not produce any sell signal

        for freq in [120, 240, 500]:
            print(f"all: step5 close{freq}...")
            # rolling norm
            topn_close_name = Alpha.rollingnorm(df=df_result, freq=freq, abase="r5:topn_index", inplace=True)

            # is max
            df_result["topn_emax"] = df_result["r5:topn_index"].expanding().max()
            is_top_pct = Alpha.ismax(df=df_result, abase="r5:topn_index", emax="topn_emax", inplace=True, q=0.85, score=1)

            # 2.1 Buy if past normalized return is < 0.2
            df_helper = df_result.loc[(df_result[topn_close_name] < 0.20)]
            df_result[f"r5:topn_close{freq}_buy"] = 1 - df_helper[topn_close_name]
            r5_freq_buy_result += [f"r5:topn_close{freq}_buy"]

            # 2.2 Sell if they are not at top 15% and there is no buy signal = bear but not bear enough
            df_helper = df_result[(df_result[is_top_pct] == -1) & (df_result[f"r5:topn_close{freq}_buy"].isna())]
            df_helper["sell_helper"] = 1
            df_result[f"r5:topn_close{freq}_sell"] = df_helper["sell_helper"]
            r5_freq_sell_result += [f"r5:topn_close{freq}_sell"]

            if debug < 2:
                del df_result[topn_close_name]
                del df_result["topn_emax"]
                del df_result[is_top_pct]

        # combine all freq into one
        counter = 0
        for freq_result in r5_freq_buy_result:
            df_result["r5:buy"] = df_result["r5:buy"].add(df_result[freq_result], fill_value=0)
            counter += 1
            if debug < 1: del df_result[freq_result]

        counter = 0
        for freq_result in r5_freq_sell_result:
            df_result["r5:sell"] = df_result["r5:sell"].add(df_result[freq_result], fill_value=0)
            counter += 1
            if debug < 1: del df_result[freq_result]

        df_result["r5:buy"] = df_result["r5:buy"] / counter
        df_result["r5:sell"] = df_result["r5:sell"] / counter

        # for now exclude sell result
        # df_result["r5:sell"] = 0.0

        # combine buy and sell
        df_result["r5:buy_sell"] = df_result[f"r5:buy"].add(df_result[f"r5:sell"] * (-1), fill_value=0)

        # adjust with volatility

        if debug < 2:
            del df_result["r5:buy"]
            del df_result["r5:sell"]
            # del df_result["r5:topn_index"]
        return

    def step5_single_v2(df_result, debug=0):
        # 2. Generate step 5 buy sell signal using macd. Because MACD buys on uptrend, sell on downtrend. goes very well with good stocks that are uptrend most of the time.

        # create all macd
        a_results_col = []
        for sfreq in [120, 180, 240]:
            for bfreq in [180, 240, 300, 360, 500]:
                if sfreq < bfreq:
                    print(f"all: step5 sfreq{sfreq} bfreq{bfreq}")
                    a_cols = macd(df=df_result, abase=f"r5:topn_index", freq=sfreq, freq2=bfreq, inplace=True, type=4, score=1)
                    a_results_col += [a_cols[0]]
                    if debug < 2:
                        for counter in range(1, len(a_cols)):
                            del df_result[a_cols[counter]]

        # add all macd results together
        df_result["r5:buy_sell"] = 0.0
        for counter, result_col in enumerate(a_results_col):
            df_result["r5:buy_sell"] = df_result["r5:buy_sell"].add(df_result[result_col], fill_value=0)
            if debug < 2:
                del df_result[result_col]

        # normalize
        df_result["r5:buy_sell"] = df_result["r5:buy_sell"] / counter

        # adjust with sh_index volatility
        df_result["r5:buy_sell"] = df_result["r5:buy_sell"] * df_result["sh_volatility"]

        return

    # 1. Generate top n index
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

    a_ts_codes = ["600519.SH", "000568.SZ", "600887.SH", "600036.SH", "600585.SH", "600272.SH", "000538.SZ", "002032.SZ", "000651.SZ"]

    df_result["step5_counter"] = 0.0
    df_result["step5_topn_pct_chg"] = 0.0

    for ts_code, df_asset in d_preload.items():
        if ts_code in a_ts_codes:
            # add counter together
            df_asset["step5_counter"] = 1
            df_result["step5_counter"] = df_result["step5_counter"].add(df_asset["step5_counter"], fill_value=0)

            # add gain together
            df_result["step5_topn_pct_chg"] = df_result["step5_topn_pct_chg"].add(df_asset["pct_chg"], fill_value=0)

    df_result["step5_topn_pct_chg"] = df_result["step5_topn_pct_chg"] / df_result["step5_counter"]
    df_result["r5:topn_index"] = Alpha.comp_chg(df=df_result, abase="step5_topn_pct_chg", inplace=False, start=100)

    if debug < 2:
        del df_result["step5_counter"]
        del df_result["step5_topn_pct_chg"]

    step5_single_v2(df_result=df_result, debug=debug)






def cy_mode(df_result, abase="close"):
    """
    this function detects in what mode/phase the cy stock is
    """

    # add all freq of rolling norm together
    df_result["fol_close_norm"] = 0.0
    a_del_cols = []
    counter = 0
    for freq in range(10, 510, 10):
        print(f"freq is {freq}")
        name = Alpha.rollingnorm(df=df_result, abase=abase, freq=freq, inplace=True)
        df_result["fol_close_norm"] = df_result["fol_close_norm"] + df_result[name]
        counter += 1
        a_del_cols += [name]
    df_result["fol_close_norm"] = df_result["fol_close_norm"] / counter

    # produce bull or bear market. 1 means bull, -1 means bear.
    bull_bear = 0.0
    for trade_date in df_result.index:

        # loop over each day
        signal = df_result.at[trade_date, "fol_close_norm"]
        if signal > 0.8:  # bull
            bull_bear = 1
        elif signal < 0.2:
            bull_bear = -1  # reset portfolio to 0
        else:
            # variation 1: no nothing and use previous high as guideline
            # variation 2: interpret it as sell signal if previous signal was buy. interpret as buy if previous signal was sell.
            # variation 3: use a low freq strategy to take a deeper look into it
            pass

        # assign value at end of day
        df_result.at[trade_date, "bull_bear"] = bull_bear

    df_result.drop(a_del_cols, axis=1, inplace=True)
    # df_result.to_csv("egal.csv")


def all(withupdate=False):
    """
    Goal: Predict market, generate concrete detailed buy and sell signals
    1. When to buy/sell/: If market is good => predict macro market bull or bear
    2. How much to buy/sell: If market is micro market overbought or underbought
    3. What to buy/sell: Check stocks, etfs, industries, concepts
    """


    # Step 0: UPDATE DATA and INIT
    if withupdate: DB.update_all_in_one_cn_v2()
    df_result = DB.get_stock_market_overview()
    df_result["sh_volatility"] = Alpha.detect_cycle(df=df_result, abase=f"close_sh", inplace=False)  # use sh index to calculate volatility no matter what
    df_result["cy_trend"] = Alpha.detect_bull(df=df_result, abase=f"close_cy", inplace=False)  # use cy index to detect macro trends
    df_result["market_trend"]=0.0# reserved for usage later
    df_result_copy = df_result.copy()
    d_preload = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")


    # Step 1.1: predict SH index Portfolio Quota: defines how much portfolio you can max spend
    #predict_cyclemode(df_result=df_result, index="sh", d_preload=d_preload)

    # Step 1.2: predict CY index Portfolio Quota: defines how much portfolio you can max spend
    predict_trendmode(df_result=df_result, index="cy", d_preload=d_preload)

    # Step 1.3: predict SZ index
    #predict_cyclemode(df_result=df_result, index="sz", d_preload=d_preload)
    #predict_trendmode(df_result=df_result, index="sz", d_preload=d_preload)



    #defines the max portfolio size the can be used during a time. It depends on the market mood.



    # Inudstry
    #df_final_industry_rank = predict_industry(df_result_copy=df_result_copy)



    #Portfolio max size
    #based on previous statistic. 6-8 years 1 cycle. bull market = 2.5 year, neutral = 1.5, bear = 2 year
    #the longer the bull, the crazier it becomes


    # Step 5: Select Concept



    # Step 6: Select Stock
    #according to its industry rank 1-100
    #accoding to its fundamental 1-100
    #according to fund holding 1-100
    #according to its current technical analysis 1-100


    # Step 7.1: Save Predict Table
    a_path = LB.a_path(f"Market/CN/PredictMarket/Predict")
    LB.to_csv_feather(df=df_result, a_path=a_path)

    # Step 7.2: Save Industry Score
    a_path = LB.a_path(f"Market/CN/PredictMarket/Industry_Score")
    #LB.to_csv_feather(df=df_final_industry_rank, a_path=a_path)






def _deprecated_seasonal_effect(df_result, debug=0):
    """
    currently no use of seasonal effect because they are too periodic.
    Seasonal effect are interesting, but deviation are too big.
    Hence it makes the stats useless

    1. overlay of chinese month of year effect
    2. overlay of first month prediction effect
    3. overlay of day of month effect
    """
    # PART 1
    # init
    df_trade_date = DB.get_trade_date()

    df_result["year"] = df_trade_date["year"]
    df_result["month"] = df_trade_date["month"]
    df_result["day"] = df_trade_date["day"]
    df_result["weekofyear"] = df_trade_date["weekofyear"]
    df_result["dayofweek"] = df_trade_date["dayofweek"]
    df_result["r8:buy_sell"] = 0.0

    # overlay of all divisions are NOT IN USE
    for division in ["month", "weekofyear"]:
        # overlay of seasonal effect
        df_division = DB.get(a_path=LB.a_path(f"Market/CN/ATest/seasonal_stock/{division}"), set_index=division)
        df_result[division] = df_result[division].astype(int)
        df_result[division] = df_result[division].replace(df_division["pct_chg"].to_dict())
        df_result[division] = df_result[division].astype(float)
        # df_result["r8:buy_sell"]+=df_result[division]

    # PART 2
    df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sh = LB.trade_date_to_calender(df_sh)
    # overlay of chinese new year effect(compare ny gain against others. If strong then the whole year is strong)
    # in order to give a more real prediction, we conduct the prediction step by step from the past

    df_sh_helper = df_sh[df_sh["month"] == 2]
    df_result = df_sh_helper.groupby("year").mean()
    df_result.to_csv("test.csv")
    # todo unfinished because I feel it will not be better than other existing signals
    # overlay of first month (compare first month gain against others. If strong then the whole year is strong)

    # overlay first and last week of year


def _deprecated_rsi(df_result, debug=0):
    """
    rsi freq: this step is to check if different freq combination of rsi would make a better rsi signal
    """

    df_result["r10:buy"] = 0
    df_result["r10:sell"] = 0
    for counter, freq in enumerate([20, 40, 60, 80, 100, 120, 180, 240, 300, 360]):
        rsi_name = Alpha.rsi(df=df_result, abase="close_sh", freq=freq, inplace=True)

        # create buy signal
        df_helper = df_result.loc[(df_result[rsi_name] < 50)]
        df_result[f"r10:close_sh{freq}_buy"] = df_helper[rsi_name]

        # create sell signal
        df_helper = df_result.loc[(df_result[rsi_name] > 50)]
        df_result[f"r10:close_sh{freq}_sell"] = df_helper[rsi_name]

        df_result["r10:buy"] = df_result["r10:buy"].add(df_result[f"r10:close_sh{freq}_buy"], fill_value=0)
        df_result["r10:sell"] = df_result["r10:sell"].add(df_result[f"r10:close_sh{freq}_sell"], fill_value=0)

        if debug < 1:
            del df_result[rsi_name]

    df_result["r10:buy"] = df_result["r10:buy"] / (counter + 1)
    df_result["r10:sell"] = df_result["r10:sell"] / (counter + 1)


def _deprecated_volatility(df_result, debug=0, index="sh"):
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
    # 1. Check volatility AGAINST PAST
    # 1.1 check time with PRICE volatility AGAINST PAST
    # result -> can predict crazy and normal time

    # normalize price
    df_result[f"r11:buy"] = 0.0
    df_result[f"r11:sell"] = 0.0
    divideby = 1
    for freq in [120, 240, 500]:
        print(f"step11 {index} close{freq}", )
        # normalize close
        norm_close_name = Alpha.rollingnorm(df=df_result, abase=f"close_{index}", freq=freq, inplace=True)

        # calcualte close std
        df_result[f"close_std{freq}"] = df_result[f"close_{index}"].rolling(freq).std()

        # normalize result(dirty way, should not be like that because it knows future)
        norm_close_std_name = Alpha.norm(df=df_result, abase=f"close_std{freq}", inplace=True)

        # generate buy signal: volatiliy is low and past price is low = Buy
        # volatility < 0.2, past gain < 0.4: buy. indicates turning point
        df_helper = df_result[(df_result[norm_close_name] < 0.3) & (df_result[norm_close_std_name] < 0.2)]
        df_result[f"r11:norm{freq}_buy"] = (1 - df_helper[norm_close_name]) + (1 - df_helper[norm_close_std_name])
        df_result[f"r11:buy"] = df_result[f"r11:buy"].add(df_result[f"r11:norm{freq}_buy"], fill_value=0)

        # generate Sell signal: volatiliy is low and past price is high = Sell
        # volatility < 0.2, past gain > 0.8: buy. indicates turning point
        df_helper = df_result[(df_result[norm_close_name] > 0.7) & (df_result[norm_close_std_name] < 0.2)]
        df_result[f"r11:norm{freq}_sell"] = (df_helper[norm_close_name]) + (1 - df_helper[norm_close_std_name])
        df_result[f"r11:sell"] = df_result[f"r11:sell"].add(df_result[f"r11:norm{freq}_sell"], fill_value=0)

        # increment divideby
        divideby += 1

        # debug
        if debug < 2:
            del df_result[norm_close_name]
            del df_result[f"close_std{freq}"]
            del df_result[norm_close_std_name]
            del df_result[f"r11:norm{freq}_buy"]
            del df_result[f"r11:norm{freq}_sell"]

    # normalize
    df_result[f"r11:buy"] = df_result[f"r11:buy"] / (divideby * 2)
    df_result[f"r11:sell"] = df_result[f"r11:sell"] / (divideby * 2)

    # generate sell signal: volatiliy is low and past price is high = Sell

    # 1.2 check time with VOL volatility AGAINST PAST
    # Result: ok but not as good as price std (because volume is not mean normalized?)
    """
    func_vol_partial = partial(func, abase="vol_sh")
    LB.frequency_ovelay(df=df_result, func=func_vol_partial, a_freqs=[[20, 40, 60, 120, 240]], a_names=["sh_vol_std", "vol_sh"], debug=debug)
    df_result.loc[df_result["year"] < 2000, "sh_vol_std"] = 0.0
    """

    # 2. Volatility against other stock

    # 3. Volatility intraday


def _deprecated_support_resistance():
    """
    two things  to be done with support and resistance:
    1. calculate actual support and resistance
    A: This can be looked on

    2. produce signals using minmax. IF current low is higher than last low and current high is higher than last high.
    A: Problem is that this method is slower than macd freq overlay. IT also does not support freq overlay that good. In addition, it produces more whipsaw. Also, baidu index data start from 2010, google trend does not work for CN stocks. In general. data is hard to get and same as index volume.
    A: We just use macd freq overlay instead of this.

    """


def _deprecated_search_engine():
    """
    using baidu, weibo, google trend data to predict.

    the problem :
    1. They are simultaneously occuring the stock market volume. So they don't predict in ADVANCE but at the same time. This makes them less useful as I can just watch the volume.
    2. They are sort of external signals. Some times high search volume does not mean bull but bear. And some times high search volume does not move the market at all.

    :return:
    """


def _deprecated_fundamentals():
    """
    Fundamentals
    This is a big topic, lets break it down to think what indicators could be useful

    Problem:
    1. fundamental indicators are lagging, the market knows first and bed before the market is even bull
    2. fundamental are publicit not very frequently.
    3. fundamentals can be faked
    4. Insider trades even pushes

    Use:
    1. Use fundamentals in a very very long term to see if a stock is stable or not
    2. the keyword is stability. The more stable the fundamental data, the better predictable the future in very very long run.
    3. But using the method in 2. will lose chance to many opportunistic investments
    4. use fundamentals also limits the stock to be very old. New stocks can not be viewed


    Note that we want to find the extrem values first as they are easier to find and have the most predictive power.

    1. PE
    2. PB
    3.

    :return:
    """

def _deprecated_cummulative_volume_vs_sh():
    """
    create a sumarized time series to see the overall vol of the complete market

    RESULT: The summarized volume is Exactly same as the sh index volume

    """

    # load in all the stocks
    d_preload = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")

    df_market = DB.get_asset(ts_code="000001.SH", asset="I", freq="D", market="CN")
    df_market = df_market[["close", "vol", "amount"]]
    df_market["agg_abs_amount"] = 0.0
    df_market["agg_rel_amount"] = 0.0
    df_market["agg_stocks"] = 0.0

    print("lengh of preload", len(d_preload))

    for ts_code, df_asset in d_preload.items():
        print(ts_code)

        df_asset["count_helper"] = 1.0
        df_market["agg_stocks"] = df_market["agg_stocks"].add(df_asset["count_helper"], fill_value=0)
        df_market["agg_abs_amount"] = df_market["agg_abs_amount"].add(df_asset["amount"], fill_value=0)

    df_market["agg_rel_amount"] = df_market["agg_abs_amount"] / df_market["agg_stocks"]
    a_path = LB.a_path(f"Market/CN/PredictMarket/Market")
    LB.to_csv_feather(df=df_market, a_path=a_path)



if __name__ == '__main__':
    all(withupdate=False)








"""

strategy analysis:
macd: buy bull, sell bear. good at volatile time. either buy or not buy
rollingnorm: buy bull sell bear. gradiently give signals.

CY Model:
continous phase of these 4 states: like in the buddha 金刚经
-normal uptrend
-turning point (= crazy time) = crazy up and crazy down
-normal downtrend
-turning point (= silent turning point)

Most important things to find in the first place:
1. start of crazy time
2. end of crazy time

"""
