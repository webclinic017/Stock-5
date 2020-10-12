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
from functools import partial



def update_data():

    """
    1. update index
    2. update all stocks
    """

    # DB.update_all_in_one_cn_v2()

    return

def predict(df_result, d_preload,debug=0,index="sh"):

    def normaltime_signal(df_result):
        divideby = 0
        for counter in range(-50, 50):
            if counter not in [0]:  # because thats macd
                if f"r{counter}:buy_sell" in df_result.columns:
                    df_result["r:buy_sell"] = df_result["r:buy_sell"].add(df_result[f"r{counter}:buy_sell"], fill_value=0)
                    divideby += 1
        df_result["r:buy_sell"] = df_result["r:buy_sell"] / divideby

    def alltime_signal(df_result):
        # when switching between normal time strategy and crazy time strategy, there is no way to gradually switch. You either choose one or the other because crazy time is very volatile. In this time. I choose macd for crazy time.

        df_result["ra:buy_sell"] = 0.0
        for divideby, thresh in enumerate([0.35]):
            df_result[f"ra:buy_sell{thresh}"] = 0.0
            df_result.loc[df_result["volatility"] <= thresh, f"ra:buy_sell{thresh}"] = df_result["r:buy_sell"]  # normal time
            df_result.loc[df_result["volatility"] > thresh, f"ra:buy_sell{thresh}"] = df_result["r0:buy_sell"]  # crazy time
            df_result["ra:buy_sell"] += df_result[f"ra:buy_sell{thresh}"]
            del df_result[f"ra:buy_sell{thresh}"]
        df_result["ra:buy_sell"] = df_result["ra:buy_sell"] / (divideby + 1)


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
    6. (todo) check if best 3 industry are doing well or not
    8. (todo relatively not included atm) overlay of the new year period month
    9. (todo currently no way access) calculate how many institution are holding stocks  http://datapc.eastmoney.com/emdatacenter/JGCC/Index?color=w&type=
    10. (done but not that useful) rsi freq combination.
    11. (todo finished partly) Price and vol volatility
    12. (todo with all stocks and industry) check volatiliy adj macd
    17. Fundamental values: sales, return data


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
    MACD crazy time: MACD good on crazy time because it buys with trend AND it is able to detect trend swap very good.
    MACD normal time: bad because there is not significant trend to detect. too much whipsaw. AND you should buy against the trend.
    Turning point: normal time anticipate BUT with tiny confirmation. Crazy time wait for turning point, also with Tiny confirmation.
    Volume: crazy time high, normal time low
    overma: crazy time high, normal time low
    
    The mistake in my previous research was to seek and define bull and bear market. When in reality. One must first define crazy and normal time.


    PORTFOLIO:
    Crazy and normal time can both have 100% portfolio. You can not choose how market gives 60% return or 600%. You can only choose your portfolio size, buy or not buy. Don't miss even if market returns 20%. 
    => The final signal is craziness adjusted portfolio size. This means that crazy time and normal time signals CAN BE COMPARED AGAINST. They are both on scala -1 to 1 to make comparison consistent.
    => This also makes portfolio decisions easier. You can directly convert the confidence into portfolio size.


    DEBUG:
    level 0 = only show combined final result like r4:buy
    level 1 = level 0 + index based like r4:sh_buy
    level 2 = level 1 + different freq like r4:sh_close120.buy
    level 3 = level 2 + junk
    
    
    TODO
    MANUAL CORRECTION
    works well on sh index, buy not so well on cy and sz index because the trend is too strong and r3+4 have not found a good time to buy. Maybe the portfolio need to be adjustd using a trend strongess
    distinguish index vs d_preload
    which index is leading the market?
    which industry is leading the market?
    tweak accuracy
    This is ONLY the market prediction. You need to combine it with industry and individual stock prediction to get a better image.
    find a better and more reliable time classifier to replace the hard coded sh version
    manage efficiency
    update data flow to become smooth and in one run. send email out when finished
    maybe interface to see more clear the final result
    Find industry or stocks that are suited for macd. = > long trend, high volatility
    What is the opposite of MACD? A signal that buys low, sells high, in cycle mode.
    MACD works very well on CY stocks 
    """


    #0 START PREDICT
    print(f"START PREDICT ---> {index} <---")
    print()

    df_result[f"r:buy_sell"]=0.0
    df_result[f"ra:buy_sell"]=0.0

    # 0 MACD  (on single index) CRAZY
    step0(df_result=df_result, index=index,debug=debug)

    # 3 VOLUME (on single index) NORMAL
    step3(df_result=df_result, index=index,debug=debug)

    # 4 OVERMA (on all stocks) NORMAL
    step4(df_result=df_result, index=index, d_preload=d_preload,debug=debug)

    # 5 TOP N Stock (on some stocks)
    #step5(df_result=df_result, d_preload=d_preload,debug=debug)

    # 6 TOP INDUSTRY
    #step6(df_result=df_result)

    # 8 SEASONAL
    #step8(df_result=df_result)

    # 10 RSI
    #step10(df_result=df_result,debug=debug)

    # 13 combine NORMAL TIME buy and sell signal into one.
    normaltime_signal(df_result)

    #Add CRAZY TIME signal into the normal time signal = > all time signal.
    alltime_signal(df_result)

    #smooth the result to have less whipsaw
    #df_result["ra:buy_sell"]=Alpha.zlema(df=df_result, abase="ra:buy_sell", freq=5, inplace=False ,gain=0)

    #portfolio strategies
    port_strat2(df_result=df_result)

    # check only after year 2000
    df_result=LB.trade_date_to_calender(df=df_result,add=["year"])
    df_result=df_result[df_result["year"] >= 2000]
    del df_result["year"]

    # Save excel
    a_path = LB.a_path(f"Market/CN/PredictMarket/Predict_{index}")
    LB.to_csv_feather(df=df_result,a_path=a_path)



def port_strat1(df_result):
    # normal time portfolio change can be seen in a longer term
    # crazy time, portfolio change must be done very frequently

    #One interpretation:
    # value from 0 to (-1). If you have portfolio, sell.
    # value from 0 to 1.If you have no portfolio buy.

    #second interpretation:
    # signals shows remaining portfolio size from 0 to 100%
    # signals shows the net add or minus pct of the portfolio

    #third interpretation:
    # use kelly formula.
    # -1 means 100% lose , 0% win
    #  1 means 100% win  , 0% lose
    # calculate portfolio using this formular then
    # naive kelly would not work as the signal is not 100% accurate. might cause man whipsaw
    # hence : 0 means 50% win, 50% lose. Kelly would bet 0 portfolio. Hence all values under 0 are useless for this method.
    # But the signals range is -1 to 1, so kelly would not match the signal and lose half the information


    #fourth interpretation:
    # the signals shows that to do with remaining portfolio.
    # 0 means 50%, -1 means 0, 1 means 100%
    # problem: even during small bear market, we don't even want to have 20%. we want to have 0% portfolio.
    #

    df_result["ra:buy_sell_diff"]=df_result["ra:buy_sell"].diff(1)
    df_result["port1"]=df_result["ra:buy_sell_diff"].cumsum()
    df_result["port1"]=df_result["port1"].clip(0, 100)


def port_strat2(df_result):
    """
    This portfolio strategy is simple: buy when > 0.2. Sell when <0.2
    buy until sell signal occurs


    """
    df_result["port2"]=0.0
    portfolio=0.0

    for trade_date in df_result.index:
        #loop over each day
        signal = df_result.at[trade_date,"ra:buy_sell"]
        if signal>0:
            portfolio=builtins.max(portfolio,signal)
        elif signal<0:
            portfolio = 0.0 #reset portfolio to 0
        elif signal==0:
            # variation 1: no nothing and use previous high as guideline
            # variation 2: interpret it as sell signal if previous signal was buy. interpret as buy if previous signal was sell.
            # variation 3: use a low freq strategy to take a deeper look into it
            pass

        #assign value at end of day
        df_result.at[trade_date, "port2"] = portfolio




def timeclassifier():
    #many many ways to define crazy time
    #external factors
    #baidu search
    #margin account
    #us stock



    #internal factors
    #volume high
    #overma high
    #price very high
    #



    return

def single_test_deprecated_1():
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




def step0(df_result, debug=0, index="sh"):

    """MACD"""

    #create all macd
    a_results_col=[]
    for sfreq in [60,120,180,240]:
        for bfreq in [180,240,300,360,500]:
            if sfreq<bfreq:
                print(f"{index}: step0 sfreq{sfreq} bfreq{bfreq}")
                a_cols=macd(df=df_result, abase=f"close_{index}", freq=sfreq, freq2=bfreq, inplace=True, type=4, score=1)
                a_results_col+=[a_cols[0]]

                #delete unessesary columns such as macd dea, diff
                if debug < 2:
                    for counter in range(1,len(a_cols)):# start from 1 because 0 is reserved for result col
                        del df_result[a_cols[counter]]

    #add all macd results together
    df_result["r0:buy_sell"]=0.0
    for counter, result_col in enumerate(a_results_col):
        df_result["r0:buy_sell"]=df_result["r0:buy_sell"].add(df_result[result_col],fill_value=0)
        if debug <2:
            del df_result[result_col]

    #normalize
    df_result["r0:buy_sell"]=df_result["r0:buy_sell"]/ (counter+1)

    #calculate overlay freq volatility: adjust the result with volatility (because macd works best on high volatile time)
    #df_result["r0:buy_sell"] = df_result["r0:buy_sell"] * df_result["volatility"]



def step3(df_result, index="sh",debug=0, ):
    """volume

    volume is best used to predict start of crazy time. in normal time, there is not so much information in volume.
    """

    def step3_single(df_result, on_index, freq_close=240, freq_vol=360, debug=0):
        """
        This can detect 3 signals:
        1. high volume and high gain -> likely to reverse to bear
        2. low volume and high gain -> even more likely to reverse to bear
        3. high volume and low gain -> ikely to reverse to bull
        """

        vol_name = f"vol_{on_index}"
        close_name = f"close_{on_index}"
        result_name = f"r3:{on_index}_vol{freq_vol}_close{freq_close}"

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

    # loop over all frequency
    df_result[f"r3:buy"] = 0.0
    df_result[f"r3:sell"] = 0.0
    result_list = []
    counter = 0
    for freq_close in [240, 500]:
        for freq_vol in [120, 500]:
            print(f"{index}: step3 close{freq_close} vol{freq_vol}...")
            counter += 1
            buy_sell_label = step3_single(df_result=df_result, freq_close=freq_close, freq_vol=freq_vol, on_index=index, debug=debug)
            result_list = result_list + [buy_sell_label]

    # combine all frequecies into one result for ONE index
    for buy_freq_signal, sell_freq_signal in result_list:
        df_result[f"r3:buy"] = df_result[f"r3:buy"].add(df_result[buy_freq_signal], fill_value=0)
        df_result[f"r3:sell"] = df_result[f"r3:sell"].add(df_result[sell_freq_signal], fill_value=0)
        if debug<2:
            del df_result[buy_freq_signal]
            del df_result[sell_freq_signal]

    # normalize the result
    df_result[f"r3:buy"] = df_result[f"r3:buy"] / (counter * 2)
    df_result[f"r3:sell"] = df_result[f"r3:sell"] / (counter * 2)

    #combine buy and sell
    df_result["r3:buy_sell"] = df_result[f"r3:buy"].add(df_result[f"r3:sell"] * (-1), fill_value=0)

    if debug < 3:
        del df_result[f"r3:buy"]
        del df_result[f"r3:sell"]






def step4(df_result, d_preload,  index="sh", a_ts_code=[], debug=0):
    """Overma"""

    if index in ["sh","sz","cy"] and not a_ts_code:
        #generate matching list of ts_code for index to be used for overma later
        if index == "sh":
            a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="主板"))
        elif index == "sz":
            a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="中小板"))
        elif index == "cy":
            a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="创业板"))

        a_ts_code=list(a_ts_code.index)
        print(a_ts_code)

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
        if f"overma{freq_overma}" not in df_result.columns:
            df_result[f"overma{freq_overma}"] = 0.0
            df_result[f"counter{freq_overma}"] = 0.0

            for ts_code, df_asset in d_preload.items():
                if ts_code in a_ts_code:
                    # calculate if stocks is over its ma
                    df_asset[f"ma{freq_overma}"] = df_asset["close"].rolling(freq_overma).mean()
                    df_asset[f"overma{freq_overma}"] = (df_asset["close"] >= df_asset[f"ma{freq_overma}"]).astype(int)
                    df_asset[f"counter{freq_overma}"] = 1

                    df_result[f"overma{freq_overma}"] = df_result[f"overma{freq_overma}"].add(df_asset[f"overma{freq_overma}"], fill_value=0)
                    # counter to see how many stocks are available
                    df_result[f"counter{freq_overma}"] = df_result[f"counter{freq_overma}"].add(df_asset[f"counter{freq_overma}"], fill_value=0)

            # finally: calculate the percentage of stocks overma
            df_result[f"overma{freq_overma}"] = df_result[f"overma{freq_overma}"] / df_result[f"counter{freq_overma}"]

        # 1.2 normalize close series
        norm_close_name = Alpha.rollingnorm(df=df_result, freq=freq_close, abase=f"close_{index}", inplace=True)

        # 1.3 generate  Buy Signal: price < 0.25 and overma < 0.25
        df_helper = df_result.loc[(df_result[f"overma{freq_overma}"] < 0.25) & (df_result[norm_close_name] < 0.25)]
        df_result[f"r4:overma{freq_overma}_close{freq_close}_{index}_buy"] = (1 - df_helper[f"overma{freq_overma}"]) + (1 - df_helper[norm_close_name])  # the lower the price, the lower overma, the better

        # 1.4 generate  Sell Signal: price > 0.75 and overma > 0.75
        df_helper = df_result.loc[(df_result[f"overma{freq_overma}"] > 0.75) & (df_result[norm_close_name] > 0.75)]
        df_result[f"r4:overma{freq_overma}_close{freq_close}_{index}_sell"] = df_helper[f"overma{freq_overma}"] + df_helper[norm_close_name]  # the lower the price, the lower overma, the better

        # 1.5 delete unessary columns
        if debug < 3:
            del df_result[f"overma{freq_overma}"]
            del df_result[f"counter{freq_overma}"]
            del df_result[norm_close_name]

        return [f"r4:overma{freq_overma}_close{freq_close}_{index}_buy", f"r4:overma{freq_overma}_close{freq_close}_{index}_sell"]


    df_result[f"r4:buy"] = 0.0
    df_result[f"r4:sell"] = 0.0

    # loop over all frequency
    result_list = []
    counter = 0
    for freq_close in [240, 500]:
        for freq_overma in [120, 500]:
            print(f"{index}: step4 close{freq_close} overma{freq_overma}...")
            buy_sell_label = step4_single(df_result=df_result, d_preload=d_preload, a_ts_code=a_ts_code, freq_close=freq_close, freq_overma=freq_overma, index=index, debug=debug)
            result_list = result_list + [buy_sell_label]
            counter += 1

    # combine all frequecies into one result for ONE index
    for buy_signal, sell_signal in result_list:
        df_result[f"r4:buy"] = df_result[f"r4:buy"].add(df_result[buy_signal], fill_value=0)
        df_result[f"r4:sell"] = df_result[f"r4:sell"].add(df_result[sell_signal], fill_value=0)
        if debug<2:
            del df_result[buy_signal]
            del df_result[sell_signal]

    # normalize the result
    df_result[f"r4:buy"] = df_result[f"r4:buy"] / (counter * 2) # why times 2 actually
    df_result[f"r4:sell"] = df_result[f"r4:sell"] / (counter * 2)

    # combine buy and sell
    df_result["r4:buy_sell"] = df_result[f"r4:buy"].add(df_result[f"r4:sell"]*(-1),fill_value=0)

    #debug
    if debug<3:
        del df_result[f"r4:buy"]
        del df_result[f"r4:sell"]





def step5(df_result,d_preload, debug=0):

    """check if top n stocks (low beta stocks stocks) are doing well or not
    If even they are bad, then the whole stock market is just bad for sure

    algorith:
    1. define top n stocks using fundamentals and technicals
    2. check if they are doing well in last freq D: 5, 20, 60

    （1. cheat, use shortage to manually define these 50 stocks)
    """


    def step5_single(df_result, debug=0):
        # 2. Generate step 5 buy sell signal using custom defined rules
        #works worse than v2 with macd
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
        #df_result["r5:sell"] = 0.0

        # combine buy and sell
        df_result["r5:buy_sell"] = df_result[f"r5:buy"].add(df_result[f"r5:sell"] * (-1), fill_value=0)

        # adjust with volatility

        if debug < 2:
            del df_result["r5:buy"]
            del df_result["r5:sell"]
            #del df_result["r5:topn_index"]
        return


    def step5_single_v2(df_result, debug=0):
        # 2. Generate step 5 buy sell signal using macd. Because MACD buys on uptrend, sell on downtrend. goes very well with good stocks that are uptrend most of the time.

        # create all macd
        a_results_col = []
        for sfreq in [ 120, 180, 240]:
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

        #adjust with sh_index volatility
        df_result["r5:buy_sell"] = df_result["r5:buy_sell"] * df_result["volatility"]

        return



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

    step5_single_v2(df_result=df_result,debug=debug)



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

def step13():
    """
    two things  to be done with support and resistance:
    1. calculate actual support and resistance
    A: This can be looked on

    2. produce signals using minmax. IF current low is higher than last low and current high is higher than last high.
    A: Problem is that this method is slower than macd freq overlay. IT also does not support freq overlay that good. In addition, it produces more whipsaw. Also, baidu index data start from 2010, google trend does not work for CN stocks. In general. data is hard to get and same as index volume.
    A: We just use macd freq overlay instead of this.

    """


def step16():
    """
    using baidu, weibo, google trend data to predict.

    the problem :
    1. They are simultaneously occuring the stock market volume. So they don't predict in ADVANCE but at the same time. This makes them less useful as I can just watch the volume.
    2. They are sort of external signals. Some times high search volume does not mean bull but bear. And some times high search volume does not move the market at all.

    :return:
    """

def step17():
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

def predict_industry():
    return

def predict_stock():
    return



def all():

    #init before predict
    d_preload = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")
    df_result = DB.get_stock_market_overview()
    df_result["volatility"] = Alpha.timeclassifier(df=df_result, abase=f"close_sh", inplace=False)# use sh index to calculate volatility no matter what

    #index
    for index in ["sh","sz","cy"]:
        predict(df_result=df_result, index=index, d_preload=d_preload)

    #industry


    #individual stock (now thats complicated)


if __name__ == '__main__':
    all()





