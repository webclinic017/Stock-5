import numpy as np
import cProfile
import LB
import time
import threading
import DB
# set global variable flag
from ICreate import *
import builtins as bi
import Plot
from scipy.stats import gmean
from scipy.stats.mstats import gmean
from scipy.stats import entropy
import sys

sys.setrecursionlimit(1000000)

array = [2, 5, 10, 20, 40, 60, 120, 240]


def trend(df: pd.DataFrame, ibase: str, thresh_log=-0.043, thresh_rest=0.7237, market_suffix: str = ""):
    a_all = [1] + array
    a_small = [str(x) for x in a_all][:-1]
    a_big = [str(x) for x in a_all][1:]

    rsi_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}rsi")
    phase_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}phase")
    rsi_abv = standard_indi_name(ibase=ibase, deri=f"{market_suffix}rsi_abv")
    turnpoint_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}turnpoint")
    under_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}under")
    trend_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}{IDeri.trend.value}")

    func = talib.RSI
    # RSI and CMO are the best. CMO is a modified RSI
    # RSI,CMO,MOM,ROC,ROCR100,TRIX

    # df[f"detrend{ibase}"] = signal.detrend(data=df[ibase])
    for i in a_all:  # RSI 1
        try:
            if i == 1:
                df[f"{rsi_name}{i}"] = (df[ibase].pct_change() > 0).astype(int)
                print(df[f"{rsi_name}{i}"].dtype)
                # df[ rsi_name + "1"] = 0
                # df.loc[(df["pct_chg"] > 0.0), rsi_name + "1"] = 1.0
            else:
                df[f"{rsi_name}{i}"] = func(df[ibase], timeperiod=i) / 100
                # df[f"{rsi_name}{i}"] = func(df[f"{rsi_name}{i}"], timeperiod=i) / 100

                # normalization causes error
                # df[f"{rsi_name}{i}"] = (df[f"{rsi_name}{i}"]-df[f"{rsi_name}{i}"].min())/ (df[f"{rsi_name}{i}"].max()-df[f"{rsi_name}{i}"].min())
        except Exception as e:  # if error happens here, then no need to continue
            print("error", e)
            df[trend_name] = np.nan
            return trend_name

    # Create Phase
    for i in [str(x) for x in a_all]:
        maximum = (thresh_log * math.log((int(i))) + thresh_rest)
        minimum = 1 - maximum

        high_low, high_high = list(df[f"{rsi_name}{i}"].quantile([0.70, 1]))
        low_low, low_high = list(df[f"{rsi_name}{i}"].quantile([0, 0.30]))

        # df[f"{phase_name}{i}"] = [1 if high_high > x > high_low else 0 if low_low < x < low_high else np.nan for x in df[f"{rsi_name}{i}"]]

    # rsi high abve low
    for freq_small, freq_big in zip(a_small, a_big):
        df[f"{rsi_abv}{freq_small}"] = (df[f"{rsi_name}{freq_small}"] > df[f"{rsi_name}{freq_big}"]).astype(int)

    # one loop to create trend from phase
    for freq_small, freq_big in zip(a_small, a_big):
        trendfreq_name = f"{trend_name}{freq_big}"  # freg_small is 2, freq_big is 240
        # the following reason is that none of any single indicator can capture all. one need a second dimension to create a better fuzzy logic

        # 2. find all points where freq_small is higher/over/abv freq_big
        df[f"{trendfreq_name}"] = np.nan
        df[f"{turnpoint_name}{freq_small}helper"] = df[f"{rsi_name}{freq_small}"] / df[f"{rsi_name}{freq_big}"]

        freq_small_top_low, freq_small_top_high = list(df[f"{rsi_name}{freq_small}"].quantile([0.97, 1]))
        freq_small_bot_low, freq_small_bot_high = list(df[f"{rsi_name}{freq_small}"].quantile([0, 0.03]))

        freq_big_top_low, freq_big_top_high = list(df[f"{rsi_name}{freq_big}"].quantile([0.96, 1]))
        freq_big_bot_low, freq_big_bot_high = list(df[f"{rsi_name}{freq_big}"].quantile([0, 0.04]))

        # if bottom big and small freq are at their alltime rare high
        # df.loc[(df[f"{rsi_name}{freq_big}"] > freq_big_top_low), f"{trendfreq_name}"] = 0
        # df.loc[(df[f"{rsi_name}{freq_big}"] < freq_big_bot_high), f"{trendfreq_name}"] = 1

        # df.loc[(df[f"{rsi_name}{freq_small}"] > freq_small_top_low), f"{trendfreq_name}"] = 0
        # df.loc[(df[f"{rsi_name}{freq_small}"] < freq_small_bot_high), f"{trendfreq_name}"] = 1

        # small over big
        df.loc[(df[f"{rsi_name}{freq_small}"] / df[f"{rsi_name}{freq_big}"]) > 1.01, f"{trendfreq_name}"] = 1
        df.loc[(df[f"{rsi_name}{freq_small}"] / df[f"{rsi_name}{freq_big}"]) < 0.99, f"{trendfreq_name}"] = 0

        # 2. find all points that have too big distant between rsi_freq_small and rsi_freq_big. They are the turning points
        # top_low, top_high = list(df[f"{turnpoint_name}{freq_small}helper"].quantile([0.98, 1]))
        # bot_low, bot_high = list(df[f"{turnpoint_name}{freq_small}helper"].quantile([0, 0.02]))
        # df.loc[ (df[f"{turnpoint_name}{freq_small}helper"]) < bot_high, f"{trendfreq_name}"] = 1
        # df.loc[ (df[f"{turnpoint_name}{freq_small}helper"]) > top_low, f"{trendfreq_name}"] = 0

        # 3. Create trend based on these two combined
        df[trendfreq_name] = df[trendfreq_name].fillna(method='ffill')
        # df.loc[(df[phase_name + freq_big] == 1) & (df[phase_name + freq_small] == 1), trendfreq_name] = 0
        # df.loc[(df[phase_name + freq_big] == 0) & (df[phase_name + freq_small] == 0), trendfreq_name] = 1

        # fill na based on the trigger points. bfill makes no sense here
        # df[trendfreq_name].fillna(method='ffill', inplace=True)
        # TODO MAYBE TREND can be used to score past day gains. Which then can be used to judge other indicators

    # remove RSI and phase Columns to make it cleaner
    a_remove = []
    for i in a_all:
        # a_remove.append(market_suffix + "rsi" + str(i))
        # a_remove.append(market_suffix + "phase" + str(i))
        pass
    LB.columns_remove(df, a_remove)

    # calculate final trend =weighted trend of previous TODO this need to be adjusted manually. But the weight has relative small impact
    return trend_name


def daily_stocks_abve():
    """this tells me how difficult my goal is to select the stocks > certain pct_chg every day
                get 1% everyday, 33% of stocks
                2% everyday 25% stocks
                3% everyda 19% stocks
                4% everyday 12% stocks
                5% everyday 7% stocks
                6% everyday 5%stocks
                7% everyday 3% stocks
                8% everday  2.5% Stocks
                9% everday  2% stocks
                10% everday, 1,5% Stocks  """

    df_asset = DB.preload("asset", step=2)
    df_result = pd.DataFrame()
    for ts_code, df in df_asset.items():
        print("ts_code", ts_code)
        # for pct in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        #     df_copy=df[ (100*(df["pct_chg_open"]-1) >  pct) ]
        #     df_result.at[ts_code,f"pct_chg_open > {pct} pct"]=len(df_copy)/len(df)
        #
        #     df_copy = df[(100 * (df["pct_chg_close"] - 1) > pct)]
        #     df_result.at[ts_code, f"pct_chg_close > {pct} pct"] = len(df_copy) / len(df)
        #
        #     df_copy = df[(((df["close"]/df["open"]) - 1)*100 > pct)] #trade
        #     df_result.at[ts_code, f"trade > {pct} pct"] = len(df_copy) / len(df)
        #
        #     df_copy = df[ ((df["co_pct_chg"]-1)*100 > pct)] #today open and yester day close
        #     df_result.at[ts_code, f"non trade > {pct} pct"] = len(df_copy) / len(df)
    df_result.to_csv("test.csv")


def polyfit(df, column, degree):
    s_index = df[column].index
    weights = np.polyfit(s_index, df[column], degree)
    data = pd.Series(index=s_index, data=0)
    for i, polynom in enumerate(weights):
        pdegree = degree - i
        data = data + (polynom * (s_index ** pdegree))
    df[f"{column}poly{degree}"] = data


def func(df, degree=1, column="close"):
    s_index = df[column].index
    y = df[column]
    weights = np.polyfit(s_index, y, degree)
    data = pd.Series(index=s_index, data=0)
    for i, polynom in enumerate(weights):
        pdegree = degree - i
        data = data + (polynom * (s_index ** pdegree))
    return data


def sim_no_bins_multiple():
    """the bins and no bins variation all conclude a inverse relationship. Maybe this is correct regarding law of big data"""

    def sim_no_bins_once(df_result, ts_code):
        # create target freq and target period
        for target in all_target:
            df_result[f"tomorrow{target}"] = df_result["open"].shift(-target) / df_result["open"].shift(-1)

        # pre calculate all rsi
        for column in all_column:
            for freq in all_freq:
                df_result[f"{column}.rsi{freq}"] = talib.RSI(df_result[column], timeperiod=freq)

        # go through each day and measure similarity
        for trade_date in df_result.index:
            if trade_date < start_date:
                continue

            # df_yesterday = df_result.loc[(df_result.index < trade_date)]
            trade_date_index_loc_minus_past = df_result.index.get_loc(trade_date) - 280
            date_lookback = df_result.index[trade_date_index_loc_minus_past]
            df_past_ref = df_result.loc[(df_result.index < date_lookback)]
            s_final_sim = pd.Series(data=1, index=df_past_ref.index)  # an array of days with the highest or lowest simlarity to today

            print(f"{ts_code} {trade_date}", f"reference until {date_lookback}")

            # check for each rsi column combination their absolute derivation
            for freq in all_freq:
                for column in all_column:
                    freq_today_value = df_result.at[trade_date, f"{column}.rsi{freq}"]

                    # IN yesterdays df you can see how similar one column.freq is to today
                    column_freq_sim = ((freq_today_value - df_past_ref[f"{column}.rsi{freq}"]).abs()) / 100
                    column_freq_sim = 1 + column_freq_sim
                    column_freq_sim = column_freq_sim.fillna(2)  # 1 being lowest, best. 2 or any other number higher 1 being highest, worst

                    column_freq_sim_weight = all_weight[f"{column}.{freq}"]
                    weighted_column_freq_sim = column_freq_sim ** column_freq_sim_weight
                    s_final_sim = s_final_sim.multiply(weighted_column_freq_sim, fill_value=1)

            # calculate a final similarity score (make the final score cross columns)
            # remove the last 240 items to prevent the algo know whats going on future
            nsmallest = s_final_sim.nsmallest(past_samples)
            # print(f"on trade_date {trade_date} the most similar days are: {list(nsmallest.index)}")
            df_similar = df_result.loc[nsmallest.index]
            df_result.at[trade_date, "similar"] = str(list(nsmallest.index))
            for target in all_target:
                df_result.at[trade_date, f"final.rsi.tomorrow{target}.score"] = df_similar[f"tomorrow{target}"].mean()

        LB.to_csv_feather(df_result, LB.a_path(f"sim_no_bins/result/similar_{ts_code}.{str(all_column)}"))
        return df_result

    # setting
    all_weight = {
        "close.2": 0.5,
        "close.5": 0.7,
        "close.10": 0.9,
        "close.20": 1,
        "close.40": 1.2,
        "close.60": 1.3,
        "close.120": 1.4,
        "close.240": 1.6,
        # "ivola.2": 0.0,
        # "ivola.5": 0.0,
        # "ivola.10": 0.3,
        # "ivola.20": 0.3,
        # "ivola.40": 0.3,
        # "ivola.60": 0.3,
        # "ivola.120": 0.3,
        # "ivola.240": 0.3,
    }

    all_column = ["close"]
    all_target = [2, 5, 10, 20, 40, 60, 120, 240]  # -60, -1 means future 60 days return to future 1 day, means in
    all_freq = [10, 20, 120, 240]  # 780, 20, 520， 2, 5,
    past_samples = 2
    start_date = 20050101

    df_summary = pd.DataFrame()
    df_ts_code = DB.get_ts_code()
    df_ts_code = df_ts_code[df_ts_code.index == "000001.SZ"]

    for ts_code in df_ts_code.index[::100]:
        df_result = DB.get_asset(ts_code=ts_code)
        df_result = df_result[df_result["period"] > 240]

        try:
            df_result = sim_no_bins_once(df_result, ts_code)
        except:
            continue

        for target in all_target:
            try:
                df_summary.at[ts_code, f"tomorrow{target}_pearson"] = df_result[f"tomorrow{target}"].corr(df_result[f"final.rsi.tomorrow{target}.score"])
            except:
                pass

    DB.ts_code_series_to_excel(df_ts_code=df_summary, path=f"sim_no_bins/summary.{str(all_column)}.xlsx", sort=[], asset=["E"], group_result=True)


def sim_bins():
    """the bins and no bins variation all conclude a inverse relationship. Maybe this is correct regarding law of big data"""
    df_ts_code = DB.get_ts_code()
    df_result_summary = pd.DataFrame()

    for ts_code in df_ts_code.index[::100]:

        df_result = DB.get_asset(ts_code)
        df_result = df_result[df_result["period"] > 240]
        df_result = df_result[df_result.index > 20000101]

        # setting
        all_column = ["close"]
        all_target = [2, 5, 10, 20, 40, 60, 120, 240]  # -60, -1 means future 60 days return to future 1 day, means in
        all_q = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        all_freq = [2, 5, 10, 20, 120, 260, 520]  # 780
        all_freq_comb = [(2, 120), (2, 260), (2, 520), (5, 120), (5, 260), (5, 520), (10, 120), (10, 260), (10, 520), (20, 120), (20, 260), (20, 520)]  # (2,780),(5,780),(10,780),(20,780),(120,260),(120,520),(120,780),(260,520),(260,780),(520,780)

        # create target freq and target period
        for target in all_target:
            df_result[f"tomorrow{target}"] = df_result["open"].shift(-target) / df_result["open"].shift(-1)

        # pre calculate all rsi
        for column in all_column:
            for freq in all_freq:
                # plain pct_chg pearson 0.01, plain tor pearson, 0.07, rsi dv_ttm is crap,ivola very good
                df_result[f"{column}.rsi{freq}"] = talib.RSI(df_result[column], timeperiod=freq)

        # for each day, check in which bin that rsi is
        for trade_date in df_result.index:
            df_today = df_result.loc[(df_result.index < trade_date)]
            if len(df_today) < 620:
                continue
            last_Day = df_today.index[-600]
            df_today = df_result.loc[(df_result.index < last_Day)]
            print(f"{ts_code} {trade_date}. last day is {last_Day}", )

            dict_accel = {}
            # create quantile
            for column in all_column:
                for freq in all_freq:
                    # divide all past values until today by bins/gategories/quantile using quantile:  45<50<55
                    a_q_result = list(df_today[f"{column}.rsi{freq}"].quantile(all_q))
                    for counter, item in enumerate(a_q_result):
                        df_result.at[trade_date, f"{column}.freq{freq}_q{counter}"] = item
                        dict_accel[f"{column}.freq{freq}_q{counter}"] = item

            # for each day check what category todays rsi belong
            for column in all_column:
                for freq in all_freq:
                    for counter in range(0, len(all_q) - 1):
                        under_limit = dict_accel[f"{column}.freq{freq}_q{counter}"]
                        above_limit = dict_accel[f"{column}.freq{freq}_q{counter + 1}"]
                        today_rsi_value = df_result.at[trade_date, f"{column}.rsi{freq}"]
                        if under_limit <= today_rsi_value <= above_limit:
                            df_result.at[trade_date, f"{column}.rsi{freq}_bin"] = int(counter)
                            break

            # calculate simulated value

            for column in all_column:
                dict_column_target_results = {key: [] for key in all_target}
                for small, big in all_freq_comb:
                    try:
                        small_bin = df_result.at[trade_date, f"{column}.rsi{small}_bin"]
                        big_bin = df_result.at[trade_date, f"{column}.rsi{big}_bin"]
                        df_filtered = df_today[(df_today[f"{column}.rsi{big}_bin"] == big_bin) & (df_today[f"{column}.rsi{small}_bin"] == small_bin)]
                        for target in dict_column_target_results.keys():
                            dict_column_target_results[target] = df_filtered[f"tomorrow{target}"].mean()
                    except Exception as e:
                        print("error", e)

                for target, a_estimates in dict_column_target_results.items():
                    df_result.at[trade_date, f"{column}.score{target}"] = pd.Series(data=a_estimates).mean()

        # create sume of all results
        for freq in all_freq:
            try:
                df_result[f"sum.score{freq}"] = bi.sum([df_result[f"{x}.score{freq}"] for x in all_column])
            except:
                pass

        # initialize setting result
        LB.to_csv_feather(df_result, LB.a_path(f"trade/result/trading_result_{ts_code}.{str(all_column)}"))

        for target in all_target:
            try:
                pearson = df_result[f"{column}.score{target}"].corr(df_result[f"tomorrow{target}"])
                df_result_summary.at[ts_code, f"pearson_{target}"] = pearson
            except:
                pass

    LB.to_csv_feather(df_result_summary, LB.a_path(f"trade/summary.{str(all_column)}"))


"""
useful functions to be transfered to icreate later start from here
"""


def normalize_vector(series, min=0, max=1):
    """Normalize vector"""
    series_min = series.min()
    series_max = series.max()
    new_series = (((max - min) * (series - series_min)) / (series_max - series_min)) + min
    return new_series


def normalize_apply(series, min=0, max=1):
    """Normalize apply"""
    return normalize_vector(series, min, max).iat[-1]


def FT(s, min=0, max=1):
    """Fisher Transform vector
    Mapps value to -inf to inf.
    Creates sharp distance between normal value and extrem values
    """
    # s=normalize_vector(s, min=min, max=max)
    expression = (1.000001 + s) / (1.000001 - s)
    result = 0.5 * np.log(expression)
    return result


def FT_APPYL(s):
    """Fisher Transform apply"""
    return FT(s).iat[-1]


def IFT(s, min=0, max=1):
    """Inverse Fisher Transform vector
    Mapps value to -1 to 1.
    Creates smooth distance between normal value and extrem values
    """

    # normalize_vector(result, min=min, max=max)
    exp = np.exp(s * 2)
    result = (exp - 1) / (exp + 1)
    return result


def IFT_Apply(s):
    """Inverse Fisher Transform Apply"""
    return IFT(s).iat[-1]


def trendslope_apply(s):
    """Trend apply = 1 degree polynomials"""
    index = range(0, len(s))
    normalized_s = normalize_vector(s, min=0, max=len(s))
    return LB.get_linear_regression_slope(index, normalized_s)


def MESA(df):
    """Using Sine, leadsine to find turning points"""
    ts_code = "000002.SZ"
    df = DB.get_asset(ts_code=ts_code)
    df = df[df.index > 20000101]
    dict_mean = {}
    for target in [5, 10, 20, 120, 240]:
        df[f"tomorrow{target}"] = df["open"].shift(-target) / df["open"].shift(-1)
        period_mean = df[f"tomorrow{target}"].mean()
        dict_mean[target] = period_mean
        print(f"ts_code {ts_code} {target} mean is {period_mean}")

    df["rsi5"] = talib.RSI(df["close"], timeperiod=5)
    df["rsi10"] = talib.RSI(df["close"], timeperiod=10)
    df["rsi20"] = talib.RSI(df["close"], timeperiod=20)
    df["rsi60"] = talib.RSI(df["close"], timeperiod=60)
    df["rsi120"] = talib.RSI(df["close"], timeperiod=120)
    df["rsi240"] = talib.RSI(df["close"], timeperiod=240)
    df["rsi480"] = talib.RSI(df["close"], timeperiod=480)
    # part 1 instantaneous trendline + Kalman Filter
    df["trend"] = talib.HT_TRENDLINE(df["close"])  # long and slow
    df["kalman"] = df["close"].rolling(5).mean()  # fast and short. for now use ma as kalman

    # df["dphase"]=talib.HT_DCPHASE(df["close"])# dominant phase
    df["dperiod"] = talib.HT_DCPERIOD(df["close"])  # dominant phase
    df["mode"] = talib.HT_TRENDMODE(df["close"])  # dominant phase
    df["inphase"], df["quadrature"] = talib.HT_PHASOR(df["close"])  # dominant phase

    df["sine"], df["leadsine"] = talib.HT_SINE(df["close"])
    # print("rsi 5,10")
    # df["sine"], df["leadsine"]= (talib.RSI(df["close"],timeperiod=5),talib.RSI(df["close"],timeperiod=10))
    # df.to_csv("mesa.csv")

    # trend mode general
    for mode in [1, 0]:
        df_filtered = df[df["mode"] == mode]
        for target in [5, 10, 20, 120, 240]:
            mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
            print(f"trend vs cycle mode. trend mode = {mode}. {target} mean {mean}")
    # conclusion. Trendmode earns more money than cycle mode. Trend contributes more to the overall stock gain than cycle mode.

    # uptrend price vs trend
    df_filtered = df[(df["mode"] == 1) & (df["close"] > df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"uptrend mode. {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["close"] < df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"downtrend mode. {target} mean {mean}")
    # conclusion uptrendmode is better than downtrendmode. Downtrend mode is better than Cycle mode which is strange

    # sine vs lead sine
    df_filtered = df[(df["mode"] == 1) & (df["sine"] > df["leadsine"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"sine  above lead {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["sine"] < df["leadsine"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"sine  under lead  {target} mean {mean}")
    # conclusion  uptrend (sine>leadsine) >  uptrend (close>trend) > uptrend

    # ma vs hilbert trendline
    df_filtered = df[(df["mode"] == 1) & (df["kalman"] > df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"kalman over trend {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["kalman"] < df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"kalman under trend {target} mean {mean}")

    df_filtered = df[(df["mode"] == 1) & (df["sine"] > df["leadsine"]) & (df["close"] > df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"trend mode. sine  above lead {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["sine"] < df["leadsine"]) & (df["close"] < df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / dict_mean[target]
        print(f"trend mode. sine  under lead  {target} mean {mean}")
    # conclusion additive: trend+sine>leadsine+close>trend

    df["marker"] = 0
    # df.loc[(df["mode"] == 1) & (df["sine"] > df["leadsine"])& (df["close"] > df["trend"]), "marker"]=10

    length = 2
    block_length = int(len(df) / length)
    a_max_index = []
    a_min_index = []
    for period in range(0, length):
        start = block_length * period
        end = (block_length * period) + block_length
        print("period is", start, end)
        df_step = df[start:end]
        print("last day", df_step.index[-1])
        that_period_max = df_step["rsi480"].max()
        that_period_min = df_step["rsi480"].min()
        max_index = df_step[df_step["rsi480"] == that_period_max].index[0]
        min_index = df_step[df_step["rsi480"] == that_period_min].index[0]
        a_max_index.append(max_index)
        a_min_index.append(min_index)
        df.at[max_index, "marker"] = 100
        df.at[min_index, "marker"] = 50

    # df=df[["close","trend","kalman","sine","leadsine","mode", "marker", "rsi60", "dperiod"]]#dphase quadrature   "dperiod", "inphase",
    df = df[["close", "trend", "kalman", "sine", "leadsine", "mode", "marker", "rsi240"]]  # dphase quadrature   "dperiod", "inphase",
    # df=df[["close", "rsi5","rsi10","marker"]]#dphase quadrature   "dperiod", "inphase",

    df.reset_index(inplace=True, drop=True)
    df.plot(legend=True)
    plt.show()
    plt.close()


def ent(data):
    import scipy.stats
    """Calculates entropy of the passed pd.Series"""
    p_data = data.value_counts()  # counts occurrence of each value
    return scipy.stats.entropy(p_data)  # get entropy from counts


def macd_tor(df, ibase, sfreq):
    name = f"{ibase}.vol.{sfreq}"

    df[f"ibase_{name}"] = zlema((df[ibase]), sfreq, 1.8)
    df[f"tor_{name}"] = zlema((df["turnover_rate"]), sfreq, 1.8)
    #
    # df[f"ibase_{name}"] = df[ibase] - highpass(df[ibase], int(sfreq))
    # df[f"tor_{name}"] = df[ibase] - highpass(df[ibase], int(sfreq))

    df[f"macd_tor_diff{name}"] = df[f"ibase_{name}"] - df[f"tor_{name}"]
    df[f"macd_tor_diff{name}"] = df[f"macd_tor_diff{name}"] * 1

    df[f"macd_tor_dea_{name}"] = df[f"macd_tor_diff{name}"] - supersmoother_3p(df[f"macd_tor_diff{name}"], int(sfreq / 2))
    df[f"macd_tor_dea_{name}"] = df[f"macd_tor_dea_{name}"] * 1

    df.loc[(df[f"macd_tor_dea_{name}"] > 0), f"macd_tor_{name}"] = 80
    df.loc[(df[f"macd_tor_dea_{name}"] <= 0), f"macd_tor_{name}"] = -80

    df[f"macd_tor_{name}"] = df[f"macd_tor_{name}"].fillna(method="ffill")

    return [f"macd_tor_{name}", f"macd_tor_diff{name}", f"macd_tor_dea_{name}"]


def custommacd(df, ibase, sfreq, bfreq, type=1, score=10):
    """ using ehlers zero lag EMA used as MACD cross over signal instead of conventional EMA
        on daily chart, useable freqs are 12*60, 24*60 ,5*60

        IMPORTNAT:
        Use a very responsive indicator for EMA! Currently, ema is the most responsive
        Use a very smoother indicator for smooth! Currently, Supersmoother 3p is the most smoothest.
        NOT other way around
        Laguerre filter is overall very good, but is neither best at any discipline. Hence using EMA+ supersmoother 3p is better.
        Butterwohle is also very smooth, but it produces 1 more curve at turning point. Hence super smoother is better for this job.


        Works good on high volatile time, works bad on flat times.
        TODO I need some indicator to have high volatility in flat time so I can use this to better identify trend with macd
        """
    name = f"{ibase}.{type, sfreq, bfreq}"
    df[f"zldif_{name}"] = 0
    df[f"zldea_{name}"] = 0
    if type == 0:  # standard macd with ma. conventional ma is very slow
        df[f"ema1_{name}"] = df[ibase].rolling(sfreq).mean()
        df[f"ema2_{name}"] = df[ibase].rolling(bfreq).mean()
        df[f"zldif_{name}"] = df[f"ema1_{name}"] - df[f"ema2_{name}"]
        df[f"zldiff_ss_big_{name}"] = supersmoother_3p(df[f"zldif_{name}"], int(sfreq/2))
        df[f"zldea_{name}"] = df[f"zldif_{name}"] - df[f"zldiff_ss_big_{name}"]
        df.loc[(df[f"zldea_{name}"] > 0), f"zlmacd_{name}"] = score
        df.loc[(df[f"zldea_{name}"] < 0), f"zlmacd_{name}"] = -score
    if type == 1:  # standard macd but with zlema
        df[f"ema1_{name}"] = zlema((df[ibase]), sfreq, 1.8)
        df[f"ema2_{name}"] = zlema((df[ibase]), bfreq, 1.8)
        df[f"zldif_{name}"] = df[f"ema1_{name}"] - df[f"ema2_{name}"]
        df[f"zldiff_ss_big_{name}"] = supersmoother_3p(df[f"zldif_{name}"], int(sfreq))
        df[f"zldea_{name}"] = df[f"zldif_{name}"] - df[f"zldiff_ss_big_{name}"]
        df.loc[(df[f"zldea_{name}"] > 0), f"zlmacd_{name}"] = score
        df.loc[(df[f"zldea_{name}"] < 0), f"zlmacd_{name}"] = -score
    elif type == 2:  # macd where price > ema1 and ema2
        df[f"ema1_{name}"] = zlema((df[ibase]), sfreq, 1.8)
        df[f"ema2_{name}"] = zlema((df[ibase]), bfreq, 1.8)
        df.loc[(df[ibase] > df[f"ema1_{name}"]) & (df[ibase] > df[f"ema2_{name}"]), f"zlmacd_{name}"] = score
        df.loc[(df[ibase] < df[f"ema1_{name}"]) & (df[ibase] < df[f"ema2_{name}"]), f"zlmacd_{name}"] = -score
    elif type == 3:  # standard macd : ema1 > ema2. better than type 2 because price is too volatile
        df[f"ema1_{name}"] = zlema((df[ibase]), sfreq, 1.8)
        df[f"ema2_{name}"] = zlema((df[ibase]), bfreq, 1.8)
        df.loc[(df[f"ema1_{name}"] > df[f"ema2_{name}"]), f"zlmacd_{name}"] = score
        df.loc[(df[f"ema1_{name}"] < df[f"ema2_{name}"]), f"zlmacd_{name}"] = -score
    elif type == 4:  # macd with lowpass constructed from highpass
        df[f"ema1_{name}"] = df[ibase] - highpass(df[ibase], int(sfreq))
        df[f"ema2_{name}"] = df[ibase] - highpass(df[ibase], int(bfreq))
        df.loc[(df[f"ema1_{name}"] > df[f"ema2_{name}"]), f"zlmacd_{name}"] = score
        df.loc[(df[f"ema1_{name}"] < df[f"ema2_{name}"]), f"zlmacd_{name}"] = -score

        # patch other
        # df.loc[(df[f"ema1_{name}"] < df[f"ema2_{name}"]) & (df[f"ema1_{name}"] > df[f"ma120"]) & (df[f"ema2_{name}"] > df["ma120"]) , f"zlmacd_{name}"] = score
        # df.loc[(df[f"ema1_{name}"] > df[f"ema2_{name}"]) & (df[f"ema1_{name}"] < df[f"ma120"]) & (df[f"ema2_{name}"] < df["ma120"]) , f"zlmacd_{name}"] = -score

    elif type == 5:  # price and tor
        df[f"ema1_{name}"] = zlema((df[ibase]), sfreq, 1.8)
        df[f"ema2_{name}"] = zlema((df["turnover_rate"]), sfreq, 1.8)
        df[f"zldif_{name}"] = df[f"ema1_{name}"] - df[f"ema2_{name}"]
        df[f"zldea_{name}"] = df[f"zldif_{name}"] - supersmoother_3p(df[f"zldif_{name}"], int(sfreq))
        df.loc[(df[f"zldea_{name}"] > 0), f"zlmacd_{name}"] = score
        df.loc[(df[f"zldea_{name}"] <= 0), f"zlmacd_{name}"] = -score
    df[f"zlmacd_{name}"] = df[f"zlmacd_{name}"].fillna(method="ffill")
    return [f"zlmacd_{name}", f"ema1_{name}", f"ema2_{name}", f"zldif_{name}", f"zldea_{name}"]


def slopecross(df, ibase, sfreq, bfreq, smfreq):
    """using 1 polynomial slope as trend cross over. It is much more stable and smoother than using ma
        slope is slower, less whipsaw than MACD, but since MACD is quicker, we can accept the whipsaw. In general usage better then slopecross
    """
    name = f"{ibase, sfreq, bfreq, smfreq}"
    df[f"slope1_{name}"] = df[ibase].rolling(sfreq).apply(trendslope_apply, raw=False)
    df[f"slope2_{name}"] = df[ibase].rolling(bfreq).apply(trendslope_apply, raw=False)

    # df[f"slopediff_{name}"] = df[f"slope1_{name}"] - df[f"slope2_{name}"]
    # df[f"slopedea_{name}"] = df[f"slopediff_{name}"] - my_best_ec(df[f"slopediff_{name}"], smfreq)

    df[f"slope_{name}"] = 0
    df.loc[(df[f"slope1_{name}"] > df[f"slope2_{name}"]) & (df[f"slope1_{name}"] > 0), f"slope_{name}"] = 10
    df.loc[(df[f"slope1_{name}"] <= df[f"slope2_{name}"]) & (df[f"slope1_{name}"] <= 0), f"slope_{name}"] = -10


def indicator_test():
    a_period = [60, 120][::-1]
    df_result = pd.DataFrame()
    df_ts_code = DB.get_ts_code()
    # df_ts_code = df_ts_code[df_ts_code.index == "600276.SH"]
    df_ts_code = df_ts_code[df_ts_code.index == "000002.SZ"]
    plot = True
    save = False
    for ts_code in df_ts_code.index[::1]:
        print("ts_code", ts_code)
        df = DB.get_asset(ts_code=ts_code)
        df = df[df["period"] > 240]
        df = df[df.index > 20000101].reset_index()
        # df = LB.timeseries_to_season(df).reset_index()
        # df = LB.timeseries_to_week(df).reset_index()
        # df = df[3000:]
        if len(df) < 500:
            continue

        for period in [2]:
            df[f"tomorrow{period}"] = df["open"].shift(-period) / df["open"].shift(-1)

        # window_min = 200
        # for i in range(0, len(df), 1):
        #     if i < window_min * 2:
        #         continue
        #     if i+ window_min> len(df)-1:
        #         continue
        #     start_date = df.index[0]
        #     end_date = df.index[i + window_min]
        #     df_past = df.loc[start_date:end_date]
        #
        #     print("today", i)
        #
        #     close_name=zlmacd(df_past, ibase="close", sfreq=12*days, bfreq=20*days, smfreq=4*days)
        #     today_signal=df_past.at[end_date,f"zlmacd_{close_name}"]
        #     df.at[end_date,f"zlmacd_{close_name}"]=today_signal
        #
        # df[ f"zlmacd_{close_name}"]=df[ f"zlmacd_{close_name}"].fillna(method="ffill")
        #

        # close_name = zlmacd(df, ibase="close", sfreq=12*days, bfreq=20*days, smfreq=4*days)
        # ivola_name = zlmacd(df, ibase="close", sfreq=12*days, bfreq=26*days, smfreq=12*days)
        # df["trade"] = 0
        # df.loc[(df[f"zlmacd_{close_name}"] == 10) & (df[f"zlmacd_{ivola_name}"] == 10), "trade"] = 10

        ibase = "close"

        freq = 240

        # mode
        a_ma = []
        a_rsi = []
        a_slope = []
        a_max = []
        a_min = []
        freqs = [5, 10, 20, 60, 120, 240]
        df["overall_rsi"] = 0
        for freq in freqs:
            df[f"ma{freq}"] = df[ibase].rolling(freq).mean()
            df[f"rsi{freq}"] = talib.RSI(df[ibase], freq)
            df[f"rsi{freq}"] = supersmoother_3p(df[f"rsi{freq}"], 240)
            df[f"rsi_bull{freq}"] = (df[f"rsi{freq}"] > 50).astype(int)
            df["overall_rsi"] = df[f"overall_rsi"] + df[f"rsi_bull{freq}"]
            # df[f"slope{freq}"]=df[ibase].rolling(freq).apply(trendslope_apply, raw=False)
            df[f"rolling_max{freq}"] = df[ibase].rolling(freq).max()
            df[f"rolling_min{freq}"] = df[ibase].rolling(freq).min()
            a_max.append(f"rolling_max{freq}")
            a_min.append(f"rolling_min{freq}")
            a_ma.append(f"ma{freq}")
            a_rsi.append(f"rsi{freq}")
            a_slope.append(f"slope{freq}")

        df["overall_rsi"] = (df["overall_rsi"] / len(freqs))

        # init detrend
        # df["detrend"]=  supersmoother_3p(df[ibase], int(freq/4))
        # df["detrend"]=  (df[ibase]- df["detrend"])/ df["detrend"]
        # df["detrend"]=IFT(df["detrend"],min= -1,max=1)
        # df["detrend_buy"]=0
        # df.loc[df["detrend"] > 0.05, "detrend_buy"] = 5
        # df.loc[df["detrend"] < -0.05, "detrend_buy"] = -5

        #
        df["highpass"] = highpass(df[ibase], int(freq * 1))
        df["de_highpass"] = highpass(df[ibase], int(freq * 1))
        df["de_highpass"] = df["de_highpass"] / df[ibase]
        df["de_highpass"] = df["de_highpass"] * 10
        # the result is a very responsive- rsi like oscilator

        rolling_period = 60

        # standard rolling mean variation
        df["expanding_de_highpass"] = df["pct_chg"].expanding(rolling_period).mean()
        # df["rolling_de_highpass"] = zlema(df["de_highpass"], int(rolling_period*2), 0)
        df["rolling_de_highpass"] = df["pct_chg"].rolling(rolling_period).mean()
        df["rolling_de_highpass"] = zlema(df["rolling_de_highpass"], int(rolling_period / 2), 0)
        df["expand_roll_diff_highpass"] = df["rolling_de_highpass"] - df["expanding_de_highpass"].shift(rolling_period)

        high_lastvalue = df.at[len(df) - 1, "expanding_de_highpass"]
        handcalculated = df["de_highpass"].mean()
        # variation using ema instead of m

        df.loc[(df["expand_roll_diff_highpass"] > 0), "r_gt_e_de_highpass"] = 30
        df.loc[(df["expand_roll_diff_highpass"] < 0), "r_gt_e_de_highpass"] = -30
        df["r_gt_e_de_highpass"] = df["r_gt_e_de_highpass"].fillna(method="ffill")

        general_gmean = df["tomorrow2"].mean()
        gmean_de_highpass = df.loc[df["r_gt_e_de_highpass"] == 30, "tomorrow2"].mean()
        print(f"general {general_gmean}. gmean de high pass", gmean_de_highpass)

        rolling_period = 240
        df["expanding_gmean"] = (df["tomorrow2"]).expanding(120).mean()
        df["rolling_gmean"] = (df["tomorrow2"]).rolling(rolling_period).mean()
        df["expand_roll_diff"] = df["rolling_gmean"] - df["expanding_gmean"].shift(rolling_period)

        df.loc[df["expand_roll_diff"] > 0, "r_gt_e_gmean"] = 30
        df.loc[df["expand_roll_diff"] < 0, "r_gt_e_gmean"] = -30

        # df["de_highpass"]=supersmoother_3p(df["de_highpass"],int(freq/4))
        #
        # df["rel_close240"]=df[ibase].rolling(240).apply(normalize_apply, raw=False)
        # df["rel_close20"]=df[ibase].rolling(20).apply(normalize_apply, raw=False)
        # df["rel_close20"]=FT(df["rel_close20"])
        # init=["detrend", "detrend_buy"]
        df["rsi"] = talib.RSI(df[ibase], freq)
        df["ema_of_rsi"] = zlema(df["rsi"], int(freq), 1.0)
        df["rsi2"] = df["rsi"] - df["ema_of_rsi"]
        init = ["de_highpass", "rolling_de_highpass", "overall_rsi", "expanding_de_highpass", "rolling_max120", "rolling_min120"]  # r_gt_e_de_highpass

        # lowpass/Smoother
        # df["zlema"] = zlema_best_gain(df[ibase], freq)
        # df["ss_2p"] = supersmoother_2p(df[ibase], freq)
        # df["butter_2p"] = butterworth_2p(df[ibase], freq)
        # df["butter_3p"] = butterworth_3p(df[ibase], freq)
        # df["my_inst"] = inst_trend(df[ibase],freq)
        # df["ehlers"] = ehlers_filter_unstable(df[ibase], freq)
        # df["mama"], df["fama"] = talib.MAMA(df[ibase])
        # stable_lowpass=["zlema","butter_3p","my_inst","butter_2p","ss_2p"]
        stable_lowpass = []

        # highpass/oscilator
        # df["highpass"] = highpass(df[ibase], freq)
        # df["roofing"] = roofing_filter(df[ibase], freq,  int(freq/2))
        # df["bandpass"], df["bandpass_lead"] = bandpass_filter_with_lead(df[ibase], freq)
        # df["bandpass_buy"]=0
        # df.loc[df["bandpass"] > 0, "bandpass_buy"] = 10
        # df.loc[df["bandpass"] < -0, "bandpass_buy"] = -10
        stable_bandpass = []

        # oscilator strategy
        # df["buy"] = 0
        # df["helper"] =df["roofing"]- df["highpass"]
        # df["helper"]=IFT(df["helper"],min=-1, max=1)
        # df.loc[df["helper"]> 0.5, "buy"]=10
        # df.loc[df["helper"]< -0.5, "buy"]=-10
        # stable_highpass = ["highpass", "buy"]
        stable_highpass = []

        # unstable period low pass
        # df["talib_inst"] = talib.HT_TRENDLINE(df["close"])
        # df["laguerre"] = laguerre_filter_unstable(df[ibase])
        unstable_lowpass = []

        # unstable period high pass
        unstable_highpass = []

        # unstable period bandpass pass
        unstable_bandpass = []

        # trend vs cycle
        # df["trend_mode"]=extract_trend(df[ibase],4)
        # macd_trend_mode=zlmacd(df,"trend_mode",freq, freq*2 ,freq)
        a_trend = ["trend_mode"]
        a_trend = []

        # 1. add volatility, 2nd use dominant period, maximum spectral, 3. Pattern recognition

        # oscilator

        # df["rvi"], df["rvi_sig"]=RVI(df["open"], df["close"],df["high"], df["low"], int(freq))
        # df["rvi"]=IFT(df["rvi"], min=-1, max=1)
        # df["slope"]=df[ibase].rolling(freq).apply(trendslope_apply, raw=False)
        # df["slope2"]=(df[ibase]*100).rolling(freq).apply(trendslope_apply, raw=False)
        # df["slope"]=df["slope"]*10
        # df["slope2"]=df["slope2"]*10
        # df["slope_mean"]=supersmoother_3p(df["slope"],freq)
        # df["slope_mean2"]=supersmoother_3p(df["slope2"],freq)
        #
        # df["cg"]=cg_Oscillator(df[ibase], int(freq/4))
        # df["cg"]=IFT(df["cg"],min=-1, max=1)
        #
        # df["cg_buy"] = 0
        # df.loc[df["cg"] > 0.5, "cg_buy"] = -10
        # df.loc[df["cg"] < -0.5, "cg_buy"] = 10
        #
        # df["leading"], df["net_lead"]=leading(df[ibase],freq)
        #
        # #df["laguerre_rsi"]=laguerre_RSI(df[ibase])
        # df["cc"]=cybercycle(df[ibase], int(freq/4))
        # df["talib_rsi1"]= normalize_vector(talib.RSI(df[ibase], timeperiod=int(freq/4)))
        # df["talib_rsi1"]= FT(df["talib_rsi1"],min=-1, max=1)
        #
        # df["talib_rsi2"] = normalize_vector(talib.RSI(df[ibase], timeperiod=int(freq / 2)))
        # df["talib_rsi2"] = FT(df["talib_rsi2"], min=-1, max=1)
        #
        # df["rsi_buy"]=0
        # df.loc[(df["talib_rsi1"]> 0.7) & (df["talib_rsi2"]> 0.7), "rsi_buy"]=-10
        # df.loc[(df["talib_rsi1"]< -0.7) & (df["talib_rsi2"]< -0.7), "rsi_buy"]=10

        ma = 240
        score_base = df[ibase].mean()
        final_score_at = (1 + 1.05 + 1.1 + 1.15 + 1.2) * score_base

        df[f"ma_buy{ma}"] = -10
        df.loc[df[ibase] > df[f"ma{ma}"], f"ma_buy{ma}"] = 10
        df["normalma"] = df[ibase].rolling(14).mean()
        df["zero"] = 0

        df["expand_max"] = df[ibase].expanding(freq).max()
        df["is_max"] = ((df[ibase].rolling(10).mean() / df["expand_max"]) > 0.85).astype(int)
        df["is_max"] = df["is_max"] * 1.25 * score_base
        custom_macd_name1, macd1_ema1, macd1_ema2, macd1_diff, macd1_dea = custommacd(df=df, ibase=ibase, sfreq=freq, bfreq=freq * 2, type=4, score=score_base * 1)
        # custom_macd_name5, macd5_ema1, macd5_ema2, macd5_diff, macd5_dea = custommacd(df=df, ibase=ibase, sfreq=freq, bfreq=freq * 2, type=5, score=score_base * 1.05)
        # custom_macd_name2= custommacd(df=df, ibase=ibase, sfreq=freq, bfreq=freq*2, type=2, score=score_base*1.05)
        # custom_macd_name3, macd3_ema1,macd3_ema2, macd3_diff,macd3_dea= custommacd(df=df, ibase=ibase, sfreq=freq, bfreq=freq*2, type=3, score=score_base*1.1)
        # custom_macd_name4= custommacd(df=df, ibase=ibase, sfreq=freq/2, bfreq=freq*2, type=4, score=score_base*1.15)
        # custom_macd_name5, macd5_ema1,macd5_ema2, macd5_diff,macd5_dea= custommacd(df=df, ibase=ibase, sfreq=freq, bfreq=freq*2, type=5, score=score_base*1.2)

        a_custom_macd_names = [custom_macd_name1]
        df["final_score"] = 0
        for name in a_custom_macd_names:
            df["final_score"] = df["final_score"] + df[name]

        df["price_abv_ma20"] = (df[macd1_ema1] > df[macd1_ema2]).astype(int)
        average_crossover_1 = LB.trend_swap(df, "r_gt_e_de_highpass", 30)
        average_crossover_0 = LB.trend_swap(df, "r_gt_e_de_highpass", -30)
        print("aveage", average_crossover_1, average_crossover_0)

        oscilator = a_custom_macd_names + ["zero", "ma120", "ma240", macd1_ema1, macd1_ema2]  # ,f"ma_buy{ma}"]

        # math
        # df["ATR1"]=talib.ATR(df["high"], df["low"], df["close"], timeperiod=freq)
        # df["ATR_pct"]=df["ATR1"]/ df["close"]
        # df["ATR_pct"]=IFT(df["ATR_pct"])*10
        # df["NATR"]=talib.NATR(df["high"], df["low"], df["close"], timeperiod=5)
        # df["TRANGE"]=talib.TRANGE(df["high"], df["low"], df["close"])
        # a_vola=["ATR1","NATR","TRANGE", "ATR_pct"]
        a_vola = []

        # trend vs cycle mode
        # df["d_period"]=talib.HT_DCPERIOD(df[ibase])
        # df["d_period_ss"]=supersmoother_2p(df["d_period"], freq)
        # #df["d_phase"]=talib.HT_DCPHASE(df[ibase])
        # df["sine"], df["leadsine"]=talib.HT_SINE(df[ibase])
        # #df["mode"]=talib.HT_TRENDMODE(df[ibase]) # mode is ueseless
        # df["sine_buy"]= (df["sine"]> df["leadsine"]).astype(int)
        # df["sine_buy"]=df["sine_buy"].diff()*5
        # df["adjust_ma"]=adjust_ma(df,ibase)
        a_hilbert = []

        # df_helper=df[ (df[dhp1].notna()) & (df[dhp2].notna())]
        #

        final_score_mean = df[df["r_gt_e_de_highpass"] == 30]
        final_score_mean = final_score_mean["tomorrow2"].mean()
        general_mean = df["tomorrow2"].mean()
        general_gmean = gmean(df["tomorrow2"].dropna()) - 1
        period = len(df)
        days_abv_ma5 = (df[ibase] > df[ibase].rolling(5).mean()).astype(int)
        days_abv_ma5 = days_abv_ma5.mean()
        days_abv_ma240 = (df[ibase] > df[ibase].rolling(240).mean()).astype(int)
        days_abv_ma240 = days_abv_ma240.mean()
        close_to_max = len(df[df["is_max"] == 70]) / period
        de_highpass_std = df["de_highpass"].std

        # ma_buy=gmean(df_helper.loc[df_helper[f"ma_buy{ma}"] == 10, f"tomorrow2"].dropna()) - 1
        # ma_notbuy=gmean(df_helper.loc[df_helper[f"ma_buy{ma}"] == -10, f"tomorrow2"].dropna()) - 1
        # dhp_buy = gmean(df_helper.loc[df_helper[dh_buy] == 10, f"tomorrow2"].dropna()) - 1
        # dhp_not_buy = gmean(df_helper.loc[df_helper[dh_buy] == -10, f"tomorrow2"].dropna()) - 1
        # volatility=df_helper["de_highpass"].dropna().mean() # the smaller the better
        # close_entropy=entropy(df[ibase].dropna())
        # de_highpass_entropy=entropy(df["de_highpass"].dropna())
        #
        df_result.at[ts_code, "period"] = period
        df_result.at[ts_code, "general_gmean"] = general_gmean
        df_result.at[ts_code, "general_mean"] = general_mean
        df_result.at[ts_code, "days_abv_ma5"] = days_abv_ma5
        df_result.at[ts_code, "days_abv_ma240"] = days_abv_ma240
        df_result.at[ts_code, "close_to_max"] = close_to_max
        df_result.at[ts_code, "final_score_mean"] = final_score_mean
        df_result.at[ts_code, "filtered_percent"] = final_score_mean / general_mean
        df_result.at[ts_code, "de_highpass_std"] = de_highpass_std

        #
        # print(f"gmean {general_gmean}. volatility{volatility}. dhp_buy {dhp_buy}. dhp_notbuy {dhp_not_buy}. ma_buy {ma_buy}. ma_notbuy {ma_notbuy}. percent {dhp_buy/general_gmean}")
        #

        # df[f"trade"] = 0
        # df.loc[(df[f"sine_{name}"] < -0.2) & (df[f"mom_{name}"] < -0.1) & (df[f"icc"] < -0.1), "trade"] = -10
        # df.loc[(df[f"sine_{name}"] >= 0.2) & (df[f"mom_{name}"] >= 0.1) & (df[f"icc"] >= 0.1), "trade"] = 10

        df_copy = df[["close"] + stable_lowpass + stable_highpass + stable_bandpass + unstable_lowpass + unstable_highpass + unstable_bandpass + oscilator + a_vola + a_hilbert + init + a_trend]

        if save:
            df.to_excel("instnace.xlsx")

        if plot:
            df_copy.plot(legend=True)
            plt.show()
            plt.close()

        # save to excel
        # df_copy = df[[f"close", "trade"]]
        # df_copy.plot(legend=True)
        # plt.show()
        # plt.savefig(f"divide/plot/bruteforce_adaptive_period{ts_code}.jpg")
        # df.to_excel(f"divide/xlsx/bruteforce_adaptive_period{ts_code}.xlsx")
        # plt.close()

    df_result["final_bullishness_rank"] = df_result["days_abv_ma5"].rank(ascending=False) + df_result["days_abv_ma240"].rank(ascending=False) + df_result["close_to_max"].rank(ascending=False) + df_result["general_gmean"].rank(ascending=False)
    DB.ts_code_series_to_excel(df_result, path="dhp_result.xlsx", sort=[])


def ema_re(s, n):
    """rekursive EMA. Should be same as talib and any other standard EMA"""
    a_result = []

    def my_ema_helper(s, n=n):
        if len(s) > n:
            k = 2 / (n + 1)
            last_day_ema = my_ema_helper(s[:-1], n)
            # result = last_day_ema + k * (s.iloc[-1] - last_day_ema) #tadoc formula
            result = k * s.iloc[-1] + (1 - k) * last_day_ema  # ehlers formula
            a_result.append(result)
            return result
        else:
            # first n values is sma
            result = s.mean()
            a_result.append(result)
            return result

    my_ema_helper(s, n)  # run through and calculate
    final_result = [np.nan] * (n - 1) + a_result  # fill first n values with nan
    return final_result


def ema(s, n):
    """Iterative EMA. Should be same as talib and any other standard EMA. Equals First order polynomial filter. Might be not as good as higher order polynomial filters
        1st order ema = alpha * f * z/(z-(1-alpha))
        2nd order ema = kf /(z**2 + az + b)
        3rd order ema = kf / (z**3 + az**2 + bz +c)
        Higher order give better filtering

        delay formula = N* p / np.pi**2 (N is order, P is cutoff period)
        The ema is basically a butterworth filter. It leas low frequencies untouched and filters high frequencies in not sharply, but smoothly
        The higher the degree in polynom, the more lag. This is the tradeoff
        Higher order give better filtering for the same amount of lag! basically should not use ema at all !!
        The higher the alpha, the more weight for recent value
        https://de.wikipedia.org/wiki/Butterworth-Filter
        EMA is a low pass filter. It lets low pass frequencies pass and rejects high frequencies

    """
    a_result = []
    alpha = 2 / (n + 1)
    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        elif i > n - 1:
            result1 = a_result[-1]
            result = alpha * s.iloc[i] + (1 - alpha) * result1  # ehlers formula
            a_result.append(result)
    return a_result


# CAT 1 Stable Period Low Pass Filter
def zlema_re(s, n, gain):
    """rekursive Zero LAG EMA. from john ehlers"""
    a_result = []

    def my_ec_helper(s, n, gain):
        if len(s) > n:
            k = 2 / (n + 1)
            last_day_ema = my_ec_helper(s[:-1], n, gain)
            today_close = s.iloc[-1]
            result = k * (today_close + gain * (today_close - last_day_ema)) + (1 - k) * last_day_ema  # ehlers formula
            a_result.append(result)
            return result
        else:
            result = s.mean()
            a_result.append(result)
            return result

    my_ec_helper(s, n, gain)  # run through and calculate
    final_result = [np.nan] * (n - 1) + a_result  # fill first n values with nan
    return final_result


def zlema(s, n, gain):
    """iterative Zero LAG EMA. from john ehlers"""
    a_result = []
    k = 2 / (n + 1)
    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        elif i > n - 1:
            result1 = a_result[-1]
            if np.isnan(result1):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                result = k * (close + gain * (close - result1)) + (1 - k) * result1  # ehlers formula
                a_result.append(result)
    return a_result


def zlema_best_gain(s, n, gain_limit=20):
    """to try all combinations to find the best gain for zlema"""
    least_error = 1000000
    best_gain = 0
    for value1 in range(-gain_limit, gain_limit, 2):
        gain = value1 / 10
        ec = zlema(s, n, gain)
        error = (s - ec).mean()
        if abs(error) < least_error:
            least_error = abs(error)
            best_gain = gain
    return zlema(s, n, best_gain)


def butterworth_2p(s, n):
    """2 pole iterative butterworth. from john ehlers
    butterworth and super smoother are very very similar
    butter_3p >  butter_2p = ss_2p > ss_3p
    """
    a_result = []
    a1 = np.exp(-1.414 * 3.1459 / n)
    b1 = 2 * a1 * math.cos(1.414 * np.radians(180) / n)  # when using 180 degree, the super smoother becomes a super high pass filter

    coeff2 = b1
    coeff3 = -a1 * a1
    coeff1 = (1 - b1 + a1 ** 2) / 4

    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1 or i == n:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = coeff1 * (close + 2 * close1 + close2) + coeff2 * result1 + coeff3 * result2  # ehlers formula
                a_result.append(result)
    return a_result


def butterworth_3p(s, n):
    """3 pole iterative butterworth. from john ehlers"""
    a_result = []
    a1 = np.exp(-3.14159 * 3.1459 / n)
    b1 = 2 * a1 * math.cos(1.738 * np.radians(180) / n)  # when using 180 degree, the super smoother becomes a super high pass filter
    c1 = a1 ** 2

    coeff2 = b1 + c1
    coeff3 = -(c1 + b1 * c1)
    coeff4 = c1 ** 2
    coeff1 = (1 - b1 + c1) * (1 - c1) / 8

    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1 or i == n:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            result3 = a_result[-3]
            if np.isnan(result1) or np.isnan(result2) or np.isnan(result3):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                close3 = s.iloc[i - 3]
                result = coeff1 * (close + 3 * close1 + 3 * close2 + close3) + coeff2 * result1 + coeff3 * result2 + coeff4 * result3  # ehlers formula
                a_result.append(result)
    return a_result


def supersmoother_2p(s, n):
    """2 pole iterative Super Smoother. from john ehlers"""
    a_result = []
    a1 = np.exp(-1.414 * 3.1459 / n)
    b1 = 2 * a1 * math.cos(1.414 * np.radians(180) / n)  # when using 180 degree, the super smoother becomes a super high pass filter
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1 or i == n:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        else:
            result1 = a_result[-1]
            resul2 = a_result[-2]
            if np.isnan(result1) or np.isnan(resul2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                result = c1 * (close + close1) / 2 + c2 * result1 + c3 * resul2  # ehlers formula
                a_result.append(result)
    return a_result


def supersmoother_3p(s, n):
    """3 pole iterative Super Smoother. from john ehlers
        lags more than 2p , is a little bit smoother. I think 2p is better
    """
    a_result = []
    a1 = np.exp(-3.1459 / n)
    b1 = 2 * a1 * math.cos(1.738 * np.radians(180) / n)  # when using 180 degree, the super smoother becomes a super high pass filter
    c1 = a1 ** 2

    coeff2 = b1 + c1
    coeff3 = -(c1 + b1 * c1)
    coeff4 = c1 ** 2
    coeff1 = 1 - coeff2 - coeff3 - coeff4

    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1 or i == n:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            result3 = a_result[-3]
            if np.isnan(result1) or np.isnan(result2) or np.isnan(result3):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                result = coeff1 * close + coeff2 * result1 + coeff3 * result2 + coeff4 * result3  # ehlers formula
                a_result.append(result)
    return a_result


def inst_trend(s, n):
    """http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/iTrend_Ind.html
    """
    a_result = []
    alpha = 2 / (n + 1)
    for i in range(0, len(s)):
        if i < n - 1:
            close = s.iloc[i]
            close1 = s.iloc[i - 1]
            close2 = s.iloc[i - 2]
            result = (close + 2 * close1 + close2) / 4
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = (alpha - (alpha / 2) ** 2) * close + (0.5 * alpha ** 2) * close1 - (alpha - (3 * alpha ** 2) / 4) * close2 + 2 * (1 - alpha) * result1 - (1 - alpha) ** 2 * result2
                a_result.append(result)
    return a_result


def gaussian_filter():
    pass


# CAT 4: UNSTABLE PERIOD FILTER
def laguerre_filter_unstable(s):
    """ Non linear laguere is a bit different than standard laguere
        http://www.mesasoftware.com/seminars/TradeStationWorld2005.pdf

        REALLY good, better than any other smoother. Is close to real price and very smooth
        currently best choice
    """
    s_price = s
    gamma = 0.8  # daming factor. The higher the more damping. If daming is 0, then it is a FIR

    # calculate L0
    L0 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L0[i] = 0
        else:
            L0[i] = (1 - gamma) * s_price.iloc[i] + gamma * L0[i - 1]

    # calcualte L1
    L1 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L1[i] = 0
        else:
            L1[i] = (- gamma) * L0[i] + L0[i - 1] + gamma * L1[i - 1]

    # calcualte L2
    L2 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L2[i] = 0
        else:
            L2[i] = (- gamma) * L1[i] + L1[i - 1] + gamma * L2[i - 1]

    # calcualte L2
    L3 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L3[i] = 0
        else:
            L3[i] = (- gamma) * L2[i] + L2[i - 1] + gamma * L3[i - 1]

    filter = (pd.Series(L0) + 2 * pd.Series(L1) + 2 * pd.Series(L2) + pd.Series(L3)) / 6

    return list(filter)


def ehlers_filter_unstable(s, n):
    """
    version1: http://www.mesasoftware.com/seminars/TradeStationWorld2005.pdf
    version2:  http://www.mesasoftware.com/papers/EhlersFilters.pdf

    Although it contains N, it is a non linear Filter

    two different implementation. Version1 is better because it is more general. version two uses 5 day as fixed date
    has unstable period. FIR filter

    I believe ehlers filter has unstable period because all outcomes for different freqs are the same
    The ehlers filter is WAAAAY to flexible and smooth, much much more flexile than lagguere or EMA, It almost seems like it is a 5 freq, and other are 60 freq. How come?
    """
    s_smooth = (s + 2 * s.shift(1) + 2 * s.shift(2) + s.shift(3)) / 6

    a_result = []
    for i in range(0, len(s)):
        if i < n:
            a_result.append(np.nan)
        else:
            smooth = s_smooth.iloc[i]
            smooth_n = s_smooth.iloc[i - n]

            a_coeff = []
            for count in range(0, n - 1):
                a_coeff.append(abs(smooth - smooth_n))

            num = 0
            sumcoeff = 0
            for count in range(0, n - 1):
                if not np.isnan(smooth):
                    num = num + a_coeff[count] * smooth
                    sumcoeff = sumcoeff + a_coeff[count]
            result = num / sumcoeff if sumcoeff != 0 else 0
            a_result.append(result)

    return a_result


# CAT 5: TREND MODE DECOMPOSITOR
def cybercycle(s, n):
    """Sometimes also called simple cycle. Is an oscilator. not quite sure what it does. maybe alpha between 0 and 1. the bigger the smoother
    https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf

    cybercycle gives very noisy signals compared to other oscilators. maybe I am using it wrong
    """
    s_price = s
    alpha = 0.2
    s_smooth = (s_price + 2 * s_price.shift(1) + 2 * s_price.shift(2) + s_price.shift(3)) / 6
    a_result = []

    for i in range(0, len(s_price)):
        if i < n + 1:
            result = (s_price.iloc[i] - 2 * s_price.iloc[i - 1] + s_price.iloc[i - 2]) / 4
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s_smooth[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                smooth = s_smooth.iloc[i]
                smooth1 = s_smooth.iloc[i - 1]
                smooth2 = s_smooth.iloc[i - 2]
                result = (1 - 0.5 * alpha) ** 2 * (smooth - 2 * smooth1 + smooth2) + 2 * (1 - alpha) * result1 - (1 - alpha) ** 2 * result2  # ehlers formula
                a_result.append(result)

    cycle = pd.Series(data=a_result)
    icycle = (cycle)  # according to formula. IFT should be used here, but then my amplitude is too small, so I left it away
    return list(icycle)


def extract_trend(s, n):
    """it is exactly same as bandpass filter except the last two lines """
    delta = 0.1
    beta = math.cos(np.radians(360) / n)
    gamma = 1 / math.cos(np.radians(720) * delta / n)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n or i == n + 1:
            result = s[0:i].mean()
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = 0.5 * (1 - alpha) * (close - close2) + beta * (1 + alpha) * result1 - alpha * result2
                a_result.append(result)

    trend = pd.Series(data=a_result)
    trend = trend.rolling(2 * n).mean()
    return list(trend)


def mode_decomposition(s, s_high, s_low, n):
    """https://www.mesasoftware.com/papers/EmpiricalModeDecomposition.pdf
    it is exactly same as bandpass filter except the last 10 lines
    I dont understand it really. Bandpass fitler is easier, more clear than this. Maybe I am just wrong
    """
    delta = 0.1
    beta = math.cos(np.radians(360) / n)
    gamma = 1 / math.cos(np.radians(720) * delta / n)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n or i == n + 1:
            result = s[0:i].mean()
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = 0.5 * (1 - alpha) * (close - close2) + beta * (1 + alpha) * result1 - alpha * result2
                a_result.append(result)

    trend = pd.Series(data=a_result)
    trend = trend.rolling(2 * n).mean()

    a_peak = list(s_high.shift(1))
    a_valley = list(s_low.shift(1))
    for i in range(0, len(s)):
        if a_result[i] == np.nan:
            pass
        else:
            result = a_result[i]
            result1 = a_result[i - 1]
            result2 = a_result[i - 2]
            if result1 > result and result1 > result2:
                a_peak[i] = result1

            if result1 < result and result1 < result2:
                a_valley[i] = result1

    avg_peak = pd.Series(a_peak).rolling(n).mean()
    avg_valley = pd.Series(a_valley).rolling(n).mean()

    return [list(trend), list(avg_peak), list(avg_valley)]


def cycle_measure(s, n):
    """http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/Cycle_Measure.html

    This is just too complicated to calculate, may contain too many errors
    """
    Imult = 0.365
    Qmult = 0.338

    a_inphase = []
    value3 = s - s.shift(n)

    # calculate inphase
    for i in range(0, len(s)):
        if i < n:
            a_inphase.append(np.nan)
        else:
            inphase3 = a_inphase[-3]
            if np.isnan(inphase3):
                inphase = s[0:1].mean()
                a_inphase.append(inphase)
            else:
                value3_2 = value3.iloc[i - 2]
                value3_4 = value3.iloc[i - 4]
                inphase = 1.25 * (value3_4 - Imult * value3_2) + Imult * inphase3
                a_inphase.append(inphase)

    a_quadrature = []
    # calculate quadrature
    for i in range(0, len(s)):
        if i < n:
            a_quadrature.append(np.nan)
        else:
            quadrature2 = a_quadrature[-2]
            if np.isnan(quadrature2):
                quadrature = s[0:1].mean()
                a_quadrature.append(quadrature)
            else:
                value3 = value3.iloc[i]
                value3_2 = value3.iloc[i - 2]
                quadrature = value3_2 - Qmult * value3 + Qmult * quadrature2
                a_quadrature.append(quadrature)


# CAT 6: OSCILATOR
# CAT 2: STABLE PERIOD HIGH PASS FILTER
def highpass(s, n):
    """high pass. from john ehlers. if frequency too short like n = 2, then it will produce overflow.
    basically you can use high pass filter as an RSI
    everything detrended is an oscilator
    """
    a_result = []
    alpha1 = (math.cos(np.radians(360) * 0.707 / n) + math.sin(np.radians(360) * 0.707 / n) - 1) / (math.cos(np.radians(360) * 0.707 / n))

    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n - 1 or i == n:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = (1 - alpha1 / 2) * (1 - alpha1 / 2) * (close - 2 * close1 + close2) + 2 * (1 - alpha1) * result1 - (1 - alpha1) * (1 - alpha1) * result2  # ehlers formula
                a_result.append(result)

    #first n highpass are always too high. make them none in order not to disturb corret values
    #a_result[:n*2] = [np.nan] * n*2
    return a_result


def roofing_filter(s, hp_n, ss_n):
    """  usually hp_n > ss_n. highpass period should be longer than supersmother period.

     1. apply high pass filter
     2. apply supersmoother (= low pass)
     """
    a_result = []
    s_hp = highpass(s, hp_n)

    a1 = np.exp(-1.414 * 3.1459 / ss_n)
    b1 = 2 * a1 * math.cos(1.414 * np.radians(180) / ss_n)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    for i in range(0, len(s)):
        if i < ss_n - 1:
            a_result.append(np.nan)
        elif i == ss_n - 1 or i == ss_n:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                hp = s_hp[i]
                hp1 = s_hp[i - 1]
                result = c1 * (hp + hp1) / 2 + c2 * result1 + c3 * result2  # ehlers formula
                a_result.append(result)
    return a_result


def bandpass_filter(s, n):
    """ http://www.mesasoftware.com/seminars/ColleaguesInTrading.pdf
        Can help MACD reduce whipsaw
        NOTE: ONLY works on sinoid charts like price, and not on RSI or oscillator
        = Detrended lowpass filter
        = oscilator

        => identifies cycle mode

        It is basically a detrended oscilator since removing lowpass is detrend
        or like a low pass but detrended

        standard usage:
        1. filter the high and low pass and only let middle band pass
        2. so only care about the midterm trend + midterm noise

        cool idea usage:
        1. when the current frequency is known (via hilbert or FFT) use bandpass filter to only target the known filter.
        2. calculate the derivation of the bandpass to see the future

        General note:
        remove low frequency = remove trend
        remove high frequency = remove noise
    """

    """http://www.mesasoftware.com/seminars/TrendModesAndCycleModes.pdf

    NOTE: This function returns 2 variables cycle and amplitude
    This might only be a cycle indicator on a bandpass filter
    So. it creates a bandpass filter???

    The higher the delta, the higher the amplitude
    """

    delta = 0.9
    beta = math.cos(np.radians(360) / n)
    gamma = 1 / math.cos(np.radians(720) * delta / n)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n or i == n + 1:
            result = s[0:i].mean()
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = 0.5 * (1 - alpha) * (close - close2) + beta * (1 + alpha) * result1 - alpha * result2
                a_result.append(result)

    return a_result


def bandpass_filter_with_lead(s, n):
    delta = 0.9
    beta = math.cos(np.radians(360) / n)
    gamma = 1 / math.cos(np.radians(720) * delta / n)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < n - 1:
            a_result.append(np.nan)
        elif i == n or i == n + 1:
            result = s[0:i].mean()
            a_result.append(result)
        else:
            result1 = a_result[-1]
            result2 = a_result[-2]
            if np.isnan(result1) or np.isnan(result2):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                close1 = s.iloc[i - 1]
                close2 = s.iloc[i - 2]
                result = 0.5 * (1 - alpha) * (close - close2) + beta * (1 + alpha) * result1 - alpha * result2
                a_result.append(result)

    s_result = pd.Series(a_result)
    lead = (n / 6.28318) * (s_result - s_result.shift(1))
    return [a_result, lead]


def laguerre_RSI(s):
    """
    http://www.mesasoftware.com/papers/TimeWarp.pdf
    Same as laguerre filter, laguerre RSI has unstable period
    It is too sensible and is useable for short term but not for longterm
    swings very sharply to both poles, produces a lot of signals
    """

    s_price = s
    gamma = 0.2

    # calculate L0
    L0 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L0[i] = 0
        else:
            L0[i] = (1 - gamma) * s_price.iloc[i] + gamma * L0[i - 1]

    # calcualte L1
    L1 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L1[i] = 0
        else:
            L1[i] = (- gamma) * L0[i] + L0[i - 1] + gamma * L1[i - 1]

    # calcualte L2
    L2 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L2[i] = 0
        else:
            L2[i] = (- gamma) * L1[i] + L1[i - 1] + gamma * L2[i - 1]

    # calcualte L2
    L3 = [0] * len(s)
    for i in range(0, len(s_price)):
        if i == 0:
            L3[i] = 0
        else:
            L3[i] = (- gamma) * L2[i] + L2[i - 1] + gamma * L3[i - 1]

    df_helper = pd.DataFrame()
    df_helper["L0"] = L0
    df_helper["L1"] = L1
    df_helper["L2"] = L2
    df_helper["L3"] = L3

    df_helper["CU"] = 0
    df_helper["CD"] = 0

    df_helper.loc[df_helper["L0"] >= df_helper["L1"], "CU"] = df_helper["L0"] - df_helper["L1"]
    df_helper.loc[df_helper["L0"] < df_helper["L1"], "CD"] = df_helper["L1"] - df_helper["L0"]

    df_helper.loc[df_helper["L1"] >= df_helper["L2"], "CU"] = df_helper["CU"] + df_helper["L1"] - df_helper["L2"]
    df_helper.loc[df_helper["L1"] < df_helper["L2"], "CD"] = df_helper["CD"] + df_helper["L2"] - df_helper["L1"]

    df_helper.loc[df_helper["L2"] >= df_helper["L3"], "CU"] = df_helper["CU"] + df_helper["L2"] - df_helper["L3"]
    df_helper.loc[df_helper["L2"] < df_helper["L3"], "CD"] = df_helper["CD"] + df_helper["L3"] - df_helper["L2"]

    RSI = df_helper["CU"] / (df_helper["CU"] + df_helper["CD"])

    return list(RSI)


def cg_Oscillator(s, n):
    """http://www.mesasoftware.com/papers/TheCGOscillator.pdf
    Center of gravity

    The CG oscilator is the only one that is FUCKING RELATIVE to the prise
    THIS MEANS you can apply FT and IF while others are not suitable for FT and IFT


    Similar to ehlers filter
    Should be better than conventional RSI
    """
    a_result = []
    for i in range(0, len(s)):
        if i < n:
            a_result.append(np.nan)
        else:
            num = 0
            denom = 0
            for count in range(0, n - 1):
                close = s.iloc[i - count]
                if not np.isnan(close):
                    num = num + (1 + count) * close
                    denom = denom + close
            result = -num / denom if denom != 0 else 0
            a_result.append(result)
    return a_result


def RVI(s_open, s_close, s_high, s_low, n):
    """
        http://www.stockspotter.com/Files/rvi.pdf
        Relative vigor index
        Price close higher in upmarket, close lower on down market
        RVI = (close - open) / (high - low)
        but basically this should be similar to another kind of indicator

        It is useable, but it is generally much slower than MACD and it can create much more noise than macd.
        Too many crossings
        Since MACD is using the convergence of two lines, which basically is a 1st order derivation. RVI is not any derivation at all.
        Either use a shorter time period e.g. freq/2


    """

    value1 = (s_close - s_open) + 2 * (s_close.shift(1) - s_open.shift(1)) + 2 * (s_close.shift(2) - s_open.shift(2)) + (s_close.shift(3) - s_open.shift(3)) / 6
    value2 = (s_high - s_low) + 2 * (s_high.shift(1) - s_low.shift(1)) + 2 * (s_high.shift(2) - s_low.shift(2)) + (s_high.shift(3) - s_low.shift(3)) / 6

    a_result = []
    for i in range(0, len(s_open)):
        if i < n:
            a_result.append(np.nan)
        else:
            num = 0
            denom = 0
            for count in range(0, n - 1):
                value1_helper = value1.iloc[i - count]
                value2_helper = value2.iloc[i - count]

                if not np.isnan(value1_helper):
                    num = num + value1_helper

                if not np.isnan(value2_helper):
                    denom = denom + value2_helper

            if denom != 0:
                result = num / denom
            else:
                result = 0
            a_result.append(result)

    RVI = pd.Series(a_result)
    RVISig = (RVI + 2 * RVI.shift(1) + 2 * RVI.shift(2) + RVI.shift(3)) / 6
    return [list(RVI), list(RVISig)]


def leading(s, n):
    """
        http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/Leading_Ind.html

        could be just a interpolation of the future price
        Looks just like a high pass of price,
        Not very useful when in cycle mode.

    """
    alpha1 = 0.25
    alpha2 = 0.33

    a_result = []
    for i in range(0, len(s)):
        if i < n:
            a_result.append(np.nan)
        else:
            result1 = a_result[-1]
            close = s.iloc[i]
            close1 = s.iloc[i - 1]
            if np.isnan(result1):
                result = s[0:i].mean()
                a_result.append(result)
            else:
                result = 2 * close + (alpha1 - 2) * close1 + (1 - alpha1) * result1
                a_result.append(result)

    a_netlead = []
    for i in range(0, len(s)):
        if i < n:
            a_netlead.append(np.nan)
        else:
            result1 = a_netlead[-1]
            lead = a_result[i - 1]
            if np.isnan(result1):
                result = s[0:i].mean()
                a_netlead.append(result)
            else:
                result = alpha2 * lead + (1 - alpha2) * result1
                a_netlead.append(result)

    return [a_result, a_netlead]


# UNFINISHED WORK
def linear_kalman_filter():
    """http://www.mesasoftware.com/seminars/PredictiveIndicators.pdf
    Guesses the error caused by phast values = difference of past_close and past kalman
    something like this
    kalman = alpha* today_close + (1-alpha)* yesterday_kalman + y* (today_close-yesterday_close)
       """


def nonlinear_kalman_filter():
    """http://www.mesasoftware.com/seminars/PredictiveIndicators.pdf
    basically = decompose close price using ema and close- ema. Since ema is low pass filter, close-ema is high pass filter.
    Then smooth the high pass filter(close-ema) and add it back to ema. The result is a price with smoothed high frequency only. Low frequency untouched.
    This is an interesting way of creating a zero lag filter: namely by smoothing the high frequency only.
       """


def pure_predictor():
    """http://www.mesasoftware.com/seminars/PredictiveIndicators.pdf
    1. use a low pass filter (3 degree polynomial)
    2. find low of that low pass filter
    3. period1 = 0.707 * dominant cycle
    4. period2 = 1.414 * dominant cycle ( this has 2x lag as the period1 )
    5. take difference of period1 and period2 (the result is in phase with cycle component of the price)
    6. Tjos os cycle component of the price. = Bandpass filter ???

    7. Take the momentum of the bandpass filter ( = diff today - yesterday)
    8. normalized by multiply momentum with p/(2*np.pi)
    9. multiply normalized momentum by 0.577 (=tan(30))

    This is basically lead sine
           """


def FIR_filter():
    """http://www.mesasoftware.com/seminars/TAOTN2002.pdf
    http://www.mesasoftware.com/papers/TimeWarp.pdf

    FIr=finite impluse response filter
    filt=  (price + 2*price1 +2*price2 +price3)/6
    fir filter is basically a smoother
    """


def zero_lag_filter():
    """http://www.mesasoftware.com/seminars/PredictiveIndicators.pdf
    calculation using phasor A- phasor B
    """


def predictfive_filter():
    """http://www.mesasoftware.com/seminars/PredictiveIndicators.pdf
        calculation using phasor A- phasor B
        """


def goertzel_filter():
    """http://www.mesasoftware.com/seminars/ColleaguesInTrading.pdf
    a form of fft where one tries each frequency individually
    """


def DFT_MUSIC():
    """http://stockspotter.com/Files/fouriertransforms.pdf
    a DFT with MUSIC algorithm to find the useful and dominant spectrum
    """


def swissarmy_helper():
    pass


def swissarmy(type, s_high, s_low, n):
    """http://www.mesasoftware.com/papers/SwissArmyKnifeIndicator.pdf
        Various indicators together in one place
    """
    delta = 0.1
    N = 0
    s_price = (s_high + s_low) / 2

    a_result = []
    if type == "EMA":

        for i in range(0, len(s_high)):
            if i < n:
                result = s_price.iloc[i]
                alpha = (math.cos(np.radians(360) / n) + math.sin((np.radians(360) / n)) - 1) / math.cos(np.radians(360) / n)
                b0 = alpha
                a1 = 1 - alpha


def kalman_filter():
    """

        From stack overflow, might be just wrong. Kalman filter is generally very noisy. Not smooth at all.
        Reasons why kalman filter is not the best for measure stock price:
        1. The assmumes the true value exist, Whereas stock price true value might always have a lot of noise, so no true value
        2. The kalman filter itself deviates a lot with a lot of error


        Parameters:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """

    def kalman(x, P, measurement, R, motion, Q, F, H):
        '''


        Parameters:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

        Return: the updated and predicted new values for (x, P)

        See also http://en.wikipedia.org/wiki/Kalman_filter

        This version of kalman can be applied to many different situations by
        appropriately defining F and H
        '''
        # UPDATE x, P based on measurement m
        # distance between measured and current position-belief
        y = np.matrix(measurement).T - H * x
        S = H * P * H.T + R  # residual convariance
        K = P * H.T * S.I  # Kalman gain
        x = x + K * y
        I = np.matrix(np.eye(F.shape[0]))  # identity matrix
        P = (I - K * H) * P

        # PREDICT x, P based on motion
        x = F * x + motion
        P = F * P * F.T + Q
        return x, P

    def kalman_xy(x, P, measurement, R, motion=np.matrix('0. 0. 0. 0.').T, Q=np.matrix(np.eye(4))):
        """
        Parameters:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """
        return kalman(x, P, measurement, R, motion, Q, F=np.matrix('''  1. 0. 1. 0.;
                                                                          0. 1. 0. 1.;
                                                                          0. 0. 1. 0.;
                                                                          0. 0. 0. 1.
                                                                          '''),
                      H=np.matrix('''
                                                                          1. 0. 0. 0.;
                                                                          0. 1. 0. 0.'''))

    x = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4)) * 1000  # initial uncertainty

    df = DB.get_asset()

    N = len(df["close"])
    observed_x = range(0, N)
    observed_y = df["close"]
    plt.plot(observed_x, observed_y)
    result = []
    R = 800 ** 2  # the bigger the noise, the more the lag . Otherwise too close to actual price, no need to filter
    for meas in zip(observed_x, observed_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    plt.plot(kalman_x, kalman_y)
    plt.show()


def adjust_ma(df, ibase):
    a_d_mean = []
    for index, row in df.iterrows():
        d_period = row["d_period"]
        d_period = int(d_period) if not np.isnan(d_period) else 0

        if d_period == 0:
            a_d_mean.append(np.nan)
        else:
            d_past = df[ibase].iloc[index - d_period:index]
            mean = supersmoother_3p(d_past, d_period)[-1]
            a_d_mean.append(mean)
            print(d_period)

    return a_d_mean


def find_peaks_array(s, n=60):
    from scipy.signal import argrelextrema

    # Generate a noisy AR(1) sample
    np.random.seed(0)
    xs = [0]
    for r in s:
        xs.append(xs[-1] * 0.9 + r)
    df = pd.DataFrame(xs, columns=[s.name])
    # Find local peaks
    df[f'bot{n}'] = df.iloc[argrelextrema(df[s.name].values, np.less_equal, order=n)[0]][s.name]
    df[f'peak{n}'] = df.iloc[argrelextrema(df[s.name].values, np.greater_equal, order=n)[0]][s.name]

    df[f"bot{n}"].update(df[f"peak{n}"].notna())

    return df[f"bot{n}"]


def find_peaks(df, ibase, a_n=[60]):
    """
    :param s: pd.series
    :param n: how many n should observe before and after to check if it is peak
    :return:

    Strengh of resistance support are defined by:
    1. how long it remains a resistance or support (remains good for n = 20,60,120,240?)
    2. How often the price can not break through it. (occurence)

    """
    from scipy.signal import argrelextrema

    # Generate a noisy AR(1) sample

    a_bot_name = []
    a_peak_name = []

    np.random.seed(0)
    s = df[ibase]
    xs = [0]
    for r in s:
        xs.append(xs[-1] * 0.9 + r)
    df = pd.DataFrame(xs, columns=[s.name])
    for n in a_n:
        # Find local peaks
        df[f'bot{n}'] = df.iloc[argrelextrema(df[s.name].values, np.less_equal, order=n)[0]][s.name]
        df[f'peak{n}'] = df.iloc[argrelextrema(df[s.name].values, np.greater_equal, order=n)[0]][s.name]
        a_bot_name.append(f'bot{n}')
        a_peak_name.append(f'peak{n}')

    # checks strenght of a maximum and miminum
    df["support_strengh"] = df[a_bot_name].count(axis=1)
    df["resistance_strengh"] = df[a_peak_name].count(axis=1)

    # puts all maxima and minimaxs togehter
    dict_all_rs = {}
    for n in a_n:
        dict_value_index_pairs = df.loc[df[f'bot{n}'].notna(), f'bot{n}'].to_dict()
        dict_all_rs.update(dict_value_index_pairs)

        dict_value_index_pairs = df.loc[df[f'peak{n}'].notna(), f'peak{n}'].to_dict()
        dict_all_rs.update(dict_value_index_pairs)

    dict_final_rs = {}
    for index_1, value_1 in dict_all_rs.items():
        keep = True
        for index_2, value_2 in dict_final_rs.items():
            closeness = value_2 / value_1
            if 0.95 < closeness < 1.05:
                keep = False
        if keep:
            dict_final_rs[index_1] = value_1

    # count how many rs we have. How many support is under price, how many support is over price
    df["total_support_resistance"] = 0
    df["abv_support"] = 0
    df["und_resistance"] = 0
    a_rs_names = []
    for counter, (resistance_index, resistance_val) in enumerate(dict_final_rs.items()):
        print("unique", resistance_index, resistance_val)
        df.loc[df.index >= resistance_index, f"rs{counter}"] = resistance_val
        a_rs_names.append(f"rs{counter}")
        if counter == 23:
            df.loc[(df["close"] / df[f"rs{counter}"]).between(0.98, 1.02), f"rs{counter}_challenge"] = 100
            df[f"rs{counter}_challenge"] = df[f"rs{counter}_challenge"].fillna(0)

        df["total_support_resistance"] = df["total_support_resistance"] + (df[f"rs{counter}"].notna()).astype(int)
        df["abv_support"] = df["abv_support"] + (df["close"] > df[f"rs{counter}"]).astype(int)
        df["und_resistance"] = df["und_resistance"] + (df["close"] < df[f"rs{counter}"]).astype(int)

    a_trend_name = []
    for index1, index2 in LB.custom_pairwise_overlap([*dict_final_rs]):
        print(f"pair {index1, index2}")
        value1 = dict_final_rs[index1]
        value2 = dict_final_rs[index2]
        df.loc[df.index == index1, f"{value1, value2}"] = value1
        df.loc[df.index == index2, f"{value1, value2}"] = value2
        df[f"{value1, value2}"] = df[f"{value1, value2}"].interpolate()
        a_trend_name.append(f"{value1, value2}")

    # Create trend support resistance lines using two max and two mins

    df.to_csv("test.csv")

    # Plot results
    # plt.scatter(df.index, df['min'], c='r')
    # plt.scatter(df.index, df['max'], c='g')
    df[["close"] + a_rs_names + a_trend_name + ["total_support_resistance", "abv_support", "und_resistance", "rs23_challenge"]].plot(legend=True)
    plt.show()


def find_flat(df, ibase):
    """
    go thorugh ALL possible indicators and COMBINE them together to an index that defines up, down trend or no trend.

    cast all indicator to 3 values: -1, 0, 1 for down trend, no trend, uptrend.
    :param df:
    :param ibase:
    :return:
    """
    a_freq = [240]
    df["ma20"] = df[ibase].rolling(20).mean()
    a_stable = []

    # unstable period
    # momentum
    df[f"bop"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])  # -1 to 1

    # volume
    df[f"ad"] = talib.AD(df["high"], df["low"], df["close"], df["vol"])
    df[f"obv"] = talib.OBV(df["close"], df["vol"])
    a_unstable = ["bop", "ad", "obv"]
    # stable period
    for freq in a_freq:
        df[f"rsi{freq}"] = talib.RSI(df[ibase], timeperiod=freq)

        """ idea: 
        1.average direction of past freq up must almost be same as average direction of past freq down. AND price should stay somewhat the same.
        2. count the peak and bot value of an OSCILATOR. IF last peak and bot are very close, then probably flat time
        """


        # MOMENTUM INDICATORS
        """ok 0 to 100, rarely over 60 https://www.fmlabs.com/reference/ADX.htm similar to dx"""
        df[f"adx{freq}"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=freq)
        df.loc[df[f"adx{freq}"] < 6, f"adx{freq}_trend"] = 0
        df.loc[df[f"adx{freq}"] > 6, f"adx{freq}_trend"] = 10

        """osci too hard difference between two ma, not normalized. need to normalize first https://www.fmlabs.com/reference/default.htm?url=PriceOscillator.htm"""
        df[f"apo{freq}"] = talib.APO(df["close"], freq, freq * 2)
        df[f"apo{freq}_trend"] = 10
        df.loc[(df[f"apo{freq}"].between(-1, 1)) & (df[f"apo{freq}"].shift(int(freq / 2)).between(-2, 2)), f"apo{freq}_trend"] = 0

        """osci too hard -100 to 100 https://www.fmlabs.com/reference/default.htm?url=AroonOscillator.htm"""
        df[f"aroonosc{freq}"] = talib.AROONOSC(df["high"], df["low"], freq)
        df[f"aroonosc{freq}_trend"] = 10
        df.loc[df[f"aroonosc{freq}"].between(-70, 70), f"aroonosc{freq}_trend"] = 0

        """osci too hard -100 to 100 https://www.fmlabs.com/reference/default.htm?url=CCI.htm"""
        df[f"cci{freq}"] = talib.CCI(df["high"], df["low"], df["close"], freq)  # basically modified rsi
        df[f"cci{freq}_trend"] = 10
        df.loc[df[f"cci{freq}"].between(-70, 70), f"cci{freq}_trend"] = 0

        """osci too hard 0 to 100 but rarely over 60 https://www.fmlabs.com/reference/default.htm?url=CMO.htm"""
        df[f"cmo{freq}"] = talib.CMO(df["close"], freq)  # -100 to 100
        df[f"cmo{freq}_trend"] = 10
        df.loc[df[f"cmo{freq}"].between(-70, 70), f"cmo{freq}_trend"] = 0

        """osci too hard 0 to 100 https://www.fmlabs.com/reference/default.htm?url=DX.htm"""
        df[f"dx{freq}"] = talib.DX(df["high"], df["low"], df["close"], timeperiod=freq)
        df[f"dx{freq}_trend"] = 10
        df.loc[df[f"dx{freq}"].between(-70, 70), f"dx{freq}_trend"] = 0

        """0 to 100 https://www.fmlabs.com/reference/default.htm?url=MFI.htm"""
        df[f"mfi{freq}"] = talib.MFI(df["high"], df["low"], df["close"], df["vol"], timeperiod=freq)
        df[f"mfi{freq}_trend"] = 10
        df.loc[df[f"mfi{freq}"].between(-70, 70), f"mfi{freq}_trend"] = 0

        """https://www.fmlabs.com/reference/default.htm?url=DI.htm"""
        df[f"minus_di{freq}"] = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=freq)
        df[f"plus_di{freq}"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=freq)

        """no doc"""
        df[f"minus_dm{freq}"] = talib.MINUS_DM(df["high"], df["low"], timeperiod=freq)
        df[f"plus_dm{freq}"] = talib.PLUS_DM(df["high"], df["low"], timeperiod=freq)

        """abs value, not relative http://www.fmlabs.com/reference/Momentum.htm"""
        df[f"mom{freq}"] = talib.MOM(df["close"], timeperiod=freq)

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"ppo{freq}"] = talib.PPO(df["close"], freq, freq * 2)
        df["test"] = find_peaks_array(df[f"ppo{freq}"], freq)
        df["test"] = df["test"].fillna(method="ffill")
        print(df["test"].notna())


        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"roc{freq}"] = talib.ROC(df["close"], freq)

        """0 to 100 rsi"""
        df[f"rsi{freq}"] = talib.RSI(df["close"], freq)
        df[f"rsi{freq}"] = supersmoother_3p(df[f"rsi{freq}"], int(freq / 4))
        df[f"rsi{freq}_diff"] = df[f"rsi{freq}"].diff()

        df[f"rsi{freq}_trend"] = 10
        df.loc[(df[f"rsi{freq}"].between(-48, 52)) & (df[f"rsi{freq}_diff"].between(-0.3, 0.3)), f"rsi{freq}_trend"] = 0

        """ """
        df[f"stochrsi_fastk{freq}"], df[f"stochrsi_fastd{freq}"] = talib.STOCHRSI(df["close"], freq, int(freq / 2), int(freq / 3))

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"ultiosc{freq}"] = talib.ULTOSC(df["high"], df["low"], df["close"], int(freq / 2), freq, int(freq * 2))

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"willr{freq}"] = talib.WILLR(df["high"], df["low"], df["close"], freq)

        # VOLUME INDICATORS
        df[f"adosc{freq}"] = talib.ADOSC(df["high"], df["low"], df["close"], df["vol"], freq, freq * 3)

        # volatility indicators
        df[f"atr{freq}"] = talib.ATR(df["high"], df["low"], df["close"], freq)
        df[f"atr_helper{freq}"] = df[f"atr{freq}"].rolling(60).mean()

        df[f"rsi{freq}_true"] = (df[f"rsi{freq}"].between(45, 55)).astype(int)
        df[f"adx{freq}_true"] = (df[f"adx{freq}"] < 10).astype(int)
        df[f"atr{freq}_true"] = (df[f"atr{freq}"] < df[f"atr_helper{freq}"]).astype(int)

    df[f"flat{freq}"] = (df[f"rsi{freq}_true"] + df[f"adx{freq}_true"] + df[f"atr{freq}_true"]) * 10

    a_stable = a_stable + [f"adx{freq}", f"adx{freq}_trend"]

    df[["close"] + a_stable].plot(legend=True)
    plt.show()


# test if stocks that are currently at their best, will last for the next freq.
def hypothesis_test():
    """
    1. go through all date
    2. find the highest ranking stocks for that date
    3. check their future gain
    """

    df_trade_date = DB.get_trade_date()
    df_result = pd.DataFrame()
    for trade_date in df_trade_date.index:
        try:
            print("trade_date", trade_date)
            if trade_date < 20000101:
                continue

            df_date = DB.get_date(trade_date=trade_date)
            df_date_expanding = DB.get_date_expanding(trade_date=trade_date)
            df_date["final_rank"] = df_date_expanding["final_rank"]

            for key, df_quant in custom_quantile(df_date, "final_rank", p_setting=[0, 1]).items():
                # for key, df_quant in get_quantile(df_date, "final_rank",p_setting=[(0, 0.05),(0.05, 0.18), (0.18, 0.5), (0.5, 0.82), (0.82, 0.95),(0.95, 1)]).items():
                for freq in [2]:
                    df_result.at[trade_date, f"q_{key}.open.fgain{freq}_gmean"] = gmean(df_quant[f"open.fgain{freq}"])
                    df_result.at[trade_date, f"q_{key}.member"] = ",".join(list(df_quant.index))

        except Exception as e:
            print("exception lol", e)

    df_result.to_csv("hypo_test.csv")


def macd_for_all(a_freqs=[5, 10, 20, 40, 60, 120, 180, 240, 300, 500], type=4, preload="stocks"):

    if preload=="stocks":
        dict_preload=DB.preload(load="asset",step=1)
    elif preload=="groups":
        dict_preload=DB.preload_groups()

    for sfreq,bfreq in LB.custom_pairwise_combination(a_freqs,2):
        if sfreq<bfreq:
            path=f"Market/CN/Backtest_Single/macd_for_all_sfreq{sfreq}_bfreq{bfreq}_type{type}_{preload}.csv"
            if os.path.exists(path):
                continue

            df_result = pd.DataFrame()
            for ts_code, df_asset in dict_preload.items():
                print(f"{ts_code}, sfreq {sfreq}, bfreq {bfreq}")
                if len(df_asset)<2000:
                    continue

                macd_name=custommacd(df=df_asset,ibase="close",sfreq=sfreq,bfreq=bfreq,type=type,score=20)[0]

                df_asset["tomorrow1"]=df_asset["close"].shift(-1)/df_asset["close"]
                df_result.at[ts_code,"period"]=len(df_asset)
                df_result.at[ts_code,"gmean"]=gmean(df_asset["tomorrow1"].dropna())

                df_macd_buy=df_asset[df_asset[macd_name]==20]
                df_result.at[ts_code,"uptrend_gmean"]=gmean(df_macd_buy["tomorrow1"].dropna())

                df_macd_sell = df_asset[df_asset[macd_name] == -20]
                df_result.at[ts_code,"downtrend_gmean"]=gmean(df_macd_sell["tomorrow1"].dropna())

            df_result["up_better_mean"]=(df_result["uptrend_gmean"]>df_result["gmean"]).astype(int)
            df_result["up_better_sell"]=(df_result["uptrend_gmean"]>df_result["downtrend_gmean"]).astype(int)
            df_result.to_csv(path,encoding='utf-8_sig')

    #create summary for all
    df_summary=pd.DataFrame()
    for sfreq, bfreq in LB.custom_pairwise_combination(a_freqs, 2):
        if sfreq < bfreq:
            path=f"macd_for_all_sfreq{sfreq}_bfreq{bfreq}_type{type}_{preload}.csv"
            df_macd=pd.read_csv(path)
            df_summary.at[path,"gmean"]=df_macd["gmean"].mean()
            df_summary.at[path,"uptrend_gmean"]=df_macd["uptrend_gmean"].mean()
            df_summary.at[path,"downtrend_gmean"]=df_macd["downtrend_gmean"].mean()
            df_summary.at[path,"up_better_mean"]=df_macd["up_better_mean"].mean()
            df_summary.at[path,"up_better_sell"]=df_macd["up_better_sell"].mean()
    df_summary.to_excel(f"Market/CN/Backtest_Single/macd_for_all_summary_type{type}_{preload}.xlsx")



def macd_for_one(sfreq=240,bfreq=750,ts_code="000002.SZ",type=1,score=20):
    df_asset=DB.get_asset(ts_code=ts_code)
    print("ts_code",ts_code)
    macd_labels=custommacd(df=df_asset,ibase="close",sfreq=sfreq,bfreq=bfreq,type=type,score=score)
    df_asset=df_asset[macd_labels+["close"]]
    Plot.plot_chart(df_asset,df_asset.columns)
    #df_asset.to_csv(f"macd_for_one_{ts_code}.csv")

if __name__ == '__main__':


    macd_for_all(type=1, preload="stocks")
    macd_for_one(sfreq=5,bfreq=10,type=1, score=200,ts_code="600519.SH")
    #hypothesis_test()


    # find_peaks(df=df,ibase="close", a_n=[120])

    # find_flat(df, "close")

    # df_result=support_resistance_horizontal_expansive(df_asset=df)
    # df_result.to_csv("support.csv")

    # indicator_test()

"""

useful techniqes
1. MACD with EMA and SS
2. Bandpass filter (as oscilator) + macd (bandpass is always later than MACD) 
3. highpass as RSI 
4. A trendline on oscilator like bandpass seems to really good to indicate the next low or high. So use momentum to see the indicator if it is already deviating from price
5. close price to rsi but better: 1. better_rsi= (close - ss)/ss



IDEAS
0. See stock chart as wave
- Use methods from eletrical engineering
- use methods from quantum physic: superposition, heisenberg uncertainty. The smaller the days, the more random are the price jumps 

Use super position and heiserbergishe ungleichung on stocks


In uptrend: Buy until ATR drops
In downtrend: wait until ATR goes up and then buy. 


1.0 ATR is a pretty good volatility measure. When the price is at bottom. The True range is at highest because buyer and seller are very mixed. In an uptrend, The ATR should decrease
1.1 Most time dominant period is 15-40 which means this is the best period to do sell or buy. 

1. Average True Range to set threshhold
calculate average true range over past period
add a fraction of the average true range to nonlinear/ unstable period filters. 
by doing this, you allow the price to swing within the past true range. If it exceeds, then it is likely an special case. 

My thought:
ATR + STD = True Range

2. The general problem of all unstable period indicator is that they are too volatile and need a bigger frequency to compensate

2. Voting system
Use uncorrelated pairs: 
momentum: ma, instand_trend
Cycle:
oscilator: RSI

5:1 spread of the same indicator in different time can also be used 

3. How to combine two indicators?
The Fucking answer is Bertrands ballot theorem
Candicate A gets a votes and candicate B gets b votes. Then the probabily of A ahead of B is:
(a-b)/(a+b)

4. For example, an RSI theoretically performs best when the computation period is just one half of a cycle period

5. What JOHN Ehler did not incooperate:
volume, ANN, fundamentals, comparative best stock using same technique at one day, industry comparison


6. Against pattern recognition
1. past price has not influencen on future price
2. short term price is random

For pattern recognition
1. magnus carlsen 
2. Should work on long term



6. Expanding values are stocks maximum value at a given time. This is useful to predict the overall stock  

Expanding:
1. Entryopy
2. gmean
3. many other values


6. Relationships
the bigger the freq the more predictable, the more useless the data
The smoother your signal, the longer the lag (solveable by insta trendline, kalman filter)
the earlier you want to detect the turnpoint, the more sensitive the indicator needs to be, the more whiplas it creates. (Basically, you need to adjust the period based on probability for the next turning point signal)
the better the trend the easier to make money.
the more normalize, the morr sensitive, the more recent is the data,   《==》  the more whipsaw, the more noise.
In Up Trend move, buy and hold
In cycle mode buy and sell
in downtrend mode, Shortsell and hold
hyper overbought= long period + short period overbought
hyper udnerbought = long period + short period underbought

"""

"""

very significant:
1. good stock 年线 半年线 cross
2. good stock have no resistance at top
3. past period trend the less normal, the better
4. how often a stock breaks support and resistance
4. use different macd to eliminate whipsaws of each other
5. the mean RSI of a stock can also indicator if it is bullish or not
5. DONT buy MACD at the start!  BECAUSE it needs an up signal to confirm. So BUY MACD after price has dropped, then start buy
6. START ANTICIPATING THE TURNING POINT. otherwise signal has lag, and then + 1 day more lag because of delay



How to distinguish between turn of trend vs temporal signal (fir finite impulse response)
similarities:
1. both can have very high pass


difference:
1. at turnpoint, the volume would divergence
2. 



4. cases:
1. price incease, vol increase
2. price decrease, vol decrease
3. price increase, vol decrease
4. price decrease, vol increase


5. Check volume to see if a resistance can be broken or not

6. Bullishness measure. 1. gmean, 2. overma 3. how often it stays at 80% of history max price
MACD + ma method = better MACD?


7. The problem of finding a trend up or going down in ADVANCE is that high pass AND lowpass together form the price. 
Sometimes High pass leads
Sometimes low pass leads the trend
high pass vs a lot of low pass. =80% vs 20%. Who do you trust?



In an continous field, who to draw the line between buy and sell? 
Basically. How to create border in continous world
A: only relative comparison is useful. absolute values are not

1. Use mean. Mean price, mean std
2. Use quantile. 4 quantile
3. 


Strategy:
1. Anticipate turning point
1. Buy at (good) stock RSI extrem extrem low in a very long time and bet against the mean. e.g. at rsi 240 at -20 or top 10 most lowest case.


2. wait for turning point to confirm
2. 




Properties on uptrend


"""
