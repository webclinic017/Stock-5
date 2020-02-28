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


def plot_autocorr():
    df = pd.read_csv("Stock_Market.csv")
    df = LB.get_numeric_df(df)
    for column in df.columns:
        print(column)
        plot_autocorrelation(df[column])


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


def test():
    df_date_all = DB.preload("asset", step=1)

    df_result = pd.DataFrame()
    for asset, df in df_date_all.items():
        print("asset", asset)
        df["tomorrow_pct_chg"] = df["pct_chg"].shift(-1)
        df["tomorrow_co_pct_chg"] = df["co_pct_chg"].shift(-1)
        df_result.at[asset, "tomorrow_pct_chg"] = df.loc[df["pjup"] == 1.0, "tomorrow_pct_chg"].mean()
        df_result.at[asset, "fgain2"] = df.loc[df["pjup"] == 1.0, "fgain2"].mean()
        df_result.at[asset, "tomorrow_co_pct_chg"] = df.loc[df["pjup"] == 1.0, "tomorrow_co_pct_chg"].mean()

    df_result.to_csv("asset pjup.csv")

    df_date_all = DB.preload("asset", step=1)


# this tells me how difficult my goal is to select the stocks > certain pct_chg every day
def daily_stocks_abve():
    df_asset = DB.preload("asset", step=2)
    df_result = pd.DataFrame()
    for ts_code, df in df_asset.items():
        # or in other words

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


def rsi_onrsi():
    df = DB.get_asset(ts_code="000002.SZ")
    df["rsi240"] = talib.RSI(df["close"], timeperiod=240)
    df["rsirsi240"] = talib.RSI(df["rsi240"], timeperiod=240)
    df["rsirsirsi240"] = talib.RSI(df["rsirsi240"], timeperiod=240)
    df = df[["rsirsirsi240", "rsirsi240", "rsi240", "close"]]
    df.reset_index(inplace=True, drop=True)
    df.plot(legend=True)
    plt.show()
    plt.close()


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


def plot_polynomials():
    df_asset = DB.get_asset(ts_code="000938.SZ")
    df_asset = df_asset.reset_index()
    window = 265
    step = 5
    for i in range(0, 6000, step):
        df = df_asset[i:i + window]
        trade_date = df_asset.at[i, "trade_date"]

        df["poly1"] = func(df=df, degree=1, column="close")
        df["poly2"] = func(df=df, degree=2, column="close")
        df["poly3"] = func(df=df, degree=3, column="close")
        df["poly4"] = func(df=df, degree=4, column="close")
        df["poly5"] = func(df=df, degree=5, column="close")
        df = df[["close", "poly1", "poly2", "poly3", "poly4", "poly5"]]
        df.reset_index(inplace=True, drop=True)
        newpath = f"Media/Plot/stock/000938.SZ/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(newpath + f"{trade_date}.jpg")
        df.plot(legend=True)
    plt.close()


def rsi_abv_under_test():
    df_ts_codes = DB.get_ts_code()
    df_result = pd.DataFrame()
    for index in df_ts_codes.index[::8]:
        df_asset = DB.get_asset(ts_code=index)
        df_asset = df_asset[df_asset["period"] > 240]
        if len(df_asset) < 240:
            continue

        df_asset["tomorrow"] = df_asset["open"].shift(-2) / df_asset["open"].shift(-1)
        # df_asset["52_high"]=df_asset["close"].rolling(52).max()
        # df_asset["52_low"]=df_asset["close"].rolling(52).min()

        print(index)

        all = [3, 5, 10, 20, 40, 60, 130, 260, 520, 770]
        a_small = [x for x in all][:-1]
        a_big = [x for x in all][1:]

        # calculate RSI
        dict_rsi_mean = {}
        for freq in all:
            df_asset[f"rsi{freq}"] = talib.RSI(df_asset["close"], timeperiod=freq)

            q1_low, q1_high = list(df_asset[f"rsi{freq}"].quantile([0.0, 0.3]))
            q2_low, q2_high = list(df_asset[f"rsi{freq}"].quantile([0.3, 0.7]))
            q3_low, q3_high = list(df_asset[f"rsi{freq}"].quantile([0.7, 1]))

            dict_rsi_mean[f"{freq}.1"] = [q1_low, q1_high]
            dict_rsi_mean[f"{freq}.2"] = [q2_low, q2_high]
            dict_rsi_mean[f"{freq}.3"] = [q3_low, q3_high]
            # top_low, top_high = list(df_asset[f"rsi{freq}"].quantile([0.75, 1]))
            # bot_low, bot_high = list(df_asset[f"rsi{freq}"].quantile([0, 0.35]))
            # dict_rsi_mean[f"rsi{freq}.gt"]=top_low
            # dict_rsi_mean[f"rsi{freq}.lt"]=bot_high

        # score
        for small in all:
            for big in all:
                if small < big:

                    for small_q in [1, 2, 3]:
                        for big_q in [1, 2, 3]:
                            small_q_low = dict_rsi_mean[f"{small}.{small_q}"][0]
                            small_q_high = dict_rsi_mean[f"{small}.{small_q}"][1]

                            high_q_low = dict_rsi_mean[f"{big}.{big_q}"][0]
                            high_q_high = dict_rsi_mean[f"{big}.{big_q}"][1]

                            df_test = df_asset[df_asset[f"rsi{small}"].between(small_q_low, small_q_high) & df_asset[f"rsi{big}"].between(high_q_low, high_q_high)]
                            df_result.at[index, f"{small}.{small_q}_{big}.{big_q}"] = df_test["tomorrow"].mean()

        # check if two lower rsi is above
        # for small in a_small:
        #     for big in a_big:
        #         if small >= big:
        #             continue
        #         else:
        #             df_asset[f"rsi{small}abv{big}"]= (df_asset[f"rsi{small}"]>df_asset[f"rsi{big}"]).astype(int)
        #
        #             #evaluate result
        #             for one_zero in [1,0]:
        #                 df_part=df_asset.loc[df_asset[f"rsi{small}abv{big}"]==one_zero, [f"rsi{small}abv{big}","tomorrow"]]
        #                 df_result.at[index, f"rsi{small}abv{big}_{one_zero}"]=df_part["tomorrow"].mean()
        # df_result.at[index, f"rsi{small}abv{big}_{one_zero}_pearson"]=df_part[f"rsi{small}abv{big}"].corr(df_part["tomorrow"])

    df_result.to_csv("price over ma3.csv")


def sim_no_bins_multiple():
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


# sim_no_bins_multiple()
# sim_pairwise()

df = DB.get_asset()


# period=120
# df["ma10"] = df["close"].rolling(period).mean()
# df["mama"],df["fama"] = talib.MAMA(df["close"],0.01,0.2)
#
# df.to_csv("test_fama_mama.csv")
# df = df[["close", "mama","fama","ma10"]]
# df.reset_index(inplace=True, drop=True)
# df.plot(legend=True)
# plt.show()
# plt.close()

def plot_fft():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.fftpack
    df = DB.get_asset(ts_code="300036.SZ")

    # Number of samplepoints
    N = 2000
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    y = df["close"].to_numpy()
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.show()


def transform(s):
    eval = (1 + s) / (1 - s)
    print("val", type(s), s, eval, )
    trans = 0.5 * math.log(eval)

    return trans


def transform_no_np(val):
    trans = 0.5 * math.log((1 + val) / (1 - val))
    print("trans is", trans)
    return trans


def fisher_transform(s, timeperiod=5):
    fisher = s.copy()
    fisher = fisher.rolling(timeperiod).apply(transform)

    return fisher


ts_code = "000002.SZ"
df = DB.get_asset(ts_code=ts_code)
df["rsi50"] = talib.RSI(df["close"], timeperiod=50)
df["fisher"] = df["rsi50"].apply(transform)
df = df[["close", "fisher"]]
df.reset_index(inplace=True, drop=True)

df.plot(legend=True)
plt.show()
plt.close()


def MESA(df):
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


def movingaverages_test():
    df_ts_code = DB.get_ts_code()[::1]
    func = talib.MIDPOINT  # WMA
    func_name = func.__name__
    df_result = pd.DataFrame()
    periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    for ts_code in df_ts_code.index:
        print("ts_code", ts_code)
        df_asset = DB.get_asset(ts_code=ts_code)

        try:
            df_asset = df_asset[(df_asset["period"] > 240) & (df_asset.index > 20000101)]
            if len(df_asset) < 500:
                continue
        except:
            continue

        # calculate stock geomean
        helper = 1 + (df_asset["pct_chg"] / 100)
        df_result.at[ts_code, "geomean"] = gmean(helper)
        df_result.at[ts_code, "period"] = len(df_asset)

        # calculate future gain
        dict_mean = {}
        dict_std = {}
        for target in periods:
            df_asset[f"tomorrow{target}"] = df_asset["open"].shift(-target) / df_asset["open"].shift(-1)
            dict_mean[target] = df_asset[f"tomorrow{target}"].mean()
            dict_std[target] = df_asset[f"tomorrow{target}"].std()

        for period in periods:
            df_asset[f"{func_name}.{period}"] = func(df_asset["close"], timeperiod=period)
            # df_asset[f"{func_name}.{period}"]=func(df_asset["high"],df_asset["low"], timeperiod=period)

        # calculate buy over or under 1 ma
        for period in periods:
            df_asset[f"{func_name}.{period}.abv"] = df_asset["close"] > df_asset[f"{func_name}.{period}"]

        for period in periods:
            for one_zero in [1, 0]:
                df_filtered = df_asset[df_asset[f"{func_name}.{period}.abv"] == one_zero]
                df_result.at[ts_code, f"{func_name}.{period}.abv{one_zero}_{period}mean"] = df_filtered[f"tomorrow{period}"].mean() / dict_mean[period]
                df_result.at[ts_code, f"{func_name}.{period}.abv{one_zero}_{period}std"] = df_filtered[f"tomorrow{period}"].std() / dict_std[period]

    label_mean = []
    label_std = []
    for period in periods:
        for one_zero in [1]:
            label_mean.append(f"{func_name}.{period}.abv{one_zero}_{period}mean")
            label_std.append(f"{func_name}.{period}.abv{one_zero}_{period}std")

    for period in periods:
        for one_zero in [0]:
            label_mean.append(f"{func_name}.{period}.abv{one_zero}_{period}mean")
            label_std.append(f"{func_name}.{period}.abv{one_zero}_{period}std")

    df_result = df_result[["period", "geomean"] + label_mean + label_std]
    DB.ts_code_series_to_excel(df_ts_code=df_result, path=f"ma_test/summary.{func_name}.xlsx", sort=[], asset=["E"], group_result=True)


#movingaverages_test()
#
# rsi_abv_under_test()
#
#
# columns=["close"]
# for i in  array:
#     for label in ["rsi","trend"]:
#         columns.append(f"close.{label}{i}")
#
# df.to_csv("test.csv")
# df=df[columns]
#
#
# for low in  array:
#     for high in array:
#         if high >= low:
#             continue
#
#         print(f"high{high}, low{low}")
#         df_copy=df.copy()
#         df_copy[f"close.rsi{low}"]=df_copy[f"close.rsi{low}"]*35
#         df_copy[f"close.rsi{high}"]=df_copy[f"close.rsi{high}"]*35
#         df_copy[f"close.trend{high}"]=df_copy[f"close.trend{high}"]*35
#         df_copy.reset_index(inplace=True, drop=True)
#         df_copy.plot(legend=True)
#         plt.show()
#         plt.close()
#
#
MESA(df)
