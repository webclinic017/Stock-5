import numpy as np
import cProfile
import LB
import time
import threading
import DB
# set global variable flag
from ICreate import *
import builtins as bi


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


def simulate_trade_basedon_rsi():
    df = DB.get_asset()
    df = df[df.index > 20030101]
    df["tomorrow"] = df["open"].shift(-2) / df["open"].shift(-1)
    start_date = 20160101
    df_trading_result = df.copy()
    df_trading_result = df_trading_result[df_trading_result.index > start_date]
    df_trading_result["score"] = 0.0
    df_past_rsi_table = pd.DataFrame()
    df_today_rsi = pd.DataFrame()
    all = [3, 5, 10, 20, 120, 260, 520]

    start_date = 20180101
    for trade_date in df.index[::2]:
        print("trade_date", trade_date)
        if trade_date < start_date:
            continue

        df_today = df[df.index <= trade_date]

        # calculate RSI
        dict_rsi_mean = {}
        for freq in all:
            df_today[f"rsi{freq}"] = talib.RSI(df_today["close"], timeperiod=freq)

            q1, q2, q3, q4, q5, q6, q7, q8, q9 = list(df_today[f"rsi{freq}"].quantile([0.0, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1]))
            dict_rsi_mean[f"{freq}.1"] = [q1, q2]
            dict_rsi_mean[f"{freq}.2"] = [q2, q3]
            dict_rsi_mean[f"{freq}.3"] = [q3, q4]
            dict_rsi_mean[f"{freq}.4"] = [q4, q5]
            dict_rsi_mean[f"{freq}.5"] = [q5, q6]
            dict_rsi_mean[f"{freq}.6"] = [q6, q7]
            dict_rsi_mean[f"{freq}.7"] = [q7, q8]
            dict_rsi_mean[f"{freq}.8"] = [q8, q9]

        # first update past rsi table
        for small in all:
            for big in all:

                if small < big:
                    for small_q in [1, 2, 3, 4, 5, 6, 7, 8]:
                        for big_q in [1, 2, 3, 4, 5, 6, 7, 8]:
                            small_q_low = dict_rsi_mean[f"{small}.{small_q}"][0]
                            small_q_high = dict_rsi_mean[f"{small}.{small_q}"][1]

                            high_q_low = dict_rsi_mean[f"{big}.{big_q}"][0]
                            high_q_high = dict_rsi_mean[f"{big}.{big_q}"][1]

                            df_test = df_today[df_today[f"rsi{small}"].between(small_q_low, small_q_high) & df_today[f"rsi{big}"].between(high_q_low, high_q_high)]
                            df_past_rsi_table.at[trade_date, f"{small}.{small_q}_{big}.{big_q}"] = (df_test["tomorrow"].mean() - 1) * 100

                            # df_past_rsi_table.at[trade_date, f"{small}.{small_q}_{big}.{big_q}"] = (df_today.loc[df_today[f"rsi{small}"].between(small_q_low, small_q_high) & df_today[f"rsi{big}"].between(high_q_low, high_q_high), "tomorrow"]-1)*100

        # check to which category todays rsi belongs to
        for freq in all:
            today_rsi_freq = df_today.at[trade_date, f"rsi{freq}"]
            for q in [1, 2, 3, 4, 5, 6, 7, 8]:
                if dict_rsi_mean[f"{freq}.{q}"][0] <= today_rsi_freq <= dict_rsi_mean[f"{freq}.{q}"][1]:
                    df_today_rsi.at[trade_date, f"rsi{freq}"] = q
                    break
            else:
                df_today_rsi.at[trade_date, f"rsi{freq}"] = 2  # assign middle rsi if no solution was found

        df_today_rsi = df_today_rsi.astype(int)

        # create todays score
        a_today_score = []
        for small, big in [(3, 120), (3, 260), (3, 520), (5, 120), (5, 260), (5, 520), (10, 120), (10, 260), (10, 520), (20, 120), (20, 260), (20, 520)]:  # (120,520),(120,780),(260,520),(260,780),(520,780)
            if small < big:
                today_small_q = df_today_rsi.at[trade_date, f"rsi{small}"]
                today_big_q = df_today_rsi.at[trade_date, f"rsi{big}"]

                today_mean_based_on_past = df_past_rsi_table.at[trade_date, f"{small}.{today_small_q}_{big}.{today_big_q}"]
                a_today_score.append(today_mean_based_on_past)

        df_trading_result.at[trade_date, "score"] = pd.Series(data=a_today_score).mean()

    LB.to_csv_feather(df_trading_result, LB.a_path("trading_result"))
    LB.to_csv_feather(df_past_rsi_table, LB.a_path("df_past_rsi_table"))
    LB.to_csv_feather(df_today_rsi, LB.a_path("df_today_rsi"))


simulate_trade_basedon_rsi()

df = DB.get_asset()
df["ma10"] = df["close"].rolling(10).mean()
df["trix10"] = talib.TRIX(df["close"], timeperiod=10)

df = df[["close", "open"]]

#
# rsi_abv_under_test()
#
#
#
#
# #plot_autocorr()
# df=DB.get_asset(ts_code="000001.SZ")
# #df=pd.read_csv("Stock_Market.csv")
# trend(df=df,ibase="close")
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
