import math
import pandas as pd
import time
import numpy as np
import LB
import os
import datetime
import glob
import talib
import itertools
from multiprocessing import Process
from scipy import signal
import inspect
import matplotlib.pyplot as plt
import enum
from sklearn.preprocessing import MinMaxScaler
from LB import *
pd.options.mode.chained_assignment = None  # default='warn'


# BOTTLE NECK modify here
class IBase(enum.Enum):  # every Indicator base should take no argument in create process. The tester should always find best argument by hand. Any other argument should be put into deri.
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    pct_chg = "pct_chg"
    co_pct_chg = "co_pct_chg"
    # fgain = "fgain"
    # # pgain = "pgain"
    pjup = "pjup"
    pjdown = "pjdown"
    ivola = "ivola"
    # cdl = "cdl"

    # fun
    pe_ttm = "pe_ttm"
    pb = "pb"
    ps_ttm = "ps_ttm"
    dv_ttm = "dv_ttm"
    total_cur_assets = "total_cur_assets"
    total_assets = "total_assets"
    total_cur_liab = "total_cur_liab"
    total_liab = "total_liab"
    n_cashflow_act = "n_cashflow_act"
    n_cashflow_inv_act = "n_cashflow_inv_act"
    n_cash_flows_fnc_act = "n_cash_flows_fnc_act"
    profit_dedt = "profit_dedt"
    netprofit_yoy = "netprofit_yoy"
    or_yoy = "or_yoy"
    grossprofit_margin = "grossprofit_margin"
    netprofit_margin = "netprofit_margin"
    debt_to_assets = "debt_to_assets"
    turn_days = "turn_days"

    # oth
    period = "period"
    # total_share = "total_share"
    total_mv = "total_mv"
    pledge_ratio = "pledge_ratio"
    vol = "vol"
    turnover_rate = "turnover_rate"


class IDeri(enum.Enum):  #first level Ideri = IDeri that only uses ibase and no other IDeri
    # create = "create"
    # count = "count"
    # sum = "sum"
    # mean = "mean"
    # median = "median"
    # var = "var"
    # std = "std"
    # min = "min"
    # max = "max"
    # corr = "corr"
    # cov = "cov"
    # skew = "skew"
    # kurt = "kurt"

    # technical Derivation
    #rsi = "rsi"
    # mom = "mom"
    # rocr = "rocr"
    # # ppo = "ppo" for some reason not existing in talib
    # cmo = "cmo"
    # apo = "apo"
    # boll="boll"
    # ema="ema"
    # sma="sma"

    # transform = normalize and standardize
    # net="net"
    # rank="rank"
    # pct_change="pct_change"
    # divmean="divmean"
    # divmabs="divabs"
    # scale ="scale" #normalize value to 1 and 0
    # abv="abv"
    # cross="cross"


    # second level IDERI, need other functions as argument
    # trend = "trend"  # RSI CMO
    overma = "overma"
    crossma = "crossma"
    # rs="rs"
    # cross over
    # divergence
    #overma

# clip,autocorr,cummax
def get_func(name: str):
    return globals()[name]

class Trend2Weight(enum.Enum):
    t8 = 0.08
    t16 = 0.16
    t32 = 0.32
    t64 = 0.64
    t128 = 1.28

class RE(enum.Enum):
    r = "r"
    e = "e"

def open(df: pd.DataFrame, ibase: str): return ibase
def high(df: pd.DataFrame, ibase: str): return ibase
def close(df: pd.DataFrame, ibase: str): return ibase
def low(df: pd.DataFrame, ibase: str): return ibase
def pct_chg(df: pd.DataFrame, ibase: str): return ibase
def pe_ttm(df: pd.DataFrame, ibase: str): return ibase
def pb(df: pd.DataFrame, ibase: str): return ibase
def ps_ttm(df: pd.DataFrame, ibase: str): return ibase
def dv_ttm(df: pd.DataFrame, ibase: str): return ibase
def n_cashflow_act(df: pd.DataFrame, ibase: str): return ibase
def n_cashflow_inv_act(df: pd.DataFrame, ibase: str): return ibase
def n_cash_flows_fnc_act(df: pd.DataFrame, ibase: str): return ibase
def profit_dedt(df: pd.DataFrame, ibase: str): return ibase
def netprofit_yoy(df: pd.DataFrame, ibase: str): return ibase
def or_yoy(df: pd.DataFrame, ibase: str): return ibase
def grossprofit_margin(df: pd.DataFrame, ibase: str): return ibase
def netprofit_margin(df: pd.DataFrame, ibase: str): return ibase
def debt_to_assets(df: pd.DataFrame, ibase: str): return ibase
def total_mv(df: pd.DataFrame, ibase: str): return ibase
def vol(df: pd.DataFrame, ibase: str): return ibase
def turnover_rate(df: pd.DataFrame, ibase: str): return ibase
def pledge_ratio(df: pd.DataFrame, ibase: str): return ibase


# TODO test jump up with different gap threshhold


def oc_pct_chg(df: pd.DataFrame):
    add_to = "co_pct_chg"
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = (df["open"] / df["close"].shift(1))
    return add_to


def pjup(df: pd.DataFrame):
    add_to = "pjup"
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = ((df["low"] > df["high"].shift(1)) & (df["pct_chg"] >= 2)).astype(int)  # today low bigger than yesterday high and pct _chg > 2
    return add_to


def pjdown(df: pd.DataFrame):
    add_to = "pjdown"
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = ((df["high"] < df["low"].shift(1)) & (df.pct_chg <= -2)).astype(int)  # yesterday low bigger than todays high and pct _chg < -2
    return add_to


def period(df: pd.DataFrame):
    add_to = "period"
    add_column(df, add_to, "ts_code", 1)
    df[add_to] = (range(1, len(df.index) + 1))
    return add_to


def ivola(df: pd.DataFrame):
    add_to = "ivola"
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = df[["close", "high", "low", "open"]].std(axis=1)
    return add_to


def pgain(df: pd.DataFrame, freq: BFreq):
    add_to = f"pgain{freq}"
    add_column(df, add_to, "pct_chg", 1)
    # df[add_to+"test"] = (1 + (df["pct_chg"] / 100)).rolling(rolling_freq).apply(pd.Series.prod, raw=False)
    try:
        df[add_to] = quick_rolling_prod((1 + (df["pct_chg"] / 100)).to_numpy(), freq)
    except:
        df[add_to] = np.nan
    return add_to

def fgain(df: pd.DataFrame, freq: BFreq):
    add_to = f"fgain{freq}"
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = df[f"pgain{freq}"].shift(int(-freq))
    return add_to

def cdl(df: pd.DataFrame, ibase: str):
    a_positive_columns = []
    a_negative_columns = []

    # create candle stick column
    for key, array in c_candle().items():
        if (array[1] != 0) or (array[2] != 0):  # if used at any, calculate the pattern
            func = array[0]
            df[key] = func(open=df["open"], high=df["high"], low=df["low"], close=df["close"]).replace(0, np.nan)

            if (array[1] != 0):  # candle used as positive pattern
                a_positive_columns.append(key)
                if (array[1] == -100):  # talib still counts the pattern as negative: cast it positive
                    df[key].replace(-100, 100, inplace=True)

            if (array[2] != 0):  # candle used as negative pattern
                a_negative_columns.append(key)
                if (array[2] == 100):  # talib still counts the pattern as positive: cast it negative
                    df[key].replace(100, -100, inplace=True)

    df[ibase] = (df[df[a_positive_columns] == 100].sum(axis='columns') + df[df[a_negative_columns] == -100].sum(axis='columns')) / 100
    # IMPORTANT! only removing column is the solution because slicing dataframe does not modify the original df
    columns_remove(df, a_positive_columns + a_negative_columns)
    return ibase


def crossma(df: pd.DataFrame, ibase: str, Sfreq1: SFreq, Sfreq2: SFreq):
    add_to = LB.standard_indi_name(ibase=ibase, deri="crossma", dict_variables={"Sfreq1": SFreq, "Sfreq2": SFreq})
    add_column(df, add_to, ibase, 1)
    df[add_to] = (df[ibase].rolling(Sfreq1.value).mean() > df[ibase].rolling(Sfreq2.value).mean()).astype(float)
    df[add_to] = (df[add_to].diff()).replace(0, np.nan)
    return add_to


def overma(df: pd.DataFrame, ibase: str, Sfreq1: SFreq, Sfreq2: SFreq):
    add_to = LB.standard_indi_name(ibase=ibase, deri="overma", dict_variables={"Sfreq1": SFreq, "Sfreq2": SFreq})
    add_column(df, add_to, ibase, 1)
    df[add_to] = (df[ibase].rolling(Sfreq1.value).mean() > df[ibase].rolling(Sfreq2.value).mean()).astype(float)
    return add_to


# ONE OF THE MOST IMPORTANT KEY FUNNCTION I DISCOVERED
# 1.Step Create RSI or Abv_ma
# 2.Step Create Phase
# 3 Step Create Trend
# 4 Step calculate trend pct_chg
# 5 Step Calculate Step comp_chg
# variables:1. function, 2. threshhold 3. final weight 4. combination with other function
def trend(df: pd.DataFrame, ibase: str, thresh_log=-0.043, thresh_rest=0.7237, market_suffix: str = ""):
    a_all = [1] + c_bfreq()
    a_low = [str(x) for x in a_all][:-1]
    a_high = [str(x) for x in a_all][1:]

    rsi_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}rsi")
    phase_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}phase")
    trend_name = standard_indi_name(ibase=ibase, deri=f"{market_suffix}{IDeri.trend.value}")

    func = talib.RSI
    # RSI and CMO are the best. CMO is a modified RSI
    # RSI,CMO,MOM,ROC,ROCR100,TRIX

    #df[f"detrend{ibase}"] = signal.detrend(data=df[ibase])
    for i in a_all:  # RSI 1
        try:
            if i == 1:
                df[f"{rsi_name}{i}"] = (df[ibase].pct_change() > 0).astype(int)
                # df[ rsi_name + "1"] = 0
                #df.loc[(df["pct_chg"] > 0.0), rsi_name + "1"] = 1.0
            else:
                df[f"{rsi_name}{i}"] = func(df[ibase], timeperiod=i) / 100

                # normalization causes error
                # df[f"{rsi_name}{i}"] = (df[f"{rsi_name}{i}"]-df[f"{rsi_name}{i}"].min())/ (df[f"{rsi_name}{i}"].max()-df[f"{rsi_name}{i}"].min())
        except Exception as e:  # if error happens here, then no need to continue
            print("error" ,e)
            df[trend_name] = np.nan
            return trend_name

    # Create Phase
    for i in [str(x) for x in a_all]:
        maximum = (thresh_log * math.log(int(i)) + thresh_rest)
        minimum = 1 - maximum
        df[f"{phase_name}{i}"] = [1 if x > maximum else 0 if x < minimum else np.nan for x in df[f"{rsi_name}{i}"]]

    # one loop to create trend from phase
    for freq_low, freq_high in zip(a_low, a_high):
        trendfreq_name = f"{trend_name}{freq_high}"
        df.loc[(df[phase_name + freq_high] == 1) & (df[phase_name + freq_low] == 1), trendfreq_name] = 1
        df.loc[(df[phase_name + freq_high] == 0) & (df[phase_name + freq_low] == 0), trendfreq_name] = 0

        # fill na based on the trigger points. bfill makes no sense here
        df[trendfreq_name].fillna(method='ffill', inplace=True)

    # remove RSI and phase Columns to make it cleaner
    a_remove = []
    for i in a_all:
        # a_remove.append(market_suffix + "rsi" + str(i))
        # a_remove.append(market_suffix + "phase" + str(i))
        pass
    LB.columns_remove(df, a_remove)

    # calculate final trend =weighted trend of previous TODO this need to be adjusted manually. But the weight has relative small impact
    df[trend_name] = df[f"{trend_name}2"] * 0.80 + df[f"{trend_name}5"] * 0.12 + df[f"{trend_name}10"] * 0.04 + df[f"{trend_name}20"] * 0.02 + df[f"{trend_name}60"] * 0.01 + df[f"{trend_name}240"] * 0.01
    return trend_name


def support_resistance_horizontal(start_window=240, rolling_freq=1, step=10, thresh=[4, 0.2], bins=8, dict_rs={"abv": 8, "und": 8}, df_asset=pd.DataFrame(), delay=3):
    def support_resistance_acc(abv_und, max_rs, s_minmax, end_date, f_end_date, df_asset):
        # 1. step calculate all relevant resistance = relevant earlier price close to current price
        current_price = df_asset.at[end_date, "close"]

        if abv_und == "abv":
            s_minmax = s_minmax[(s_minmax / current_price < thresh[0]) & ((s_minmax / current_price > 1))]
        elif abv_und == "und":
            s_minmax = s_minmax[(s_minmax / current_price < 1) & ((s_minmax / current_price > thresh[1]))]

        # 2.step find the max occurence of n values
        try:
            s_occurence_bins = s_minmax.value_counts(bins=bins)
            a_rs = []
            for counter, (index, value) in enumerate(zip(s_occurence_bins.iteritems())):
                a_rs.append(index.left)

            # 3. step sort the max occurence values and assign them as rs
            a_rs.sort()  # small value first 0 ,1, 2
            for i, item in enumerate(a_rs):
                df_asset.loc[end_date:f_end_date, f"rs{abv_und}{i}"] = item
        except:
            pass

    if len(df_asset) > start_window:
        #  calculate all min max for acceleration used for later simulation
        try:
            s_minall = df_asset["close"].rolling(rolling_freq).min()
            s_maxall = df_asset["close"].rolling(rolling_freq).max()
        except:
            return

        # iterate over past data as window. This simulates rs for past times/frames
        for row in range(0, len(df_asset), step):
            if row + start_window > len(df_asset) - 1:
                break

            start_date = df_asset.index[0]
            end_date = df_asset.index[row + start_window]
            for abv_und, max_rs in dict_rs.items():
                if row + start_window + step > len(df_asset) - 1:
                    break

                f_end_date = df_asset.index[row + start_window + step]
                s_minmax = (s_minall.loc[start_date:end_date]).append(s_maxall.loc[start_date:end_date])
                support_resistance_acc(abv_und=abv_und, max_rs=max_rs, s_minmax=s_minmax, end_date=end_date, f_end_date=f_end_date, df_asset=df_asset)

        # calcualte individual rs score
        for abv_und, count in dict_rs.items():
            for i in range(0, count):
                try:
                    # first fill na for the last period because rolling does not reach
                    df_asset[f"rs{abv_und}{i}"].fillna(method="ffill", inplace=True)
                    df_asset[f"rs{abv_und}{i}"].fillna(method="bfill", inplace=True)

                    df_asset[f"rs{abv_und}{i}_abv"] = (df_asset[f"rs{abv_und}{i}"] > df_asset["close"]).astype(int)
                    df_asset[f"rs{abv_und}{i}_cross"] = df_asset[f"rs{abv_und}{i}_abv"].diff().fillna(0).astype(int)
                except Exception as e:
                    print("error resistance 1: probably new stock", e)

        df_asset["rs_abv"] = 0  # for detecting breakout to top: happens often
        df_asset["rs_und"] = 0  # for detecting breakout to bottom: happens rare
        for abv_und, max_rs in dict_rs.items():
            for i in range(0, max_rs):
                try:
                    df_asset[f"rs_{abv_und}"] = df_asset[f"rs_{abv_und}"] + df_asset[f"rs{abv_und}{i}_cross"].abs()
                except Exception as e:
                    print("error resistance 2", e)

        # optional #TODO configure it nicely
        df_asset["rs_abv"] = df_asset["rs_abv"].rolling(delay).max().fillna(0)
        df_asset["rs_und"] = df_asset["rs_und"].rolling(delay).max().fillna(0)
    else:
        print(f"df is len is under {start_window}. probably new stock")
        for abv_und, max_rs in dict_rs.items():
            for i in range(0, max_rs):  # only consider rs und 2,3,4,5, evenly weighted. middle ones have most predictive power
                df_asset[f"rs{abv_und}{i}"] = np.nan
                df_asset[f"rs{abv_und}{i}_abv"] = np.nan
                df_asset[f"rs{abv_und}{i}_cross"] = np.nan

        df_asset["rs_abv"] = 0
        df_asset["rs_und"] = 0

    return df_asset


# TODO this article
# https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
# def rsi(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
#     add_to = LB.standard_indi_name(ibase=ibase, deri=IDeri.rsi.value, dict_variables={"freq": freq, "re": re.value})
#     try:
#         df[add_to]=talib.RSI(df[ibase], timeperiod=freq.value)
#     except:
#         df[add_to]=np.nan
#     return add_to


def cmo(df: pd.DataFrame, ibase: str, freq: BFreq):
    return deri_tec(df=df, ibase=ibase, ideri=IDeri.cmo, func=talib.CMO, timeperiod=freq.value)


def apo(df: pd.DataFrame, ibase: str, ffreq: SFreq, sfreq: SFreq):
    return deri_tec(df=df, ibase=ibase, ideri=IDeri.apo, func=talib.APO, fastperiod=ffreq.value, slowperiod=sfreq.value)


# ((slowMA-fastMA)*100)/fastMA
# def ppo (df: pd.DataFrame, ibase: str, ffreq: Freq, sfreq: Freq):
#     return deri_tec(df=df, ibase=ibase, ideri=IDeri.PPO , func=talib.PPO, fastperiod=ffreq.value,slowperiod=sfreq.value, matype=0)

# ((close_today - close_ndays ago)/close_ndays_ago) * 100
def rocr(df: pd.DataFrame, ibase: str, freq: BFreq):
    return deri_tec(df=df, ibase=ibase, ideri=IDeri.rocr, func=talib.ROCR, timeperiod=freq.value)


# close_today - close_ndays_ago
def mom(df: pd.DataFrame, ibase: str, freq: BFreq):
    return deri_tec(df=df, ibase=ibase, ideri=IDeri.mom, func=talib.MOM, timeperiod=freq.value)


def rsi(df: pd.DataFrame, ibase: str, freq: BFreq):
    return deri_tec(df=df, ibase=ibase, ideri=IDeri.rsi, func=talib.RSI, timeperiod=freq.value)


@try_ignore  # try ignore because all talib function can not handle nan input values. so this wrapper ignores all nan input values and creates add_to_column at one point
def deri_tec(df: pd.DataFrame, ibase: str, ideri: IDeri, func, **kwargs):
    add_to = LB.standard_indi_name(ibase=ibase, deri=ideri.value, dict_variables=kwargs)
    add_column(df, add_to, ibase, 1)
    df[add_to] = func(df[ibase], **kwargs)
    return add_to


def count(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.count)


def sum(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.sum)


def mean(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.mean)


def median(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.median)


def var(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.var)


def std(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.std)


def min(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.min)


def max(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.max)


def corr(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.corr)


def cov(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.cov)


def skew(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.skew)


def kurt(df: pd.DataFrame, ibase: str, freq: BFreq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, ideri=IDeri.kurt)


# this funciton should not exist. One should be able to pass function, but apply on rolling is slow and pandas.core.window.rolling is private. So Only if else case here is possible
def deri_sta(df: pd.DataFrame, ibase: str, ideri: IDeri, freq: BFreq, re: RE):
    # enum to value
    freq = freq.value
    ideri = ideri.value
    reFunc = pd.Series.rolling if re == RE.r else pd.Series.expanding

    add_to = LB.standard_indi_name(ibase=ibase, deri=ideri, dict_variables={"freq": freq, "re": re.value})
    add_column(df, add_to, ibase, 1)

    # https://pandas.pydata.org/pandas-docs/stable/reference/window.html
    if ideri == "count":
        df[add_to] = reFunc(df[ibase], freq).count()
    elif ideri == "sum":
        df[add_to] = reFunc(df[ibase], freq).sum()
    elif ideri == "mean":
        df[add_to] = reFunc(df[ibase], freq).mean()
    elif ideri == "median":
        df[add_to] = reFunc(df[ibase], freq).median()
    elif ideri == "var":
        df[add_to] = reFunc(df[ibase], freq).var()
    elif ideri == "std":
        df[add_to] = reFunc(df[ibase], freq).std()
    elif ideri == "min":
        df[add_to] = reFunc(df[ibase], freq).min()
    elif ideri == "max":
        df[add_to] = reFunc(df[ibase], freq).max()
    elif ideri == "corr":
        from DB import get_asset
        df_sh = get_asset(ts_code="000001.SH", asset="I")
        df[add_to] = reFunc(df[ibase], freq).corr(df_sh["close"])
    elif ideri == "cov":
        df[add_to] = reFunc(df[ibase], freq).cov()
    elif ideri == "skew":
        df[add_to] = reFunc(df[ibase], freq).skew()
    elif ideri == "kurt":
        df[add_to] = reFunc(df[ibase], freq).kurt()
    return add_to



# input a dict with all variables. Output a list of all possible combinations
def explode_settings(dict_one_indicator_variables):
    # 1. only get values form above dict
    # 2. create cartesian product of the list
    # 3. create dict out of list

    # first list out all possible choices
    for key, value in dict_one_indicator_variables.items():
        print(f"Input ---> {key}: {value}")

    a_product_result = []
    for one_combination in itertools.product(*dict_one_indicator_variables.values()):
        dict_result = dict(zip(dict_one_indicator_variables.keys(), one_combination))
        a_product_result.append(dict_result)
    print(f"there are that many combinations: {len(a_product_result)}")
    return a_product_result


def function_all_combinations(func):
    signature = inspect.getfullargspec(func).annotations
    result_dict = {}
    # get function annotation with variable and type
    for variable, enum_or_class in signature.items():
        if issubclass(enum_or_class, enum.Enum):  # ignore everything else that is not a enum. All Variable types MUST be custom defined Enum
            result_dict[variable] = enum_or_class
    return explode_settings(result_dict)


def trendtest():
    import DB
    df = DB.get_asset()
    df = df[["close"]]
    df_copy = df.copy()

    trend(df=df, ibase="close")
    df.to_csv("all in one calc.csv")

    startday = 19920508
    for trade_day in df_copy.index:
        if int(trade_day) < startday:
            continue
        df_loc = df_copy.loc[19910129:trade_day]

        trend(df=df_loc, ibase="close")

        for i in LB.c_bfreq():
            # rsi
            prev = df.loc[19910129:trade_day, f"close.trend{i}"]
            now = df_loc[f"close.trend{i}"]

            if prev.equals(now):
                pass
                print(f"{trade_day} different trend same")
            else:
                where_not_same = (prev != (now))
                print(where_not_same.index)
                print(f"{trade_day} different trend NOT same")
                df_loc.to_csv("real simulated.csv")
                return




        
if __name__ == '__main__':
    # TODO test jump up with different gap threshhold
    # TODO Try to get trend better with only catching the phase 1 and not phase 3. need some differatiating between phase
    # TODO make a statistical test to see if phase 1 is really better
    # TODO E.G fgain if all phases are strict 1.
    #trendtest()

    # cross over brute force
    # define all main deri function. scale it if nessesary
    # define second cross option: ibase:str, deri, variable

    # if crossover then buy / sell signal

    #if both are 1 then buy buy signal




    # first only add all ideri that uses one column
    # then add all ideri that uses multiple columns
    # import DB
    # scaler=MinMaxScaler()
    #
    # df = DB.get_asset()
    # tren_mean_return = df.loc[df["trend2"] == 1, "fgain2"].mean()
    # print("trend mean", tren_mean_return)
    # df=df[["close","fgain2"]]
    # for i in [2,5,10,20,60,240]:
    #     df[f"cmo{i}"]=talib.CMO(df["close"],timeperiod=i)
    #     df[f"cmo{i}_mean"]=df[f"cmo{i}"].mean()
    #     df[f"cmo{i}_buy"]=(df[f"cmo{i}"]> df[f"cmo{i}_mean"]).astype(int)
    #
    # mean_return=df.loc[(df["cmo2_buy"]==1)& (df["cmo5_buy"]==1),"fgain2"].mean()
    # print("oocc", len(df[(df["cmo2_buy"]==1)]))
    # print("cmo 2，5 mean",mean_return)
    #
    #
    # mean_return = df.loc[(df["cmo5_buy"] == 1) & (df["cmo10_buy"] == 1), "fgain2"].mean()
    #
    # print("cmo 5,10 mean", mean_return)
    # df = df[["close", "cmo240","cmo240_buy"]]
    # df.reset_index(inplace=True, drop=True)
    # df.plot(legend=True)
    # plt.show()

    #then bruteforce ideri that can be used on ideri
    pass
