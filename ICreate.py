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
import inspect
import enum
from LB import *
pd.options.mode.chained_assignment = None  # default='warn'


# BOTTLE NECK modify here
class IBase(enum.Enum):
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    pct_chg = "pct_chg"
    # fgain = "fgain"
    # # pgain = "pgain"
    # pjup = "pjup"
    # pjdown = "pjdown"
    ivola = "ivola"
    # cdl = "cdl"

    # fun
    pe_ttm = "pe_ttm"
    pb = "pb"
    ps_ttm = "ps_ttm"
    dv_ttm = "dv_ttm"
    n_cashflow_act = "n_cashflow_act"
    n_cashflow_inv_act = "n_cashflow_inv_act"
    n_cash_flows_fnc_act = "n_cash_flows_fnc_act"
    profit_dedt = "profit_dedt"
    netprofit_yoy = "netprofit_yoy"
    or_yoy = "or_yoy"
    grossprofit_margin = "grossprofit_margin"
    netprofit_margin = "netprofit_margin"
    debt_to_assets = "debt_to_assets"

    # oth
    period = "period"
    # total_share = "total_share"
    total_mv = "total_mv"
    pledge_ratio = "pledge_ratio"
    vol = "vol"
    turnover_rate = "turnover_rate"

class IDeri(enum.Enum):
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
    rsi = "rsi"
    mom = "mom"
    rocr = "rocr"
    # ppo = "ppo" for some reason not existing in talib
    cmo = "cmo"
    apo = "apo"
    # boll="boll"
    # ema="ema"
    # sma="sma"

    # transform = normalize and standardize
    # net="net"
    # rank="rank"
    # pct_change="pct_change"
    # divmean="divmean"
    # divmabs="divabs"
    # abv="abv"
    # cross="cross"

    # custom
    # trend="trend"
    # rs="rs"

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


def pjup(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "pct_chg", 1)
    condition_1 = df["low"] > df["high"].shift(1)  # today low bigger thann yesterday high
    condition_2 = df["pct_chg"] >= 2
    df[add_to] = (condition_1 & condition_2).astype(int)
    return add_to


def pjdown(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "pct_chg", 1)
    condition_1 = df["high"] < df["low"].shift(1)  # yesterday low bigger than todays high
    condition_2 = df.pct_chg <= -2
    df[add_to] = (condition_1 & condition_2).astype(int)
    return add_to


def period(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "ts_code", 1)
    df[add_to] = (range(1, len(df.index) + 1))
    return add_to


def ivola(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = df[["close", "high", "low", "open"]].std(axis=1)
    return add_to


def pgain(df: pd.DataFrame, freq: LB.BFreq):
    add_to = f"pgain{freq}"
    add_column(df, add_to, "pct_chg", 1)
    # df[add_to+"test"] = (1 + (df["pct_chg"] / 100)).rolling(rolling_freq).apply(pd.Series.prod, raw=False)
    try:
        df[add_to] = quick_rolling_prod((1 + (df["pct_chg"] / 100)).to_numpy(), freq)
    except:
        df[add_to] = np.nan
    return add_to


def fgain(df: pd.DataFrame, freq: LB.BFreq):
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

    for i in a_all:  # RSI 1
        try:
            if i == 1:  # TODO RSI 1 need to be generallized for every indicator. if rsi1 > RSI2, then it is 1, else 0. something like that
                df.loc[(df["pct_chg"] > 0.0), rsi_name + "1"] = 1.0
            else:
                df[f"{rsi_name}{i}"] = talib.RSI(df[ibase], timeperiod=i) / 100
        except:  # if error happens here, then no need to continue
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

        # fill na based on the trigger points
        df[trendfreq_name].fillna(method='bfill', inplace=True)
        last_trade = df.at[df.last_valid_index(), trendfreq_name]
        fill = 0 if last_trade == 1 else 1
        df[trendfreq_name].fillna(value=fill, inplace=True)

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


# TODO this article
# https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
# def rsi(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
#     add_to = LB.standard_indi_name(ibase=ibase, deri=IDeri.rsi.value, dict_variables={"freq": freq, "re": re.value})
#     try:
#         df[add_to]=talib.RSI(df[ibase], timeperiod=freq.value)
#     except:
#         df[add_to]=np.nan
#     return add_to

def cmo(df: pd.DataFrame, ibase: str, ffreq: SFreq, sfreq: SFreq):
    return deri_tec(df=df, ibase=ibase, ideri=IDeri.cmo, func=talib.CMO, fastperiod=ffreq.value, slowperiod=sfreq.value)


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
        df[add_to] = reFunc(df[ibase], freq).corr()
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


if __name__ == '__main__':
    # first only add all ideri that uses one column
    # then add all ideri that uses multiple columns
    import DB

    df = DB.get_asset()

    #then bruteforce ideri that can be used on ideri
    pass
