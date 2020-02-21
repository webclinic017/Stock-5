import math

import tushare as ts
import pandas as pd
import time
import os.path
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import LB
import os
import datetime
import copy
import imageio
import glob
import talib
import itertools
from multiprocessing import Process
import inspect
import enum
from LB import *

pd.options.mode.chained_assignment = None  # default='warn'


class IDeri(enum.Enum):
    # statistical derivation
    create = "create"
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
    # rsi="rsi"
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
def get_deri_func(deri_name: IDeri):
    dict = {
        # "create": create, # IMPORTANT the create function should never be requested here. it should be get from get_create_func instead
        "count": count,
        "sum": sum,
        "mean": mean,
        "median": median,
        "var": var,
        "std": std,
        "min": min,
        "max": max,
        "corr": corr,
        "cov": cov,
        "skew": skew,
        "kurt": kurt,
    }
    return dict[deri_name.value]


def get_create_func(ibase):
    dict = {
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "pct_chg": pct_chg,
        "ivola": ivola,
        "pgain": pgain,
        "fgain": fgain,
        "pjup": pjup,
        "pjdown": pjdown,
        "cdl": cdl,
        "trend": trend,

        "pe_ttm": pe_ttm,
        "pb": pb,
        "ps_ttm": ps_ttm,
        "dv_ttm": dv_ttm,
        "n_cashflow_act": n_cashflow_act,
        "n_cashflow_inv_act": n_cashflow_inv_act,
        "n_cash_flows_fnc_act": n_cash_flows_fnc_act,
        "profit_dedt": profit_dedt,
        "netprofit_yoy": netprofit_yoy,
        "or_yoy": or_yoy,
        "grossprofit_margin": grossprofit_margin,
        "netprofit_margin": netprofit_margin,
        "debt_to_assets": debt_to_assets,

        "period": period,
        # "total_share":total_share,
        "total_mv": total_mv,
        "pledge_ratio": pledge_ratio,
        "vol": vol,
        "turnover_rate": turnover_rate

    }
    return dict[ibase.value]


# BOTTLE NECK modify here
class IBase(enum.Enum):
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    pct_chg = "pct_chg"
    # fgain = "fgain"
    # # pgain = "pgain"
    pjup = "pjup"
    pjdown = "pjdown"
    ivola = "ivola"
    cdl = "cdl"
    trend = "trend"

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


class SApply(enum.Enum):
    count = "count"
    sum = "sum"
    mean = "mean"
    median = "median"
    var = "var"
    std = "std"
    min = "min"
    max = "max"
    corr = "corr"
    cov = "cov"
    skew = "skew"
    kurt = "kurt"


class Trend2Weight(enum.Enum):
    t1 = 0.01
    t2 = 0.02
    t4 = 0.04
    t8 = 0.08
    t16 = 0.16
    t32 = 0.32
    t64 = 0.64

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


# ONE OF THE MOST IMPORTANT KEY FUNNCTION I DISCOVERED
# 1.Step Create RSI or Abv_ma
# 2.Step Create Phase
# 3 Step Create Trend
# 4 Step calculate trend pct_chg
# 5 Step Calculate Step comp_chg

def trend(df: pd.DataFrame, ibase: str, t2w: Trend2Weight = Trend2Weight.t1, t5w: Trend2Weight = Trend2Weight.t1, thresh_log=-0.043, thresh_rest=0.7237, market_suffix: str = ""):
    a_all = [1] + c_freq()
    a_low = [str(x) for x in a_all][:-1]  # should be [5, 20,60]
    a_high = [str(x) for x in a_all][1:]  # should be [20,60,240]
    on_column = IBase.close

    # variables:1. function, 2. threshhold 3. final weight 4. combination with other function
    for i in a_all:  # RSI 1
        if i == 1:  # TODO RSI 1 need to be generallized for every indicator. if rsi1 > RSI2, then it is 1, else 0. something like that
            df[market_suffix + "rsi1"] = 0.0
            df.loc[(df["pct_chg"] > 0.0), market_suffix + "rsi1"] = 1.0
        else:
            df[market_suffix + "rsi" + str(i)] = talib.RSI(df[on_column.value], timeperiod=i) / 100

    # Create Phase
    for i in [str(x) for x in a_all]:
        maximum = (thresh_log * math.log(int(i)) + thresh_rest)
        minimum = 1 - maximum

        # df[market_suffix + f"phase{i}"] = [1 if x > maximum else 0 if x < minimum else np.nan for x in df[market_suffix + "rsi" + i]]
        # df[market_suffix + f"phase{i}"] = df[market_suffix + "rsi" + i].apply(lambda x: 1 if x > maximum else 0 if x < minimum else np.nan )
        df[market_suffix + f"phase{i}"] = [1 if x > maximum else 0 if x < minimum else np.nan for x in df[market_suffix + "rsi" + i]]

    # one loop to create trend from phase
    for freq_low, freq_high in zip(a_low, a_high):
        trend_name = market_suffix + f"trend{freq_high}"
        df[trend_name] = np.nan
        df.loc[(df[market_suffix + f"phase{freq_high}"] == 1) & (df[market_suffix + "phase" + freq_low] == 1), trend_name] = 1
        df.loc[(df[market_suffix + f"phase{freq_high}"] == 0) & (df[market_suffix + "phase" + freq_low] == 0), trend_name] = 0

        # fill na based on the trigger points
        df[trend_name].fillna(method='bfill', inplace=True)
        last_trade = df.loc[df.last_valid_index(), trend_name]
        fill = 0 if last_trade == 1 else 1
        df[trend_name].fillna(value=fill, inplace=True)

    # remove RSI and phase Columns to make it cleaner
    a_remove = []
    for i in a_all:
        # a_remove.append(market_suffix + "rsi" + str(i))
        # a_remove.append(market_suffix + "phase" + str(i))
        pass
    LB.columns_remove(df, a_remove)

    # calculate final trend =weighted trend of previous TODO this need to be adjusted manually
    df[market_suffix + ibase] = df[market_suffix + "trend2"] * t2w.value + df[market_suffix + "trend5"] * t5w.value + df[market_suffix + "trend20"] * 0.05 + df[market_suffix + "trend60"] * 0.05 + df[market_suffix + "trend240"] * 0.05
    # df[market_suffix + ibase] = df[market_suffix + "trend2"]
    return market_suffix + ibase


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

    df[ibase] = (df[df[a_positive_columns] == 100].sum(axis='columns') + df[df[a_negative_columns] == -100].sum(axis='columns'))/100

    # IMPORTANT! only removing column is the solution because slicing dataframe does not modify the original df
    columns_remove(df, a_positive_columns + a_negative_columns)
    return ibase


def pjup(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "pct_chg", 1)
    yesterday_high = df["high"].shift(1)
    today_low = df["low"]
    condition_1 = today_low > yesterday_high
    condition_2 = df["pct_chg"] >= 2
    df[add_to] = condition_1 & condition_2
    df[add_to] = df[add_to].astype(int)
    return add_to


def pjdown(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "pct_chg", 1)
    yesterday_low = df["low"].shift(1)
    today_high = df["high"]
    condition_1 = today_high < yesterday_low
    condition_2 = df.pct_chg <= -2
    df[add_to] = condition_1 & condition_2
    df[add_to] = df[add_to].astype(int)
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


def pgain(df: pd.DataFrame, freq: LB.Freq):
    add_to = f"pgain{freq}"
    add_column(df, add_to, "pct_chg", 1)
    # df[add_to+"test"] = (1 + (df["pct_chg"] / 100)).rolling(rolling_freq).apply(pd.Series.prod, raw=False)
    try:
        df[add_to] = quick_rolling_prod((1 + (df["pct_chg"] / 100)).to_numpy(), freq)
    except:
        df[add_to] = np.nan
    return add_to


def fgain(df: pd.DataFrame, freq: LB.Freq):
    add_to = f"fgain{freq}"
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = df[f"pgain{freq}"].shift(int(-freq))
    return add_to


def rsi(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):  # TODO
    # add_to = LB.standard_indi_name(ibase=ibase, deri=, dict_variables={"freq": freq, "re": re.value})
    # func=talib.RSI()
    return


def count(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.count)


def sum(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.sum)


def mean(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.mean)


def median(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.median)


def var(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.var)


def std(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.std)


def min(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.min)


def max(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.max)


def corr(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.corr)


def cov(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.cov)


def skew(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.skew)


def kurt(df: pd.DataFrame, ibase: str, freq: Freq, re: RE):
    return deri_sta(df=df, freq=freq, ibase=ibase, re=re, apply=SApply.kurt)


# this funciton should not exist. One should be able to pass function, but apply on rolling is slow and pandas.core.window.rolling is private. So Only if else case here is possible
def deri_sta(df: pd.DataFrame, ibase: str, freq: Freq, re: RE, apply: SApply):
    # enum to value
    freq = freq.value
    apply = apply.value
    reFunc = pd.Series.rolling if re == RE.r else pd.Series.expanding

    add_to = LB.standard_indi_name(ibase=ibase, deri=apply, dict_variables={"freq": freq, "re": re.value})
    add_column(df, add_to, ibase, 1)

    # https://pandas.pydata.org/pandas-docs/stable/reference/window.html
    if apply == "count":
        df[add_to] = reFunc(df[ibase], freq).count()
    elif apply == "sum":
        df[add_to] = reFunc(df[ibase], freq).sum()
    elif apply == "mean":
        df[add_to] = reFunc(df[ibase], freq).mean()
    elif apply == "median":
        df[add_to] = reFunc(df[ibase], freq).median()
    elif apply == "var":
        df[add_to] = reFunc(df[ibase], freq).var()
    elif apply == "std":
        df[add_to] = reFunc(df[ibase], freq).std()
    elif apply == "min":
        df[add_to] = reFunc(df[ibase], freq).min()
    elif apply == "max":
        df[add_to] = reFunc(df[ibase], freq).max()
    elif apply == "corr":
        df[add_to] = reFunc(df[ibase], freq).corr()
    elif apply == "cov":
        df[add_to] = reFunc(df[ibase], freq).cov()
    elif apply == "skew":
        df[add_to] = reFunc(df[ibase], freq).skew()
    elif apply == "kurt":
        df[add_to] = reFunc(df[ibase], freq).kurt()

    return add_to


# TODO remove all APPLY functions in pandas. Generally avoid apply whereever possible


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
    pass
