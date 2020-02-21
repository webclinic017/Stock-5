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
    count = "count"
    sum = "sum"
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
        "create": create,
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


class IBase(enum.Enum):
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    pct_chg = "pct_chg"
    # fgain = "fgain"
    # pgain = "pgain"
    pjump_up = "pjump_up"
    pjump_down = "pjump_down"
    ivola = "ivola"

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
    total_share = "total_share"
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

class RE(enum.Enum):
    r = "r"
    e = "e"



def add_ivola(df, df_saved, complete_new_update=True):
    add_to = "ivola"
    add_column(df, add_to, "pct_chg", 1)

    df[add_to] = df[["close", "high", "low", "open"]].std(axis=1)
    for rolling_freq in [2, 5]:
        if complete_new_update:
            df[add_to + str(rolling_freq)] = df[add_to].rolling(rolling_freq).mean()
        else:
            fast_add_rolling(df, add_from=add_to, add_to=add_to + str(rolling_freq), rolling_freq=rolling_freq, func=pd.Series.mean)


def add_pgain(df, rolling_freq, complete_new_update=True):
    add_to = "pgain" + str(rolling_freq)
    add_column(df, add_to, "pct_chg", 1)

    # df[add_to+"test"] = (1 + (df["pct_chg"] / 100)).rolling(rolling_freq).apply(pd.Series.prod, raw=False)
    try:
        df[add_to] = quick_rolling_prod((1 + (df["pct_chg"] / 100)).to_numpy(), rolling_freq)
    except:
        df[add_to] = np.nan


def add_fgain(df, rolling_freq, complete_new_update=True):
    add_to = "fgain" + str(rolling_freq)
    add_column(df, add_to, "pct_chg", 1)
    df[add_to] = df["pgain" + str(rolling_freq)].shift(int(-rolling_freq))


def add_candle_signal(df, complete_new_update=True):
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
                    df[key] = df[key].replace(-100, 100)

            if (array[2] != 0):  # candle used as negative pattern
                a_negative_columns.append(key)
                if (array[2] == 100):  # talib still counts the pattern as positive: cast it negative
                    df[key] = df[key].replace(100, -100)

    df["candle_pos"] = (df[df[a_positive_columns] == 100].sum(axis='columns') / 100)
    df["candle_neg"] = (df[df[a_negative_columns] == -100].sum(axis='columns') / 100)
    df["candle_net_pos"] = (df["candle_pos"] + df["candle_neg"])

    # remove candle stick column
    # IMPORTANT! only removing column is the solution because slicing dataframe does not modify the original df
    columns_remove(df, a_positive_columns + a_negative_columns)

    # last step add rolling
    for rolling_freq in [2, 5]:
        if complete_new_update:
            df["candle_net_pos" + str(rolling_freq)] = df["candle_net_pos"].rolling(rolling_freq).sum()
        else:
            df["candle_net_pos" + str(rolling_freq)] = df["candle_net_pos"].rolling(rolling_freq).sum()
            # fast_add_rolling(df=df, add_from="candle_net_pos",add_to="candle_net_pos"+str(rolling_freq), rolling_freq=rolling_freq, func=pd.Series.sum)


def add_pjump_up(df, complete_new_update=True):
    add_to = "pjump_up"
    add_column(df, add_to, "pct_chg", 1)

    yesterday_high = df["high"].shift(1)
    today_low = df["low"]
    condition_1 = today_low > yesterday_high
    condition_2 = df["pct_chg"] >= 2
    df[add_to] = condition_1 & condition_2
    df[add_to] = df[add_to].astype(int)

    for rolling_freq in [5, 10]:
        if complete_new_update:
            df[add_to + str(rolling_freq)] = df[add_to].rolling(rolling_freq).sum()
        else:
            fast_add_rolling(df=df, add_from=add_to, add_to=add_to + str(rolling_freq), rolling_freq=rolling_freq, func=pd.Series.sum)


def add_pjump_down(df, complete_new_update=True):
    add_to = "pjump_down"
    add_column(df, add_to, "pct_chg", 1)

    yesterday_low = df["low"].shift(1)
    today_high = df["high"]
    condition_1 = today_high < yesterday_low
    condition_2 = df.pct_chg <= -2
    df[add_to] = condition_1 & condition_2
    df[add_to] = df[add_to].astype(int)

    for rolling_freq in [5, 10]:
        if complete_new_update:
            df[add_to + str(rolling_freq)] = df[add_to].rolling(rolling_freq).sum()
        else:
            fast_add_rolling(df=df, add_from=add_to, add_to=add_to + str(rolling_freq), rolling_freq=rolling_freq, func=pd.Series.sum)


def period(df: pd.DataFrame, ibase: str):
    add_to = ibase
    add_column(df, add_to, "ts_code", 1)
    df[add_to] = (range(1, len(df.index) + 1))


def create(df: pd.DataFrame, ibase: str):
    if ibase == "period":
        period(df, ibase)
    elif ibase == "egal":
        pass
    return ibase


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
