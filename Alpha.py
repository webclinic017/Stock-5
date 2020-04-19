import math
import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gmean

import DB
import LB
import os
import datetime
import glob
import inspect
import talib
import itertools
from multiprocessing import Process
from scipy import signal
import inspect
import matplotlib.pyplot as plt
import enum
import Plot
from enum import auto
from sklearn.preprocessing import MinMaxScaler
from LB import *

pd.options.mode.chained_assignment = None  # default='warn'

"""
For all functions that create series
1. inplace: modify inplace with standard_name
2. not inplace: modify at a copy and return created series. user can assign new name to original df
"""

"""Point: normalizer for all alpha functions to avoid repetition in code"""


def alpha_norm1(func):
    """
    this constrains that all alpha functions must use kwargs
    standardizes for all alpha creation
    """

    def invisible(*args, **kwargs):
        try:
            # 1. create standard name for all functions based on variables
            if "inplace" not in kwargs:
                kwargs["inplace"] = False
            kwargs["name"] = LB.name_norm(kwargs, func.__name__)

            # 2. swap DF if not modify inplace
            if kwargs["inplace"]:
                # do nothing, modify in place
                pass
            else:
                # inplace holds the original df to be later compared with
                # df is just a copy in order not to modify inplace
                kwargs["inplace"] = kwargs["df"].copy()
                kwargs["df"] = kwargs["df"].copy()
            kwargs["cols"] = kwargs["df"].columns
            return func(*args, **kwargs)
        except:
            print(f"!!! ERROR at {func.__name__}")
            traceback.print_stack()

    # change func name to original name
    result = invisible
    result.__name__ = func.__name__
    return result


def alpha_norm2(d_locals):
    if "inplace" not in d_locals:
        print(d_locals)
        raise AssertionError

    df = d_locals["df"]
    new_cols = list(df.columns.difference(d_locals["cols"]))

    # not inplace=return series
    if type(d_locals["inplace"]) == pd.DataFrame:
        return df[new_cols]

    # inplace=return name
    else:
        if len(new_cols) == 0:
            return d_locals["name"]
        elif len(new_cols) == 1:
            return new_cols[0]
        elif len(new_cols) > 1:
            return new_cols[0]


"""Point: vanilla functions = creation = hardcoded = no abase"""


@alpha_norm1
def co_pct_chg(df, inplace, name, cols):
    df[name] = (df["open"] / df["close"].shift(1))
    return alpha_norm2(locals())


@alpha_norm1
def pjup(df, inplace, name, cols):  # TODO test if 2 pct gap is better
    df[name] = 0
    df[name] = ((df["low"] > df["high"].shift(1)) & (df["pct_chg"] >= 2)).astype(int)  # today low bigger than yesterday high and pct _chg > 2
    return alpha_norm2(locals())


@alpha_norm1
def pjdown(df, inplace, name, cols):
    df[name] = 0
    df[name] = ((df["high"] < df["low"].shift(1)) & (df.pct_chg <= -2)).astype(int)  # yesterday low bigger than todays high and pct _chg < -2
    return alpha_norm2(locals())


@alpha_norm1
def period(df, inplace, name, cols):
    df[name] = range(1, len(df) + 1)
    print(df[name])
    return alpha_norm2(locals())


@alpha_norm1
def ivola(df, inplace, name, cols):
    df[name] = df[["close", "high", "low", "open"]].std(axis=1)
    return alpha_norm2(locals())


"""Point: custom generic vector functions """


# past n days until today. including today
# pct_chg always +1
@alpha_norm1
def pgain(df, abase, freq, inplace, name, cols):
    df[name] = 1 + df[abase].pct_change(periods=freq)
    return alpha_norm2(locals())


# future n days from today on. e.g. open.fgain1 for 20080101 is 20080102/20080101
# CAUTION. if today is signal day, you trade TOMORROW and sell ATOMORROW. Which means you need the fgain1 from tomorrow
# day1: Signal tells you to buy. day2: BUY. day3. SELL
@alpha_norm1
def fgain(df, abase, freq, inplace, name, cols):
    df[name] = df[f"{abase}.pgain{freq.value}"].shift(-int(freq))
    return alpha_norm2(locals())


@alpha_norm1
def pct_chg(df, abase, freq, inplace, name, cols):
    df[name] = (1 + df[abase].pct_change())
    return alpha_norm2(locals())


# possible another version is x.mean()**2/x.std()
@alpha_norm1
def sharp(df, abase, freq, inplace, name, cols):
    df[name] = df[abase].rolling(freq).apply(lambda x: x.mean() / x.std())
    return alpha_norm2(locals())


@alpha_norm1
def comp_chg(df, abase, inplace, name, cols):
    """this comp_chg is for version pct_chg in form of 30 means 30%"""
    df[name] = (1 + (df[abase] / 100)).cumprod()
    return alpha_norm2(locals())


"""NOTE!: This function requires df to have range index and not date!"""


@alpha_norm1
def poly_fit(df, abase, inplace, name, cols, degree=1):
    weights = np.polyfit(df[abase].index, df[abase], degree)
    data = pd.Series(index=df[abase].index, data=0)
    for i, polynom in enumerate(weights):
        data += (polynom * (df[abase].index ** (degree - i)))
    df[name] = data
    return alpha_norm2(locals())


@alpha_norm1
def norm(df, abase, inplace, name, cols, min=0, max=1):
    series_min = df[abase].min()
    series_max = df[abase].max()
    df[name] = (((max - min) * (df[abase] - series_min)) / (series_max - series_min)) + min
    return alpha_norm2(locals())


@alpha_norm1
def FT(df, abase, inplace, name, cols, min=0, max=1, ):
    """Fisher Transform vector
    Mapps value to -inf to inf.
    Creates sharp distance between normal value and extrem values
    """
    # s=normalize_vector(s, min=min, max=max)
    expression = (1.000001 + df[abase]) / (1.000001 - df[abase])
    df[name] = 0.5 * np.log(expression)
    return alpha_norm2(locals())


@alpha_norm1
def IFT(df, abase, inplace, name, cols, min=0, max=1):
    """Inverse Fisher Transform vector
    Mapps value to -1 to 1.
    Creates smooth distance between normal value and extrem values
    """
    # normalize_vector(result, min=min, max=max)
    exp = np.exp(df[abase] * 2)
    df[name] = (exp - 1) / (exp + 1)
    return alpha_norm2(locals())


"""Point: generic apply functions (no generic norm needed)"""


def poly_fit_apply(s):
    """Trend apply = 1 degree polynomials"""
    """ maybe normalize s before apply"""
    index = range(0, len(s))
    return LB.get_linear_regression_slope(index, s)


def norm_apply(series, min=0, max=1):
    series_min = series.min()
    series_max = series.max()
    new_series = (((max - min) * (series - series_min)) / (series_max - series_min)) + min
    return new_series.iat[-1]


def FT_APPYL(s):
    """Fisher Transform apply"""
    return FT(s).iat[-1]


def IFT_Apply(s):
    """Inverse Fisher Transform Apply"""
    return IFT(s).iat[-1]


def sharp_apply(series):
    try:
        return series.mean() / series.std()
    except Exception as e:
        return np.nan


def gmean_apply(series):
    return gmean((series / 100) + 1)


def mean_apply(series):
    return ((series / 100) + 1).mean()


def std_apply(series):
    return ((series / 100) + 1).std()


def mean_std_diff_apply(series):
    new_series = (series / 100) + 1
    series_mean = new_series.mean()
    series_std = new_series.std()
    return series_mean - series_std


"""Point: talib functions"""


# close_today - close_ndays_ago
@alpha_norm1
def mom(df, abase, freq, inplace, name, cols):
    df[name] = talib.MOM(df[abase], timeperiod=freq)
    return alpha_norm2(locals())


@alpha_norm1
def rsi(df, abase, freq, inplace, name, cols):
    df[name] = talib.RSI(df[abase], timeperiod=freq)
    return alpha_norm2(locals())


# ((close_today - close_ndays ago)/close_ndays_ago) * 100
@alpha_norm1
def rocr(df, abase, freq, inplace, name, cols):
    df[name] = talib.ROCR(df[abase], timeperiod=freq)
    return alpha_norm2(locals())


# ((slowMA-fastMA)*100)/fastMA
@alpha_norm1
def ppo(df, abase, freq, freq2, inplace, name, cols):
    df[name] = talib.PPO(df[abase], freq, freq2)
    return alpha_norm2(locals())


@alpha_norm1
def cmo(df, abase, freq, inplace, name, cols):
    df[name] = talib.CMO(df[abase], freq)
    return alpha_norm2(locals())


@alpha_norm1
def apo(df, abase, freq, freq2, inplace, name, cols):
    df[name] = talib.APO(df[abase], freq, freq2)
    return alpha_norm2(locals())


"""Point: pandas Series rolling/expanding alphas"""


@alpha_norm1
def max(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).max()
    return alpha_norm2(locals())


@alpha_norm1
def min(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).min()
    return alpha_norm2(locals())


@alpha_norm1
def median(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).median()
    return alpha_norm2(locals())


@alpha_norm1
def var(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).var()
    return alpha_norm2(locals())


@alpha_norm1
def std(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).std()
    return alpha_norm2(locals())


@alpha_norm1
def skew(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).skew()
    return alpha_norm2(locals())


@alpha_norm1
def kurt(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).kurt()
    return alpha_norm2(locals())


@alpha_norm1
def sum(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).sum()
    return alpha_norm2(locals())


@alpha_norm1
def count(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).count()
    return alpha_norm2(locals())


@alpha_norm1
def mean(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).mean()
    return alpha_norm2(locals())


@alpha_norm1
def cov(df, abase, freq, re, inplace, name, cols):
    df[name] = re(df[abase], freq).cov()
    return alpha_norm2(locals())


# @alpha_norm2 TODO global access
# def corr(df, abase, freq, re, corr_ts_code,inplace,name,cols):
#     df[name] = re(df[abase], freq).corr()
#     return alpha_norm2(locals())


# input a dict with all variables. Output a list of all possible combinations
def explode_settings(d_one_indicator_variables):
    # 1. only get values form above dict
    # 2. create cartesian product of the list
    # 3. create dict out of list

    # first list out all possible choices
    for key, value in d_one_indicator_variables.items():
        print(f"Input ---> {key}: {value}")

    a_product_result = []
    for one_combination in itertools.product(*d_one_indicator_variables.values()):
        d_result = dict(zip(d_one_indicator_variables.keys(), one_combination))
        a_product_result.append(d_result)
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


"""complex functions """


def MESA(df):
    """Using Sine, leadsine to find turning points"""
    ts_code = "000002.SZ"
    df = DB.get_asset(ts_code=ts_code)
    df = df[df.index > 20000101]
    d_mean = {}
    for target in [5, 10, 20, 120, 240]:
        df[f"tomorrow{target}"] = df["open"].shift(-target) / df["open"].shift(-1)
        period_mean = df[f"tomorrow{target}"].mean()
        d_mean[target] = period_mean
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
            mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
            print(f"trend vs cycle mode. trend mode = {mode}. {target} mean {mean}")
    # conclusion. Trendmode earns more money than cycle mode. Trend contributes more to the overall stock gain than cycle mode.

    # uptrend price vs trend
    df_filtered = df[(df["mode"] == 1) & (df["close"] > df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"uptrend mode. {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["close"] < df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"downtrend mode. {target} mean {mean}")
    # conclusion uptrendmode is better than downtrendmode. Downtrend mode is better than Cycle mode which is strange

    # sine vs lead sine
    df_filtered = df[(df["mode"] == 1) & (df["sine"] > df["leadsine"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"sine  above lead {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["sine"] < df["leadsine"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"sine  under lead  {target} mean {mean}")
    # conclusion  uptrend (sine>leadsine) >  uptrend (close>trend) > uptrend

    # ma vs hilbert trendline
    df_filtered = df[(df["mode"] == 1) & (df["kalman"] > df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"kalman over trend {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["kalman"] < df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"kalman under trend {target} mean {mean}")

    df_filtered = df[(df["mode"] == 1) & (df["sine"] > df["leadsine"]) & (df["close"] > df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
        print(f"trend mode. sine  above lead {target} mean {mean}")
    # vs downtrend
    df_filtered = df[(df["mode"] == 1) & (df["sine"] < df["leadsine"]) & (df["close"] < df["trend"])]
    for target in [5, 10, 20, 120, 240]:
        mean = df_filtered[f"tomorrow{target}"].mean() / d_mean[target]
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


@alpha_norm1
def ema(df, abase, freq, inplace, name, cols):
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
    s = df[abase]
    a_result = []
    alpha = 2 / (freq + 1)
    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        elif i > freq - 1:
            result1 = a_result[-1]
            result = alpha * s.iloc[i] + (1 - alpha) * result1  # ehlers formula
            a_result.append(result)
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def ema_re(df, abase, freq, inplace, name, cols):
    """rekursive EMA. Should be same as talib and any other standard EMA"""
    s = df[abase]
    a_result = []

    def my_ema_helper(s, n=freq):
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

    my_ema_helper(s, freq)  # run through and calculate
    final_result = [np.nan] * (freq - 1) + a_result  # fill first n values with nan
    df[name] = final_result
    return alpha_norm2(locals())


@alpha_norm1
def zlema_re(df, abase, freq, gain, inplace, name, cols):
    """rekursive Zero LAG EMA. from john ehlers"""
    s = df[abase]
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

    my_ec_helper(s, freq, gain)  # run through and calculate
    final_result = [np.nan] * (freq - 1) + a_result  # fill first n values with nan
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def zlema(df, abase, freq, gain, inplace, name, cols):
    """iterative Zero LAG EMA. from john ehlers"""
    s = df[abase]
    a_result = []
    k = 2 / (freq + 1)
    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        elif i > freq - 1:
            result1 = a_result[-1]
            if np.isnan(result1):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                close = s.iloc[i]
                result = k * (close + gain * (close - result1)) + (1 - k) * result1  # ehlers formula
                a_result.append(result)
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def butterworth_2p(df, abase, freq, inplace, name, cols):
    """2 pole iterative butterworth. from john ehlers
    butterworth and super smoother are very very similar
    butter_3p >  butter_2p = ss_2p > ss_3p
    """
    s = df[abase]
    a_result = []
    a1 = np.exp(-1.414 * 3.1459 / freq)
    b1 = 2 * a1 * math.cos(1.414 * np.radians(180) / freq)  # when using 180 degree, the super smoother becomes a super high pass filter

    coeff2 = b1
    coeff3 = -a1 * a1
    coeff1 = (1 - b1 + a1 ** 2) / 4

    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1 or i == freq:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def butterworth_3p(df, abase, freq, inplace, name, cols):
    """3 pole iterative butterworth. from john ehlers"""
    s = df[abase]
    a_result = []
    a1 = np.exp(-3.14159 * 3.1459 / freq)
    b1 = 2 * a1 * math.cos(1.738 * np.radians(180) / freq)  # when using 180 degree, the super smoother becomes a super high pass filter
    c1 = a1 ** 2

    coeff2 = b1 + c1
    coeff3 = -(c1 + b1 * c1)
    coeff4 = c1 ** 2
    coeff1 = (1 - b1 + c1) * (1 - c1) / 8

    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1 or i == freq:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def supersmoother_2p(df, abase, freq, inplace, name, cols):
    """2 pole iterative Super Smoother. from john ehlers"""
    s = df[abase]
    a_result = []
    a1 = np.exp(-1.414 * 3.1459 / freq)
    b1 = 2 * a1 * math.cos(1.414 * np.radians(180) / freq)  # when using 180 degree, the super smoother becomes a super high pass filter
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1 or i == freq:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def supersmoother_3p(df, abase, freq, inplace, name, cols):
    """3 pole iterative Super Smoother. from john ehlers
        lags more than 2p , is a little bit smoother. I think 2p is better
    """
    s = df[abase]
    a_result = []
    a1 = np.exp(-3.1459 / freq)
    b1 = 2 * a1 * math.cos(1.738 * np.radians(180) / freq)  # when using 180 degree, the super smoother becomes a super high pass filter
    c1 = a1 ** 2

    coeff2 = b1 + c1
    coeff3 = -(c1 + b1 * c1)
    coeff4 = c1 ** 2
    coeff1 = 1 - coeff2 - coeff3 - coeff4

    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1 or i == freq:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def inst_trend(df, abase, freq, inplace, name, cols):
    """http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/iTrend_Ind.html
    """
    s = df[abase]
    a_result = []
    alpha = 2 / (freq + 1)
    for i in range(0, len(s)):
        if i < freq - 1:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def laguerre_filter_unstable(df, abase, inplace, name, cols):
    """ Non linear laguere is a bit different than standard laguere
        http://www.mesasoftware.com/seminars/TradeStationWorld2005.pdf

        REALLY good, better than any other smoother. Is close to real price and very smooth
        currently best choice
    """
    s = df[abase]
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

    df[name] = (pd.Series(L0) + 2 * pd.Series(L1) + 2 * pd.Series(L2) + pd.Series(L3)) / 6
    return alpha_norm2(locals())


@alpha_norm1
def ehlers_filter_unstable(df, abase, freq, inplace, name, cols):
    """
    version1: http://www.mesasoftware.com/seminars/TradeStationWorld2005.pdf
    version2:  http://www.mesasoftware.com/papers/EhlersFilters.pdf

    Although it contains N, it is a non linear Filter

    two different implementation. Version1 is better because it is more general. version two uses 5 day as fixed date
    has unstable period. FIR filter

    I believe ehlers filter has unstable period because all outcomes for different freqs are the same
    The ehlers filter is WAAAAY to flexible and smooth, much much more flexile than lagguere or EMA, It almost seems like it is a 5 freq, and other are 60 freq. How come?
    """
    s = df[abase]
    s_smooth = (s + 2 * s.shift(1) + 2 * s.shift(2) + s.shift(3)) / 6

    a_result = []
    for i in range(0, len(s)):
        if i < freq:
            a_result.append(np.nan)
        else:
            smooth = s_smooth.iloc[i]
            smooth_n = s_smooth.iloc[i - freq]

            a_coeff = []
            for count in range(0, freq - 1):
                a_coeff.append(abs(smooth - smooth_n))

            num = 0
            sumcoeff = 0
            for count in range(0, freq - 1):
                if not np.isnan(smooth):
                    num = num + a_coeff[count] * smooth
                    sumcoeff = sumcoeff + a_coeff[count]
            result = num / sumcoeff if sumcoeff != 0 else 0
            a_result.append(result)
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def cybercycle(df, abase, freq, inplace, name, cols):
    """Sometimes also called simple cycle. Is an oscilator. not quite sure what it does. maybe alpha between 0 and 1. the bigger the smoother
    https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf

    cybercycle gives very noisy signals compared to other oscilators. maybe I am using it wrong
    """
    s = df[abase]
    s_price = s
    alpha = 0.2
    s_smooth = (s_price + 2 * s_price.shift(1) + 2 * s_price.shift(2) + s_price.shift(3)) / 6
    a_result = []

    for i in range(0, len(s_price)):
        if i < freq + 1:
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
    df[name] = (cycle)  # according to formula. IFT should be used here, but then my amplitude is too small, so I left it away
    return alpha_norm2(locals())


@alpha_norm1
def extract_trend(df, abase, freq, inplace, name, cols):
    """it is exactly same as bandpass filter except the last two lines """
    s = df[abase]
    delta = 0.1
    beta = math.cos(np.radians(360) / freq)
    gamma = 1 / math.cos(np.radians(720) * delta / freq)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq or i == freq + 1:
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

    df[name] = pd.Series(data=a_result)
    df[name] = df[name].rolling(2 * freq).mean()
    return alpha_norm2(locals())


@alpha_norm1
def mode_decomposition(df, abase, s_high, s_low, freq, inplace, name, cols):
    """https://www.mesasoftware.com/papers/EmpiricalModeDecomposition.pdf
    it is exactly same as bandpass filter except the last 10 lines
    I dont understand it really. Bandpass fitler is easier, more clear than this. Maybe I am just wrong
    """
    s = df[abase]
    delta = 0.1
    beta = math.cos(np.radians(360) / freq)
    gamma = 1 / math.cos(np.radians(720) * delta / freq)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq or i == freq + 1:
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

    df[f"{name}_trend"] = pd.Series(data=a_result)
    df[f"{name}_trend"] = df[f"{name}_trend"].rolling(2 * freq).mean()

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

    df[f"{name}_avg_peak"] = pd.Series(a_peak).rolling(freq).mean()
    df[f"{name}_avg_valley"] = pd.Series(a_valley).rolling(freq).mean()
    return alpha_norm2(locals())


@alpha_norm1
def cycle_measure(df, abase, freq, inplace, name, cols):
    """http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/Cycle_Measure.html

    This is just too complicated to calculate, may contain too many errors
    """
    s = df[abase]
    Imult = 0.365
    Qmult = 0.338

    a_inphase = []
    value3 = s - s.shift(freq)

    # calculate inphase
    for i in range(0, len(s)):
        if i < freq:
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
        if i < freq:
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
    df[name] = a_quadrature
    return alpha_norm2(locals())


@alpha_norm1
def highpass(df, abase, freq, inplace, name, cols):
    """high pass. from john ehlers. if frequency too short like n = 2, then it will produce overflow.
    basically you can use high pass filter as an RSI
    everything detrended is an oscilator
    """
    s = df[abase]
    a_result = []
    alpha1 = (math.cos(np.radians(360) * 0.707 / freq) + math.sin(np.radians(360) * 0.707 / freq) - 1) / (math.cos(np.radians(360) * 0.707 / freq))

    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq - 1 or i == freq:
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

    # first n highpass are always too high. make them none in order not to disturb corret values
    # a_result[:n*2] = [np.nan] * n*2
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def roofing_filter(df, abase, hp_freq, ss_freq, inplace, name, cols):
    """  usually hp_n > ss_n. highpass period should be longer than supersmother period.

     1. apply high pass filter
     2. apply supersmoother (= low pass)
     """
    s = df[abase]
    a_result = []
    s_hp = highpass(s, hp_freq)

    a1 = np.exp(-1.414 * 3.1459 / ss_freq)
    b1 = 2 * a1 * math.cos(1.414 * np.radians(180) / ss_freq)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    for i in range(0, len(s)):
        if i < ss_freq - 1:
            a_result.append(np.nan)
        elif i == ss_freq - 1 or i == ss_freq:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def bandpass_filter(df, abase, freq, inplace, name, cols):
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
    s = df[abase]
    delta = 0.9
    beta = math.cos(np.radians(360) / freq)
    gamma = 1 / math.cos(np.radians(720) * delta / freq)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq or i == freq + 1:
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
    df[name] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def bandpass_filter_with_lead(df, abase, freq, inplace, name, cols):
    s = df[abase]
    delta = 0.9
    beta = math.cos(np.radians(360) / freq)
    gamma = 1 / math.cos(np.radians(720) * delta / freq)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)

    a_result = []
    for i in range(0, len(s)):
        if i < freq - 1:
            a_result.append(np.nan)
        elif i == freq or i == freq + 1:
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

    df[name] = pd.Series(a_result)
    df[f"{name}_lead"] = (freq / 6.28318) * (df[name] - df[name].shift(1))
    return alpha_norm2(locals())


@alpha_norm1
def laguerre_RSI(df, abase, inplace, name, cols):
    """
    http://www.mesasoftware.com/papers/TimeWarp.pdf
    Same as laguerre filter, laguerre RSI has unstable period
    It is too sensible and is useable for short term but not for longterm
    swings very sharply to both poles, produces a lot of signals
    """
    s = df[abase]
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

    df[name] = df_helper["CU"] / (df_helper["CU"] + df_helper["CD"])
    return alpha_norm2(locals())


@alpha_norm1
def cg_Oscillator(df, abase, freq, inplace, name, cols):
    """http://www.mesasoftware.com/papers/TheCGOscillator.pdf
    Center of gravity

    The CG oscilator is the only one that is FUCKING RELATIVE to the price
    THIS MEANS you can apply FT and IF while others are not suitable for FT and IFT


    Similar to ehlers filter
    Should be better than conventional RSI
    """
    s = df[abase]
    a_result = []
    for i in range(0, len(s)):
        if i < freq:
            a_result.append(np.nan)
        else:
            num = 0
            denom = 0
            for count in range(0, freq - 1):
                close = s.iloc[i - count]
                if not np.isnan(close):
                    num = num + (1 + count) * close
                    denom = denom + close
            result = -num / denom if denom != 0 else 0
            a_result.append(result)
    df[abase] = a_result
    return alpha_norm2(locals())


@alpha_norm1
def RVI(df, abase, freq, inplace, name, cols):
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

    s_open = df["open"]
    s_close = df["close"]
    s_high = df["high"]
    s_low = df["low"]
    value1 = (s_close - s_open) + 2 * (s_close.shift(1) - s_open.shift(1)) + 2 * (s_close.shift(2) - s_open.shift(2)) + (s_close.shift(3) - s_open.shift(3)) / 6
    value2 = (s_high - s_low) + 2 * (s_high.shift(1) - s_low.shift(1)) + 2 * (s_high.shift(2) - s_low.shift(2)) + (s_high.shift(3) - s_low.shift(3)) / 6

    a_result = []
    for i in range(0, len(s_open)):
        if i < freq:
            a_result.append(np.nan)
        else:
            num = 0
            denom = 0
            for count in range(0, freq - 1):
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

    df[name] = pd.Series(a_result)
    df[f"{name}_sig"] = (RVI + 2 * RVI.shift(1) + 2 * RVI.shift(2) + RVI.shift(3)) / 6
    return alpha_norm2(locals())


@alpha_norm1
def leading(df, abase, freq, inplace, name, cols):
    """
        http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/Leading_Ind.html

        could be just a interpolation of the future price
        Looks just like a high pass of price,
        Not very useful when in cycle mode.

    """
    s = df[abase]
    alpha1 = 0.25
    alpha2 = 0.33

    a_result = []
    for i in range(0, len(s)):
        if i < freq:
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
        if i < freq:
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

    df[name] = a_result
    df[f"{name}_netlead"] = a_netlead
    return alpha_norm2(locals())


"""Point: complex functions """


@alpha_norm1
def cdl(df, abase):
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

    df[abase] = (df[df[a_positive_columns] == 100].sum(axis='columns') + df[df[a_negative_columns] == -100].sum(axis='columns')) / 100
    # IMPORTANT! only removing column is the solution because slicing dataframe does not modify the original df
    columns_remove(df, a_positive_columns + a_negative_columns)
    return abase


@alpha_norm1
def crossma(df, abase, Sfreq1: SFreq, Sfreq2: SFreq):
    add_to = LB.indi_name(abase=abase, deri="crossma", d_variables={"Sfreq1": Sfreq1, "Sfreq2": Sfreq2})
    add_column(df, add_to, abase, 1)
    df[add_to] = (df[abase].rolling(Sfreq1.value).mean() > df[abase].rolling(Sfreq2.value).mean()).astype(float)
    df[add_to] = (df[add_to].diff()).fillna(0)
    return add_to


@alpha_norm1
def overma(df, abase, Sfreq1: SFreq, Sfreq2: SFreq):
    add_to = LB.indi_name(abase=abase, deri="overma", d_variables={"Sfreq1": Sfreq1, "Sfreq2": Sfreq2})
    add_column(df, add_to, abase, 1)
    df[add_to] = (df[abase].rolling(Sfreq1.value).mean() > df[abase].rolling(Sfreq2.value).mean()).astype(float)
    return add_to


@alpha_norm1
def zlmacd(df, abase, sfreq, bfreq, smfreq):
    name = f"{sfreq, bfreq, smfreq}"
    df[f"zlema1_{name}"] = my_best_ec((df[abase]), sfreq)
    df[f"zlema2_{name}"] = my_best_ec((df[abase]), bfreq)

    df[f"zldif_{name}"] = df[f"zlema1_{name}"] - df[f"zlema2_{name}"]
    # df[f"zldea_{name}"]= df[f"zldif_{name}"] -df[f"zldif_{name}"].rolling(smfreq).mean() # ma as smoother, but tradeoff is lag
    df[f"zldea_{name}"] = df[f"zldif_{name}"] - my_best_ec(df[f"zldif_{name}"], smfreq)

    df.loc[df[f"zldea_{name}"] > 0, f"zlmacd_{name}"] = 10
    df.loc[df[f"zldea_{name}"] <= 0, f"zlmacd_{name}"] = -10


@alpha_norm1
def my_ec_it(s, n, gain):
    a_result = []
    k = 2 / (n + 1)
    counter = 0
    for i in range(0, len(s)):
        if i < n - 1:
            counter += 1
            a_result.append(np.nan)
        elif i == n - 1:
            result = s[0:i].mean()  # mean of an array
            a_result.append(result)
        elif i > n - 1:
            last_day_ema = a_result[-1]
            if np.isnan(last_day_ema):  # if the first n days are also nan
                result = s[0:i].mean()  # mean of an array
                a_result.append(result)
            else:
                today_close = s.iloc[i]
                result = k * (today_close + gain * (today_close - last_day_ema)) + (1 - k) * last_day_ema  # ehlers formula
                a_result.append(result)

    return a_result


@alpha_norm1
def my_best_ec(s, n, gain_limit=50):
    least_error = 1000000
    best_gain = 0
    for value1 in range(-gain_limit, gain_limit, 4):

        gain = value1 / 10
        ec = my_ec_it(s, n, gain)
        error = (s - ec).mean()
        print(value1, len(s), len(ec), (error), n)
        if abs(error) < least_error:
            least_error = abs(error)
            best_gain = gain
    best_ec = my_ec_it(s, n, best_gain)
    return best_ec


def support_resistance_horizontal_expansive(start_window=240, rolling_freq=5, step=10, spread=[4, 0.2], bins=10, d_rs={"abv": 10}, df_asset=pd.DataFrame(), delay=3):
    """
    KEY DIFFERENCE BETWEEN THESE TWO CALCULATION:
    1. This one calculates the EXPANDING RS. d.h. top 10 RS from IPO until today.
    2. and it displays all resistance.
    3. By doing this you can calculate how many rs are above or under the price. By increasing the bin, you allow resistance to be closer

    Measure RS:
    1. Based on occurence
    2. Base on how long this price has been max or low or some price

    start_window: when iterating in the past, how big should the minimum window be. not so relevant actually
    rolling_freq: when creating rolling min or max, how long should past be considered
    step: then simulating past rs, how big is the step
    spread: distance rs-price: when creating multiple rs, how big should the spread/distance between price and rs be. The bigger the spread, the more far away they are.
    bins: distance rs-rs: when picking rs from many occurnece, How far should distance between resistance be. You should let algo naturally decide which rs is second strongest and not set limit to them by hand.
    d_rs: how many rs lines you need. Actually deprecated by bins
    df_asset: the df containing close price
    delay: optional


    General idea:
    1. rolling past and find max and minimum
    2. put max and min together in one array
    3. count max occurecne by using bins (cateogirzed)
    4. max occurence value under price is support, max occurence value above price is resistance
    5. iterate through past to simulate past rs for each period

    """

    def support_resistance_acc(abv_und, max_rs, s_minmax, end_date, f_end_date, df_asset):
        # 1. step calculate all relevant resistance = relevant earlier price close to current price
        current_price = df_asset.at[end_date, "close"]

        # 2.step find the max occurence of n values
        print("abv_und is", abv_und)

        s_occurence_bins = s_minmax.value_counts(bins=bins)
        a_rs = []
        for counter, (index, value) in enumerate(s_occurence_bins.iteritems()):
            a_rs.append(index.left)

        # 3. step sort the max occurence values and assign them as rs
        a_rs.sort()  # small value first 0 ,1, 2
        for i, item in enumerate(a_rs):
            if i < max_rs:
                df_asset.loc[end_date:f_end_date, f"rs{abv_und}{i}"] = item

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

            # basically, this is a manual expand, using previous calculated values
            start_date = df_asset.index[0]
            end_date = df_asset.index[row + start_window]
            print("start end", start_date, end_date)
            for abv_und, max_rs in d_rs.items():
                if row + start_window + step > len(df_asset) - 1:
                    break

                f_end_date = df_asset.index[row + start_window + step]
                s_minmax = (s_minall.loc[start_date:end_date]).append(s_maxall.loc[start_date:end_date])  # create an array of all pst min and max
                support_resistance_acc(abv_und=abv_und, max_rs=max_rs, s_minmax=s_minmax, end_date=end_date, f_end_date=f_end_date, df_asset=df_asset)

        # calcualte individual rs score
        for abv_und, count in d_rs.items():
            for i in range(0, count):
                # first fill na for the last period because rolling does not reach
                df_asset[f"rs{abv_und}{i}"].fillna(method="ffill", inplace=True)
                df_asset[f"rs{abv_und}{i}"].fillna(method="bfill", inplace=True)

                df_asset[f"rs{abv_und}{i}_abv"] = (df_asset[f"rs{abv_und}{i}"] > df_asset["close"]).astype(int)
                df_asset[f"rs{abv_und}{i}_cross"] = df_asset[f"rs{abv_und}{i}_abv"].diff().fillna(0).astype(int)

        df_asset["rs_abv"] = 0  # for detecting breakout to top: happens often
        df_asset["rs_und"] = 0  # for detecting breakout to bottom: happens rare
        for abv_und, max_rs in d_rs.items():
            for i in range(0, max_rs):
                try:
                    df_asset[f"rs_{abv_und}"] += df_asset[f"rs{abv_und}{i}_cross"].abs()
                except Exception as e:
                    print("error resistance 2", e)

        # optional #TODO configure it nicely
        df_asset["rs_abv"] = df_asset["rs_abv"].rolling(delay).max().fillna(0)
        df_asset["rs_und"] = df_asset["rs_und"].rolling(delay).max().fillna(0)
    else:
        print(f"df is len is under {start_window}. probably new stock")
        for abv_und, max_rs in d_rs.items():
            for i in range(0, max_rs):  # only consider rs und 2,3,4,5, evenly weighted. middle ones have most predictive power
                df_asset[f"rs{abv_und}{i}"] = np.nan
                df_asset[f"rs{abv_und}{i}_abv"] = np.nan
                df_asset[f"rs{abv_und}{i}_cross"] = np.nan

        df_asset["rs_abv"] = 0
        df_asset["rs_und"] = 0

    return df_asset


def support_resistance_horizontal_responsive(start_window=240, rolling_freq=5, step=10, spread=[4, 0.2], bins=10, d_rs={"abv": 4, "und": 4}, df_asset=pd.DataFrame(), delay=3):
    """
    start_window: when iterating in the past, how big should the minimum window be. not so relevant actually
    rolling_freq: when creating rolling min or max, how long should past be considered
    step: then simulating past rs, how big is the step
    spread: distance rs-price: when creating multiple rs, how big should the spread/distance between price and rs be. The bigger the spread, the more far away they are.
    bins: distance rs-rs: when picking rs from many occurnece, How far should distance between resistance be. You should let algo naturally decide which rs is second strongest and not set limit to them by hand.
    d_rs: how many rs lines you need. Actually deprecated by bins
    df_asset: the df containing close price
    delay: optional


    General idea:
    1. rolling past and find max and minimum
    2. put max and min together in one array
    3. count max occurecne by using bins (cateogirzed)
    4. max occurence value under price is support, max occurence value above price is resistance
    5. iterate through past to simulate past rs for each period

    """

    def support_resistance_acc(abv_und, max_rs, s_minmax, end_date, f_end_date, df_asset):
        # 1. step calculate all relevant resistance = relevant earlier price close to current price
        current_price = df_asset.at[end_date, "close"]

        if abv_und == "abv":
            s_minmax = s_minmax[(s_minmax / current_price < spread[0]) & ((s_minmax / current_price >= 1))]
        elif abv_und == "und":
            s_minmax = s_minmax[(s_minmax / current_price <= 1) & ((s_minmax / current_price > spread[1]))]

        # 2.step find the max occurence of n values
        print("abv_und is", abv_und)

        s_occurence_bins = s_minmax.value_counts(bins=bins)
        a_rs = []
        for counter, (index, value) in enumerate(s_occurence_bins.iteritems()):
            a_rs.append(index.left)

        # 3. step sort the max occurence values and assign them as rs
        a_rs.sort()  # small value first 0 ,1, 2
        for i, item in enumerate(a_rs):
            if i < max_rs:
                df_asset.loc[end_date:f_end_date, f"rs{abv_und}{i}"] = item

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
            print("start end", start_date, end_date)
            for abv_und, max_rs in d_rs.items():
                if row + start_window + step > len(df_asset) - 1:
                    break

                f_end_date = df_asset.index[row + start_window + step]
                s_minmax = (s_minall.loc[start_date:end_date]).append(s_maxall.loc[start_date:end_date])  # create an array of all pst min and max
                support_resistance_acc(abv_und=abv_und, max_rs=max_rs, s_minmax=s_minmax, end_date=end_date, f_end_date=f_end_date, df_asset=df_asset)

        # calcualte individual rs score
        for abv_und, count in d_rs.items():
            for i in range(0, count):
                # first fill na for the last period because rolling does not reach
                df_asset[f"rs{abv_und}{i}"].fillna(method="ffill", inplace=True)
                df_asset[f"rs{abv_und}{i}"].fillna(method="bfill", inplace=True)

                df_asset[f"rs{abv_und}{i}_abv"] = (df_asset[f"rs{abv_und}{i}"] > df_asset["close"]).astype(int)
                df_asset[f"rs{abv_und}{i}_cross"] = df_asset[f"rs{abv_und}{i}_abv"].diff().fillna(0).astype(int)

        df_asset["rs_abv"] = 0  # for detecting breakout to top: happens often
        df_asset["rs_und"] = 0  # for detecting breakout to bottom: happens rare
        for abv_und, max_rs in d_rs.items():
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
        for abv_und, max_rs in d_rs.items():
            for i in range(0, max_rs):  # only consider rs und 2,3,4,5, evenly weighted. middle ones have most predictive power
                df_asset[f"rs{abv_und}{i}"] = np.nan
                df_asset[f"rs{abv_und}{i}_abv"] = np.nan
                df_asset[f"rs{abv_und}{i}_cross"] = np.nan

        df_asset["rs_abv"] = 0
        df_asset["rs_und"] = 0

    return df_asset


# ONE OF THE MOST IMPORTANT KEY FUNNCTION I DISCOVERED
# 1.Step Create RSI or Abv_ma
# 2.Step Create Phase
# 3 Step Create Trend
# 4 Step calculate trend pct_chg
# 5 Step Calculate Step comp_chg
# variables:1. function, 2. threshhold 3. final weight 4. combination with other function
def trend(df, abase, thresh_log=-0.043, thresh_rest=0.7237, market_suffix: str = ""):
    a_all = [1] + c_bfreq()
    a_low = [str(x) for x in a_all][:-1]
    a_high = [str(x) for x in a_all][1:]

    rsi_name = indi_name(abase=abase, deri=f"{market_suffix}rsi")
    phase_name = indi_name(abase=abase, deri=f"{market_suffix}phase")
    trend_name = indi_name(abase=abase, deri=f"{market_suffix}{ADeri.trend.value}")

    func = talib.RSI
    # RSI and CMO are the best. CMO is a modified RSI
    # RSI,CMO,MOM,ROC,ROCR100,TRIX

    # df[f"detrend{abase}"] = signal.detrend(data=df[abase])
    for i in a_all:  # RSI 1
        try:
            if i == 1:
                df[f"{rsi_name}{i}"] = (df[abase].pct_change() > 0).astype(int)
                # df[ rsi_name + "1"] = 0
                # df.loc[(df["pct_chg"] > 0.0), rsi_name + "1"] = 1.0
            else:
                df[f"{rsi_name}{i}"] = func(df[abase], timeperiod=i) / 100

                # normalization causes error
                # df[f"{rsi_name}{i}"] = (df[f"{rsi_name}{i}"]-df[f"{rsi_name}{i}"].min())/ (df[f"{rsi_name}{i}"].max()-df[f"{rsi_name}{i}"].min())
        except Exception as e:  # if error happens here, then no need to continue
            print("error", e)
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
        # TODO MAYBE TREND can be used to score past day gains. Which then can be used to judge other indicators

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


"""Point: Not implemented"""


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


if __name__ == '__main__':
    df = DB.get_asset()
    df = LB.ohlcpp(df).reset_index()
    # df["lol_inplace"]=period(df=df,inplace=False)
    name = poly_fit(df=df, abase="close", inplace=True, degree=4)
    Plot.plot_chart(df, ["close", name])
    pass
