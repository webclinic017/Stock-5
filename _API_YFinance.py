import tushare as ts
import pandas as pd
import time
import math
import timeit
from datetime import timedelta
import os.path
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import glob
import os
import sys
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from timeit import timeit

import sys
from itertools import groupby
from operator import itemgetter
from timeit import timeit

import pandas as pd
import numpy as np
import time
import math
import talib

import DB
import LB
import yfinance as yf

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def convert_date_tushare_to_yfinance(date):
    result = date[0:4] + "_" + date[4:6] + "_" + date[6:8]
    return result


import yfinance as yf

msft = yf.Ticker("MSFT")

# get stock info
msft.info

# get historical market data
hist = msft.history(period="max")

print("hist", hist)

# show actions (dividends, splits)
msft.actions

print("msft.sustainability", msft.sustainability)
