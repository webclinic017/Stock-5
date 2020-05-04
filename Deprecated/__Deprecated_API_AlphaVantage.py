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

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

ts = TimeSeries(key='ZYLHZGV7CDAWIHWQ', output_format='pandas')
# data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
data, meta_data = ts.get_daily_adjusted(symbol='MSFT', outputsize='full')
print("finished")
print(data)

data['4. close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.show()
