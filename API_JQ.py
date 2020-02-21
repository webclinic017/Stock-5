import tushare as ts
import pandas as pd
import time
import LB
import traceback
from jqdatasdk import *

auth('13817373362', '373362')

if __name__ == '__main__':
    df = get_price('000001.XSHE', start_date='2015-01-01', end_date='2015-01-31 23:00:00', frequency='minute', fields=['open', "high", "low", 'close'])
    print(df)
    df.to_csv("testmindata.csv")
    pass
