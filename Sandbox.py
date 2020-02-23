import numpy as np
import cProfile
import LB
import time
import threading
import DB
# set global variable flag
from ICreate import *

# df = DB.get_asset(ts_code="603999.SH",asset="E")
# df=df[["open","close","pct_chg"]]


df = DB.get_asset(ts_code="000001.SH", asset="I")
trend(df=df, ibase="close")

df.to_csv("test2.csv")
