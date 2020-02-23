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

df = DB.get_asset(ts_code="000002.SZ", asset="E")
print("df", df)

df2 = DB.get_asset(ts_code="000001.SH", asset="I")

df2["rolling"] = df2["close"].rolling(5).corr(df["close"])
print(df2)
df.to_csv("test2.csv")
