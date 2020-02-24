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
df2 = DB.get_asset(ts_code="000001.SZ", asset="E")
for i in range(3000):
    print(i)
    # 0:02:16.116418
    # df3=pd.merge(df,df2,how="left",on="trade_date")

    # 0:02:43.022969
    df[list(df2.columns)] = df2[list(df2.columns)]

# df.to_csv("test2.csv")
