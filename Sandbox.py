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

df = pd.read_csv("test.csv")
print("df", df)

df2 = pd.DataFrame(index=df.index, columns=["egal"], data=1)

df2 = pd.concat(objs=[df, df2], sort=False, axis=1)
print(df2)
df.to_csv("test2.csv")
