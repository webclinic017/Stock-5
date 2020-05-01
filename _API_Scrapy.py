import pysnowball as ball


import pandas as pd
import numpy as np
import DB,LB



import requests
response = requests.get('http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol=sz000651&t1=all&p=1')
print (response.status_code)
print (response.content)
html=response.content.decode("ascii")
print(html)
print("格力电器" in html)