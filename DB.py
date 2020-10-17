import pandas as pd
import numpy as np
import _API_Tushare
import _API_JQ
import _API_Investpy
import _API_Scrapy
import LB
import os.path
import cProfile
from tqdm import tqdm
import traceback
import Alpha
import datetime

pd.options.mode.chained_assignment = None  # default='warn'


# pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
# ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")

def tushare_limit_breaker(func, kwargs, limit=1000):
    """for some reason tushare only allows fund，forex to be given at max 1000 entries per request"""
    df = func(**kwargs)
    len_df_this = len(df)
    df_last = df
    while len_df_this == limit:
        kwargs["end_date"] = df_last.at[len(df_last) - 1, "trade_date"]
        df_this = func(**kwargs)
        if (df_this.equals(df_last)):
            break
        df = df.append(df_this, sort=False, ignore_index=True).drop_duplicates(subset="trade_date")
        len_df_this = len(df_this)
        df_last = df_this
    return df


def update_trade_cal(start_date="19900101", end_date="30250101", market="CN"):
    if market == "CN":
        exchange = "SSE"
    elif market == "HK":
        exchange = "XHKG"

    df = _API_Tushare.my_trade_cal(start_date=start_date, end_date=end_date, exchange=exchange).set_index(keys="cal_date", inplace=False, drop=True)
    LB.to_csv_feather(df, LB.a_path(f"Market/{market}/General/trade_cal_D"))


def update_trade_date(freq="D", market="CN", start_date="00000000", end_date=LB.today()):
    def update_trade_date_stockcount(df):
        for asset in LB.c_asset():
            df_ts_codes = get_ts_code(a_asset=[asset])
            df_ts_codes = df_ts_codes.rename(columns={"list_date": "trade_date"})
            df_grouped = df_ts_codes[["name", "trade_date"]].groupby(by="trade_date").count()
            # vecorized approach faster than loop over individual date
            df[f"{asset}_count"] = df_grouped["name"].astype(int).cumsum()
            df[f"{asset}_count"] = df[f"{asset}_count"].fillna(method="ffill")
            df[f"{asset}_count"] = df[f"{asset}_count"].fillna(0)
        return df

    def update_trade_date_indexcount(df):
        df_ts_codes = get_ts_code(a_asset=["E"])
        df_ts_codes = df_ts_codes.rename(columns={"list_date": "trade_date"})

        for index in ["主板","中小板","创业板","科创板"]:
            df_helper=df_ts_codes[df_ts_codes["market"]==index]
            df_grouped = df_helper[["name", "trade_date"]].groupby(by="trade_date").count()
            # vecorized approach faster than loop over individual date
            index_name=LB.c_index_name()[index]
            df[f"{index_name}_count"] = df_grouped["name"].astype(int).cumsum()
            df[f"{index_name}_count"] = df[f"{index_name}_count"].fillna(method="ffill")
            df[f"{index_name}_count"] = df[f"{index_name}_count"].fillna(0)

        return df


    def update_trade_date_seasonal_score(df_trade_date, freq="D", market="CN"):
        """this requires the date_seasonal matrix to be updated first """
        # get all indicator for each day
        path_indicator = "Market/CN/Atest/seasonal/all_date_seasonal.xlsx"
        xls = pd.ExcelFile(path_indicator)

        # get all different groups
        a_groups = [[LB.trade_date_to_datetime_dayofweek, "dayofweek"],  # 1-5
                    [LB.trade_date_to_datetime_m, "monthofyear"],  # 1-12
                    [LB.trade_date_to_datetime_d, "dayofmonth"],  # 1-31
                    [LB.get_trade_date_datetime_weekofyear, "weekofyear"],  # 1-51
                    [LB.trade_date_to_dayofyear, "dayofyear"],  # 1-365
                    [LB.trade_date_to_datetime_s, "seasonofyear"],  # 1-365
                    ]

        # transform all trade_date into different format
        for group in a_groups:
            df_trade_date[group[1]] = df_trade_date["trade_date"].apply(lambda x: group[0](x)).astype(str)
            df_score_board = pd.read_excel(xls, sheet_name=group[1], converters={group[1]: lambda x: str(x)})
            df_score_board = df_score_board.fillna(0.0)  # if no information about certain period, assume the change is 0
            df_score_board[group[1]] = df_score_board[group[1]].astype(str)

            df_score_board = df_score_board.rename(columns={"pct_chg": f"{group[1]}_score"})
            df_trade_date[group[1]] = df_trade_date[group[1]].astype(str)
            df_trade_date = pd.merge(df_trade_date, df_score_board, how='left', on=group[1], suffixes=["", ""], sort=False)
            df_trade_date[f"{group[1]}_score"] = df_trade_date[f"{group[1]}_score"].fillna(0.0)  # if information not enough, assume the pct_chg is 0.0

        df_trade_date["seasonal_score"] = df_trade_date["dayofweek_score"] * 0.15 + \
                                          df_trade_date["dayofmonth_score"] * 0.25 + \
                                          df_trade_date["weekofyear_score"] * 0.20 + \
                                          df_trade_date["dayofyear_score"] * 0.05 + \
                                          df_trade_date["monthofyear_score"] * 0.35

        df_trade_date["seasonal_score"] = df_trade_date["seasonal_score"].rolling(2).mean()
        return df_trade_date



    def update_ny(df):
        # calculate new year on gregorian calendar
        # find all dates that are not trading
        df_trade_cal_D = get_trade_cal_D(market=market, a_is_open=[0, 1])

        # count the non trading sequence
        df_trade_cal_D["is_open_counter"] = Alpha.consequtive_count(df=df_trade_cal_D, abase="is_open", count=0, inplace=False)

        # shift 1 because we want to find the first day AFTER HOLIDAZ
        df_trade_cal_D["is_open_counter"] = df_trade_cal_D["is_open_counter"].shift(1)
        df_trade_cal_D.to_csv("df_trade_cal_D.csv")

        # search for the highest non trading sequence in JAN or FEB for each year
        df_trade_cal_D = LB.trade_date_to_calender(df_trade_cal_D)
        df_trade_cal_D = df_trade_cal_D[df_trade_cal_D["month"] < 4]

        # find all distinct years
        s_years = df_trade_cal_D["year"].unique()

        df["new_year"] = 0

        # end of NY: loop over each year to find the max non trading sequence
        for year in s_years:
            df_this_year = df_trade_cal_D[df_trade_cal_D["year"] == year]
            NY_end = df_this_year["is_open_counter"].idxmax()
            df.at[str(NY_end), "new_year"] = -1

        # start of NY:
        copy_helper = df["new_year"].copy()
        copy_helper = copy_helper.shift(-1)
        copy_helper = copy_helper * (-1)
        df["new_year"] += copy_helper
        df["new_year"] = df["new_year"].fillna(0)

        return df

    # here function starts
    if freq in ["D"]:
        a_path = LB.a_path(f"Market/{market}/General/trade_date_{freq}")
        df = get_trade_cal_D(market=market)
        df = LB.df_between(df, start_date, end_date)

        if market == "CN":
            df = LB.trade_date_to_calender(df) # add year, month, day, weekofmonth etc
            df = update_trade_date_stockcount(df)  # adds E,I,FD count
            df = update_trade_date_indexcount(df)  # adds sh,sz,cy count
            df = update_ny(df) # add new year data

        LB.to_csv_feather(df, a_path, index_relevant=True)


def update_ts_code(asset="E", market="CN", night_shift=True):
    print("start update ts_code ", market, asset)
    if market == "CN":
        if asset == "E":
            df = _API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="").set_index("ts_code")

            # add SW industry info for each stock
            for level in [1, 2, 3]:
                df_member = get_ts_code(a_asset=[f"sw_industry{level}"])
                df[f"sw_industry{level}"] = df_member[f"sw_industry{level}"]

            # add JQ industry info for each stock
            for level in [1, 2]:
                df_member = get_ts_code(a_asset=[f"jq_industry{level}"])
                df_member = df_member[~df_member.index.duplicated(keep="first")]
                df[f"jq_industry{level}"] = df_member[f"jq_industry{level}"]

            # add ZJ industry info for each stock
            for level in [1]:
                df_member = get_ts_code(a_asset=[f"zj_industry{level}"])
                df[f"zj_industry{level}"] = df_member[f"zj_industry{level}"]

            # add concept
            df_concept = get_ts_code(a_asset=["concept"])
            df_grouped_concept = df_concept.groupby("ts_code").agg(lambda column: ", ".join(column))
            df["concept"] = df_grouped_concept["concept"]
            df["concept_code"] = df_grouped_concept["code"]

            # add State Government for each stock
            df["state_company"] = False
            #for now exclude them
            """for ts_code in df.index:
                print("update state_company", ts_code)
                df_government = get_asset(ts_code=ts_code, freq="top10_holders", market=market, a_columns=["holder_name", "hold_ratio"])
                if df_government.empty:  # if empty, assume it is False
                    continue
                df_government_grouped = df_government.groupby(by="holder_name").mean()
                df_government_grouped = df_government_grouped["hold_ratio"].nlargest(n=1)  # look at the top 4 share holders

                counter = 0
                for top_holder_name in df_government_grouped.index:
                    if ("公司" in top_holder_name) or (len(top_holder_name) > 3):
                        counter += 1
                df.at[ts_code, "state_company"] = True if counter >= 1 else False
"""
        elif asset == "I":
            df_SSE = _API_Tushare.my_index_basic(market='SSE')
            df_SZSE = _API_Tushare.my_index_basic(market='SZSE')
            df = df_SSE.append(df_SZSE, sort=False).set_index("ts_code")

        elif asset == "FD":
            df_E = _API_Tushare.my_fund_basic(market='E')
            df_O = _API_Tushare.my_fund_basic(market='O')
            df = df_E.append(df_O, sort=False).set_index("ts_code")

        elif asset == "G":
            df = pd.DataFrame()
            for on_asset in LB.c_asset():
                for group, a_instance in LB.c_d_groups(assets=[on_asset]).items():
                    for instance in a_instance:
                        df.at[f"{group}_{instance}", "name"] = f"{group}_{instance}"
                        df.at[f"{group}_{instance}", "on_asset"] = on_asset
                        df.at[f"{group}_{instance}", "group"] = str(group)
                        df.at[f"{group}_{instance}", "instance"] = str(instance)
            df.index.name = "ts_code"

        elif asset == "F":
            df = _API_Tushare.my_fx_daily(start_date=LB.today())
            df["name"] = df["ts_code"]
            df = df.set_index("ts_code")
            df = df.loc[~df.index.duplicated(keep="last")]
            df = df[["name"]]

        elif asset == "B":
            # 可转债，相当于股票。可以选择换成股票，也可以选择换成利率。网上信息很少，几乎没人玩
            df = _API_Tushare.my_cb_basic().set_index("ts_code")
            # 中债，国债
            # only yield curve, no daily data

        elif asset in ["sw_industry1", "sw_industry2", "sw_industry3"]:
            # industry member list
            df_member = _API_Tushare.my_index_classify(f"L{asset[-1]}")
            df_member = df_member.rename(columns={"industry_name": f"{asset}"}).set_index("index_code")

            # industry instance
            a_df_instances = []
            for index in df_member.index:
                df_instance = _API_Tushare.my_index_member(index)
                df_instance.rename(columns={"con_code": "ts_code"}, inplace=True)
                a_df_instances.append(df_instance)
            df = pd.concat(a_df_instances, sort=False)
            df = pd.merge(df, df_member, how="left", on=["index_code"], suffixes=[False, False], sort=False)
            df = df.set_index("ts_code", drop=True)
            LB.to_csv_feather(df, a_path=LB.a_path(f"Market/{market}/General/ts_code_{asset}"))
            return

        elif asset in ["zj_industry1", "jq_industry1", "jq_industry2"]:
            d_name = {"zj_industry1": "zjw", "jq_industry1": "jq_l1", "jq_industry2": "jq_l2"}
            df_member = _API_JQ.my_get_industries(name=d_name[asset])
            a_df_instances = []
            for industry_code in df_member.index:
                a_industry = _API_JQ.my_get_industry_stocks(industry_code)
                a_industry = [LB.df_switch_ts_code(x) for x in a_industry]
                df_industry = pd.DataFrame(index=a_industry)
                df_industry.index.name = "ts_code"
                df_industry["index_code"] = industry_code
                df_industry[asset] = df_member.at[industry_code, "name"]
                df_industry["start_date"] = df_member.at[industry_code, "start_date"]
                a_df_instances.append(df_industry)
            df = pd.concat(a_df_instances, sort=False)
            LB.to_csv_feather(df, a_path=LB.a_path(f"Market/{market}/General/ts_code_{asset}"))
            return

        elif asset == "concept":
            df_member = _API_Tushare.my_concept()
            a_concepts = []
            for code in df_member["code"]:
                df_instance = _API_Tushare.my_concept_detail(id=code)
                a_concepts.append(df_instance)
            df = pd.concat(a_concepts, sort=False)

            # remove column name it is in both df
            df_member_col = [x for x in df_member if x not in df.columns]
            df["code"] = df["id"]
            df = pd.merge(df, df_member[df_member_col], how="left", on=["code"], suffixes=[False, False], sort=False)

            # change name with / to:
            df.rename(columns={"concept_name": "concept"}, inplace=True)
            df["concept"] = df["concept"].str.replace(pat="/", repl="_")
            df = df.set_index("ts_code", drop=True)
            LB.to_csv_feather(df, a_path=LB.a_path(f"Market/{market}/General/ts_code_concept"))
            return

    # HK MARKET
    elif market == "HK":
        if asset == "E":
            df = _API_Tushare.my_hk_basic(list_status="L").set_index("ts_code")
            df = df.loc[:, df.columns != "curr_type"]

    elif market == "US":
        """interestingly enough, not one single API or website can give you a list
            instead, they all try to load bit by bit which is very annoying
            -finviz has E industry, sector but no FD,I
            -investing all E,FD,D but no industry
            -nasdaqtraded has E but no industry but need manual download
            -tradingview has industry,sector, but need premium to download, infinite scroll    
        """
        if "I" == asset:
            df = _API_Investpy.my_get_indices(country="united states")
            df.columns = ["country", "name", "full_name", "ts_code", "currency", "class", "market"]
        elif "E" == asset:
            df = _API_Investpy.my_get_stocks(country="united states")
            df.columns = ["country", "name", "full_name", "isin", "currency", "ts_code"]
        elif "FD" == asset:
            # df_fund = _API_Investpy.my_get_funds(country="united states")
            # df_fund["etf"]=False

            #for now, only consider etf as fund
            df_etf = _API_Investpy.my_get_etf(country="united states")
            df=df_etf
            df = df.rename({'symbol': 'ts_code'}, axis='columns')

        df = df.set_index("ts_code")

    df["asset"] = asset
    if "list_date" not in df.columns:
        df["list_date"] = np.nan
    df["list_date"] = df["list_date"].fillna(method='ffill') if asset not in ["G", "F"] else np.nan
    a_path = LB.a_path(f"Market/{market}/General/ts_code_{asset}")
    LB.to_csv_feather(df, a_path)


def update_asset_US(asset="E", step=1, api="investpy"):
    """maybe in future merge both functions. For NOW this is minimized version"""
    if api == "yfinance":
        import _API_YFinance

    df_ts_code = pd.read_csv(f"Market/US/General/ts_code_{asset}.csv").set_index("ts_code")
    df_ts_code = df_ts_code.loc[~df_ts_code.index.duplicated(keep="last")]

    for ts_code, name in zip(df_ts_code.index[::step],df_ts_code["name"][::step]):
        print(f"investpy download US step{step}", asset, ts_code)
        a_path = LB.a_path(f"Market/US/Asset/{asset}/D/{ts_code}")
        if os.path.isfile(a_path[1]):
            continue

        try:
            if api == "investpy":
                d_func = {"I": _API_Investpy.my_index_historical_data,
                          "E": _API_Investpy.my_stock_historical_data,
                          "FD": _API_Investpy.my_etf_historical_data }

                """really weird. Stock use ticker, index and fund use name"""
                d_symbol={"I":name, "E":ts_code, "FD":name }
                df = d_func[asset](d_symbol[asset], "united states", "01/01/1880", LB.trade_date_to_investpy(LB.today()))
                df = df.rename({'Open': 'open', 'High': 'high',"Low":"low","Close":"close","Volume":"vol","Currency":"currency"}, axis='columns')


            elif api == "yfinance":
                df, sector, industry = _API_YFinance.download_asset(ts_code)
                df.columns = ["open", "high", "low", "close", "vol", "dv", "split"]
            else:
                df = pd.DataFrame()

        except Exception as e:
            print(e)
            continue

        # post processing to standardize names
        df["period"] = Alpha.period(df=df, inplace=False)
        df["pct_chg"] = df["close"].pct_change() * 100
        df["trade_date"] = df.index
        # df["trade_date"]=df["trade_date"].apply(LB.switch_trade_date)
        df["trade_date"] = df["trade_date"].dt.strftime('%Y%m%d')
        df = df.set_index("trade_date", drop=True)
        LB.to_csv_feather(df=df, a_path=a_path, skip_csv=True)


def update_asset_CNHK(asset="E", freq="D", market="CN", step=1, night_shift=True, miniver=True):
    def merge_saved_df_helper(df, df_saved):
        df.set_index("trade_date", inplace=True)
        df.index = df.index.astype(int)  # IMPORTANT
        df_saved.index = df_saved.index.astype(int)  # IMPORTANT therewise there will be a str 20180101 and int 20180101 in index
        return df_saved.append(df, sort=False)
        # df = LB.df_drop_duplicated_reindex(df, "trade_date")

    def set_index_helper(df):
        df.set_index(keys="trade_date", drop=True, inplace=True)
        df = df.loc[~df.index.duplicated(keep="last")]  # important
        df = df[df.index.notna()]
        df.index = df.index.astype(int)
        return df

    def helper_EI(ts_code, freq, asset, start_date, end_date, adj="qfq", market="CN"):
        if ".SZ" in ts_code or ".SH" in ts_code:
            df = _API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
        elif ".HK" in ts_code:
            df = _API_Tushare.my_hk_daily(ts_code=ts_code, start_date=str(start_date), end_date=str(end_date))

        df = LB.df_reverse_reindex(df)
        LB.df_columns_remove(df, ["pre_close", "change"])
        return df

    def helper_F(ts_code, start_date, end_date):
        df = tushare_limit_breaker(func=_API_Tushare.my_fx_daily, kwargs={"ts_code": ts_code, "start_date": f"{start_date}", "end_date": f"{end_date}"}, limit=1000)
        df = LB.df_reverse_reindex(df)
        for column in ["open", "high", "low", "close"]:
            df[column] = (df[f"bid_{column}"] + df[f"ask_{column}"]) / 2
        df["pct_chg"] = df["close"].pct_change() * 100
        return df[["trade_date", "ts_code", "open", "high", "low", "close", "pct_chg", "tick_qty"]]

    def helper_FD(ts_code, freq, asset, start_date, end_date, adj=None, market="CN"):
        if ".OF" in str(ts_code):
            # 场外基金
            df = tushare_limit_breaker(func=_API_Tushare.my_fund_nav, kwargs={"ts_code": ts_code}, limit=1000)
            LB.df_reverse_reindex(df)

            df_helper = pd.DataFrame()
            for ohlc in ["open", "high", "low", "close"]:
                df_helper[ohlc] = df["unit_nav"].astype(int)
            df_helper["trade_date"] = df["ann_date"]
            df_helper["pct_chg"] = df_helper["open"].pct_change()
            df = df_helper
            return LB.df_reverse_reindex(df)

        else:
            # 场内基金
            df = tushare_limit_breaker(func=_API_Tushare.my_fund_daily, kwargs={"ts_code": ts_code, "start_date": f"{start_date}", "end_date": f"{end_date}"}, limit=1000)
            if df.empty:
                print("not good")

            """
            不复权 = 无
            前复权 = 当日收盘价 × 当日复权因子 / 最新复权因子 (This is the standard Xueqiu also uses)
            后复权 = 当日收盘价 x 当日复权因子
            Fund 前复权was tested and compared to xueqiu and it was CORRECT
            """
            if not df.empty:
                df.index = df["trade_date"]
                df_adj_factor = _API_Tushare.my_query(api_name='fund_adj', ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df_adj_factor.empty:
                    print(asset, ts_code, freq, "has no adj_factor, skip")
                else:
                    df_adj_factor.index = df_adj_factor.index.astype(int)
                    latest_adj = df_adj_factor.at[0, "adj_factor"]
                    df_adj_factor.index = df_adj_factor["trade_date"]
                    df["adj_factor"] = df_adj_factor["adj_factor"]
                    df["adj_factor"] = df["adj_factor"].interpolate()  # interpolate between alues because some dates are missing from tushare

                    # df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * df.adj_factor / latest_adj
                    for column in ["open", "high", "low", "close"]:
                        # for debug include 不复权and后复权
                        # df[f"{column}_不复权"]=df[column]
                        # df[f"{column}_后复权"]=df[column] * df["adj_factor"]
                        df[column] = df[column] * df["adj_factor"] / latest_adj

                df = LB.df_reverse_reindex(df)
                LB.remove_columns(df, ["pre_close", "amount", "change", "adj_factor"])
            return df

    # add technical indicators. but us stock still might use it .maybe change later
    def update_asset_point(df, asset):
        Alpha.ivola(df=df, inplace=True)  # 0.890578031539917 for 300 loop
        Alpha.period(df=df, inplace=True)  # 0.2 for 300 loop
        Alpha.pjup(df=df, inplace=True)  # 1.0798187255859375 for 300 loop
        Alpha.pjdown(df=df, inplace=True)  # 1.05 independend for 300 loop

    # For all Pri indices and derivates. ordered after FILO
    def update_asset_re(df, asset):
        LB.time_counter()
        # close pgain fgain
        for freq in [5, 10, 20, 60, 120, 240, 500]:
            Alpha.pgain(df=df, abase="close", freq=freq, inplace=True)  # past gain includes today = today +yesterday comp_gain
        for freq in [5, 10, 20, 60, 120, 240, 500]:
            Alpha.fgain(df=df, abase="close", freq=freq, inplace=True)  # future gain does not include today = tomorrow+atomorrow comp_gain
        LB.time_counter("close fgain")
        # 52w high and low
        for freq in [20, 60, 240, 500]:
            df[f"{freq}d_high"] = Alpha.max(df=df, abase="close", freq=freq, re=pd.Series.rolling, inplace=False)
        for freq in [20, 60, 240, 500]:
            df[f"{freq}d_low"] = Alpha.min(df=df, abase="close", freq=freq, re=pd.Series.rolling, inplace=False)
        LB.time_counter("52w high/low")
        # open pgain fgain
        for freq in [1, 2, 5]:
            Alpha.pgain(df=df, abase="open", freq=freq, inplace=True)  # past gain includes today = today +yesterday comp_gain
        for freq in [1, 2, 5]:
            Alpha.fgain(df=df, abase="open", freq=freq, inplace=True)  # future gain does not include today = tomorrow+atomorrow comp_gain
        LB.time_counter("open fgain")
        # counts how many percent of past n days are positive
        for freq in [5, 10, 20]:
            Alpha.pos(df=df, abase="pct_chg", freq=freq, inplace=True)
        LB.time_counter("pos")
        # alpha and Beta, lower the better
        if asset in ["I", "E", "FD"]:
            for ts_code in LB.c_index():
                for freq in [5, 20, 60, 240]:
                    Alpha.ab(df=df, abase="close", freq=freq, re=pd.Series.rolling, vs=ts_code, inplace=True)
        LB.time_counter("beta")
        # sharp ratio
        for freq in [5, 20, 60, 240]:
            Alpha.sharp(df=df, abase="pct_chg", re=pd.Series.rolling, freq=freq, inplace=True)
        Alpha.sharp(df=df, abase="pct_chg", re=pd.Series.expanding, freq=240, inplace=True)
        LB.time_counter("sharp")
        # gmean. 5 sec too slow
        # for freq in [5,20, 60, 240][::-1]:
        #     Alpha.geomean(df=df, abase="pct_chg", re=pd.Series.rolling, freq=freq, inplace=True)
        # Alpha.geomean(df=df, abase="pct_chg", re=pd.Series.expanding, freq=240, inplace=True)
        # LB.time_counter("geomean")
        # std
        for freq in [5, 20, 60, 240]:
            Alpha.std(df=df, abase="pct_chg", re=pd.Series.rolling, freq=freq, inplace=True)
        LB.time_counter("std")
        # is_max
        emax = Alpha.max(df=df, abase="close", re=pd.Series.expanding, freq=240, inplace=True)
        Alpha.ismax(df=df, abase="close", emax=emax, inplace=True)
        LB.time_counter("ismax")
        # is_min
        emin = Alpha.min(df=df, abase="close", re=pd.Series.expanding, freq=240, inplace=True)
        Alpha.ismin(df=df, abase="close", emin=emin, inplace=True)
        LB.time_counter("ismin")
        # entropy(waaay to slow)
        # for freq in [5, 20, 60, 240]:
        #     Alpha.entropy(df=df, abase="close", re=pd.Series.rolling, freq=freq, inplace=True)
        # Alpha.entropy(df=df, abase="close", re=pd.Series.expanding, freq=240, inplace=True)
        # LB.time_counter("entropy")
        # ta
        for sfreq, bfreq in [(10, 20), (20, 60), (120, 240), (240, 500)]:
            Alpha.macd(df=df, abase="close", freq=sfreq, freq2=bfreq, type=1, score=10, inplace=True)
        LB.time_counter("macd")
        for freq in [5, 20, 60, 240]:
            Alpha.rsi(df=df, abase="close", freq=freq, inplace=True)
        LB.time_counter("rsi")
        for freq in [5, 20, 60, 240]:
            Alpha.abv_ma(df=df, abase="close", freq=freq, inplace=True)
        LB.time_counter("abv_ma")
        Alpha.laguerre_rsi_u(df=df, abase="close", inplace=True)
        LB.time_counter("laguerre")
        for freq in [5, 20, 60, 240]:
            Alpha.cj(df=df, abase="close", freq=freq, inplace=True)
        LB.time_counter("cj")
        # Testing - 高送转
        # 1. total share/ first day share
        # 2. difference between today share and tomorrow share
        if "total_share" in df.columns:
            df["norm_total_share"] = df["total_share"] / df.at[df["total_share"].first_valid_index(), "total_share"]
            df["pct_total_share"] = df["norm_total_share"].pct_change()
            df["e_pct_total_share"] = df["pct_total_share"].expanding(240).mean()
        LB.time_counter("share")

        # 3. trend swap. how long a trend average lasts too slow
        # for freq in a_freqs:
        #     #expanding
        #     for until_index,df_expand in custom_expand(df,freq).items():
        #         df.at[until_index,f"e_abv_ma_days{freq}"]=LB.trend_swap(df_expand, f"abv_ma{freq}", 1)
        # print(f"{ts_code} abv_ma_days finished")


    #MAIN FUNCTION STARTS HERE
    def run(df_ts_codes):
        # iteratve over ts_code
        for ts_code in df_ts_codes.index[::step]:
            start_date, middle_date, end_date, complete_new_update, df_saved = ("00000000", "20050101", LB.today(), True, pd.DataFrame())  # True means update from 00000000 to today, False means update latest_trade_date to today

            ts_code = str(ts_code)
            # sometimes using slash can cause feather unicode error invalid continuation of byte
            # happens if asset=FD because /FD somehow produces unicode char

            # standard
            a_path = LB.a_path(f"Market/{market}/Asset/{asset}/{freq}/{ts_code}")

            # using raw string
            # a_path = LB.a_path(r"Market/"+market+r"/Asset/"+asset+r"/"+freq+r"/"+ts_code)

            # using double slash or backslash
            # a_path = LB.a_path(f"Market//{market}//Asset//{asset}//{ts_code}")

            # file exists--> check latest_trade_date, else update completely new
            if os.path.isfile(a_path[1]):  # or os.path.isfile(a_path[1])

                try:
                    df_saved = get_asset(ts_code=ts_code, asset=asset, freq=freq, market=market)  # get latest file trade_date
                    asset_latest_trade_date = str(df_saved.index[-1])
                except:
                    asset_latest_trade_date = start_date
                    print(asset, ts_code, freq, end_date, "EMPTY - START UPDATE", asset_latest_trade_date, " to today")

                # file exist and on latest date--> finish, else update
                if (int(asset_latest_trade_date) == int(real_latest_trade_date)):
                    print(asset, ts_code, freq, end_date, "Up-to-date", real_latest_trade_date)
                    continue
                else:  # file exists and not on latest date
                    # file exists and not on latest date, AND stock trades--> update
                    print(asset, ts_code, freq, end_date, "Exist but not on latest date", real_latest_trade_date)

                    #NORMALLY: USE OLD WAY TO UPDATE ONLY LAST FEW ROWS
                    #NEW WAY: STILL DO COMPLETE NEW UPDATE IF LAST DATE IS NOT CORRECT
                    # currently I don't truss the manual update, the manual update even takes longer than rewrite the asset with new data.
                    # no need to delete file as the new result will override it

                    complete_new_update = True # new way
                    #complete_new_update = False #old way

            else:
                print(asset, ts_code, freq, end_date, "File Not exist. UPDATE!", real_latest_trade_date)

            # file not exist or not on latest_trade_date --> update
            if (asset == "E" and freq == "D"):
                # 1.1 get df
                if complete_new_update:
                    df1 = helper_EI(ts_code, freq, asset, start_date, middle_date, adj="hfq")
                    df2 = helper_EI(ts_code, freq, asset, middle_date, end_date, adj="hfq")
                    df = df1.append(df2, ignore_index=True, sort=False)
                    df = set_index_helper(df)
                else:
                    df = helper_EI(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="hfq")

                # only for CN
                if market == "CN":
                    # 1.2 get adj factor because tushare is too dump to calculate it on its own
                    df_adj = _API_Tushare.my_query(api_name='adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
                    if df_adj.empty:
                        print(asset, ts_code, freq, start_date, end_date, "has no adj_factor yet. skipp")
                    else:
                        latest_adj = df_adj.at[0, "adj_factor"]
                        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / latest_adj


                    #skip fundamentals and other indicator in minified version
                    if not miniver:
                        # 2.1 get daily basic
                        if complete_new_update:
                            df_fun_1 = _API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=start_date, end_date=middle_date)
                            df_fun_2 = _API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=middle_date, end_date=end_date)
                            df_fun = df_fun_1.append(df_fun_2, ignore_index=True, sort=False)
                        else:
                            df_fun = _API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=asset_latest_trade_date, end_date=end_date)

                        try:  # new stock can cause error here
                            df_fun = df_fun[["trade_date", "turnover_rate", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_share", "total_mv"]]
                            df_fun["total_share"] = df_fun["total_share"] * 10000
                            df_fun["total_mv"] = df_fun["total_mv"] * 10000
                            df_fun["trade_date"] = df_fun["trade_date"].astype(int)
                            df = pd.merge(df, df_fun, how='left', on=["trade_date"], suffixes=[False, False], sort=False).set_index("trade_date")
                        except Exception as e:
                            print("error fun", e)


                        # 2.2 add FUN financial report aka fina
                        # 流动资产合计,非流动资产合计,资产合计，   流动负债合计,非流动负债合计,负债合计
                        # old df_balancesheet = get_asset_E_D_Fun("balancesheet", ts_code=ts_code, columns=["total_cur_asset", "total_asset", "total_cur_liab", "total_liab"])
                        df_balancesheet = get_asset(ts_code=ts_code, freq="balancesheet", a_columns=["total_cur_asset", "total_asset", "total_cur_liab", "total_liab"])

                        # 营业活动产生的现金流量净额	，投资活动产生的现金流量净额,筹资活动产生的现金流量净额
                        # old df_cashflow = get_asset_E_D_Fun("cashflow", ts_code=ts_code, columns=["n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"])
                        df_cashflow = get_asset(ts_code=ts_code, freq="cashflow", a_columns=["n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"])

                        # 扣除非经常性损益后的净利润,净利润同比增长(netprofit_yoy instead of q_profit_yoy on the doc)，营业收入同比增长,销售毛利率，销售净利率,资产负债率,存货周转天数(should be invturn_days,but casted to turn_days in the api for some reasons)
                        # old df_indicator = get_asset_E_D_Fun("fina_indicator", ts_code=ts_code, columns=["profit_dedt", "netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_asset", "turn_days"])
                        df_indicator = get_asset(ts_code=ts_code, freq="fina_indicator", a_columns=["profit_dedt", "netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_asset", "turn_days"])

                        # 股权质押比例
                        # old df_pledge_stat = get_asset_pledge_stat(ts_code=ts_code, columns=["pledge_ratio"])
                        df_pledge_stat = get_asset(ts_code=ts_code, freq="W_pledge_stat", a_columns=["pledge_ratio"])

                        # add fina to df
                        for df_fun, fun_name in zip([df_balancesheet, df_cashflow, df_indicator, df_pledge_stat], ["bala", "cash", "indi", "pledge"]):
                            df_fun.index.name = "trade_date"
                            df_fun.index = df_fun.index.astype(int)
                            df[list(df_fun.columns)] = df_fun[list(df_fun.columns)]
                            # df = pd.merge(df, df_fun, how="left", on="trade_date", sort=False).set_index("trade_date")  # just added might be wrong

                        # append old df and drop duplicates
                        if not df_saved.empty:
                            df = df_saved.append(df, sort=False)
                        df = df.loc[~df.index.duplicated(keep="last")]  # important

                        # interpolate/fill between empty fina and pledge_stat values
                        all_report_label = list(df_balancesheet.columns.values) + list(df_cashflow.columns.values) + list(df_indicator.columns.values) + ["pledge_ratio"]
                        for label in all_report_label:
                            try:
                                df[label] = df[label].fillna(method='ffill')
                            except:
                                pass


            elif (asset == "I" and freq == "D"):
                if complete_new_update:
                    df = helper_EI(ts_code, freq, asset, start_date, end_date, adj="qfq")
                    df = set_index_helper(df)
                else:
                    df = helper_EI(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="qfq")
                    df = merge_saved_df_helper(df=df, df_saved=df_saved)

            elif (asset == "FD" and freq == "D"):
                if complete_new_update:
                    df = helper_FD(ts_code, freq, asset, start_date, end_date)
                    df = set_index_helper(df)
                else:
                    df = helper_FD(ts_code, freq, asset, asset_latest_trade_date, end_date)
                    df = merge_saved_df_helper(df=df, df_saved=df_saved)
            elif (asset == "F" and freq == "D"):
                if complete_new_update:
                    df = helper_F(ts_code, start_date, end_date)
                    df = set_index_helper(df)
                else:
                    df = helper_F(ts_code, asset_latest_trade_date, end_date)
                    df = merge_saved_df_helper(df=df, df_saved=df_saved)

            # duplicated index ward
            df = df.loc[~df.index.duplicated(keep="first")]

            """the problem above is df(new) has trade date as column. df_saved has trade_date as index right?"""
            # 3. add my derivative indices and save it
            if not df.empty and not miniver:
                update_asset_point(df=df, asset=asset)
                update_asset_re(df=df, asset=asset)
            else:
                #add period for all stocks
                Alpha.period(df=df, inplace=True)

            LB.to_csv_feather(df=df, a_path=a_path, skip_csv=True)  # save space
            print(asset, ts_code, freq, end_date, "UPDATED!", real_latest_trade_date)
            print("=" * 50)
            print()

    # REAL FUNCTION START

    # init
    df_ts_codes = get_ts_code(a_asset=[asset], market=market)

    # TODO needs to adjusted for other markets like hk and us
    real_latest_trade_date = get_last_trade_date(freq, market=market)

    # if index does not exist, update index first:
    d_index = preload(asset="I", d_queries_ts_code=LB.c_index_queries())
    if len(d_index) < 3:
        previousasset = asset
        asset = "I"
        run(df_ts_codes=pd.DataFrame(index=LB.c_index(), data=LB.c_index()))
        asset = previousasset

    run(df_ts_codes=df_ts_codes)


def update_asset_G(asset=["E"], night_shift=True, step=1):
    """
    for each possible G instance:
        for each member of G instance:
            add together and save
    """
    df_ts_code = get_ts_code(a_asset=["E"])
    d_preload = preload(asset="E")
    example_column = get_example_column(asset="E", freq="D", numeric_only=True, notna=False)

    # preparation. operations that has to be done for all stocks ONCE
    for ts_code, df_asset in d_preload.items():
        print("pre processing ts_code", ts_code)

        df_nummeric = LB.df_to_numeric(df_asset)

        df_nummeric["divide_helper"] = 1  # counts how many asset are there at one day
        d_preload[ts_code] = df_nummeric

    # loop over all stocks
    for group, a_instance in LB.c_d_groups(assets=["E"]).items():
        for instance in a_instance:

            # get member of that instance
            if group == "concept":
                # TODO check in and out concept date
                df_members = df_ts_code[df_ts_code["concept"].str.contains(instance) == True]
            else:
                df_members = df_ts_code[df_ts_code[group] == instance]


            # loop over all members individually
            df_instance = pd.DataFrame()
            for counter, ts_code in enumerate(df_members.index):
                try:
                    df_asset = d_preload[ts_code]
                except Exception as e:
                    pass  # in df_ts_code but not in preload. maybe not met condition of preload like period > 240

                if not df_asset.empty:
                    print(f"creating group: {group, instance}: {counter} - {ts_code} - len {len(df_asset)}")
                    # all_columns=df_asset.columns
                    # df_asset = LB.get_numeric_df(df_asset)
                    # not_numeric_columns = [x for x in all_columns if x not in df_asset.columns]
                    # print(not_numeric_columns)

                    # added in init preload
                    # df_asset["divide_helper"] = 1  # counts how many asset are there at one day

                    #in case there are inf or na values.
                    df_asset=df_asset.replace([np.inf, -np.inf], np.nan)

                    #add together
                    df_instance = df_instance.add(df_asset, fill_value=0)


            # put all instances together
            if not df_instance.empty:
                df_instance = df_instance.div(df_instance["divide_helper"], axis=0)
                df_instance = df_instance[example_column]  # align columns
                df_instance.insert(0, "ts_code", f"{group}_{instance}")

            #if this is not the case, then to feather will be error
            df_instance.index=df_instance.index.astype(int)

            LB.to_csv_feather(df_instance, LB.a_path(f"Market/CN/Asset/G/D/{group}_{instance}"), skip_csv=True)
            print(f"{group}_{instance} UPDATED")


def update_asset_G2(asset=["E"], night_shift=True, step=1):
    """
    DEPRECATED!

    there are 2 approach to do this
    1. Asset to industry group and then create date

    This approach is using procedure 2. But both approach should produce same result
    Requirement: df_date
    """

    LB.interrupt_start()

    # initialize all group as dict
    d_group_instance_update = {}  # dict of array
    d_group_instance_saved = {}  # dict of array
    for group, a_instance in LB.c_d_groups(asset=asset).items():
        print("what", group, a_instance)
        for instance in a_instance:
            d_group_instance_update[f"{group}_{instance}"] = []
            d_group_instance_saved[f"{group}_{instance}"] = get_asset(f"{group}_{instance}", asset="G")

    # # get last saved trade_date on df_saved
    # last_trade_date = get_last_trade_date("D")
    # example_df = d_group_instance_saved["asset_E"]
    # try:
    #     last_saved_date = example_df.index[-1]
    # except:
    #     last_saved_date = "19990101"
    #
    #
    # # if small update and saved_date==last trade_date
    # if last_trade_date == last_saved_date and (not night_shift):
    #     return print("ALL GROUP Up-to-date")
    #
    # #     # initialize trade_date
    df_trade_date = get_trade_date(end_date=LB.today(), freq="D")
    # df_trade_date = df_trade_date[df_trade_date.index > int(last_saved_date)]
    # print("START UPDATE GROUP since", last_saved_date)
    #
    #     # create a dict for concept in advance
    #     d_concept = {}
    #     df_grouped_concept = get_ts_code(a_asset=["concept"]).groupby("concept")
    #     for concept, row in df_grouped_concept:
    #         d_concept[concept] = row.index
    #
    #     # loop over date and get mean
    for trade_date in df_trade_date.index:  # for each day

        if LB.interrupt_confirmed():
            break

        print(trade_date, "Updating GROUP")
        df_date = get_date(trade_date=trade_date, a_asset=asset, freq="D")
        for group, a_instance in LB.c_d_groups(asset=asset).items():  # for each group

            # one stock can have many concept groups. 1:N Relation
            if group == "concept":
                continue
                for instance in a_instance:
                    try:
                        df_date_grouped = df_date.loc[d_concept[instance]]
                    except Exception as e:
                        print(f"{trade_date} {group} {instance}: {e}")
                        continue

                    if df_date_grouped.empty:
                        continue
                    else:
                        row = df_date_grouped.mean()
                        d_group_instance_update[f"{group}_{instance}"].append(row)

            # one stock can only be in one group. 1:1 Relation
            else:
                df_date_grouped = df_date.groupby(by=group).mean()
                for instance, row in df_date_grouped.iterrows():
                    if f"{group}_{instance}" != "asset_G":
                        d_group_instance_update[f"{group}_{instance}"].append(row)

    # save all to df
    for (key, a_update_instance), (key_saved, df_saved) in zip(d_group_instance_update.items(), d_group_instance_saved.items()):
        df_update = pd.DataFrame(a_update_instance)
        if df_update.empty:
            print(key, "EMPTY. CONTINUE")
            continue
        else:
            df_update.set_index(keys="trade_date", drop=True, inplace=True)  # reset index after group

        if not df_saved.empty:
            df_update = pd.concat(objs=[df_saved, df_update], sort=False, ignore_index=True)

        if "ts_code" not in df_update:
            df_update.insert(0, "ts_code", key)
        LB.to_csv_feather(df_update, LB.a_path(f"Market/CN/Asset/G/D/{key}"))
        print(key, "UPDATED")


def update_asset_stock_market_all(start_date="00000000", end_date=LB.today(), asset=["E"], freq="D", market="CN", comparison_index=["000001.SH", "399001.SZ", "399006.SZ"], night_shift=False):
    """ In theory this should be exactly same as Asset_E
    """  # TODO make this same as asset_E

    # check if big update and if the a previous saved file exists
    last_saved_date = "19990104"
    last_trade_date = get_last_trade_date("D")
    if not night_shift:
        try:
            df_saved = get_stock_market_all()
            last_saved_date = df_saved.index[len(df_saved) - 1]
            df_saved = df_saved.reset_index()
            print(last_saved_date)

            if last_saved_date == last_trade_date:
                print("Date_Base Up-to_date")
                return
        except:
            df_saved = pd.DataFrame()
    else:
        df_saved = pd.DataFrame()

    print("last saved day is", last_saved_date)

    # meta preparation
    a_path = LB.a_path(f"Market/{market}/Asset/G/D/all_stock_market")
    a_result = []
    df_sh_index = get_asset(ts_code="000001.SH", asset="I", freq=freq, market="CN")
    df_sh_index = df_sh_index.loc[int(last_saved_date):int(last_trade_date)]
    df_sh_index = df_sh_index.loc[df_sh_index.index > int(last_saved_date)]  # exclude last saved date from update

    # loop through all trade dates and add them together
    for trade_date, sh_pct_chg in zip(df_sh_index.index, df_sh_index["pct_chg"]):
        print(trade_date, "being added to all_stock_market_base")
        df_date = get_date(str(trade_date), a_asset=asset, freq=freq, market="CN")
        df_date = df_date[df_date["period"] >= 240]  # IMPORTANT disregard ipo stocks

        trading_stocks = len(df_date)
        df_date_mean = df_date.mean()
        df_date_mean["trading"] = trading_stocks
        df_date_mean["trade_date"] = trade_date

        df_date_mean["up_limit"] = len(df_date[df_date["pct_chg"] >= 8.0])
        df_date_mean["down_limit"] = len(df_date[df_date["pct_chg"] <= -8.0])
        df_date_mean["net_limit_ratio"] = (df_date_mean["up_limit"] - df_date_mean["down_limit"]) / trading_stocks

        df_date_mean["winner"] = len(df_date[df_date["pct_chg"] > 0]) / trading_stocks
        df_date_mean["loser"] = len(df_date[df_date["pct_chg"] < 0]) / trading_stocks
        df_date_mean["beat_sh_index"] = len(df_date[df_date["pct_chg"] >= sh_pct_chg]) / trading_stocks

        a_result.append(df_date_mean)

    # array to dataframe
    df_result = pd.DataFrame(a_result)

    # if small update, append new data to old data
    if (not df_saved.empty) and (not night_shift):
        df_result = df_saved.append(df_result, sort=False)

    # add comp chg and index
    df_result["comp_chg"] = Alpha.comp_chg(df=df_result, abase="pct_chg", inplace=False)
    for ts_code in comparison_index:
        df_result = add_asset_comparison(df=df_result, freq=freq, asset="I", ts_code=ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"])
        df_result[f"comp_chg_{ts_code}"] = Alpha.comp_chg(df=df_result, abase=f"pct_chg_{ts_code}", inplace=False)

    LB.to_csv_feather(df_result, a_path, index_relevant=False)
    print("Date_Base UPDATED")



def update_asset_bundle(bundle_name, bundle_func, market="CN", night_shift=True, a_asset=["E"], step=1, skip_csv=True):
    """updates other information from tushare such as blocktrade,sharefloat
    check out LB.c_asset_E_bundle() for more
    """
    df_ts_code = get_ts_code(a_asset=a_asset, market=market)
    for ts_code in df_ts_code.index[::step]:
        a_path = LB.a_path(f"Market/{market}/Asset/{a_asset[0]}/{bundle_name}/{ts_code}")
        if os.path.isfile(a_path[1]) and (not night_shift):
            print(ts_code, bundle_name, "Up-to-date")
        else:
            df = bundle_func(ts_code)
            if not df.empty:
                df = LB.df_reverse_reindex(df)
                df = LB.set_index(df,set_index="ts_code")
                LB.to_csv_feather(df=df, a_path=a_path, skip_csv=skip_csv)
                print(ts_code, bundle_name, "UPDATED")


def update_asset_qdii():
    _API_Scrapy.qdii_research()
    _API_Scrapy.qdii_grade()


def update_asset_xueqiu(asset="E", market="CN"):
    """loads xueqiu data from jq"""
    df_ts_code = get_ts_code(a_asset=[asset])
    for ts_code in df_ts_code.index:
        print(f"update xueqiu {ts_code}")
        a_path = LB.a_path(f"Market/{market}/Asset/E/xueqiu_raw/{ts_code}")
        if not os.path.isfile(a_path[1]):
            code = LB.df_switch_ts_code(ts_code)
            df_xueqiu = _API_JQ.break_jq_limit_helper_xueqiu(code=code)

            if df_xueqiu.empty:
                continue
            df_xueqiu["trade_date"] = df_xueqiu["day"].apply(LB.df_switch_trade_date)
            df_xueqiu["trade_date"] = df_xueqiu["trade_date"].astype(int)
            df_xueqiu = df_xueqiu[["trade_date", "follower", "new_follower", "discussion", "new_discussion", "trade", "new_trade"]]
            df_xueqiu = df_xueqiu.set_index("trade_date", drop=True)

            # df = pd.merge(df_asset, df_xueqiu, on="trade_date", how="left", sort=False)
            a_path = LB.a_path(f"Market/{market}/Asset/E/xueqiu/{ts_code}")
            LB.to_csv_feather(df_xueqiu, a_path=a_path)
            LB.to_csv_feather(df=df_xueqiu, a_path=a_path)


def update_asset_intraday(asset="I", freq="15m"):
    df_ts_code = ["000001.SH", "399001.SZ", "399006.SZ"]
    for ts_code in df_ts_code:
        print(f"update intraday {asset, freq, ts_code}")
        a_path = LB.a_path(f"Market/CN/Asset/{asset}/{freq}/{ts_code}")
        if not os.path.isfile(a_path[1]):
            jq_code = LB.df_switch_ts_code(ts_code)
            df = _API_JQ.my_get_bars(jq_code=jq_code, freq=freq)
            df["ts_code"] = ts_code
            df = df.set_index("date", drop=True)
            LB.to_csv_feather(df=df, a_path=a_path)


def update_asset_beta_table(asset1="I", asset2="E", freq=240):
    """
    This calculates pairwise beta between all stocks for all days
    for all trade_dates
        for all asset
            for all past beta frequencies

    Mini version = all 3 index with E,FD,G
    """

    # init preload both type of asset
    if asset1 == "G":
        d_query1 = LB.c_G_queries()
    elif asset1 == "I":
        d_query1 = LB.c_index_queries()
    else:
        d_query1 = {}

    if asset2 == "G":
        d_query2 = LB.c_G_queries()
    elif asset2 == "I":
        d_query2 = LB.c_index_queries()
    else:
        d_query2 = {}

    d_preload1 = preload(asset=asset1, step=1, d_queries_ts_code=d_query1)
    d_preload2 = preload(asset=asset2, step=1, d_queries_ts_code=d_query2)

    # loop over and calculate corr
    for asset_counter1, (ts_code1, df_asset1) in enumerate(d_preload1.items()):
        a_path = LB.a_path(f"Market/CN/Asset/Beta/{asset1}_{asset2}/{freq}/{ts_code1}")
        if os.path.isfile(a_path[1]):
            continue

        df_result = df_asset1[["close"]]
        for asset_counter2, (ts_code2, df_asset2) in enumerate(d_preload2.items()):
            print(f"freq: {freq}. beta: {ts_code1} - {ts_code2}")
            df_result[ts_code2] = df_result["close"].rolling(freq, min_periods=2).corr(df_asset2["close"])

        # create summary
        print(df_result.index)
        for column in df_result.columns:
            try:
                df_result.at[00000000, column] = df_result[column].mean()
            except:
                pass
        LB.to_csv_feather(df=df_result, a_path=a_path, skip_csv=True)


def update_date(asset="E", freq="D", market="CN", night_shift=True, step=1, start_date="000000", end_date=LB.today(), naive=False):
    """step -1 might be wrong if trade dates and asset are updated seperately. then they will not align
        step 1 always works
        naive: approach always works, but is extremly slow
    """
    if step not in [1, -1]:
        raise BaseException("STEP only 1 or -1 !!!!")

    # latest example column
    example_column = get_example_column(asset=asset, freq=freq, market=market, numeric_only=True, notna=False)

    # init df
    df_static_data = get_ts_code(a_asset=[asset])
    df_trade_dates = get_trade_date(start_date=start_date, end_date=end_date, freq=freq, market=market)

    # init dict
    d_list_date = {ts_code: row["list_date"] for ts_code, row in get_ts_code(a_asset=[asset]).iterrows()}
    d_queries_ts_code = LB.c_G_queries() if asset == "G" else {}
    d_preload = preload(asset=asset, freq=freq, step=1, period_abv=240, d_queries_ts_code=d_queries_ts_code)
    d_lookup_table = {ts_code: (0 if step == 1 else len(df) - 1) for ts_code, df in d_preload.items()}

    for trade_date in df_trade_dates.index[::step]:  # IMPORTANT! do not modify step, otherwise lookup will not work
        a_path = LB.a_path(f"Market/{market}/Date/{asset}/{freq}/{trade_date}")
        a_date_result = []

        # date file exists AND not night_shift. If night_shift, then always overwrite
        if os.path.isfile(a_path[1]) and (not night_shift):

            if naive:  # fallback strategies is the naive approach
                continue
            else:
                # update lookup table before continue. So that skipped days still match
                for ts_code, df_asset in d_preload.items():
                    row_number = d_lookup_table[ts_code]
                    if step == 1 and row_number > len(df_asset) - 1:
                        continue
                    if step == -1 and row_number < 0:
                        continue
                    if int(df_asset.index[row_number]) == int(trade_date):
                        d_lookup_table[ts_code] += step
                print(asset, freq, trade_date, "date file Up-to-date")

        # date file does not exist or night_shift
        else:
            for counter, (ts_code, df_asset) in enumerate(d_preload.items()):

                if naive:
                    try:
                        a_date_result.append(df_asset.loc[trade_date].to_numpy().flatten())
                    except:
                        # it is totally normal to have not that day for each df_asset
                        pass
                else:
                    if len(df_asset) == 0:
                        print(f"{trade_date} skip ts_code {ts_code} because len is 0")
                        continue

                    # if asset list date is in future: not IPO yet
                    list_date = d_list_date[ts_code]
                    if type(list_date) in [str, int]:
                        if int(list_date) > int(trade_date):
                            # print(f"{trade_date} skip ts_code {ts_code} because IPO in future")
                            continue

                    row_number = d_lookup_table[ts_code]  # lookup table can not be changed while iterating over it.

                    # this df_asset is already at last row. Is finished
                    if step == 1 and row_number > len(df_asset) - 1:
                        continue
                    if step == -1 and row_number < 0:
                        continue

                    if int(df_asset.index[row_number]) == int(trade_date):

                        a_date_result.append(df_asset.loc[trade_date].to_numpy().flatten())
                        d_lookup_table[ts_code] += step
                        # print(f"IN {trade_date} counter {counter}, step {step}, ts_code {ts_code}, len {len(df_asset)}  row number {row_number}")
                    else:
                        pass
                        # print(f"{int(df_asset.index[row_number])}",trade_date)
                        # print(f"OUT {trade_date} counter {counter}, step {step}, ts_code {ts_code}, len {len(df_asset)}  row number {row_number} associated date {int(df_asset.index[row_number])}")

            # create df_date from a_date_result
            df_date = pd.DataFrame(data=a_date_result, columns=example_column)

            # remove duplicate columns that also exist in static data. Then merge
            no_duplicate_cols = df_date.columns.difference(df_static_data.columns)
            df_date = pd.merge(df_date[no_duplicate_cols], df_static_data, how='left', on=["ts_code"], suffixes=["", ""], sort=False).set_index("ts_code")  # add static data
            df_date.insert(loc=0, column='trade_date', value=int(trade_date))

            LB.to_csv_feather(df_date, a_path, skip_csv=True)
            print(asset, freq, trade_date, "date updated")


# measures which day of week/month performs generally better
#TODO add seasonal stats
def update_date_seasonal_stats(group_instance="asset_E"):
    """this needs to be performed first before using the seasonal matrix"""
    path = f"Market/CN/Atest/seasonal/all_date_seasonal_{group_instance}.xlsx"
    pdwriter = pd.ExcelWriter(path, engine='xlsxwriter')

    # perform seasonal stats for all stock market or for some groups only
    df_group = get_stock_market_all().reset_index() if group_instance == "" else get_asset(ts_code=group_instance, asset="G").reset_index()

    # get all different groups
    a_groups = [[LB.trade_date_to_datetime_dayofweek, "dayofweek"],
                [LB.trade_date_to_datetime_d, "dayofmonth"],
                [LB.get_trade_date_datetime_weekofyear, "weekofyear"],
                [LB.trade_date_to_dayofyear, "dayofyear"],
                [LB.trade_date_to_datetime_m, "monthofyear"],
                [LB.trade_date_to_datetime_s, "seasonofyear"], ]

    # transform all trade_date into different date format
    for group in a_groups:
        df_group[group[1]] = df_group["trade_date"].apply(lambda x: group[0](x))

    # OPTIONAL filter all ipo stocks and pct_chg greater than 10
    df_group = df_group[(df_group["pct_chg"] < 11) & (df_group["pct_chg"] > -11)]

    # create and sort single groups
    for group in a_groups:
        df_result = df_group[[group[1], "pct_chg"]].groupby(group[1]).agg(["mean", "std", Alpha.apply_gmean, Alpha.apply_mean, Alpha.apply_std, Alpha.apply_mean_std_diff])
        df_result.to_excel(pdwriter, sheet_name=group[1], index=True, encoding='utf-8_sig')

    pdwriter.save()


# TODO should be replaced and removed by tushare

# def update_date_margin_total():
#     """jq total margin of the whole stock market together"""
#     df_trade_date = get_trade_date()
#     a_result = []
#     for trade_date in df_trade_date.index:
#         print(f"update_margin_total {trade_date}")
#         a_path = LB.a_path(f"Market/CN/Asset/E/margin_total/{trade_date}")
#         if not os.path.isfile(a_path[1]):
#             df_margintotal = _API_JQ.my_margin(date=trade_date)
#             LB.to_csv_feather(df_margintotal, a_path=a_path)
#         else:
#             df_margintotal = get(a_path=a_path)
#         a_result.append(df_margintotal)
#
#     df_result = pd.concat(a_result, sort=False)
#     df_sh = get_asset("000001.SH", asset="I")
#     df_sh = LB.ohlcpp(df_sh)
#     for market in [f"XSHE", f"XSHG"]:
#         df_market = df_result[df_result["exchange_code"] == market]
#         df_market["trade_date"] = df_market["date"].apply(LB.switch_trade_date)
#         df_market["trade_date"] = df_market["trade_date"].astype(int)
#         df_market = pd.merge(df_sh, df_market, how="left", on="trade_date", sort=False)
#         a_path = LB.a_path(f"Market/CN/Asset/E_Market/margin_total/{market}")
#         LB.to_csv_feather(df=df_market, a_path=a_path)


def update_date_news(a_news=["jq_cctv_news", "ts_major_news", "ts_cctv"]):
    """used and tested, but does not seem to be very predictive, not used atm"""
    d_news = {"jq_cctv_news": _API_JQ.my_cctv_news,
              "ts_major_news": _API_Tushare.my_major_news,
              "ts_cctv": _API_Tushare.my_cctv_news}

    df_sh = get_asset("000001.SH", asset="I")
    df_sh = df_sh[["open", "high", "low", "close"]]

    df_trade_date = get_trade_date()
    for news in a_news:
        # downloads news for every trade_date
        for today in df_trade_date.index[::-1]:
            print(today)
            a_path = LB.a_path(f"Market/CN/Date/News/{news}/{today}")
            if not os.path.isfile(a_path[1]):
                df = d_news[news](LB.df_switch_trade_date(today))
                if not df.empty:
                    LB.to_csv_feather(df=df, a_path=a_path, skip_feather=True)

        # count and aggregate the news to one file
        for trade_date in df_sh.index:
            df_news = get_date(trade_date, a_asset=["E"], freq=news)
            str_df = df_news.to_string()
            for keyword in ["股市", "股份", "股份公司", "增长", "好", "涨幅"]:
                df_sh.at[trade_date, keyword] = str_df.count(keyword) / len(str_df)
        LB.to_csv_feather(df=df_sh, a_path=LB.a_path(f"Market/CN/Date/E/{news}/count/count"))


def get(a_path=[], set_index=""):  # read feather first
    """bottle neck function to get all df at one place"""
    for counter, func in [(1, pd.read_feather), (0, pd.read_csv)]:
        try:
            return LB.set_index(func(a_path[counter]), set_index=set_index)
        except Exception as e:
            print(f"read error {func.__name__}", e)
    else:
        print("DB READ File Not Exist!", f"{a_path[0]}.feather")
        return pd.DataFrame()

def get_trade_cal_D(start_date="00000000", end_date="30000000", a_is_open=[1], market="CN"):
    df = get(LB.a_path(f"Market/{market}/General/trade_cal_D"), set_index="cal_date")
    df.index.name = "trade_date"
    df.index= df.index.astype(int)
    return df[(df["is_open"].isin(a_is_open)) & (df.index >= int(start_date)) & (df.index <= int(end_date))]


def get_trade_date(start_date="000000", end_date=LB.today(), freq="D", market="CN"):
    df = get(LB.a_path(f"Market/{market}/General/trade_date_{freq}"), set_index="trade_date")
    # return df[(df.index >= int(start_date)) & (df.index <= int(end_date))]
    df= LB.df_between(df=df, start_date=start_date, end_date=end_date)
    df.index=df.index.astype(int)
    return df

def get_last_trade_date(freq="D", market="CN", type=str):
    df_trade_date = get_trade_date(start_date="00000000", end_date=LB.today(), freq=freq, market=market)

    #Depends on the hour: if it is 2am morning, then it is latest trade date, but market has not opended yet.
    today_hour=datetime.datetime.now().hour
    if today_hour > 16:
        # after 1 hour of stock close. Should get latest data
        return type(df_trade_date.index[-1])
    else:
        #return the yesterdays trade date as the latest
        return type(df_trade_date.index[-2])


def get_next_trade_date(freq="D", market="CN"):
    df = get_trade_cal_D(a_is_open=[1])
    last_trade_date = get_last_trade_date(freq, market)
    df = df[df.index > int(last_trade_date)].reset_index()
    return df.at[0, "trade_date"]



def get_ts_code(a_asset=["E"], market="CN", d_queries={}):
    """
    d_query: only entries that are TRUE.
    Example: {"E": ["industry1 == '医疗设备'", "period > 240 "]}"""
    a_result = []
    for asset in a_asset:
        df = get(LB.a_path(f"Market/{market}/General/ts_code_{asset}"), set_index="ts_code")
        if df.empty:
            continue

        if (asset == "FD") and market == "CN":
            df = df[df["delist_date"].isna()]
            # df = df[df["type"]=="契约型开放式"] #契约型开放式 and 契约型封闭式 都可以买 在线交易，封闭式不能随时赎回，但是可以在二级市场上专卖。 开放式更加资本化，发展的好可以扩大盘面，发展的不好可以随时赎回。所以开放式的盘面大小很重要。越大越稳重
            df = df[df["market"] == "E"] #Only consider ETF and LOF because they can be publicly traded

        if d_queries:
            a_queries = d_queries[asset]
            for query in a_queries:
                # when query index use name or "index"? A: both are working
                df = df.query(query)

        a_result.append(df)
    return pd.concat(a_result, sort=False) if a_result else pd.DataFrame()


def get_asset(ts_code="000002.SZ", asset="E", freq="D", market="CN", by_ts_code=True, a_columns=[]):
    """higher level function to get df_asset by ts_code"""
    if not by_ts_code:  # costly and should avoided whereas possible
        result = df_ts_code[df_ts_code["name"] == ts_code]
        ts_code = list(result.index)[0]
    df = get(LB.a_path(f"Market/{market}/Asset/{asset}/{freq}/{ts_code}"))

    if a_columns:
        try:
            return df[a_columns]
        except:
            return df
    else:
        return df


def get_date(trade_date, a_asset=["E"], freq="D", market="CN"):  # might need get_date_all for efficiency
    a_df = []
    for asset in a_asset:
        a_df.append(get(LB.a_path(f"Market/{market}/Date/{asset}/{freq}/{trade_date}"), set_index="ts_code"))
    return pd.concat(a_df, sort=False) if len(a_df) > 1 else a_df[0]





def get_stock_market_all(market="CN"):
    return get(LB.a_path(f"Market/{market}/Asset/G/D/all_stock_market"), set_index="trade_date")

def get_stock_market_overview(market="CN",fields=["close","vol"]):
    df_sh=get_asset(ts_code="000001.SH",asset="I")
    df_sz=get_asset(ts_code="399001.SZ",asset="I")
    df_cy=get_asset(ts_code="399006.SZ",asset="I")
    df_sh=df_sh[fields].rename(columns={key:f"{key}_sh" for key in fields})
    df_sz=df_sz[fields].rename(columns={key:f"{key}_sz" for key in fields})
    df_cy=df_cy[fields].rename(columns={key:f"{key}_cy" for key in fields})
    df_result= pd.merge(df_sh, df_sz, how='left', on="trade_date", suffixes=["", ""], sort=False)
    df_result= pd.merge(df_result, df_cy, how='left', on="trade_date", suffixes=["", ""], sort=False)
    return df_result

def get_example_column(asset="E", freq="D", numeric_only=False, notna=True, market="CN"):
    """get the latest column of the asset file"""
    if asset == "E":
        ts_code = "000001.SZ"
    elif asset == "I":
        ts_code = "000001.SH"
    elif asset == "FD":
        ts_code = "150008.SZ"
    elif asset == "F":
        ts_code = "AUDCAD.FXCM"
    elif asset == "G":
        ts_code = "area_安徽"
    else:
        ts_code = "000001.SZ"

    df = get_asset(ts_code, asset, freq, market=market)
    if notna:
        df = df.dropna(how="all", axis=1)

    # nummeric only or not
    return list(LB.df_to_numeric(df).columns) if numeric_only else list(df.columns)


# path =["column_name", True]
def to_excel_with_static_data(df_ts_code, path, sort: list = [], a_asset=["I", "E", "FD", "F", "G"], group_result=True, market="CN"):
    df_ts_code = add_static_data(df_ts_code, asset=a_asset, market=market)
    d_df = {"Overview": df_ts_code}

    # tab group
    if group_result:
        for group, a_instance in LB.c_d_groups(a_asset, market=market).items():
            if group == "concept":
                df_group = pd.DataFrame()
                for instance in a_instance:
                    df_instance = df_ts_code[df_ts_code["concept"].str.contains(instance) == True]
                    s = df_instance.mean()
                    s["count"] = len(df_instance)
                    s.name = instance
                    df_group = df_group.append(s, sort=False)
                df_group.index.name = "concept"
                df_group = df_group[["count"] + list(LB.df_to_numeric(df_ts_code).columns)]
            else:
                df_groupbyhelper = df_ts_code.groupby(group)
                df_group = df_groupbyhelper.mean()
                if not df_group.empty:
                    df_group["count"] = df_groupbyhelper.size()

            if sort and sort[0] in df_group.columns:
                df_group = df_group.sort_values(by=sort[0], ascending=sort[1])
            d_df[group] = df_group
    LB.to_excel(path=path, d_df=d_df, index=True)


# needs to be optimized for speed and efficiency
def add_static_data(df, asset=["E", "I", "FD"], market="CN"):
    df_result = pd.DataFrame()
    for asset in asset:
        df_asset = get_ts_code(a_asset=[asset], market=market)
        if df_asset.empty:
            continue
        df_result = df_result.append(df_asset, sort=False, ignore_index=False)
    df.index.name = "ts_code"  # important, otherwise merge will fail
    return pd.merge(df, df_result, how='left', on=["ts_code"], suffixes=[False, False], sort=False)


def add_asset_comparison(df, freq, asset, ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"]):
    """ require: trade_date
        function: adds another asset close price"""
    d_rename = {column: f"{column}_{ts_code}" for column in a_compare_label}
    df_compare = get_asset(ts_code, asset, freq)[a_compare_label]
    df_compare.rename(columns=d_rename, inplace=True)
    LB.remove_columns(df, [f"{label}_{ts_code}" for label in a_compare_label])
    return pd.merge(df, df_compare, how='left', on=["trade_date"], suffixes=["", ""], sort=False)


def preload(asset="E", freq="D", on_asset=True, step=1, query_df="", period_abv=240, d_queries_ts_code={}, reset_index=False,market="CN"):
    """
    query_on_df: filters df_asset/df_date by some criteria. If the result is empty dataframe, it will NOT be included in d_result
    """


    d_result = {}
    df_index = get_ts_code(a_asset=[asset], d_queries=d_queries_ts_code, market=market)[::step] if on_asset else get_trade_date(start_date="20000000")[::step]
    func = get_asset if on_asset else get_date
    kwargs = {"asset": asset, "freq": freq, "market":market} if on_asset else {"a_asset": [asset],"market":market}

    bar = tqdm(range(len(df_index)))
    for index, i in zip(df_index.index, bar):
        bar.set_description(f"{i}: {asset}: {index}")
        try:
            df = func(index, **kwargs)
            if asset in ["E", "I", "FD"]:  # not work for G, F
                df = df[(df["period"] > period_abv)]
            if query_df:
                df = df.query(expr=query_df)
            if df.empty:
                continue
            else:  # only take df that satisfy ALL conditions and is non empty
                d_result[index] = df.reset_index() if reset_index else df
        except Exception as e:
            print(e)
            pass
            # print("preload exception", e)
    bar.close()

    # print not loaded index
    a_notloaded = [print(f"not loaded: {x}") for x in df_index.index if x not in d_result]
    print(f"LOADED : {len(d_result)}")
    print(f"NOT LOADED : {len(a_notloaded)}")
    return d_result

def preload_index(market="CN",step=1):
    return preload(asset="I", freq="D", on_asset=True, step=step, query_df="", period_abv=240, d_queries_ts_code=LB.c_index_queries(market=market), reset_index=False,market=market)

def update_all_in_one_us(night_shift=False):
    # ts_code
    for asset in ["I", "E", "FD"]:
        update_ts_code(asset=asset, market="US")


    #LB.multi_process(func=update_asset_US, a_kwargs={"asset": "E", }, a_partial=LB.multi_steps(2))  # big: smart decide - small: smart decide
    LB.multi_process(func=update_asset_US, a_kwargs={"asset": "FD", }, a_partial=LB.multi_steps(2))  # big: smart decide - small: smart decide
    #LB.multi_process(func=update_asset_US, a_kwargs={"asset": "I", }, a_partial=LB.multi_steps(2))  # big: smart decide - small: smart decide

    # update trade_date (using index)


def update_all_in_one_hk(night_shift=False):
    update_trade_cal(market="HK")
    update_trade_date(market="HK")
    update_ts_code(asset="E", market="HK")
    LB.multi_process(func=update_asset_CNHK, a_kwargs={"market": "HK", "asset": "E", "freq": "D", "night_shift": False}, a_partial=LB.multi_steps(2))  # SMART


def update_all_in_one_cn(night_shift=False):
    # 0. ALWAYS UPDATE
    # 1. ONLY ON BIG UPDATE: OVERRIDES EVERY THING EVERY TIME
    # 2. ON BOTH BIG AND SMALL UPDATE: OVERRIDES EVERYTHING EVERY TIME
    # 3. SMART: BIG OR SMALL UPDATE DOES NOT MATTER, ALWAYS CHECK IF FILE NEEDS TO BE UPDATED

    # 1.0. ASSET - Indicator bundle
    if False:
        # E: update each E asset one after another
        for asset in ["E","FD"]:
            for counter, (bundle_name, bundle_func) in enumerate(LB.c_asset_E_bundle(asset=asset).items()):
                LB.multi_process(func=update_asset_bundle, a_kwargs={"bundle_name": bundle_name, "bundle_func": bundle_func, "night_shift": False, "a_asset":[asset]}, a_partial=LB.multi_steps(2))  # SMART does not alternate step, but alternates fina_name+fina_function


    else:
        # update all E asset at same time
        for asset in ["FD"]:
            a_partial = [{"bundle_name": bundle_name, "bundle_func": bundle_func} for bundle_name, bundle_func in LB.c_asset_E_bundle(asset=asset).items()]
            LB.multi_process(func=update_asset_bundle, a_kwargs={"step": 1, "night_shift": False, "a_asset":[asset]}, a_partial=a_partial)  # SMART does not alternate step, but alternates fina_name+fina_function


    # 1.0. GENERAL - CAL_DATE
    # update_trade_cal()  # always update

    # # 1.3. GENERAL - TS_CODE
    # for asset in ["sw_industry1", "sw_industry2","sw_industry3","jq_industry1","jq_industry2","zj_industry1","concept"] + c_asset() + ["G","F"]:
    # update_ts_code(asset)  # ALWAYS UPDATE

    # # 1.5. GENERAL - TRADE_DATE (later than ts_code because it counts ts_codes)
    # for freq in ["D"]:  # Currently only update D and W, because W is needed for pledge stats
    #    update_trade_date(freq)  # ALWAYS UPDATE

    # 2.2. ASSET
    # LB.multi_process(func=update_asset_EIFD_D, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "night_shift": False}, a_partial=LB.multi_steps(4))  # SMART
    # LB.multi_process(func=update_asset_EIFD_D, a_kwargs={"asset": "FD", "freq": "D", "market": "CN", "night_shift": False}, a_partial=LB.multi_steps(4))  # SMART
    # LB.multi_process(func=update_asset_EIFD_D, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "night_shift": False}, a_partial=LB.multi_steps(8))  # SMART
    # update_asset_G(night_shift=night_shift)  # update concept is very very slow. = Night shift
    # multi_process(func=update_asset_EIFD_D, a_kwargs={"asset": "F", "freq": "D", "market": "CN", "night_shift": False}, a_partial=LB.multi_steps(1))  # SMART

    # 3.2. DATE
    # date_step = [-1, 1] if night_shift else [-1, 1]
    # LB.multi_process(func=update_date, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "night_shift": False, "naive":False}, a_partial=LB.multi_steps(1))  # SMART
    # LB.multi_process(func=update_date, a_kwargs={"asset": "FD", "freq": "D", "market": "CN", "night_shift": False, "naive":True}, a_partial=LB.multi_steps(1))  # SMART
    # LB.multi_process(func=update_date, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "night_shift": False, "naive":False}, a_partial=LB.multi_steps(1))  # SMART
    # LB.multi_process(func=update_date, a_kwargs={"asset": "G", "freq": "D", "market": "CN", "night_shift": False, "naive":False}, a_partial=LB.multi_steps(1))  # SMART
    # LB.multi_process(func=update_date, a_kwargs={"asset": "F", "freq": "D", "market": "CN", "night_shift": False, "naive":True}, a_partial=LB.multi_steps(1))  # SMART

    # # 3.3. DATE - BASE
    # update_asset_stock_market_all(start_date="19990101", end_date=today(), night_shift=night_shift, asset=["E"])  # SMART


def update_all_in_one_cn_v2(night_shift=False):
    # 0. ALWAYS UPDATE
    # 1. ONLY ON BIG UPDATE: OVERRIDES EVERY THING EVERY TIME
    # 2. ON BOTH BIG AND SMALL UPDATE: OVERRIDES EVERYTHING EVERY TIME
    # 3. SMART: BIG OR SMALL UPDATE DOES NOT MATTER, ALWAYS CHECK IF FILE NEEDS TO BE UPDATED

    # 1.0. ASSET - Indicator bundle (MANUALLY UPDATE)


    # 1.0. ASSET - Indicator bundle
    """if False:
        # E: update each E asset one after another
        for asset in ["E", "FD"]:
            for counter, (bundle_name, bundle_func) in enumerate(LB.c_asset_E_bundle(asset=asset).items()):
                LB.multi_process(func=update_asset_bundle, a_kwargs={"bundle_name": bundle_name, "bundle_func": bundle_func, "night_shift": False, "a_asset": [asset]}, a_partial=LB.multi_steps(2))  # SMART does not alternate step, but alternates fina_name+fina_function


    else:
        # update all E asset at same time
        for asset in ["FD"]:
            a_partial = [{"bundle_name": bundle_name, "bundle_func": bundle_func} for bundle_name, bundle_func in LB.c_asset_E_bundle(asset=asset).items()]
            LB.multi_process(func=update_asset_bundle, a_kwargs={"step": 1, "night_shift": False, "a_asset": [asset]}, a_partial=a_partial)  # SMART does not alternate step, but alternates fina_name+fina_function
"""


    # 1.0. GENERAL - CAL_DATE
    update_trade_cal()  # always update

    # # 1.3. GENERAL - TS_CODE
    for asset in ["I","E","G"]:#"F" sometimes bug
        update_ts_code(asset)  # ALWAYS UPDATE

    for asset in ["sw_industry1", "sw_industry2","sw_industry3","concept"]:
        if night_shift: update_ts_code(asset)  # SOMETIMES UPDATE

    # # 1.5. GENERAL - TRADE_DATE (later than ts_code because it counts ts_codes)
    for freq in ["D"]:  # Currently only update D and W, because W is needed for pledge stats
        update_trade_date(freq)  # ALWAYS UPDATE

    # 2.2. ASSET
    LB.multi_process(func=update_asset_CNHK, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "night_shift": False, "miniver":True}, a_partial=LB.multi_steps(4))  # 40 mins
    LB.multi_process(func=update_asset_CNHK, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "night_shift": False, "miniver":True}, a_partial=LB.multi_steps(4))  # 60 mins
    update_asset_G(night_shift=night_shift)  # update concept is very very slow. = Night shift





fast_load = {}
df_ts_code = get_ts_code(a_asset=["E", "I", "FD", "F", "G"])


def get_fast_load(ts_code,market="CN"):
    if ts_code in fast_load:
        df = fast_load[ts_code]
        if not df.empty:
            return df

    asset = df_ts_code.at[ts_code, "asset"]
    fast_load[ts_code] = get_asset(ts_code=ts_code, asset=asset)
    return fast_load[ts_code]




if __name__ == '__main__':
    # TODO in general, save empty df or not safe?
    # TODO generate test cases
    # TODO update all in one so that it can run from zero to hero in one run
    pr = cProfile.Profile()
    pr.enable()
    try:
        night_shift = False

        # update_all_in_one_hk()
        # update_all_in_one_us()
        #update_all_in_one_us()
        #update_asset_stock_market_all()
        for asset in ["FD"]:
            a_partial = [{"bundle_name": bundle_name, "bundle_func": bundle_func} for bundle_name, bundle_func in LB.c_asset_E_bundle(asset=asset).items()]
            LB.multi_process(func=update_asset_bundle, a_kwargs={"step": 1, "night_shift": False, "a_asset": [asset], "skip_csv":False}, a_partial=a_partial)  # SMART does not alternate step, but alternates fina_name+fina_function

        #update_all_in_one_cn()







    except Exception as e:
        traceback.print_exc()
        print(e)
        LB.sound("error.mp3")
    pr.disable()
    # pr.print_stats(sort='file')

# slice
# a[-1]    # last item in the array
# a[-6:]   # get last 6 elements

# a[:-2]   # everything except the last two items

# a[::-1]    # all items in the array, reversed
# a[1::-1]   # the first two items, reversed
# a[:-3:-1]  # the last two items, reversed
# a[-3::-1]  # everything except the last two items, reversed
# qcut

# '…'

"""excel sheets
detect duplicates  =COUNTIF(A:A, A2)>1
"""
