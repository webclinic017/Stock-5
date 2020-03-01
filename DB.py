import pandas as pd
import tushare as ts
import numpy as np
import API_Tushare
import LB
import os.path
import inspect
from itertools import combinations
import operator
import math
from numba import njit
import traceback
import cProfile
from tqdm import tqdm
import ICreate

from LB import *

pd.options.mode.chained_assignment = None  # default='warn'
pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def update_general_trade_cal(start_date="19900101", end_date="20250101"):
    df = API_Tushare.my_trade_cal(start_date=start_date, end_date=end_date)
    a_path = LB.a_path("Market/CN/General/trade_cal_D")
    df.set_index(keys="cal_date", inplace=True, drop=True)
    LB.to_csv_feather(df, a_path)


def update_general_trade_date(freq="D", market="CN", big_update=True):
    if freq in ["D", "W"]:
        a_path = LB.a_path("Market/" + market + "/General/trade_date_" + freq)
        df = API_Tushare.my_pro_bar(ts_code="000001.SH", start_date="00000000", end_date=today(), freq=freq, asset="I")
        df = LB.df_reverse_reindex(df)

        df = df[["trade_date"]]
        df = update_general_trade_date_stockcount(df)  # adds E,I,FD count
        # df = update_general_trade_date_seasonal_score(df, freq, market)  # TODO adds seasonal score for each day
        LB.to_csv_feather(df, a_path, index_relevant=False)


def update_general_trade_date_seasonal_score(df_trade_date, freq="D", market="CN"):
    # get all indicator for each day
    path_indicator = "Market/CN/Backtest_Single/seasonal/all_date_seasonal.xlsx"
    xls = pd.ExcelFile(path_indicator)

    # get all different groups
    a_groups = [[LB.get_trade_date_datetime_dayofweek, "dayofweek"],  # 1-5
                [LB.get_trade_date_datetime_m, "monthofyear"],  # 1-12
                [LB.get_trade_date_datetime_d, "dayofmonth"],  # 1-31
                [LB.get_trade_date_datetime_weekofyear, "weekofyear"],  # 1-51
                [LB.get_trade_date_datetime_dayofyear, "dayofyear"],  # 1-365
                # TODO add [Library_Main.get_trade_date_datetime_s, "seasonofyear"]  # 1-4
                ]

    # transform all trade_date into different format
    for group in a_groups:
        df_trade_date[group[1]] = df_trade_date["trade_date"].apply(lambda x: group[0](x)).astype(str)
        df_score_board = pd.read_excel(xls, sheet_name=group[1], converters={group[1]: lambda x: str(x)})
        df_score_board = df_score_board.fillna(0.0)  # if no information about certain period, assume the change is 0
        df_score_board[group[1]] = df_score_board[group[1]].astype(str)

        df_score_board = df_score_board.rename(columns={"pct_chg": group[1] + "_score"})
        df_trade_date[group[1]] = df_trade_date[group[1]].astype(str)
        df_trade_date = pd.merge(df_trade_date, df_score_board, how='left', on=group[1], suffixes=["", ""], sort=False)
        df_trade_date[group[1] + "_score"] = df_trade_date[group[1] + "_score"].fillna(0.0)  # if information not enough, assume the pct_chg is 0.0

    df_trade_date["seasonal_score"] = df_trade_date["dayofweek_score"] * 0.15 + \
                                      df_trade_date["dayofmonth_score"] * 0.25 + \
                                      df_trade_date["weekofyear_score"] * 0.20 + \
                                      df_trade_date["dayofyear_score"] * 0.05 + \
                                      df_trade_date["monthofyear_score"] * 0.35

    df_trade_date["seasonal_score"] = df_trade_date["seasonal_score"].rolling(2).mean()
    return df_trade_date


def update_general_trade_date_stockcount(df, market="CN"):
    df.index = df["trade_date"].astype(int)
    for asset in c_assets():
        df_ts_codes = get_ts_code(asset)
        df_ts_codes = df_ts_codes.rename(columns={"list_date": "trade_date"})
        df_grouped = df_ts_codes[["name", "trade_date"]].groupby(by="trade_date").count()
        # vecorized approach faster than loop over individual date
        df[asset + "_count"] = df_grouped["name"].astype(int).cumsum()
        df[asset + "_count"] = df[asset + "_count"].fillna(method="ffill")
        df[asset + "_count"] = df[asset + "_count"].fillna(0)
    return df


def update_general_ts_code(asset="E", market="CN", big_update=True):
    print("start update general ts_code ", asset)
    if (asset == "E"):
        df = API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="").set_index("ts_code")

        # add asset
        df["asset"] = asset

        # add exchange info for each stock
        df["exchange"] = ["创业板" if x[0:3] in ["300"] else "中小板" if x[0:3] in ["002"] else "主板" if x[0:2] in ["00", "60"] else float("nan") for x in df.index]

        # add SW industry info for each stock
        for level in c_industry_level():
            df_industry_instance = pd.DataFrame()
            df_industry_member = get_industry_member(level)
            for industry_index in df_industry_member.index:
                df_industry_instance = df_industry_instance.append(get_industry_index(level=level, index=industry_index), sort=False)
            df_industry_instance.index.name = "ts_code"
            df[f"industry{level}"] = df_industry_instance["index_code"].replace(to_replace=df_industry_member[f"industry{level}"].to_dict())

        # LB.columns_remove(df, ["index_code" + x for x in c_industry_level()])

        # add State Government for each stock
        df["state_company"] = False
        for ts_code in df.index:
            print("update state_company", ts_code)
            df_government = get_assets_top_holder(ts_code=ts_code, columns=["holder_name", "hold_ratio"])
            if df_government.empty:  # if empty, assume it is False
                continue
            df_government_grouped = df_government.groupby(by="holder_name").mean()
            df_government_grouped = df_government_grouped["hold_ratio"].nlargest(n=1)  # look at the top 4 share holders

            counter = 0
            for top_holder_name in df_government_grouped.index:
                if ("公司" in top_holder_name) or (len(top_holder_name) > 3):
                    counter = counter + 1
            df.at[ts_code, "state_company"] = True if counter >= 1 else False

    elif (asset == "I"):
        df_SSE = API_Tushare.my_index_basic(market='SSE')
        df_SZSE = API_Tushare.my_index_basic(market='SZSE')
        df = df_SSE.append(df_SZSE, sort=False).set_index("ts_code")
        df["asset"] = asset
    elif (asset == "FD"):
        df_E = API_Tushare.my_fund_basic(market='E')
        df_O = API_Tushare.my_fund_basic(market='O')
        df = df_E.append(df_O, sort=False).set_index("ts_code")
        df["asset"] = asset
    else:
        df = pd.DataFrame()
    a_path = LB.a_path("Market/" + market + "/General/ts_code_" + asset)
    df["list_date"] = df["list_date"].fillna(method='ffill')
    LB.to_csv_feather(df, a_path)


def update_general_ts_code_all(market="CN"):
    a_path = LB.a_path("Market/" + market + "/General/ts_code_all")
    df = pd.DataFrame()
    for asset in c_assets():
        df_asset = get_ts_code(asset)
        df = df.append(df_asset, sort=False)
    LB.to_csv_feather(df, a_path)


@LB.only_big_update
def update_general_industry(level, market="CN", big_update=True):
    # industry member list
    df_member = API_Tushare.my_index_classify(f"L" + level)
    df = df_member[["index_code", "industry_name"]].rename(columns={"industry_name": "industry" + level}).set_index("index_code")
    LB.to_csv_feather(df, a_path=LB.a_path("Market/" + market + "/General/industry_" + level))

    # industry instance
    for index in df_member["index_code"]:
        df = API_Tushare.my_index_member(index).set_index(keys="con_code", drop=True)
        LB.to_csv_feather(df, LB.a_path("Market/" + market + "/General/industry/" + level + "/" + index))


def update_assets_EIFD_D(asset="E", freq="D", market="CN", step=1, big_update=True):
    a_path_empty = LB.a_path("Market/CN/General/ts_code_ignore")

    # ignore some ts_code to update to speed up process
    def get_ts_code_ignore():
        return pd.DataFrame()
        df_saved = get(a_path_empty)
        df_ts_code_all = get_ts_code_all()
        df_ts_code_all = df_ts_code_all[["asset"]]
        if len(df_saved.columns) != 0 and not big_update:  # update existing ignore list with latest ts_code
            df_saved = pd.merge(df_ts_code_all, df_saved, how="left").set_index("ts_code")
            df_saved["ignore"] = df_saved["ignore"].fillna(False)
            return df_saved
        else:  # create new empty file with columns
            df_ts_code_all["ignore"] = False
            df_ts_code_all["reason"] = "no reason"
            return df_ts_code_all

    def update_ignore_list(df, ts_code, df_empty_EI, real_latest_trade_date):
        return
        if df.empty:  # check if it empty first, otherwise following operation should never execute
            df_empty_EI.at[ts_code, "ignore"] = True
            df_empty_EI.at[ts_code, "reason"] = "Empty"
            LB.to_csv_feather(df_empty_EI, a_path_empty)
        # elif str(df["trade_date"].tail(1).values.tolist()[0]) != real_latest_trade_date:
        elif str(df.index[-1]) != real_latest_trade_date:
            df_empty_EI.at[ts_code, "ignore"] = True
            df_empty_EI.at[ts_code, "reason"] = "no more update"
            LB.to_csv_feather(df_empty_EI, a_path_empty)

    # init
    df_ignore_EI = get_ts_code_ignore()
    df_ignore_EI["ignore"] = False
    print("Index not to consider:", df_ignore_EI[df_ignore_EI["ignore"] == True].index.tolist())
    df_ts_codes = get_ts_code(asset)
    if asset == "I":  # acceleration process to skip I TODO remove when I is fully used
        df_ts_codes = df_ts_codes[df_ts_codes.index.isin(["000001.SH", "399006.SZ", "399001.SZ"])]
    df_ts_codes = df_ts_codes[~df_ts_codes.index.isin(df_ignore_EI[df_ignore_EI["ignore"] == True].index.tolist())]
    real_latest_trade_date = get_last_trade_date(freq)

    # iteratve over ts_code
    for ts_code in df_ts_codes.index[::step]:
        start_date, middle_date, end_date, complete_new_update, df_saved = ("00000000", "20050101", today(), True, pd.DataFrame())  # True means update from 00000000 to today, False means update latest_trade_date to today
        a_path = LB.a_path(f"Market/{market}/Asset/{asset}/{freq}/{ts_code}")

        # file exists--> check latest_trade_date, else update completely new
        if os.path.isfile(a_path[1]):  # or os.path.isfile(a_path[1])
            try:
                df_saved = get_asset(ts_code=ts_code, asset=asset, freq=freq, market=market)  # get latest file trade_date
                asset_latest_trade_date = str(df_saved.index[-1])
            except:
                asset_latest_trade_date = start_date
                print(asset, ts_code, freq, end_date, "EMPTY - START UPDATE", asset_latest_trade_date, " to today")

            # file exist and on latest date--> finish, else update
            if (str(asset_latest_trade_date) == str(real_latest_trade_date)):

                print(asset, ts_code, freq, end_date, "Up-to-date", real_latest_trade_date)
                continue
            else:  # file exists and not on latest date
                if ts_code in df_ignore_EI[df_ignore_EI["ignore"] == True].index:
                    print(asset, ts_code, freq, end_date, "FILLER INDEX AND Up-to-date", real_latest_trade_date)
                    continue
                else:  # file exists and not on latest date, AND stock trades--> update
                    complete_new_update = False

        # file not exist or not on latest_trade_date --> update
        if (asset == "E" and freq == "D"):
            # 1.1 get df
            if complete_new_update:
                print("big update from", start_date)
                df1 = update_assets_EI_D_reindexreverse(ts_code, freq, asset, start_date, middle_date, adj="hfq")
                df2 = update_assets_EI_D_reindexreverse(ts_code, freq, asset, middle_date, end_date, adj="hfq")
                df = df1.append(df2, ignore_index=True, sort=False)
            else:
                print("small update from", asset_latest_trade_date)
                df = update_assets_EI_D_reindexreverse(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="hfq")
            df.set_index(keys="trade_date", drop=True, inplace=True)
            df.index = df.index.astype(int)
            update_ignore_list(df, ts_code, df_ignore_EI, real_latest_trade_date)

            # 1.2 get adj factor because tushare is too dump to calculate it on its own
            df_adj = API_Tushare.my_query(api_name='adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df_adj.empty:
                print(asset, ts_code, freq, start_date, end_date, "has no adj_factor yet. skipp")
            else:
                latest_adj = df_adj.at[0, "adj_factor"]
                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / latest_adj

            # 2.1 get daily basic
            if complete_new_update:
                df_fun_1 = API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=start_date, end_date=middle_date)
                df_fun_2 = API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=middle_date, end_date=end_date)
                df_fun = df_fun_1.append(df_fun_2, ignore_index=True, sort=False)
            else:
                df_fun = API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=asset_latest_trade_date, end_date=end_date)

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
            df_balancesheet = get_assets_E_D_Fun("balancesheet", ts_code=ts_code, columns=["total_cur_assets", "total_assets", "total_cur_liab", "total_liab"])
            # 营业活动产生的现金流量净额	，投资活动产生的现金流量净额,筹资活动产生的现金流量净额
            df_cashflow = get_assets_E_D_Fun("cashflow", ts_code=ts_code, columns=["n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"])
            # 扣除非经常性损益后的净利润,净利润同比增长(netprofit_yoy instead of q_profit_yoy on the doc)，营业收入同比增长,销售毛利率，销售净利率,资产负债率,存货周转天数(should be invturn_days,but casted to turn_days in the api for some reasons)
            df_indicator = get_assets_E_D_Fun("fina_indicator", ts_code=ts_code, columns=["profit_dedt", "netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "turn_days"])
            # 股权质押比例
            df_pledge_stat = get_assets_pledge_stat(ts_code=ts_code, columns=["pledge_ratio"])

            # add fina to df

            for df_fun, fun_name in zip([df_balancesheet, df_cashflow, df_indicator, df_pledge_stat], ["bala", "cash", "indi", "pledge"]):
                df_fun.index.name = "trade_date"
                df_fun.index = df_fun.index.astype(int)
                df[list(df_fun.columns)] = df_fun[list(df_fun.columns)]
                # df = pd.merge(df, df_fun, how="left", on="trade_date", sort=False).set_index("trade_date")  # just added might be wrong

            # append old df and drop duplicates
            if not df_saved.empty:
                df = df_saved.append(df, sort=False)
            df = df[~df.index.duplicated(keep="last")]  # important

            # interpolate/fill between empty fina and pledge_stat values
            all_report_label = list(df_balancesheet.columns.values) + list(df_cashflow.columns.values) + list(df_indicator.columns.values) + ["pledge_ratio"]
            for label in all_report_label:
                try:
                    df[label] = df[label].fillna(method='ffill')
                except:
                    pass

        elif (asset == "I" and freq == "D"):
            if complete_new_update:
                df = update_assets_EI_D_reindexreverse(ts_code, freq, asset, start_date, end_date, adj="qfq")
            else:
                df = update_assets_EI_D_reindexreverse(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="qfq")
                df = df_saved.append(df, sort=False)
                df = LB.df_drop_duplicated_reindex(df, "trade_date")
            df.set_index(keys="trade_date", drop=True, inplace=True)
            df.index = df.index.astype(int)
            update_ignore_list(df, ts_code, df_ignore_EI, real_latest_trade_date)
        elif (asset == "FD" and freq == "D"):
            if complete_new_update:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date)
            else:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date)
                df = df_saved.append(df, sort=False)
                df = LB.df_drop_duplicated_reindex(df, "trade_date")
            df.set_index(keys="trade_date", drop=True, inplace=True)
            df.index = df.index.astype(int)
            update_ignore_list(df, ts_code, df_ignore_EI, real_latest_trade_date)

        # 3. add my derivative indices and save it
        if not df.empty:
            update_assets_EIFD_D_technical(df=df, asset=asset)
        LB.to_csv_feather(df=df, a_path=a_path)
        print(asset, ts_code, freq, end_date, "UPDATED!", real_latest_trade_date)


# For all Pri indices and derivates. ordered after FILO
def update_assets_EIFD_D_technical(df, asset="E", bfreq=c_bfreq()):
    ICreate.pct_chg_close(df=df)  # 0.890578031539917 for 300 loop
    ICreate.pct_chg_open(df=df)  # 0.890578031539917 for 300 loop

    for rolling_freq in bfreq[::-1]:
        ICreate.pgain(df=df, freq=rolling_freq)  # past gain includes today = today +yesterday comp_gain
    for rolling_freq in bfreq[::-1]:
        ICreate.fgain(df=df, freq=rolling_freq)  # future gain does not include today = tomorrow+atomorrow comp_gain
    ICreate.ivola(df=df)  # 0.890578031539917 for 300 loop
    ICreate.period(df=df)  # 0.2 for 300 loop
    ICreate.pjup(df=df)  # 1.0798187255859375 for 300 loop
    ICreate.pjdown(df=df)  # 1.05 independend for 300 loop
    ICreate.co_pct_chg(df=df)

    # ICreate.cdl(df,ibase="cdl")  # VERY SLOW. NO WAY AROUND. 120 sec for 300 loop
    if asset == "E":  # else sh_index will try to get corr wit himself during update
        ICreate.deri_sta(df=df, ibase="close", freq=BFreq.f5, ideri=ICreate.IDeri.corr, re=ICreate.RE.r)
        ICreate.deri_sta(df=df, ibase="close", freq=BFreq.f10, ideri=ICreate.IDeri.corr, re=ICreate.RE.r)
    # add trend for individual stocks
    # ICreate.trend(df=df, ibase="close")
    # df = support_resistance_horizontal(df_asset=df)


# For E,I,FD  D
def update_assets_EI_D_reindexreverse(ts_code, freq, asset, start_date, end_date, adj="qfq", market="CN"):
    df = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
    df = LB.df_reverse_reindex(df)
    LB.columns_remove(df, ["pre_close", "amount", "change"])
    return df


# For FD D
def update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj=None, market="CN"):
    df = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
    request_df_len = len(df)
    last_df2 = df
    while request_df_len == 1000:  # TODO if this is fixed or add another way to loop
        middle_date = last_df2.at[len(last_df2) - 1, "trade_date"]
        df2 = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(middle_date), adj=adj)
        if (df2.equals(last_df2)):
            break
        df = df.append(df2, sort=False, ignore_index=True)
        df = df.drop_duplicates(subset="trade_date")
        request_df_len = len(df2)
        last_df2 = df2

    # 当日收盘价 × 当日复权因子 / 最新复权因子
    if not df.empty:
        df.index = df["trade_date"]
        df_adj_factor = API_Tushare.my_query(api_name='fund_adj', ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df_adj_factor.empty:
            print(asset, ts_code, freq, "has no adj_factor, skip")
        else:
            df_adj_factor.index = df_adj_factor.index.astype(int)
            latest_adj = df_adj_factor.at[0, "adj_factor"]
            df_adj_factor.index = df_adj_factor["trade_date"]

            df["adj_factor"] = df_adj_factor["adj_factor"]
            df["adj_factor"] = df["adj_factor"].interpolate()  # interpolate between alues because some dates are missing from tushare

            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * df.adj_factor / latest_adj

        df = LB.df_reverse_reindex(df)
        LB.columns_remove(df, ["pre_close", "amount", "change", "adj_factor"])
    return df


# recommended update frequency W
def update_assets_E_W_pledge_stat(start_date="00000000", market="CN", step=1, big_update=True):
    for counter, ts_code in enumerate(get_ts_code("E").index[::step]):
        a_path = LB.a_path("Market/" + market + "/Asset/E/W_pledge_stat/" + str(ts_code))
        if os.path.isfile(a_path[1]) and (not big_update):
            print(counter, ts_code, "pledge_stat Up-to-date")
            continue
        else:  # always update completely new
            df = API_Tushare.my_pledge_stat(ts_code=ts_code)
            df = LB.df_reverse_reindex(df).set_index("end_date")
            LB.to_csv_feather(df, a_path)
            print(counter, ts_code, "pledge_stat UPDATED")


def update_assets_E_top_holder(big_update=True, market="CN", step=1):
    for counter, ts_code in enumerate(get_ts_code("E").index[::step]):
        a_path = LB.a_path("Market/" + market + "/Asset/E/D_top_holder/" + str(ts_code))
        if os.path.isfile(a_path[1]) and (not big_update):  # skipp if small update or file exist
            print(counter, ts_code, "top_holder Up-to-date")
            continue
        else:  # always update completely new
            df = API_Tushare.my_query(api_name='top10_holders', ts_code=ts_code, start_date='20190101', end_date=today())  # NO INDEX SET HERE
            df = LB.df_reverse_reindex(df)  # .set_index()
            LB.to_csv_feather(df, a_path, index_relevant=False)
            print(counter, ts_code, "top_holder UPDATED")


def update_assets_E_D_Fun(start_date="0000000", end_date=LB.today(), market="CN", step=1, big_update=True):
    for counter, ts_code in enumerate(get_ts_code("E").index[::step]):
        for fina_name, fina_function in c_assets_fina_function_dict().items():
            a_path = LB.a_path("Market/" + market + "/Asset/E/D_Fun/" + fina_name + "/" + ts_code)
            if os.path.isfile(a_path[1]) and (not big_update):  # skipp if small update or file exist
                print(counter, ts_code, f"{ts_code} {fina_name} Up-to-date")
                continue
            else:  # always update completely new
                df = fina_function(ts_code=ts_code, start_date=start_date, end_date=end_date)
                df = LB.df_reverse_reindex(df).set_index("end_date")
                LB.to_csv_feather(df, a_path)
                print(counter, ts_code, f"{ts_code} {fina_name} UPDATED")



def update_date(asset="E", freq="D", market="CN", step=1, big_update=True):
    for asset in ["E"]:
        update_date_EIFD_DWMYS(asset, freq, big_update=big_update, step=step)


def update_date_EIFD_DWMYS(asset="E", freq="D", market="CN", big_update=True, step=1):  # STEP only 1 or -1 !!!!
    df_ts_codes = get_ts_code(asset)
    df_ts_codes["list_date"] = df_ts_codes["list_date"].astype(int)
    trade_dates = get_trade_date("00000000", LB.today(), freq)

    # get the latest column of the asset file
    code = "000001.SZ" if asset == "E" else "000001.SH" if asset == "I" else "150001.SZ"
    example_column = list(get_asset(code, asset, freq).columns)

    # makes searching for one day in asset time series faster. BUT can only be used with step=1 and ONLY using ONE THREAD
    print("Update date preparing for setup. Please wait...")
    dict_list_date = {ts_code: list_date for ts_code, list_date in zip(df_ts_codes.index, df_ts_codes["list_date"])}
    dict_df = {ts_code: get_asset(ts_code=ts_code) for ts_code in df_ts_codes.index}

    if step == 1:
        dict_lookup_table = {ts_code: 0 for ts_code, df in dict_df.items()}
    elif step == -1:
        dict_lookup_table = {ts_code: len(df) - 1 for ts_code, df in dict_df.items()}
    else:
        return print("error Step msut be 1 or -1")

    for trade_date in trade_dates.index[::step]:  # IMPORTANT! do not modify step, otherwise lookup will not work
        a_path = LB.a_path("Market/" + market + "/Date/" + asset + "/" + freq + "/" + str(trade_date))
        a_date = []

        if os.path.isfile(a_path[0]) and (not big_update):  # date file exists AND not big_update. If big_update, then always overwrite
            # update lookup table before continue. So that skipped days still match
            for (ts_code, list_date), (ts_code_unused, df_asset) in zip(dict_list_date.items(), dict_df.items()):
                row_number = dict_lookup_table[ts_code]
                if (step == -1) and (row_number == -1):
                    # Case 1: step=-1. row_number=-1. This means the pointer is at row 0. which is max
                    # Case 2: step=1. row_number=len(df) This happens automatically and does not need to take care
                    continue
                try:
                    if df_asset.index[row_number] == trade_date:
                        dict_lookup_table[ts_code] = dict_lookup_table[ts_code] + step
                except:
                    pass
            print(asset, freq, trade_date, "date file Up-to-date")
            continue

        else:  # date file does not exist or big_update
            for (ts_code, list_date), (ts_code_unused, df_asset) in zip(dict_list_date.items(), dict_df.items()):
                if int(list_date) > int(trade_date):
                    continue

                row_number = dict_lookup_table[ts_code]  # lookup table can not be changed while iterating over it.
                try:
                    if int(df_asset.index[row_number]) == int(trade_date):
                        a_date.append(df_asset.loc[trade_date].to_numpy().flatten())
                        dict_lookup_table[ts_code] = dict_lookup_table[ts_code] + step
                except Exception as e:
                    print("except", e)
                    continue

            df_date = pd.DataFrame(data=a_date, columns=example_column)
            df_date.insert(loc=0, column='trade_date', value=int(trade_date))
            df_date = pd.merge(df_date, df_ts_codes, how='left', on=["ts_code"], suffixes=[False, False], sort=False).set_index("ts_code")
            LB.to_csv_feather(df_date, a_path)
            print(asset, freq, trade_date, "date updated")


def update_date_E_Oth(asset="E", freq="D", market="CN", big_update=True, step=1):
    trade_dates = get_trade_date(start_date="00000000", end_date=today(), freq=freq)
    for trade_date in trade_dates["trade_date"][::step]:
        dict_oth_names = c_date_oth()
        dict_oth_paths = {name: LB.a_path("Market/" + market + "/Date/" + asset + "/" + name + "/" + str(trade_date)) for name, function in dict_oth_names.items()}
        for name, function in dict_oth_names.items():
            if os.path.isfile(dict_oth_paths[name][0]):
                print(trade_date, asset, freq, name, "Up-to-date")
            else:
                df_oth = function(trade_date)
                LB.to_csv_feather(df_oth, dict_oth_paths[name])
                print(trade_date, asset, freq, name, "UPDATED")


def update_date_Oth_analysis_market_D(freq, market="CN"):  # TODO add it to all date
    path = "Market/" + market + "/Date/market/" + freq + ".csv"
    trade_dates = get_trade_date(start_date="00000000", end_date=today(), freq=freq)
    df_result = trade_dates.copy()

    oth = c_date_oth()  # changed from array to dict
    a_block_trade_count = []
    a_holdertrade_in_count = []
    a_holdertrade_de_count = []
    a_repurchase_count = []
    a_repurchase_amount = []
    a_share_float_count = []
    for trade_date in trade_dates.trade_date:
        print("calculating market for", trade_date)
        df_block_trade = get_date_E_oth(trade_date, oth[0])
        if (df_block_trade.empty):
            a_block_trade_count.append(0)
            a_holdertrade_in_count.append(0)
            a_holdertrade_de_count.append(0)
            a_repurchase_count.append(0)
            a_repurchase_amount.append(0)
            a_share_float_count.append(0)
            continue

        a_block_trade_count.append(df_block_trade['ts_code'].nunique())

        df_holdertrade = get_date_E_oth(trade_date, oth[1])
        df_holdertrade = df_holdertrade.drop_duplicates(subset="ts_code")
        df_count = df_holdertrade.in_de.value_counts()
        try:
            a_holdertrade_in_count.append(df_count["IN"])
        except Exception:
            a_holdertrade_in_count.append(0)

        try:
            a_holdertrade_de_count.append(df_count["DE"])
        except Exception:
            a_holdertrade_de_count.append(0)

        df_repurchase = get_date_E_oth(trade_date, oth[2])
        a_repurchase_count.append(len(df_repurchase))
        a_repurchase_amount.append(df_repurchase.amount.sum())

        df_share_float = get_date_E_oth(trade_date, oth[3])
        try:
            a_share_float_count.append(df_share_float['ts_code'].nunique())
        except Exception:
            a_share_float_count.append(0)

    df_result["q_block_trade_count"] = (pd.Series(a_block_trade_count) / df_result.E) * 100
    df_result["q_holdertrade_in_count"] = (pd.Series(a_holdertrade_in_count) / df_result.E) * 100
    df_result["q_holdertrade_de_count"] = (pd.Series(a_holdertrade_de_count) / df_result.E) * 100
    df_result["q_repurchase_count"] = (pd.Series(a_repurchase_count) / df_result.E) * 100
    df_result["q_share_float_count"] = (pd.Series(a_share_float_count) / df_result.E) * 100
    df_result.to_csv(path, index=False, encoding='utf-8_sig')


def update_date_base(start_date="00000000", end_date=today(), assets=["E"], freq="D", market="CN", comparison_index=["000001.SH", "399001.SZ", "399006.SZ"], big_update=False):
    # check if big update and if the a previous saved file exists
    last_saved_date = "19990104"
    last_trade_date = get_last_trade_date("D")
    if not big_update:
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
    a_path = LB.a_path("Market/" + market + "/Backtest_Multiple/Setup/Stock_market/all_stock_market")
    a_result = []
    df_sh_index = get_asset(ts_code="000001.SH", asset="I", freq=freq, market="CN")
    df_sh_index = df_sh_index.loc[int(last_saved_date):int(last_trade_date)]
    df_sh_index = df_sh_index.loc[df_sh_index.index > int(last_saved_date)]  # exclude last saved date from update

    # loop through all trade dates and add them together
    for trade_date, sh_pct_chg in zip(df_sh_index.index, df_sh_index["pct_chg"]):
        print(trade_date, "being added to all_stock_market_base")
        df_date = get_date(str(trade_date), assets=assets, freq=freq, market="CN")
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

        # how many stocks封板 TODO
        a_result.append(df_date_mean)

    # array to dataframe
    df_result = pd.DataFrame(a_result)

    # if small update, append new data to old data
    if (not df_saved.empty) and (not big_update):
        df_result = df_saved.append(df_result, sort=False)

    # add comp chg and index
    df_result["comp_chg"] = ICreate.column_add_comp_chg(df_result["pct_chg"])
    for ts_code in comparison_index:
        df_result = add_asset_comparison(df=df_result, freq=freq, asset="I", ts_code=ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"])
        df_result["comp_chg_" + ts_code] = ICreate.column_add_comp_chg(df_result["pct_chg_" + ts_code])

    LB.to_csv_feather(df_result, a_path, index_relevant=False)
    print("Date_Base UPDATED")


def update_custom_index(assets=["E"], big_update=True):
    # all stock market pct_chg
    update_cj_index_000001_SH()

    # initialize all group as dict
    dict_group_instance_update = {}  # dict of array
    dict_group_instance_saved = {}  # dict of array
    for group, a_instance in c_groups_dict(assets=assets).items():
        for instance in a_instance:
            dict_group_instance_update[str(group) + "_" + str(instance)] = []
            dict_group_instance_saved[str(group) + "_" + str(instance)] = get_group_instance(group + "_" + str(instance))

    # get last saved trade_date on df_saved
    last_trade_date = get_last_trade_date("D")
    example_df = dict_group_instance_saved["asset_E"]
    try:
        last_saved_date = example_df.index[-1]
    except:
        last_saved_date = "19990101"

    # if small update and saved_date==last trade_date
    if last_trade_date == last_saved_date and (not big_update):
        return print("ALL GROUP Up-to-date")

    # initialize trade_date
    df_trade_date = get_trade_date(end_date=today(), freq="D")
    df_trade_date = df_trade_date[df_trade_date.index > int(last_saved_date)]
    print("START UPDATE GROUP since", last_saved_date)

    # loop over date and get mean
    for trade_date in df_trade_date.index:  # for each day
        print(trade_date, "Updating GROUP")
        df_date = get_date(trade_date=trade_date, assets=assets, freq="D")
        for group, a_instance in c_groups_dict(assets=assets).items():  # for each group
            df_date_grouped = df_date.groupby(by=group, ).mean()  # calculate mean
            for instance, row in df_date_grouped.iterrows():  # append to previous group
                dict_group_instance_update[group + "_" + str(instance)].append(row)

    # save all to df
    for (key, a_update_instance), (key_saved, df_saved) in zip(dict_group_instance_update.items(), dict_group_instance_saved.items()):
        df_update = pd.DataFrame(a_update_instance)
        if not df_saved.empty:
            df_update = pd.concat(objs=[df_saved, df_update], sort=False, ignore_index=True)
        LB.to_csv_feather(df_update, LB.a_path("Market/CN/Backtest_Multiple/Setup/Stock_Market/Group/" + key))
        print(key, "UPDATED")


def ts_code_series_to_excel(df_ts_code, path, sort: list = ["column_name", True], asset=["I", "E", "FD"], group_result=True):
    df_ts_code = add_static_data(df_ts_code, assets=asset)
    dict_df = {"Overview": df_ts_code}

    # tab group
    if group_result:
        for group_column in LB.c_groups_dict(asset).keys():
            try:
                df_groupbyhelper = df_ts_code.groupby(group_column)
                df_group = df_groupbyhelper.mean()
                df_group["count"] = df_groupbyhelper.size()
                if sort:
                    df_group.sort_values(by=sort[0], ascending=sort[1], inplace=True)
                dict_df[group_column] = df_group
            except Exception as e:
                print("error in group results", e)
    LB.to_excel(path_excel=path, dict_df=dict_df)


def update_cj_index_000001_SH():
    df = get_stock_market_all()
    df = df[["pct_chg"]]
    a_path = LB.a_path("Market/CN/Asset/I/D/CJ000001.SH")
    LB.to_csv_feather(df, a_path)



def get(a_path=[], set_index=""):  # read feather first
    for counter,func in [(1, [pd.read_feather]), (0, pd.read_csv)]:
        try:
            df = func(a_path[counter])
            if set_index:
                if set_index in ["trade_date", "cal_date", "end_date"]:
                    df[set_index] = df[set_index].astype(int)
                df.set_index(keys=set_index, drop=True, inplace=True)
            return df
        except Exception as e:
            print(f"read error {func.__name__}", e)

    print("DB READ File Not Exist!", a_path[0])
    return pd.DataFrame()


def get_ts_code(asset="E", market="CN"):
    df = get(LB.a_path("Market/" + market + "/General/ts_code_" + asset), set_index="ts_code")
    if (asset == "FD"):
        df = df[df["delist_date"].isna()]
        df = df[df["market"] == "E"]  # for now, only consider Equity market traded funds
    return df

def get_asset(ts_code="000002.SZ", asset="E", freq="D", market="CN"):
    return get(LB.a_path("Market/" + market + "/Asset/" + str(asset) + "/" + str(freq) + "/" + str(ts_code)), set_index="trade_date")


@LB.except_empty_df
def get_file(path):
    return pd.read_csv(path)


def get_ts_code_all(market="CN"):
    return get(LB.a_path("Market/" + market + "/General/ts_code_all"), set_index="ts_code")


def get_group_instance(group_instance="asset_E", market="CN"):
    return get(LB.a_path("Market/" + market + "/Backtest_Multiple/Setup/Stock_Market/Group/" + group_instance), set_index="trade_date")


def get_group_instance_all(assets=["E"]):
    dict_result = {}
    dict_group_label_pair = c_groups_dict(assets=assets, a_ignore=["asset", "industry3"])
    for group, instance_array in dict_group_label_pair.items():
        for instance in instance_array:
            dict_result[group + "_" + str(instance)] = get_group_instance(group_instance=group + "_" + str(instance))
    return dict_result


def get_assets_E_D_Fun(query, ts_code, columns=["end_date"], market="CN"):
    df = get(LB.a_path("Market/" + market + "/Asset/E/D_Fun/" + query + "/" + ts_code), set_index="end_date")
    if df.empty:
        print("Error get_assets_E_D_Fun ", query, "not exist for", ts_code)
        return LB.empty_df(query)[columns]
    else:
        df = df[~df.index.duplicated(keep="last")]
        return df[columns]


def get_assets_pledge_stat(ts_code, columns, market="CN"):
    df = get(LB.a_path("Market/" + market + "/Asset/E/W_pledge_stat/" + ts_code), set_index="end_date")
    if df.empty:
        print("Error get_assets_pledge_stat not exist for", ts_code)
        df = LB.empty_df("pledge_stat")
    df = df[~df.index.duplicated(keep="last")]
    return df[columns]


def get_assets_top_holder(ts_code, columns, market="CN"):
    df = get(LB.a_path("Market/" + market + "/Asset/E/D_top_holder/" + ts_code), set_index="end_date")
    if df.empty:
        print("Error get_assets_top_holder not exist for", ts_code)
        df = LB.empty_df("top_holder")
    return df[columns]


def get_trade_date(start_date="000000", end_date=today(), freq="D", market="CN"):
    df = get(LB.a_path("Market/" + market + "/General/trade_date_" + freq), set_index="trade_date")
    return df[(df.index >= int(start_date)) & (df.index <= int(end_date))]


def get_last_trade_date(freq="D", market="CN"):
    df_trade_date = get_trade_date(start_date="00000000", end_date=LB.today(), freq=freq, market=market)
    return str(df_trade_date.index[-1])


def get_next_trade_date(freq="D", market="CN"):  # TODO might be wrong
    df = get_trade_cal_D(a_is_open=[1])  # todo next trade date should be set to 17:00 after tushare has released its new data
    last_trade_date = get_last_trade_date(freq, market)
    df = df[df.index > int(last_trade_date)].reset_index()
    return df.at[0, "trade_date"]


def get_trade_cal_D(start_date="19900101", end_date="88888888", a_is_open=[1]):
    df = get(LB.a_path("Market/CN/General/trade_cal_D"), set_index="cal_date")
    df.index.name = "trade_date"
    return df[(df["is_open"].isin(a_is_open)) & (df.index >= int(start_date)) & (df.index <= int(end_date))]


def get_stock_market_all(market="CN"):
    return get(LB.a_path("Market/" + market + "/Backtest_Multiple/Setup/Stock_Market/all_stock_market"), set_index="trade_date")


def get_industry_member(level, market="CN"):
    return get(LB.a_path("Market/" + market + "/General/industry_" + level), set_index="index_code")


def get_industry_index(index, level, market="CN"):
    return get(LB.a_path("Market/" + market + "/General/industry/" + level + "/" + index), set_index="con_code")


def get_date(trade_date, assets=["E"], freq="D", market="CN"):
    if len(assets) != 1:
        return get_date_all()
    else:
        return get(LB.a_path("Market/" + market + "/Date/" + assets[0] + "/" + freq + "/" + str(trade_date)), set_index="ts_code")


def get_date_all(trade_date, assets=["E", "I", "FD"], freq="D", format=".feather", market="CN"):
    if len(assets) == 1:
        return get_date(trade_date=trade_date, assets=assets, freq=freq, market=market)
    else:
        raise NotImplementedError('This function is not implemented yet')
    # TODO implement it


def get_date_E_oth(trade_date, oth_name, market="CN"):
    return get(LB.a_path("Market/" + market + "/Date/E/" + oth_name + "/" + str(trade_date)), set_index="")  # nothing to set


def get_market(freq, market="CN"):
    return get(LB.a_path("Market/" + market + "/Date/market/" + freq + ".csv"), set_index="")  # dont know what to set


# needs to be optimized for speed and efficiency
def add_static_data(df, assets=["E", "I", "FD"], market="CN"):
    df_result = pd.DataFrame()
    for asset in assets:
        df_asset = get_ts_code(asset)
        df_result = df_result.append(df_asset, sort=False, ignore_index=False)
    df.index.name = "ts_code"  # important, otherwise merge will fail
    return pd.merge(df, df_result, how='left', on=["ts_code"], suffixes=[False, False], sort=False)


# require: trade_date
# function: adds another asset close price
def add_asset_comparison(df, freq, asset, ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"]):
    dict_rename = {column: f"{column}_{ts_code}" for column in a_compare_label}
    df_compare = get_asset(ts_code, asset, freq)[a_compare_label]
    df_compare.rename(columns=dict_rename, inplace=True)
    LB.columns_remove(df, [label + "_" + ts_code for label in a_compare_label])
    return pd.merge(df, df_compare, how='left', on=["trade_date"], suffixes=["", ""], sort=False)


def add_asset_final_analysis_rank(df, assets, freq, analysis="bullishness", market="CN"):
    path = "Market/CN/Backtest_Single/" + analysis + "/EIFD_D_final.xlsx"
    df_analysis = pd.read_excel(path, sheet_name="Overview")
    final_score_label = ["ts_code"] + [s for s in df_analysis.columns if "final_" + analysis + "_rank" in s]
    df_analysis = df_analysis[final_score_label]
    return pd.merge(df, df_analysis, how='left', on=["ts_code"], suffixes=[False, False], sort=False)


def preload(load="asset", step=1, query=""):
    dict_result = {}
    df_listing = get_ts_code()[::step] if load == "asset" else get_trade_date(start_date="20000101")[::step]
    func = get_asset if load == "asset" else get_date

    bar = tqdm(range(len(df_listing)))
    bar.set_description(f"loading {load}...")
    for iterator, i in zip(df_listing.index, bar):
        try:
            df = func(iterator)
            df = df[(df["period"] > 240)]
            if query:
                df = df.query(expr=query)
            if df.empty:
                continue
            else:  # only take df that satisfy ALL conditions and is non empty
                dict_result[iterator] = df
        except:
            pass
    bar.close()
    return dict_result


def update_all_in_one(big_update=False):
    # there are 3 types of update

    # TODO update completely is to delete all folder. Else update smart
    # 0. ALWAYS UPDATE
    # 1. ONLY ON BIG UPDATE: OVERRIDES EVERY THING EVERY TIME
    # 2. ON BOTH BIG AND SMALL UPDATE: OVERRIDES EVERYTHING EVERY TIME
    # 3. SMART: BIG OR SMALL UPDATE DOES NOT MATTER, ALWAYS CHECK IF FILE NEEDS TO BE UPDATED

    big_steps = [1, 2, 3, 5, 7, 9, - 1, -2, -3, -5, -7, -9]
    middle_steps = [1, 2, 3, 5, -1, -2, -3, -5]
    small_steps = [1, 2, -1, -2]

    # 1.0. GENERAL - CAL_DATE
    # update_general_trade_cal()  # always update
    #
    # # 1.1. GENERAL - INDUSTRY
    # for level in c_industry_level():
    #     update_general_industry(level, big_update=big_update)  # ONLY ON BIG UPDATE

    # 1.2. GENERAL - TOP HOLDER
    # multi_process(func=update_assets_E_top_holder, a_kwargs={"big_update": False}, a_steps=small_steps)  # SMART
    #
    # # 1.3. GENERAL - TS_CODE
    # for asset in c_assets():
    #     update_general_ts_code(asset)  # ALWAYS UPDATE
    # update_general_ts_code_all()
    # #
    # # 1.5. GENERAL - TRADE_DATE
    # for freq in ["D", "W"]:  # Currently only update D and W, because W is needed for pledge stats
    #     update_general_trade_date(freq)  # ALWAYS UPDATE

    # 2.1. ASSET - FUNDAMENTALS
    multi_process(func=update_assets_E_D_Fun, a_kwargs={"start_date": "00000000", "end_date": today(), "big_update": False}, a_steps=big_steps)  # SMART
    multi_process(func=update_assets_E_W_pledge_stat, a_kwargs={"start_date": "00000000", "big_update": False}, a_steps=small_steps)  # SMART

    # 2.2. ASSET - DF
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": False}, a_steps=middle_steps)  # SMART

    # 3.1. DATE - OTH
    # multi_process(func=update_date_E_Oth, a_kwargs={"asset": "E", "freq": "D", "big_update": big_update}, a_steps=[1, -1])  # big: smart decide - small: smart decide

    # 3.2. DATE - DF
    # date_step = [-1, 1] if big_update else [-1, 1]
    # multi_process(func=update_date, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART
    #
    # # 3.3. DATE - BASE
    # update_date_base(start_date="19990101", end_date=today(), big_update=big_update, assets=["E"])  # SMART
    #
    # # 3.4. DATE - TREND
    # df = get_stock_market_all()  # ALWAS
    # ICreate.trend(df=df, ibase="close", market_suffix="market.")  # big: override - small: override
    # LB.to_csv_feather(df, a_path=LB.a_path("Market/CN/Backtest_Multiple/Setup/Stock_Market/all_stock_market"))
    #
    # # 4.1. CUSTOM - INDEX
    # update_custom_index(big_update=big_update)


# speed order=remove apply and loc, use vectorize where possible
# 1. vectorized
# 2. List comprehension
# 3. apply
# 4. loc
if __name__ == '__main__':
    # TODO update all in one so that it can run from zero to hero in one run
    # TODO make update general ts_code state_company faster and not update very damn time
    pr = cProfile.Profile()
    pr.enable()
    try:
        big_update = False
        update_all_in_one(big_update=big_update)
        # TODO add update ts_code back and update it make it fastr



    except Exception as e:
        traceback.print_exc()
        print(e)
        LB.sound("error.mp3")
    pr.disable()
    # pr.print_stats(sort='file')

# slice
# a[-1]    # last item in the array
# a[-2:]   # last two items in the array
# a[:-2]   # everything except the last two items
# a[::-1]    # all items in the array, reversed
# a[1::-1]   # the first two items, reversed
# a[:-3:-1]  # the last two items, reversed
# a[-3::-1]  # everything except the last two items, reversed
