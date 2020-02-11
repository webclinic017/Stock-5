import tushare as ts
import pandas as pd
import numpy as np
import talib
import API_Tushare
import Util
import os.path
import inspect
from itertools import combinations
import operator
import math
import numba
from numba import jit
from numba import njit
import traceback
import cProfile
from tqdm import tqdm
from Util import c_assets, c_rolling_freqs, c_date_oth, c_assets_fina_function_dict, c_industry_level, c_ops, c_candle, c_groups_dict, multi_process, today

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def update_general_trade_cal(start_date="19900101", end_date="20250101"):
    df = API_Tushare.my_trade_cal(start_date=start_date, end_date=end_date)
    a_path = Util.a_path("Market/CN/General/trade_cal_D")
    Util.to_csv_feather(df, a_path)


def update_general_trade_date(freq="D", market="CN", big_update=True):
    if (freq in ["D", "W"]):
        a_path = Util.a_path("Market/" + market + "/General/trade_date_" + freq)
        df = API_Tushare.my_pro_bar(ts_code="000001.SH", start_date="00000000", end_date=today(), freq=freq, asset="I")
        df = Util.df_reverse_reindex(df)

        df = df[["trade_date"]]
        df = update_general_trade_date_stockcount(df)  # adds E,I,FD count
        df = update_general_trade_date_seasonal_score(df, freq, market)  # adds seasonal score for each day
        Util.to_csv_feather(df, a_path, encoding='utf-8_sig')


def update_general_trade_date_seasonal_score(df_trade_date, freq="D", market="CN"):
    # get all indicator for each day
    path_indicator = "Market/CN/Backtest_Single/seasonal/all_date_seasonal.xlsx"
    xls = pd.ExcelFile(path_indicator)

    # get all different groups
    a_groups = [[Util.get_trade_date_datetime_dayofweek, "dayofweek"],  # 1-5
                [Util.get_trade_date_datetime_m, "monthofyear"],  # 1-12
                [Util.get_trade_date_datetime_d, "dayofmonth"],  # 1-31
                [Util.get_trade_date_datetime_weekofyear, "weekofyear"],  # 1-51
                [Util.get_trade_date_datetime_dayofyear, "dayofyear"],  # 1-365
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
        # rename list_date to trade_date and then group it by trade_date
        df_ts_codes = df_ts_codes.rename(columns={"list_date": "trade_date"})
        df_ts_codes["trade_date"] = df_ts_codes["trade_date"].astype(int)
        df_grouped = df_ts_codes[["trade_date", "ts_code"]].groupby(by="trade_date").count()

        # vecorized approach faster than loop over individual date
        df[asset + "_count"] = df_grouped["ts_code"].astype(int).cumsum()
        df[asset + "_count"] = df[asset + "_count"].fillna(method="ffill")
        df[asset + "_count"] = df[asset + "_count"].fillna(0)
    return df


def update_general_ts_code(asset="E", market="CN", big_update=True):
    print("start update general ts_code ", asset)
    a_path = Util.a_path("Market/" + market + "/General/ts_code_" + asset)

    if (asset == "E"):
        df = API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="")

        # add asset
        df["asset"] = asset

        # add exchange info for each stock
        df["exchange"] = df['ts_code'].apply(lambda x: "创业板" if x[0:3] in ["300"] else "中小板" if x[0:3] in ["002"] else "主板" if x[0:2] in ["00", "60"] else float("nan"))

        # add SW industry info for each stock
        for level in c_industry_level():
            df_industry = pd.DataFrame()
            df_industry_classify = get_industry_classify(level)
            for industry_index in df_industry_classify["index_code" + level]:
                df_industry = df_industry.append(get_industry_index(level=level, index=industry_index), sort=False)
            df_industry = df_industry[["index_code", "con_code"]]
            df_industry = df_industry.rename(columns={"con_code": "ts_code", "index_code": "index_code" + str(level)})
            df = pd.merge(df, df_industry, how='left', on=["ts_code"], suffixes=[False, False], sort=False)
            df = pd.merge(df, df_industry_classify, how='left', on=["index_code" + str(level)], suffixes=[False, False], sort=False)
        Util.columns_remove(df, ["index_code" + x for x in c_industry_level()])

        # add State Government for each stock
        df.index = df["ts_code"]  # let industry merge first, otherwise error
        df["state_company"] = False
        for ts_code in df.ts_code:
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
        df = df_SSE.append(df_SZSE, sort=False)
        df["asset"] = asset

    elif (asset == "FD"):
        df_E = API_Tushare.my_fund_basic(market='E')
        df_O = API_Tushare.my_fund_basic(market='O')
        df = df_E.append(df_O, sort=False)
        df["asset"] = asset
    else:
        df = pd.DataFrame()

    df.reset_index(drop=True, inplace=True)
    df["list_date"] = df["list_date"].fillna(method='ffill')
    Util.to_csv_feather(df, a_path, encoding='utf-8_sig')


def update_general_ts_code_all(market="CN"):
    a_path = Util.a_path("Market/" + market + "/General/ts_code_all")

    df = pd.DataFrame()
    for asset in c_assets():
        df_asset = get_ts_code(asset)
        df = df.append(df_asset, sort=False, ignore_index=True)

    Util.to_csv_feather(df, a_path, encoding='utf-8_sig')


def update_general_industry_classify(level, market="CN", big_update=True):
    if not big_update:
        return

    path_array = Util.a_path("Market/" + market + "/General/industry_" + level)
    df = API_Tushare.my_index_classify(f"L" + level)
    if not df.empty:
        df = df[["index_code", "industry_name"]].rename(columns={"industry_name": "industry" + level, "index_code": "index_code" + level})
        Util.to_csv_feather(df, a_path=path_array, encoding='utf-8_sig')
    else:
        print(inspect.currentframe().f_code.co_name, "is empty!")


def update_general_industry_index(level, market="CN", big_update=True):
    if not big_update:
        return

    df_all_industry_index = get_industry_classify(level)

    for index in df_all_industry_index["index_code" + level]:
        a_path = Util.a_path("Market/" + market + "/General/industry/" + level + "/" + index)
        df = API_Tushare.my_index_member(index)
        Util.to_csv_feather(df, a_path, encoding='utf-8_sig')


def update_assets_EIFD_D(asset="E", freq="D", market="CN", step=1, big_update=True):
    a_path_empty = Util.a_path("Market/CN/General/ts_code_ignore")

    def get_ts_code_ignore():
        df_saved = read(a_path_empty)
        df_ts_code_all = get_ts_code_all()
        df_ts_code_all = df_ts_code_all[["ts_code", "asset"]]

        if len(df_saved.columns) != 0 and not big_update:  # update existing column with latest ts_code
            df_saved = pd.merge(df_ts_code_all, df_saved, how="left")
            df_saved["ignore"] = df_saved["ignore"].fillna(False)
            return df_saved
        else:  # create new empty file with columns
            df_ts_code_all["ignore"] = False
            df_ts_code_all["reason"] = np.nan
            return df_ts_code_all

    def update_ts_code_empty(df_empty_I):
        Util.to_csv_feather(df_empty_I, a_path_empty)

    def update_ignore_list(df, df_empty_EI, real_latest_trade_date):
        if df.empty:
            df_empty_EI.loc[df_empty_EI["ts_code"] == ts_code, ["ignore", "reason"]] = (True, "empty")
            update_ts_code_empty(df_empty_EI)
        elif str(df["trade_date"].tail(1).values.tolist()[0]) != real_latest_trade_date:
            df_empty_EI.loc[df_empty_EI["ts_code"] == ts_code, ["ignore", "reason"]] = (True, "no updated anymore")
            update_ts_code_empty(df_empty_EI)

    df_ignore_EI = get_ts_code_ignore()
    print("Index not to consider:", df_ignore_EI.loc[df_ignore_EI["ignore"] == True, "ts_code"].tolist())

    df_ts_codes = get_ts_code(asset)
    df_ts_codes = df_ts_codes[~df_ts_codes["ts_code"].isin(df_ignore_EI.loc[df_ignore_EI["ignore"] == True, "ts_code"].tolist())]

    real_latest_trade_date = get_last_trade_date(freq)

    for ts_code in df_ts_codes["ts_code"][::step]:
        start_date = "00000000"
        middle_date = "20050101"
        end_date = today()

        complete_new_update = True  # True means update from 00000000 to today, False means update latest_trade_date to today

        a_path = Util.a_path("Market/" + market + "/Asset/" + asset + "/" + freq + "/" + ts_code)
        df_saved = pd.DataFrame()

        # file exists--> check latest_trade_date, else update completely new
        if os.path.isfile(a_path[0]):
            try:
                df_saved = read(a_path)  # get latest file trade_date
                asset_latest_trade_date = str(df_saved["trade_date"].tail(1).values.tolist()[0])
            except:
                asset_latest_trade_date = start_date
                print(asset, ts_code, freq, end_date, "EMPTY - START UPDATE", asset_latest_trade_date, " to today")

            # file exist and on latest date--> finish, else update
            if (str(asset_latest_trade_date) == str(real_latest_trade_date)):
                print(asset, ts_code, freq, end_date, "Up-to-date", real_latest_trade_date)
                continue
            else:  # file exists and not on latest date
                if ts_code in df_ignore_EI.loc[df_ignore_EI["ignore"] == True, "ts_code"]:
                    print(asset, ts_code, freq, end_date, "FILLER INDEX AND Up-to-date", real_latest_trade_date)
                    continue
                else:  # file exists and not on latest date, AND stock trades--> update
                    complete_new_update = False

        # file not exist or not on latest_trade_date --> update
        if (asset == "E" and freq == "D"):
            # 1.1 get df
            if complete_new_update:
                df1 = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, middle_date, adj="hfq")
                df2 = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, middle_date, end_date, adj="hfq")
                df = df1.append(df2, ignore_index=True, sort=False)
            else:
                df = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="hfq")

            update_ignore_list(df, df_ignore_EI, real_latest_trade_date)

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
                df = pd.merge(df, df_fun, how='left', on=["trade_date"], suffixes=[False, False], sort=False)
            except:
                pass

            # 2.2 add FUN financial report aka fina
            # 流动资产合计,非流动资产合计,资产合计，   流动负债合计,非流动负债合计,负债合计
            df_balancesheet = get_assets_E_D_Fun("balancesheet", ts_code=ts_code, columns=["end_date"])

            # 营业活动产生的现金流量净额	，投资活动产生的现金流量净额,筹资活动产生的现金流量净额
            df_cashflow = get_assets_E_D_Fun("cashflow", ts_code=ts_code, columns=["end_date", "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"])

            # 扣除非经常性损益后的净利润,净利润同比增长(netprofit_yoy instead of q_profit_yoy on the doc)，营业收入同比增长,销售毛利率，销售净利率,资产负债率,存货周转天数(should be invturn_days,but casted to turn_days in the api for some reasons)
            df_indicator = get_assets_E_D_Fun("fina_indicator", ts_code=ts_code, columns=["end_date", "profit_dedt", "netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_assets"])

            # 股权质押比例
            df_pledge_stat = get_assets_pledge_stat(ts_code=ts_code, columns=["end_date", "pledge_ratio"])

            # rename end_date to trade_date
            for df_fun in [df_balancesheet, df_cashflow, df_indicator, df_pledge_stat]:
                df_fun.rename(columns={'end_date': 'trade_date'}, inplace=True)

            # set all df_trade_trade_date type to int
            for df_toset in [df, df_balancesheet, df_cashflow, df_indicator, df_pledge_stat, df_saved]:
                if not df_toset.empty:
                    df_toset["trade_date"] = df_toset["trade_date"].astype(int)  # IMPORTANT we change end_date to trade_date later

            # merge fina with df
            for df_fun in [df_balancesheet, df_cashflow, df_indicator, df_pledge_stat]:
                df = pd.merge(df, df_fun, how='left', on=["trade_date"], suffixes=[False, False], sort=False)

            # append old df and drop duplicates
            if not df_saved.empty:
                df = df_saved.append(df, sort=False)
            df = Util.df_drop_duplicated_reindex(df, "trade_date")

            # interpolate/fill between empty fina and pledge_stat values
            all_report_label = list(df_balancesheet.columns.values) + list(df_cashflow.columns.values) + list(df_indicator.columns.values) + ["pledge_ratio"]
            for label in all_report_label:
                try:
                    df[label] = df[label].fillna(method='ffill')
                except:
                    pass

        elif (asset == "I" and freq == "D"):
            if complete_new_update:
                df = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj="qfq")
            else:
                df = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="qfq")
                df = df_saved.append(df, sort=False)
                df = Util.df_drop_duplicated_reindex(df, "trade_date")

            update_ignore_list(df, df_ignore_EI, real_latest_trade_date)

        elif (asset == "FD" and freq == "D"):
            if complete_new_update:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date)
            else:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date)
                df = df_saved.append(df, sort=False)
                df = Util.df_drop_duplicated_reindex(df, "trade_date")

            update_ignore_list(df, df_ignore_EI, real_latest_trade_date)
        # 3. add my derivative indices
        if not df.empty:
            df = update_assets_EIFD_D_technical(df=df, df_saved=df_saved, asset=asset)

        # save asset
        Util.to_csv_feather(df, a_path)
        print(asset, ts_code, freq, end_date, "UPDATED!", real_latest_trade_date)

    # update empty I list
    update_ts_code_empty(df_ignore_EI)


# For all Pri indices and derivates
def update_assets_EIFD_D_technical(df, df_saved, asset="E"):
    complete_new_update = True if len(df_saved) == 0 else False
    traceback_freq_big = c_rolling_freqs()
    traceback_freq_small = [2, 5]

    if asset == "E":
        Util.add_ivola(df, df_saved=df_saved, complete_new_update=complete_new_update)  # 0.890578031539917 for 300 loop
    Util.add_period(df, complete_new_update=complete_new_update)  # 0.2 for 300 loop
    Util.add_indi_rs(df)  # notimplemented
    Util.add_pjump_up(df, complete_new_update=complete_new_update)  # 1.0798187255859375 for 300 loop
    Util.add_pjump_down(df, complete_new_update=complete_new_update)  # 1.05 independend for 300 loop
    Util.add_candle_signal(df, complete_new_update=complete_new_update)  # VERY SLOW. NO WAY AROUND. 120 sec for 300 loop

    for rolling_freq in traceback_freq_small[::-1]:
        if asset == "E":
            Util.column_add_mean(df, rolling_freq, "turnover_rate", complete_new_update=complete_new_update)  # dependend
            df[f"turnover_rate_pct{rolling_freq}"] = df["turnover_rate"] / df[f"turnover_rate{rolling_freq}"]
        # Util.column_add_std(df, rolling_freq, "close", complete_new_update=complete_new_update)  # dependend

    for rolling_freq in traceback_freq_big[::-1]:
        # Util.column_add_mean(df, rolling_freq, "close", complete_new_update=complete_new_update)  # dependend
        Util.add_pgain(df, rolling_freq, complete_new_update=complete_new_update)
        Util.add_fgain(df, rolling_freq, complete_new_update=complete_new_update)

    # add trend for individual stocks
    Setup_date_trend_once(a_all=[1] + c_rolling_freqs(), df_result=df, close_label="close", index_label="", index=[], thresh=0.5, dict_ops=c_ops(), op_sign="gt", thresh_log=-0.043, thresh_rest=0.7237, for_analysis=False, market_suffix="")

    return df


# KEEPIT IN CASE YOU NEED OTHER FREQUENCY LATER
# def update_assets_EIFD_WMYS(asset="PD", freq="W", market="CN"):
#     ts_codes = get_ts_code(asset)
#     for ts_code in ts_codes.ts_code:
#         path_d = "Market/" + market + "/Asset/" + asset + "/" + "D" + "/" + ts_code + ".csv"
#         path = "Market/" + market + "/Asset/" + asset + "/" + freq + "/" + ts_code + ".csv"
#
#         # check if daily file exists, only makes sense if it does
#         if os.path.isfile(path_d):
#             df_d = pd.read_csv(path_d)
#             if df_d.empty:
#                 print("NEW STOCK: file EXISTS BUT file is EMPTY for update_assets_WMYS", asset, ts_code, "D")
#                 df_result = Library_Main.empty_asset_Tushare(asset)
#                 df_result.to_csv(path, index=False)
#                 continue
#             else:
#                 start_date = df_d.at[0, "trade_date"]
#                 end_date = df_d.at[len(df_d) - 1, "trade_date"]
#                 df_trade_date_WMYS = get_trade_date(freq, start_date, end_date)
#         else:
#             print(asset, ts_code, freq, " has no Daily file for FD! Continue")
#             continue
#
#         # if WMYS file exits
#         if os.path.isfile(path):
#             df_saved = pd.read_csv(path)
#             if (df_saved.empty):
#                 continue
#             else:
#                 # check if last db stored is the latest
#                 asset_latest_trade_date = str(int(df_saved.at[len(df_saved) - 1, "trade_date"]))
#                 real_latest_trade_date = get_last_trade_date(freq)
#                 if (str(asset_latest_trade_date) == str(real_latest_trade_date)):
#                     print(asset, ts_code, freq, start_date, end_date, "is already updated to latest trade_date", real_latest_trade_date)
#                     continue
#                 else:
#                     df_trade_date_WMYS = get_trade_date(freq, asset_latest_trade_date, real_latest_trade_date)
#                     df = update_assets_helper_WMYS(ts_code, freq, asset, start_date, end_date, df_d, df_trade_date_WMYS)
#                     df_saved = df_saved.append(df, ignore_index=True, sort=False)
#
#                     df_saved["pct_chg"] = df_saved.close.pct_change() * 100
#                     df_saved.iat[0, df_saved.columns.get_loc("pct_chg")] = df_saved.iat[0, df_saved.columns.get_loc("close")] / df_saved.iat[0, df_saved.columns.get_loc("open")]
#
#                     df_saved.to_csv(path, index=False)
#
#         # file WMYS does not exist
#         else:
#             # use D file to update for WMYS
#             df_result = update_assets_helper_WMYS(ts_code, freq, asset, start_date, end_date, df_d, df_trade_date_WMYS)
#             if (df_result.empty):
#                 df_result = Library_Main.empty_asset_Tushare(asset)
#                 df_result.to_csv(path, index=False)
#                 continue
#             df_result["pct_chg"] = df_result.close.pct_change() * 100
#             df_result.iat[0, df_result.columns.get_loc("pct_chg")] = df_result.iat[0, df_result.columns.get_loc("close")] / df_result.iat[0, df_result.columns.get_loc("open")]
#             df_result.to_csv(path, index=False)


# def update_assets_helper_WMYS(ts_code, freq, asset, start_date, end_date, df_D, df_trade_date_WMYS):
#     df_result = pd.DataFrame()
#     s_trade_date, s_open, s_high, s_low, s_close, s_vol, s_turnover_rate = (pd.Series(),) * 7
#
#     for df_wmys in D_to_WMYS_splitter(df_trade_date_WMYS, df_D):
#         if (df_wmys.empty):
#             continue
#         df_wmys = Library_Main.df_reindex(df_wmys)
#         s_trade_date = s_trade_date.append(pd.Series(df_wmys.iat[len(df_wmys) - 1, df_wmys.columns.get_loc("trade_date")]), ignore_index=True)
#         s_open = s_open.append(pd.Series(df_wmys.at[0, "open"]), ignore_index=True)
#         s_close = s_close.append(pd.Series(df_wmys.at[len(df_wmys.index) - 1, "close"]), ignore_index=True)
#         s_high = s_high.append(pd.Series(df_wmys[['open', "high", "low", "close"]].max().max()), ignore_index=True)
#         s_low = s_low.append(pd.Series(df_wmys[['open', "high", "low", "close"]].min().min()), ignore_index=True)
#         s_vol = s_vol.append(pd.Series(df_wmys.vol.sum()), ignore_index=True)
#
#         if "turnover_rate" in df_D.columns:
#             s_turnover_rate = s_turnover_rate.append(pd.Series(df_wmys.turnover_rate.sum()), ignore_index=True)
#
#     df_result["trade_date"] = s_trade_date
#     df_result["ts_code"] = ts_code
#
#     df_result = Library_Main.df_reindex(df_result)
#
#     df_result["open"] = s_open
#     df_result["high"] = s_high
#     df_result["low"] = s_low
#     df_result["close"] = s_close
#     df_result["vol"] = s_vol
#
#     df_result = df_result.round(4)
#     print(asset, ts_code, freq, start_date, end_date, "updated to path")
#     return df_result

# For E,I,FD  D
def update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj="qfq", market="CN"):
    df = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
    df = Util.df_reverse_reindex(df)
    Util.columns_remove(df, ["pre_close", "amount", "change"])
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

        df = Util.df_reverse_reindex(df)
        Util.columns_remove(df, ["pre_close", "amount", "change", "adj_factor"])
    return df


# recommended update frequency W
def update_assets_E_D_Fun(start_date="0000000", end_date=Util.today(), market="CN", step=1, big_update=True):
    if not big_update:
        return

    for ts_code, counter in zip(get_ts_code("E").ts_code[::step], range(0, len(get_ts_code("E").ts_code[::step]))):
        for fina_name, fina_function in c_assets_fina_function_dict().items():
            a_path = Util.a_path("Market/" + market + "/Asset/E/D_Fun/" + fina_name + "/" + ts_code)
            # ALWAYS UPDATES COMPLETELY because We can not detect by looking at file if it is most recent
            df = fina_function(ts_code=ts_code, start_date=start_date, end_date=end_date)
            df = Util.df_reverse_reindex(df)
            Util.to_csv_feather(df, a_path)
        print(counter, ts_code, "update FUN finished")


# recommended update frequency W
def update_assets_E_W_pledge_stat(start_date="00000000", market="CN", step=1, big_update=True):
    if not big_update:
        return

    real_latest_trade_date = get_last_trade_date("W")
    for ts_code, counter in zip(get_ts_code("E").ts_code[::step], range(0, len(get_ts_code("E").ts_code[::step]))):
        a_path = Util.a_path("Market/" + market + "/Asset/E/W_pledge_stat/" + str(ts_code))
        if os.path.isfile(a_path[0]):
            try:
                df_saved = read(a_path)
                asset_latest_trade_date = str(df_saved.at[len(df_saved) - 1, "end_date"])
            except:
                asset_latest_trade_date = start_date

            # file is on latest date--> finish
            if (str(asset_latest_trade_date) == str(real_latest_trade_date)):
                print(counter, ts_code, "pledge_stat Up-to-date", real_latest_trade_date)
                continue

        df = API_Tushare.my_pledge_stat(ts_code=ts_code)
        df = Util.df_reverse_reindex(df)
        Util.to_csv_feather(df, a_path)
        print(counter, ts_code, "pledge_stat UPDATED", real_latest_trade_date)


def update_assets_E_top_holder(big_update=True, market="CN", step=1):
    for ts_code, counter in zip(get_ts_code("E").ts_code[::step], range(0, len(get_ts_code("E").ts_code[::step]))):
        a_path = Util.a_path("Market/" + market + "/Asset/E/D_top_holder/" + str(ts_code))

        if not big_update:  # For small update. If File exists, Finish. For big Update, overwrite completely
            if os.path.isfile(a_path[0]):
                print(counter, ts_code, "top_holder Up-to-date")
                continue

        # always update completely new
        df = API_Tushare.my_query(api_name='top10_holders', ts_code=ts_code, start_date='20190101', end_date=today())
        df = Util.df_reverse_reindex(df)
        Util.to_csv_feather(df, a_path, encoding="utf-8_sig")
        print(counter, ts_code, "top_holder UPDATED")


# merges all date file of E,I,FD into one. Do Not confuse with update_all_date!
def update_date_all():
    # TODO merge all date files together (with static data)into one file
    pass


def update_date(asset="E", freq="D", market="CN", step=1, big_update=True):
    for asset in ["E"]:  # TODO add I, FD date
        update_date_EIFD_DWMYS(asset, freq, big_update=big_update, step=step)


def update_date_EIFD_DWMYS(asset="E", freq="D", market="CN", big_update=True, step=1):  # STEP only 1 or -1 !!!!
    df_ts_codes = get_ts_code(asset)
    df_ts_codes["list_date"] = df_ts_codes["list_date"].astype(int)
    trade_dates = get_trade_date("00000000", Util.today(), freq)

    # get the latest column of the asset file
    if asset == "E":
        code = "000001.SZ"
    elif asset == "I":
        code = "000001.SH"
    else:
        code = "150001.SZ"
    example_df = get_asset(code, asset, freq)
    example_column = list(example_df.columns)

    # makes searching for one day in asset time series faster. BUT can only be used with step=1 and ONLY using ONE THREAD
    print("Update date preparing for setup. Please wait...")
    dict_list_date = {ts_code: list_date for ts_code, list_date in zip(df_ts_codes["ts_code"], df_ts_codes["list_date"])}
    dict_df = {ts_code: get_asset(ts_code=ts_code) for ts_code in df_ts_codes["ts_code"]}

    if step == 1:  # step 1 means iterate forward, -1 means iterate backwards
        dict_lookup_table = {ts_code: 0 for ts_code, df in dict_df.items()}
    elif step == -1:
        dict_lookup_table = {ts_code: len(df) - 1 for ts_code, df in dict_df.items()}
    else:
        print("ERROR, STEP MUST BE 1 or -1.")
        return
    print("Update date finished preparation")

    for trade_date in trade_dates["trade_date"][::step]:  # IMPORTANT! do not modify step, otherwise lookup will not work
        a_path = Util.a_path("Market/" + market + "/Date/" + asset + "/" + freq + "/" + str(trade_date))
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
                if list_date > trade_date:
                    continue

                row_number = dict_lookup_table[ts_code]  # lookup table can not be changed while iterating over it.
                try:
                    if df_asset.index[row_number] == trade_date:
                        # a_date.append(df_asset.loc[[row_number]])#.to_numpy().flatten()
                        a_date.append(df_asset.loc[trade_date].to_numpy().flatten())
                        dict_lookup_table[ts_code] = dict_lookup_table[ts_code] + step
                except:
                    continue

            # df_date = pd.concat(objs=a_date,ignore_index=True,sort=False)
            df_date = pd.DataFrame(data=a_date, columns=example_column)
            df_date.insert(loc=0, column='trade_date', value=int(trade_date))
            df_date = pd.merge(df_date, df_ts_codes, how='left', on=["ts_code"], suffixes=[False, False], sort=False)
            Util.to_csv_feather(df_date, a_path)
            print(asset, freq, trade_date, "date updated")


def update_date_E_Oth(asset="E", freq="D", market="CN", big_update=True, step=1):
    trade_dates = get_trade_date(start_date="00000000", end_date=today(), freq=freq)
    for trade_date in trade_dates["trade_date"][::step]:
        dict_oth_names = c_date_oth()
        dict_oth_paths = {name: Util.a_path("Market/" + market + "/Date/" + asset + "/" + name + "/" + str(trade_date)) for name, function in dict_oth_names.items()}

        for name, function in dict_oth_names.items():
            if os.path.isfile(dict_oth_paths[name][0]):
                print(trade_date, asset, freq, name, "Up-to-date")
            else:
                df_oth = function(trade_date)
                Util.to_csv_feather(df_oth, dict_oth_paths[name], encoding='utf-8_sig')
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
    last_saved_date = "19990101"
    last_trade_date = get_last_trade_date("D")
    if not big_update:

        try:
            df_saved = get_stock_market_all().reset_index()
            last_saved_date = df_saved.index[len(df_saved) - 1]
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
    a_path = Util.a_path("Market/" + market + "/Backtest_Multiple/Setup/Stock_market/all_stock_market")
    a_result = []
    df_sh_index = get_asset(ts_code="000001.SH", asset="I", freq=freq, market="CN")
    df_sh_index = df_sh_index.loc[int(last_saved_date):int(last_trade_date)]
    print(df_sh_index)

    # loop through all trade dates and add them together
    for trade_date, sh_pct_chg in zip(df_sh_index.index, df_sh_index["pct_chg"]):
        print(trade_date, "being added to all_stock_market_base")
        df_date = get_date(str(trade_date), assets=assets, freq=freq, market="CN")
        df_date = df_date[df_date["period"] >= 240]  # IMPORTANT disregard ipo stocks

        trading_stocks = len(df_date)
        df_date_mean = df_date.mean()
        df_date_mean["trading"] = trading_stocks
        df_date_mean["pct_chg_std"] = df_date["pct_chg"].std()
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
    df_result["comp_chg"] = Util.column_add_comp_chg(df_result["pct_chg"])
    for ts_code in comparison_index:
        df_result = add_asset_comparison(df=df_result, freq=freq, asset="I", ts_code=ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"])
        df_result["comp_chg_" + ts_code] = Util.column_add_comp_chg(df_result["pct_chg_" + ts_code])

    print(df_result)
    Util.to_csv_feather(df_result, a_path, index=True, reset_index=False)
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
        last_saved_date = example_df.at[len(example_df) - 1, "trade_date"]
    except:
        last_saved_date = "19990101"

    # if small update and saved_date==last trade_date
    if last_trade_date == last_saved_date and (not big_update):
        print("ALL GROUP Up-to-date")
        return

    # initialize trade_date
    df_trade_date = get_trade_date(end_date=today(), freq="D")
    df_trade_date = df_trade_date[df_trade_date["trade_date"] > int(last_saved_date)]
    print("START UPDATE GROUP since", last_saved_date)

    # loop over date and get mean
    for trade_date in df_trade_date["trade_date"]:  # for each day
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

        a_path = Util.a_path("Market/CN/Backtest_Multiple/Setup/Stock_Market/Group/" + key)
        Util.to_csv_feather(df_update, a_path)
        print(key, "UPDATED")


def update_cj_index_000001_SH():
    df = get_stock_market_all()
    print(df.index)
    df = df[["pct_chg"]]
    print(df.index)
    a_path = Util.a_path("Market/CN/Asset/I/D/CJ000001.SH")
    Util.to_csv_feather(df, a_path, index=False, reset_index=True, drop=False)


def update_asset_candle(market="CN", asset="E", freq="D", step=1):
    df_ts_code = get_ts_code("E")
    dict_candle = c_candle()

    for ts_code, counter in zip(df_ts_code.ts_code[::step], range(0, len(df_ts_code))):
        df = get_asset(ts_code=ts_code, asset=asset, freq="D")
        if df.empty:
            continue

        a_path = Util.a_path("Market/" + market + "/Asset/" + str(asset) + "/" + str(freq) + "/" + str(ts_code))

        a_positive_candle_columns = []
        a_negative_candle_columns = []

        # create candle stick column
        for key, array in dict_candle.items():
            if (array[1] != 0) or (array[2] != 0):  # if used at any, calculate the pattern
                func = array[0]
                df[key] = func(open=df["open"], high=df["high"], low=df["low"], close=df["close"]).replace(0, np.nan)

                if (array[1] != 0):  # candle used as positive pattern
                    a_positive_candle_columns.append(key)
                    if (array[1] == -100):  # talib still counts the pattern as negative: cast it positive
                        df[key] = df[key].replace(-100, 100)

                if (array[2] != 0):  # candle used as negative pattern
                    a_negative_candle_columns.append(key)
                    if (array[2] == 100):  # talib still counts the pattern as positive: cast it negative
                        df[key] = df[key].replace(100, -100)

        # create a compound flag for all 12 positive and negative candle signals
        df_pos = df[a_positive_candle_columns]
        df_neg = df[a_negative_candle_columns]

        df["candle_pos"] = (df_pos[df_pos[a_positive_candle_columns] == 100].sum(axis='columns') / 100)
        df["candle_neg"] = (df_neg[df_neg[a_negative_candle_columns] == -100].sum(axis='columns') / 100)
        df["candle_net_pos"] = (df["candle_pos"] + df["candle_neg"])
        df["candle_net_pos2"] = df["candle_net_pos"].rolling(2).sum()
        df["candle_net_pos5"] = df["candle_net_pos"].rolling(5).sum()

        # remove candle stick column
        for key, array in dict_candle.items():
            Util.columns_remove(df, [key])

        Util.to_csv_feather(df, a_path, encoding='utf-8_sig')
        print(counter, ts_code, "Candlestick UPDATED")


def update_market_WMYS(freq, market="CN"):  # TODO no idea if i need this or what to do
    a_path = Util.a_path("Market/" + market + "/Date/market/" + freq)
    df_trade_dates_WMYS = get_trade_date(start_date="00000000", end_date=today(), freq=freq)
    df_result = df_trade_dates_WMYS.copy()

    df_D = get_market("D")
    s_trade_date, s_block_trade_count, s_holdertrade_in_count, s_holdertrade_de_count, s_repurchase_count, s_share_float_count = (pd.Series(),) * 6
    for df_WMYS in D_to_WMYS_splitter(df_trade_dates_WMYS, df_D):
        if (df_WMYS.empty):
            continue
        df_WMYS = Util.df_reindex(df_WMYS)
        s_block_trade_count = s_block_trade_count.append(pd.Series(df_WMYS.q_block_trade_count.sum()), ignore_index=True)
        s_holdertrade_in_count = s_holdertrade_in_count.append(pd.Series(df_WMYS.q_holdertrade_in_count.sum()), ignore_index=True)
        s_holdertrade_de_count = s_holdertrade_de_count.append(pd.Series(df_WMYS.q_holdertrade_de_count.sum()), ignore_index=True)
        s_repurchase_count = s_repurchase_count.append(pd.Series(df_WMYS.q_repurchase_count.sum()), ignore_index=True)
        s_share_float_count = s_share_float_count.append(pd.Series(df_WMYS.q_share_float_count.sum()), ignore_index=True)

    df_result["q_block_trade_count"] = pd.Series(s_block_trade_count)
    df_result["q_holdertrade_in_count"] = pd.Series(s_holdertrade_in_count)
    df_result["q_holdertrade_de_count"] = pd.Series(s_holdertrade_de_count)
    df_result["q_repurchase_count"] = pd.Series(s_repurchase_count)
    df_result["q_share_float_count"] = pd.Series(s_share_float_count)

    Util.to_csv_feather(df_result, a_path, encoding="utf-8_sig")


def D_to_WMYS_splitter(df_trade_date_WMYS, df_D):
    a_result = []
    df_helper = pd.DataFrame(float("nan"), index=[-1], columns=list(df_trade_date_WMYS.columns))
    df_helper["trade_date"] = "00000000"
    df_trade_date_WMYS = df_helper.append(df_trade_date_WMYS)
    df_trade_date_WMYS.index = df_trade_date_WMYS.index + 1

    for i in df_trade_date_WMYS.index:
        if (i == len(df_trade_date_WMYS.index) - 1):
            break
        trade_date = df_trade_date_WMYS.at[i, "trade_date"]
        next_trade_date = df_trade_date_WMYS.at[i + 1, "trade_date"]

        df_wmys = df_D.loc[(int(trade_date) < df_D["trade_date"]) & (df_D["trade_date"] <= int(next_trade_date))]
        a_result.append(df_wmys)
    return a_result


# actually a onetime use
# One of the most important functions
def Setup_date_trend_multiple(run_once_as_date_summary=True, big_update=True):
    # current winner setting
    # trend2 on trend1
    # sh_label:"",
    # close_label:"low",
    # thresh:0.0
    # thresh_log:-0.2E-16, # trend 240 and 60 do not divide and activate
    # thresh_rest:0.7,
    # op_sign:gt
    # all_com: [1, 2, 10, 11, 12, 13, 15, 18, 20]
    # minmax:0.5

    # other possible tresh
    # more curvy log  -0.043* ln(x) + 0.7351 where trend 240 also divides
    # linear -0.0008*log(x)+0.6928 # TEST: Very bad! Log is much better than linear

    setting = {
        "sh_label": "",  # ["","_000001.SH","_399001.SZ","_399006.SZ"] use close from all stocks or sh
        "close_label": "low",  # ["close","open","high","low"]  RSI base
        "thresh": 0.0,  # [0.0, 0.05, -0.05,0.03, 0.02, -0.03,-0.02, -0.01]
        "thresh_log": -0.043,
        "thresh_rest": 0.7237,
        "op_sign": "gt",  # ["<"，">"]
        "all_comb": [1] + c_rolling_freqs(),  # tested, that the higher the ma, the less useful. somewhere between is da best
        "minmax": 0.5
    }

    ops = {"plus": operator.add, "minus": operator.sub, "prod": operator.mul, "gt": operator.gt, "lt": operator.lt}

    for sh_label in [""]:
        for close_label in ["low"]:
            for op_sign in ["gt"]:
                setting["sh_label"] = sh_label
                setting["close_label"] = close_label
                setting["op_sign"] = op_sign

                thresh = setting["thresh"]
                thresh_log = setting["thresh_log"]
                thresh_rest = setting["thresh_rest"]

                # use the function as a date_summary_generator and not a tester
                if run_once_as_date_summary:
                    df_result = pd.DataFrame(columns=[str(x) + "_pct_chg_mean" for x in setting["all_comb"]] + [str(x) + "_pct_chg_std" for x in setting["all_comb"]] + [str(x) + "_comp_chg" for x in setting["all_comb"]] + [str(x) + "_trading_days" for x in setting["all_comb"]],
                                             index=setting["all_comb"])  # index is high which the trend name is after. Column is low. which the trend uses to determin is maxium
                    df = Setup_date_trend_once(a_all=setting["all_comb"], df_result=df_result, close_label=close_label, index_label=sh_label, index=[], thresh=thresh, dict_ops=ops, op_sign=op_sign, thresh_log=thresh_log, thresh_rest=thresh_rest, market_suffix="market_")

                    a_path = Util.a_path("Market/CN/Backtest_Multiple/Setup/Stock_Market/all_stock_market")
                    Util.to_csv_feather(df, a_path)
                    return

                # assign df_result
                test_index = []
                for max_len in range(3, len(setting["all_comb"])):
                    comb = combinations(setting["all_comb"], max_len)
                    for index in list(comb):
                        test_index.append(index)

                df_result = pd.DataFrame(columns=[str(x) + "_pct_chg_mean" for x in setting["all_comb"]] + [str(x) + "_pct_chg_std" for x in setting["all_comb"]] + [str(x) + "_comp_chg" for x in setting["all_comb"]] + [str(x) + "_trading_days" for x in setting["all_comb"]],
                                         index=test_index)  # index is high which the trend name is after. Column is low. which the trend uses to determin is maxium

                # loop through and calculate for each setting
                for max_len in range(3, len(setting["all_comb"])):
                    comb = combinations(setting["all_comb"], max_len)
                    for index in list(comb):
                        part_combi = [x for x in index]
                        Setup_date_trend_once(a_all=part_combi, df_result=df_result, close_label=close_label, index_label=sh_label, index=index, thresh=thresh, dict_ops=ops, op_sign=op_sign, thresh_log=thresh_log, thresh_rest=thresh_rest)
                #
                setting_path = Util.setting_to_path(setting)
                df_result.to_csv("Market/CN/Backtest_Single/trend/date_trend_" + setting_path + ".csv")


# ONE OF THE MOST IMPORTANT KEY FUNNCTION I DISCOVERED
# input df = df_date with abov_ma indicator for a period e.g. 20000101 to 20191111
def Setup_date_trend_once(minmax=0.5, market="CN", close_label="close", index_label="_000001.SH", a_all=[], df_result=pd.DataFrame(), index=[0], thresh=0.0, dict_ops={}, op_sign="gt", thresh_log=8.8, thresh_rest=0.0, for_analysis=True, market_suffix=""):
    if for_analysis:
        df = get_stock_market_all(market).reset_index()
    else:
        df = df_result
    func = talib.RSI

    # 1.Step Create RSI or Abv_ma
    # 2.Step Create Phase
    # 3 Step Create Trend
    # 4 Step calculate trend pct_chg
    # 5 Step Calculate Step comp_chg

    a_low = [str(x) for x in a_all][:-1]  # should be [5, 20,60]
    a_high = [str(x) for x in a_all][1:]  # should be [20,60,240]

    pct_chg_column = "pct_chg"
    if for_analysis:
        # always measure the whole stock market pct_chg. More direct and better than approx with index
        pct_chg_tomorrow_column = pct_chg_column + "_tomorrow"
        df[pct_chg_tomorrow_column] = df[pct_chg_column].shift(-1)

    for i in a_all:  # RSI 1
        if i == 1:
            df[market_suffix + "rsi1"] = 0.0
            op_func = dict_ops[op_sign]
            df.loc[op_func(df["pct_chg" + index_label], 0.0), market_suffix + "rsi1"] = 1.0
        else:
            df[market_suffix + "rsi" + str(i)] = func(df[close_label + index_label], timeperiod=i) / 100

    max240 = minmax
    min240 = minmax

    # 1 means uptrend
    # 0 means downtrend
    # df["phase240"] = df["rsi240"].apply(lambda x: 1 if x > max240 else 0 if x < min240 else float("nan"))
    # df["phase60"] = df["rsi60"].apply(lambda x: 1 if x > max240 - thresh60 else 0 if x < min240 + thresh60 else float("nan"))
    # df["phase20"] = df["rsi20"].apply(lambda x: 1 if x > max240 - thresh60 else 0 if x < min240 + thresh60 else float("nan"))
    # df["phase5"] = df["rsi5"].apply(lambda x: 1 if x > max240 - thresh60*4 else 0 if x < min240 + thresh60*4 else float("nan"))
    # Create Phase

    for i in [str(x) for x in a_all]:
        maximum = (thresh_log * math.log(int(i)) + thresh_rest)
        minimum = 1 - maximum
        # df["phase" + i] = df["rsi" + i].apply(lambda x: 1 if x > max240 + thresh*int(i) else 0 if x < min240 - thresh*int(i) else float("nan"))
        df[market_suffix + "phase" + i] = df[market_suffix + "rsi" + i].apply(lambda x: 1 if x > maximum else 0 if x < minimum else float("nan"))

    # one loop to create trend from phase
    for rolling_freq_low, rolling_freq_high in zip(a_low, a_high):
        trend_name = market_suffix + "trend" + rolling_freq_high
        df[trend_name] = float("nan")
        df.loc[(df[market_suffix + "phase" + rolling_freq_high] == 1) & (df[market_suffix + "phase" + rolling_freq_low] == 1), trend_name] = 1
        df.loc[(df[market_suffix + "phase" + rolling_freq_high] == 0) & (df[market_suffix + "phase" + rolling_freq_low] == 0), trend_name] = 0

        # fill na based on the trigger points
        df[trend_name].fillna(method='bfill', inplace=True)
        last_trade = df.loc[df.last_valid_index(), trend_name]
        if last_trade == 1:
            df[trend_name].fillna(value=0, inplace=True)
        else:
            df[trend_name].fillna(value=1, inplace=True)

        # calculate trend strategy pct_chg and comp_chg
        if for_analysis:
            df[trend_name + "_pct_chg"] = df.loc[(df[trend_name] == 1), pct_chg_tomorrow_column].fillna(value=0, inplace=False)
            df[trend_name + "_comp_chg"] = Util.column_add_comp_chg(df[trend_name + "_pct_chg"])

            # print_comp_chg = df.tail(1)[trend_name + "_comp_chg"].to_numpy()[0]
            # print_pct_chg_mean = df[trend_name + "_pct_chg"].mean()
            # print_pct_chg_std = df[trend_name + "_pct_chg"].std()
            # trading_days = df[trend_name].sum() / len(df)

            try:
                pass
                # df_result.at[index, market_suffix+str(rolling_freq_high) + "_pct_chg_mean"] = print_pct_chg_mean
                # df_result.at[index, market_suffix+str(rolling_freq_high) + "_pct_chg_std"] = print_pct_chg_std
                # df_result.at[index, market_suffix+ str(rolling_freq_high) + "_comp_chg"] = print_comp_chg
                # df_result.at[index, market_suffix+str(rolling_freq_high) + "_trading_days"] = trading_days
            except Exception as e:
                print(e)

    # remove RSI and phase Columns to make it cleaner
    if not for_analysis:
        a_remove = []
        for i in a_all:
            a_remove.append(market_suffix + "rsi" + str(i))
            a_remove.append(market_suffix + "phase" + str(i))
        Util.columns_remove(df, a_remove)

    # calculate final trend =weighted trend of previous
    # TODO this need to be adjusted manually
    df[market_suffix + "trend"] = df[market_suffix + "trend2"] * 0.7 + df[market_suffix + "trend5"] * 0.15 + df[market_suffix + "trend20"] * 0.05 + df[market_suffix + "trend60"] * 0.05 + df[market_suffix + "trend240"] * 0.05
    if for_analysis:
        df[market_suffix + "trend_pct_chg"] = df.loc[(df[market_suffix + "trend"] >= 0.7), pct_chg_tomorrow_column].fillna(value=0, inplace=False)
        df[market_suffix + "trend_comp_chg"] = Util.column_add_comp_chg(df[market_suffix + "trend_pct_chg"])

    return df


def read(a_path=[], step=-1):  # if step is -1, read feather first
    dict_format = {".csv": [pd.read_csv, "filepath_or_buffer", a_path[0]], ".feather": [pd.read_feather, "path", a_path[1]]}

    # reorder dict if nessesary to read csv first
    dict_final = {}
    for key in list(dict_format)[::step]:
        dict_final[key] = dict_format[key]

    # iterate over and read
    for format, array in dict_final.items():
        function = array[0]
        argument = array[1]
        path = array[2]
        kwargs = {argument: path}
        try:
            df = function(**kwargs)
            return df
        except Exception as e:
            print(format, e)

    print("DB READ File Not Exist!", a_path[0])
    return pd.DataFrame()


def get_ts_code(asset="E", market="CN"):
    a_path = Util.a_path("Market/" + market + "/General/ts_code_" + asset)
    df = read(a_path)

    if (asset == "FD"):
        # only consider still ongoing fund
        df = df[df["delist_date"].isna()]

        # for now, only consider Equity market traded funds
        df = df[df["market"] == "E"]
        df.reset_index(drop=True, inplace=True)
    return df


def get_ts_code_all(market="CN"):
    return read(Util.a_path("Market/" + market + "/General/ts_code_all"))


def get_asset(ts_code="000002.SZ", asset="E", freq="D", market="CN"):
    df = read(Util.a_path("Market/" + market + "/Asset/" + str(asset) + "/" + str(freq) + "/" + str(ts_code)))
    try:
        df["trade_date"] = df["trade_date"].astype(int)
        df.set_index(keys="trade_date", inplace=True, drop=True)
    except:
        pass
    return df


def get_group_instance(group_instance="asset_E", market="CN"):
    df = read(Util.a_path("Market/" + market + "/Backtest_Multiple/Setup/Stock_Market/Group/" + group_instance))
    return df


def get_group_instance_all(assets=["E"]):
    dict_result = {}

    dict_group_label_pair = c_groups_dict(assets=assets, a_ignore=["asset", "industry3"])
    for group, instance_array in dict_group_label_pair.items():
        for instance in instance_array:
            df = get_group_instance(group_instance=group + "_" + str(instance))
            dict_result[group + "_" + str(instance)] = df
    return dict_result


def get_assets_E_D_Fun(query, ts_code, columns=["end_date"], market="CN"):
    a_path = Util.a_path("Market/" + market + "/Asset/E/D_Fun/" + query + "/" + ts_code)
    try:
        df = read(a_path)
        df = df[columns]
        df = df.drop_duplicates(subset="end_date", keep="last")
        return df
    except Exception as e:
        print("Error get_assets_E_D_Fun ", query, "not exist for", ts_code, e)
        df = Util.empty_asset_E_D_Fun(query)
        df = df[columns]
        return df


def get_assets_pledge_stat(ts_code, columns, market="CN"):
    a_path = Util.a_path("Market/" + market + "/Asset/E/W_pledge_stat/" + ts_code)
    try:
        df = read(a_path)
        df = df[columns]
        return df
    except Exception as e:
        print("Error get_assets_pledge_stat not exist for", ts_code, e)
        df = Util.emty_asset_E_W_pledge_stat()
        df = df[columns]
        return df


def get_assets_top_holder(ts_code, columns, market="CN"):
    a_path = Util.a_path("Market/" + market + "/Asset/E/D_top_holder/" + ts_code)
    try:
        df = read(a_path)
        df = df[columns]
        return df
    except Exception as e:
        print("Error get_assets_top_holder not exist for", ts_code, e)
        df = Util.empty_asset_E_top_holder()
        df = df[columns]
        return df


def get_trade_date(start_date="000000", end_date=today(), freq="D", market="CN"):
    a_path = Util.a_path("Market/" + market + "/General/trade_date_" + freq)
    df = read(a_path)
    df["trade_date"] = df["trade_date"].astype(int)
    df = df.loc[df["trade_date"].between(int(start_date), int(end_date))]
    df = Util.df_reindex(df)
    return df


def get_last_trade_date(freq="D", market="CN"):
    df_trade_date = get_trade_date(start_date="00000000", end_date=Util.today(), freq=freq, market=market)
    latest_tade_date = str(df_trade_date["trade_date"].tail(1).values.tolist()[0])
    return latest_tade_date


def get_next_trade_date(freq="D", market="CN"):
    df = get_trade_cal_D(a_is_open=[1])  # todo next trade date should be set to 17:00 after tushare has released its new data

    df["trade_date"] = df["trade_date"].astype(str)
    last_trade_date = get_last_trade_date(freq, market)
    df = df[df["trade_date"] > str(last_trade_date)]
    df = Util.df_reindex(df)

    next_trade_date = df.at[0, "trade_date"]
    return next_trade_date


def get_trade_cal_D(start_date="19900101", end_date="88888888", a_is_open=[0, 1]):
    a_path = Util.a_path("Market/CN/General/trade_cal_D")
    df = read(a_path)
    df.rename(columns={"cal_date": "trade_date"}, inplace=True)

    df["trade_date"] = df["trade_date"].astype(int)
    df = df[(df["is_open"].isin(a_is_open)) & (df["trade_date"].between(int(start_date), int(end_date)))]
    df = df[["trade_date"]]

    return df


def get_stock_market_all(market="CN"):
    df = read(Util.a_path("Market/" + market + "/Backtest_Multiple/Setup/Stock_Market/all_stock_market"))
    try:
        df["trade_date"] = df["trade_date"].astype(int)
    except:
        pass
    df.set_index(keys="trade_date", drop=True, inplace=True)
    return df


def get_industry_classify(level, market="CN"):
    return read(Util.a_path("Market/" + market + "/General/industry_" + level))


def get_industry_index(index, level, market="CN"):
    return read(Util.a_path("Market/" + market + "/General/industry/" + level + "/" + index))


def get_date(trade_date, assets=["E"], freq="D", market="CN"):
    if len(assets) != 1:
        return get_date_all()
    else:
        df = read(Util.a_path("Market/" + market + "/Date/" + assets[0] + "/" + freq + "/" + str(trade_date)))
        df.set_index(keys="ts_code", inplace=True, drop=True)
        return df


def get_date_all(trade_date, assets=["E", "I", "FD"], freq="D", format=".feather", market="CN"):
    if len(assets) == 1:
        return get_date(trade_date=trade_date, assets=assets, freq=freq, market=market)
    else:
        raise NotImplementedError('This function is not implemented yet')
    # TODO implement it


def get_date_E_oth(trade_date, oth_name, market="CN"):
    return read(Util.a_path("Market/" + market + "/Date/E/" + oth_name + "/" + str(trade_date)))


def get_market(freq, market="CN"):
    return read(Util.a_path("Market/" + market + "/Date/market/" + freq + ".csv"))


# needs to be optimized for speed and efficiency
def add_static_data(df, assets=["E", "I", "FD"], market="CN"):
    # all static code except name
    df_result = pd.DataFrame()

    if assets == c_assets():  # accelerate the process if requesting all data
        df_result = get_ts_code_all()
    else:  # if the process requires individual asset data
        for asset in assets:
            df_asset = get_ts_code(asset)
            df_result = df_result.append(df_asset, sort=False, ignore_index=True)

    df = pd.merge(df, df_result, how='left', on=["ts_code"], suffixes=[False, False], sort=False)
    return df


# require: trade_date
# function: adds another asset close price
def add_asset_comparison(df, freq, asset, ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"]):
    df_performance_comparison = get_asset(ts_code, asset, freq)
    df_performance_comparison = df_performance_comparison[a_compare_label]

    dict_rename = {column: column + "_" + ts_code for column in a_compare_label}
    df_performance_comparison = df_performance_comparison.rename(columns=dict_rename)

    # doing groups this need int, doing backtest this need without
    # df["trade_date"] = df["trade_date"].astype(int)
    # df_performance_comparison["trade_date"] = df_performance_comparison["trade_date"].astype(int)

    Util.columns_remove(df, [label + "_" + ts_code for label in a_compare_label])
    df = pd.merge(df, df_performance_comparison, how='left', on=["trade_date"], suffixes=["", ""], sort=False)
    return df


def add_asset_final_analysis_rank(df, assets, freq, analysis="bullishness", market="CN"):
    path = "Market/CN/Backtest_Single/" + analysis + "/EIFD_D_final.xlsx"
    df_analysis = pd.read_excel(path, sheet_name="Overview")

    final_score_label = ["ts_code"] + [s for s in df_analysis.columns if "final_" + analysis + "_rank" in s]
    df_analysis = df_analysis[final_score_label]

    df = pd.merge(df, df_analysis, how='left', on=["ts_code"], suffixes=[False, False], sort=False)
    return df


def check_tushare_upto_date():
    try:
        latest_trade_date = Util.today()
        df_tushare = update_assets_EI_D_reindex_reverse(ts_code="000001.SH", freq="D", asset="I", start_date="000000", end_date=today(), adj="hfq")
        latest_tushare_date = df_tushare.at[len(df_tushare) - 1, "trade_date"]
        print("today is", latest_trade_date, ". tushare is on", latest_tushare_date)
        return str(latest_trade_date) == str(latest_tushare_date)
    except:
        return False


def preload(load="asset", step=1):
    dict_result = {}
    df_listing = get_ts_code()[::step] if load == "asset" else get_trade_date(start_date="20000101")[::step]
    key = "ts_code" if load == "asset" else "trade_date"
    func = get_asset if load == "asset" else get_date

    bar = tqdm(range(len(df_listing)))
    bar.set_description(f"loading {load}...")
    for iterator, i in zip(df_listing[key], bar):
        df = func(iterator)
        if df.empty:
            dict_result[iterator] = pd.DataFrame()
            continue
        try:
            df = df[df["period"] > 240]
        except:
            pass
        dict_result[iterator] = df
    bar.close()

    return dict_result


def update_all_in_one(big_update=False):
    # meta
    if __name__ != '__main__':
        print("update_all_in_one can only be called in main!")
        return
    else:
        Util.sound("close_excel.mp3")

    # 1.0. GENERAL - CAL_DATE
    update_general_trade_cal()

    # 1.1. GENERAL - INDUSTRY
    for level in c_industry_level():
        update_general_industry_classify(level, big_update=big_update)  # big: override - small: skip
        update_general_industry_index(level, big_update=big_update)  # big: override - small: skip

    # 1.2. GENERAL - TOP HOLDER
    # multi_process(func=update_assets_E_top_holder, a_kwargs={"big_update": big_update}, a_steps=[1])  # big: override - small: only update new files and ignore existing files

    # 1.3. GENERAL - TS_CODE
    for asset in c_assets():
        update_general_ts_code(asset, big_update=big_update)  # big: override - small: override
    update_general_ts_code_all()

    # 1.5. GENERAL - TRADE_DATE
    for freq in ["D", "W"]:  # Currently only update D and W, because W is needed for pledge stats
        update_general_trade_date(freq, big_update=big_update)  # big: override - small: override

    # 2.1. ASSET - FUNDAMENTALS
    update_assets_E_D_Fun(start_date="00000000", end_date=today(), step=1, big_update=big_update)  # big: override - small: skip
    update_assets_E_W_pledge_stat(start_date="00000000", step=1, big_update=big_update)  # big: override - small: skip

    # 2.2. ASSET - DF
    a_steps = [1, 2, 3, 5, 11, -1, -2, -3, -7]
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "big_update": big_update}, a_steps=a_steps)
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": big_update}, a_steps=a_steps)  # big: smart decide - small: smart decide

    # 3.1. DATE - OTH
    multi_process(func=update_date_E_Oth, a_kwargs={"asset": "E", "freq": "D", "big_update": big_update}, a_steps=[1, -1])  # big: smart decide - small: smart decide

    # 3.2. DATE - DF
    a_steps = [1, -1] if big_update else [1, -1]
    multi_process(func=update_date, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": big_update}, a_steps=a_steps)  # big: smart decide - small: smart decide

    # 3.3. DATE - BASE
    update_date_base(start_date="19990101", end_date=today(), big_update=big_update, assets=["E"])  # big: override - small: smart decide

    # 3.4. DATE - TREND
    Setup_date_trend_multiple(run_once_as_date_summary=True, big_update=big_update)  # big: override - small: override

    # 4.1. CUSTOM - INDEX
    update_custom_index(big_update=big_update)


@njit
def std(xs):
    # compute the mean
    mean = 0
    for x in xs:
        mean += x
    mean /= len(xs)
    # compute the variance
    ms = 0
    for x in xs:
        ms += (x - mean) ** 2
    variance = ms / len(xs)
    std = math.sqrt(variance)
    return std


if __name__ == '__main__':
    try:
        # update_all_in_one(big_update=False)
        #
        # TODO add concept
        # update_custom_index(big_update=True)

        pr = cProfile.Profile()
        pr.enable()
        update_all_in_one(False)

        pr.disable()
        pr.print_stats(sort='file')



    except Exception as e:
        traceback.print_exc()
        print(e)
        Util.sound("error.mp3")
