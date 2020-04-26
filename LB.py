import operator
from multiprocessing import Process
import tushare as ts
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import xlsxwriter
import threading
import smtplib
from email.message import EmailMessage
import math
import re
import itertools
from win32com.client import Dispatch
import traceback
import _API_Tushare
import atexit
from time import time, strftime, localtime
import time
import subprocess, os, platform
from datetime import timedelta
from playsound import playsound
from numba import jit
import numba
import enum

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")



#decorator TODO maybe remove if no one else uses
def trade_date_norm(func):
    def this_function_will_never_be_seen(*args, **kwargs):
        try:
            if "date" in kwargs:
                kwargs["date"]=switch_trade_date(kwargs["date"])
            return func(*args, **kwargs)
        except Exception as e:
            return
    return this_function_will_never_be_seen


# decorator functions must be at top
def deco_only_big_update(func):
    def this_invisible_func(*args, **kwargs):
        if "big_update" in [*kwargs]:
            if kwargs["big_update"]:
                return func(*args, **kwargs)  # only return original function if kwargs has keyword "big_update" and big_update is true
        return

    return this_invisible_func


# TODO very dangerous function, check it often
def deco_try_ignore(func):
    def this_function_will_never_be_seen(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return

    return this_function_will_never_be_seen


def deco_except_empty_df(func):
    def this_function_will_never_be_seen(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return pd.DataFrame()

    return this_function_will_never_be_seen


def deco_wrap_line(func):
    def this_function_will_never_be_seen(*args, **kwargs):
        print("=" * 50)
        result = func(*args, **kwargs)
        print("=" * 50)
        return result

    return this_function_will_never_be_seen


def today():
    return str(datetime.now().date()).replace("-", "")


# def indi_name(abase, deri, d_variables={}):
#     variables = ""
#     for key, enum_val in d_variables.items():
#         if key not in ["df", "ibase"]:
#             # if issubclass(enum_val, enum.Enum):
#             try:
#                 variables = f"{variables}{key}={enum_val.value},"
#             except:
#                 variables = f"{variables}{key}={enum_val},"
#     if variables:
#         return f"{abase}.{deri}({variables})"
#     else:
#         return f"{abase}.{deri}"


def fibonacci(n):
    if n < 0:
        print("Incorrect input")
    # First Fibonacci number is 0
    elif n == 1:
        return 1
    # Second Fibonacci number is 1
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_array(n):
    a_result = []
    for i in range(1, n + 1):
        a_result.append(fibonacci(i))
    return a_result


def fibonacci_weight(n):
    array = fibonacci_array(n)
    array = [x / sum(array) for x in array]
    return array





def empty_df(query):
    if query == "pro_bar":
        return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol"])
    elif query == "holdertrade":
        return pd.DataFrame(columns=["ts_code", "ann_date", "holder_name", "holder_type", "in_de", "change_vol", "change_ratio", "after_share", "after_ratio", "avg_price", "total_share", "begin_date", "close_date"])
    elif query == "share_float":
        return pd.DataFrame(columns=["ts_code", "ann_date", "float_date", "float_share", "float_ratio", "holder_name", "share_type"])
    elif query == "repurchase":
        return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "proc", "exp_date", "vol", "amount", "high_limit", "low_limit"])
    elif query == "block_trade":
        return pd.DataFrame(columns=["ts_code", "trade_date", "price", "vol", "amount", "buyer", "seller"])
    elif query == "pledge_stat":
        return pd.DataFrame(columns=["ts_code", "end_date", "pledge_count", "unrest_pledge", "rest_pledge", "total_share", "pledge_ratio"])
    elif query == "top_holder":
        return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "holder_name", "hold_amount", "hold_ratio"])
    elif query == "balancesheet":
        return pd.DataFrame(
            columns=["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "total_share", "cap_rese", "undistr_porfit", "surplus_rese", "special_rese", "money_cap", "trad_asset", "notes_receiv", "accounts_receiv", "oth_receiv", "prepayment", "div_receiv", "int_receiv",
                     "inventories", "amor_exp", "nca_within_1y", "sett_rsrv", "loanto_oth_bank_fi", "premium_receiv", "reinsur_receiv", "reinsur_res_receiv", "pur_resale_fa", "oth_cur_assets", "total_cur_assets", "fa_avail_for_sale", "htm_invest", "lt_eqt_invest", "invest_real_estate",
                     "time_deposits", "oth_assets", "lt_rec", "fix_assets", "cip", "const_materials", "fixed_assets_disp", "produc_bio_assets", "oil_and_gas_assets", "intan_assets", "r_and_d", "goodwill", "lt_amor_exp", "defer_tax_assets", "decr_in_disbur", "oth_nca", "total_nca", "cash_reser_cb",
                     "depos_in_oth_bfi", "prec_metals", "deriv_assets", "rr_reins_une_prem", "rr_reins_outstd_cla", "rr_reins_lins_liab", "rr_reins_lthins_liab", "refund_depos", "ph_pledge_loans", "refund_cap_depos", "indep_acct_assets", "client_depos", "client_prov", "transac_seat_fee",
                     "invest_as_receiv", "total_assets", "lt_borr", "st_borr", "cb_borr", "depos_ib_deposits", "loan_oth_bank", "trading_fl", "notes_payable", "acct_payable", "adv_receipts", "sold_for_repur_fa", "comm_payable", "payroll_payable", "taxes_payable", "int_payable", "div_payable",
                     "oth_payable", "acc_exp", "deferred_inc", "st_bonds_payable", "payable_to_reinsurer", "rsrv_insur_cont", "acting_trading_sec", "acting_uw_sec", "non_cur_liab_due_1y", "oth_cur_liab", "total_cur_liab", "bond_payable", "lt_payable", "specific_payables", "estimated_liab",
                     "defer_tax_liab", "defer_inc_non_cur_liab", "oth_ncl", "total_ncl", "depos_oth_bfi", "deriv_liab", "depos", "agency_bus_liab", "oth_liab", "prem_receiv_adva", "depos_received", "ph_invest", "reser_une_prem", "reser_outstd_claims", "reser_lins_liab", "reser_lthins_liab",
                     "indept_acc_liab", "pledge_borr", "indem_payable", "policy_div_payable", "total_liab", "treasury_share", "ordin_risk_reser", "forex_differ", "invest_loss_unconf", "minority_int", "total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int", "total_liab_hldr_eqy",
                     "lt_payroll_payable", "oth_comp_income", "oth_eqt_tools", "oth_eqt_tools_p_shr", "lending_funds", "acc_receivable", "st_fin_payable", "payables", "hfs_assets", "hfs_sales"])
    elif query == "cashflow":
        return pd.DataFrame(
            columns=["ts_code", "ann_date", "f_ann_date", "end_date", "comp_type", "report_type", "net_profit", "finan_exp", "c_fr_sale_sg", "recp_tax_rends", "n_depos_incr_fi", "n_incr_loans_cb", "n_inc_borr_oth_fi", "prem_fr_orig_contr", "n_incr_insured_dep", "n_reinsur_prem", "n_incr_disp_tfa",
                     "ifc_cash_incr", "n_incr_disp_faas", "n_incr_loans_oth_bank", "n_cap_incr_repur", "c_fr_oth_operate_a", "c_inf_fr_operate_a", "c_paid_goods_s", "c_paid_to_for_empl", "c_paid_for_taxes", "n_incr_clt_loan_adv", "n_incr_dep_cbob", "c_pay_claims_orig_inco", "pay_handling_chrg",
                     "pay_comm_insur_plcy", "oth_cash_pay_oper_act", "st_cash_out_act", "n_cashflow_act", "oth_recp_ral_inv_act", "c_disp_withdrwl_invest", "c_recp_return_invest", "n_recp_disp_fiolta", "n_recp_disp_sobu", "stot_inflows_inv_act", "c_pay_acq_const_fiolta", "c_paid_invest",
                     "n_disp_subs_oth_biz", "oth_pay_ral_inv_act", "n_incr_pledge_loan", "stot_out_inv_act", "n_cashflow_inv_act", "c_recp_borrow", "proc_issue_bonds", "oth_cash_recp_ral_fnc_act", "stot_cash_in_fnc_act", "free_cashflow", "c_prepay_amt_borr", "c_pay_dist_dpcp_int_exp",
                     "incl_dvd_profit_paid_sc_ms", "oth_cashpay_ral_fnc_act", "stot_cashout_fnc_act", "n_cash_flows_fnc_act", "eff_fx_flu_cash", "n_incr_cash_cash_equ", "c_cash_equ_beg_period", "c_cash_equ_end_period", "c_recp_cap_contrib", "incl_cash_rec_saims", "uncon_invest_loss",
                     "prov_depr_assets", "depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp", "decr_deferred_exp", "incr_acc_exp", "loss_disp_fiolta", "loss_scr_fa", "loss_fv_chg", "invest_loss", "decr_def_inc_tax_assets", "incr_def_inc_tax_liab", "decr_inventories",
                     "decr_oper_payable", "incr_oper_payable", "others", "im_net_cashflow_oper_act", "conv_debt_into_cap", "conv_copbonds_due_within_1y", "fa_fnc_leases", "end_bal_cash", "beg_bal_cash", "end_bal_cash_equ", "beg_bal_cash_equ", "im_n_incr_cash_equ"])
    elif query == "income":
        return pd.DataFrame(
            columns=["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "basic_eps", "diluted_eps", "total_revenue", "revenue", "int_income", "prem_earned", "comm_income", "n_commis_income", "n_oth_income", "n_oth_b_income", "prem_income", "out_prem", "une_prem_reser",
                     "reins_income", "n_sec_tb_income", "n_sec_uw_income", "n_asset_mg_income", "oth_b_income", "fv_value_chg_gain", "invest_income", "ass_invest_income", "forex_gain", "total_cogs", "oper_cost", "int_exp", "comm_exp", "biz_tax_surchg", "sell_exp", "admin_exp", "fin_exp",
                     "assets_impair_loss", "prem_refund", "compens_payout", "reser_insur_liab", "div_payt", "reins_exp", "oper_exp", "compens_payout_refu", "insur_reser_refu", "reins_cost_refund", "other_bus_cost", "operate_profit", "non_oper_income", "non_oper_exp", "nca_disploss", "total_profit",
                     "income_tax", "n_income", "n_income_attr_p", "minority_gain", "oth_compr_income", "t_compr_income", "compr_inc_attr_p", "compr_inc_attr_m_s", "ebit", "ebitda", "insurance_exp", "undist_profit", "distable_profit"])
    elif query == "fina_indicator":
        return pd.DataFrame(
            columns=["ts_code", "ann_date", "end_date", "eps", "dt_eps", "total_revenue_ps", "revenue_ps", "capital_rese_ps", "surplus_rese_ps", "undist_profit_ps", "extra_item", "profit_dedt", "gross_margin", "current_ratio", "quick_ratio", "cash_ratio", "ar_turn", "ca_turn", "fa_turn",
                     "assets_turn", "op_income", "ebit", "ebitda", "fcff", "fcfe", "current_exint", "noncurrent_exint", "interestdebt", "netdebt", "tangible_asset", "working_capital", "networking_capital", "invest_capital", "retained_earnings", "diluted2_eps", "bps", "ocfps", "retainedps", "cfps",
                     "ebit_ps", "fcff_ps", "fcfe_ps", "netprofit_margin", "grossprofit_margin", "cogs_of_sales", "expense_of_sales", "profit_to_gr", "saleexp_to_gr", "adminexp_of_gr", "finaexp_of_gr", "impai_ttm", "gc_of_gr", "op_of_gr", "ebit_of_gr", "roe", "roe_waa", "roe_dt", "roa", "npta",
                     "roic", "roe_yearly", "roa2_yearly", "debt_to_assets", "assets_to_eqt", "dp_assets_to_eqt", "ca_to_assets", "nca_to_assets", "tbassets_to_totalassets", "int_to_talcap", "eqt_to_talcapital", "currentdebt_to_debt", "longdeb_to_debt", "ocf_to_shortdebt", "debt_to_eqt",
                     "eqt_to_debt", "eqt_to_interestdebt", "tangibleasset_to_debt", "tangasset_to_intdebt", "tangibleasset_to_netdebt", "ocf_to_debt", "turn_days", "roa_yearly", "roa_dp", "fixed_assets", "profit_to_op", "q_saleexp_to_gr", "q_gc_to_gr", "q_roe", "q_dt_roe", "q_npta",
                     "q_ocf_to_sales", "basic_eps_yoy", "dt_eps_yoy", "cfps_yoy", "op_yoy", "ebt_yoy", "netprofit_yoy", "dt_netprofit_yoy", "ocf_yoy", "roe_yoy", "bps_yoy", "assets_yoy", "eqt_yoy", "tr_yoy", "or_yoy", "q_sales_yoy", "q_op_qoq", "equity_yoy"])


def get_trade_date_datetime(trade_date):
    return datetime.strptime(str(trade_date), '%Y%m%d')


def get_trade_date_datetime_y(trade_date):
    return get_trade_date_datetime(trade_date).year


def get_trade_date_datetime_s(trade_date):
    date = get_trade_date_datetime_m(trade_date)
    if (date in [1, 2, 3]):
        return 1
    elif (date in [4, 5, 6]):
        return 2
    elif (date in [7, 8, 9]):
        return 3
    elif (date in [10, 11, 12]):
        return 4
    else:
        return float("nan")


def get_trade_date_datetime_m(trade_date):
    return get_trade_date_datetime(trade_date).month


def get_trade_date_datetime_d(trade_date):
    return get_trade_date_datetime(trade_date).day


# D-W 1-7 in words
def get_trade_date_datetime_dayofweek(trade_date):
    return get_trade_date_datetime(trade_date).strftime("%A")


# D-Y 1-365
def get_trade_date_datetime_dayofyear(trade_date):
    return get_trade_date_datetime(trade_date).strftime("%j")


# W-Y 1-52
def get_trade_date_datetime_weekofyear(trade_date):
    return get_trade_date_datetime(trade_date).strftime("%W")

def get_trade_date_datetime_weekofmonth(trade_date):
    trade_date=get_trade_date_datetime(trade_date)
    return (trade_date.isocalendar()[1] - trade_date.replace(day=1).isocalendar()[1] + 1)


def df_reverse_reindex(df):
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.reindex(index=df.index[::-1]).set_index(pd.Series(range(0, len(df.index))))
    return df


def df_reindex(df):
    return df.reset_index(drop=True, inplace=False)


def df_drop_duplicated_reindex(df, column_name):
    df[column_name] = df[column_name].astype(int)
    df = df.drop_duplicates(subset=column_name)
    df = df_reindex(df)
    return df


@deco_try_ignore
def add_column(df, add_to, add_after, position):  # position 1 means 1 after add_after column. Position -1 means 1 before add_after column
    df.insert(df.columns.get_loc(add_after) + position, add_to, "", allow_duplicates=False)


def remove_columns(df, columns_array):
    for column in columns_array:
        try:
            df.drop(column, axis=1, inplace=True)
        except Exception as e:
            pass



def polyfit_full(s_index, s_data, degree=1):
    full = np.polyfit(s_index, s_data, degree,full=True)
    z=full[0]
    residuals=full[1]
    return [pd.Series(index=s_index, data=s_index * z[0] + z[1]),residuals]

def polyfit_slope(s_index, s_data, degree=1):
    return np.polyfit(s_index, s_data, degree)[0]  # if degree is 1, then [0] is slope

# def get_linear_regression_variables(s_index, s_data):
#     return np.polyfit(s_index, s_data, 1)

# def polyfit(s_index, s_data,degree=1):
#     z = np.polyfit(s_index, s_data, degree)
#     return pd.Series(index=s_index, data=s_index * z[0] + z[1])


# def polyfit(df, degree=1, column="close"):
#     s_index = df[column].index
#     y = df[column]
#     weights = np.polyfit(s_index, y, degree)
#     data = pd.Series(index=s_index, data=0)
#     for i, polynom in enumerate(weights):
#         pdegree = degree - i
#         data = data + (polynom * (s_index ** pdegree))
#     return data


def calculate_beta(s1, s2):  # useful, otherwise s.corr mostly returns nan because std returns nan too often
    s1_name = s1.name
    s2_name = s2.name
    s1 = s1.copy()  # nessesary for some reason . dont delete it
    s2 = s2.copy()

    # calculate beta by only using the non na days = smallest amount of days where both s1 s2 are trading
    asset_all = pd.merge(s1, s2, how='inner', on=["trade_date"], suffixes=["", ""], sort=False)
    return asset_all[s1_name].corr(asset_all[s2_name], method="pearson")


def open_file(filepath):
    filepath = f"D:/GoogleDrive/私人/私人 Stock 2.0/{filepath}"
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':  # Windows
        os.startfile(filepath)
    else:  # linux variants
        subprocess.call(('xdg-open', filepath))


@deco_try_ignore
def close_file(filepath):
    filepath = f"D:/GoogleDrive/私人/私人 Stock 2.0/{filepath}"
    xl = Dispatch('Excel.Application')
    wb = xl.Workbooks.Open(filepath)

    wb.Close(True)

    # new
    xl.DisplayAlerts = False
    xl.Quit()


# maybe create a rekursive version
def groups_d_to_string_iterable(d_groups: dict):
    result = ""
    for key, d_value in d_groups.items():
        if type(d_value) in [list, dict]:
            a_string_helper = [str(x.__name__) if callable(x) else str(x) for x in d_value]
            result = f"{result}{key}: [{', '.join(a_string_helper)}], "
        elif callable(d_value):
            result = f"{result}{d_value.__name__}: {d_value}, "
        else:  # bool, string, scalar, int
            result = f"{result}{key}: {d_value}, "
    return result


# returns only numeric volumns in a df
def get_numeric_df(df):
    return df.select_dtypes(include=[np.number])

def shutdown_windows():
    os.system('shutdown -s')

def sound(file="error.mp3"):
    playsound(f"Media/Sound/{file}")


def a_path(path: str = ""):
    return [x for x in [f"{path}.csv", f"{path}.feather"]]


def handle_save_exception(e, path):
    if type(e) in [FileNotFoundError, xlsxwriter.exceptions.FileCreateError]:
        folder = "/".join(path.rsplit("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
            time.sleep(1)
    elif type(e) == PermissionError:
        print(f"try to close {path}")
        sound("close_excel.mp3")
        close_file(path)
        time.sleep(10)
    else:
        traceback.print_exc()


# reset index no matter what becaus of feather format. Drop index depends if index is relevant. CSV store index is always false
def to_csv_feather(df, a_path, index_relevant=True, skip_feather=False, skip_csv=False):  # utf-8_sig
    df.reset_index(drop=(not index_relevant), inplace=True)  # reset index no matter what because feather can only store normal index. if index relevant then dont drop
    df = df.infer_objects()
    if not skip_csv:
        for _ in range(10):
            try:
                df.to_csv(a_path[0], index=False, encoding='utf-8_sig')
                break
            except Exception as e:
                handle_save_exception(e, a_path[0])
    if not skip_feather:
        try:
            df.to_feather(a_path[1])
        except Exception as e:
            handle_save_exception(e, a_path[1])


def to_excel(path, d_df, index=True):
    for i in range(0, 10):
        try:
            writer = pd.ExcelWriter(path,engine="xlsxwriter")
            for key, df in d_df.items():
                df.to_excel(writer, sheet_name=key, index=index, encoding='utf-8_sig')
            writer.save()
            break
        except Exception as e:
            print("excel save exception type", type(e))
            handle_save_exception(e, path)



def send_mail(trade_string="what to buy and sell"):
    sender_email = "sizhe.huang@guanyueinternational.com"
    receiver = "sizhe.huang@guanyueinternational.com"
    cc = "yang.qiong@guanyueinternational.com"
    password = "Ba22101964!"
    msg = EmailMessage()
    msg.set_content(trade_string)
    today = pd.datetime.now().date()
    msg['Subject'] = f"Stock {today.day}.{today.month}.{today.year}"
    msg['From'] = "cj@python.org"
    msg['To'] = "sizhe.huang@guanyueinternational.com"
    msg['CC'] = "yang.qiong@guanyueinternational.com"

    server = smtplib.SMTP_SSL("hwsmtp.exmail.qq.com", port=465)
    server.ehlo()
    server.login(sender_email, password)

    print("login success...")
    server.sendmail(sender_email, [receiver, cc], msg.as_string())
    server.close()
    print("successfuly send...")


def multi_process(func, a_kwargs, a_steps=[]):
    a_process = []
    for step in a_steps:
        new_dict = a_kwargs.copy()
        new_dict.update({"step": step})
        p = Process(target=func, kwargs=new_dict)
        a_process.append(p)
    [process.start() for process in a_process]
    [process.join() for process in a_process]



def c_root():
    return "D:\Stock/"

def c_root_beta():
    return "E:\Beta/"

def c_assets():
    return ["I","E","FD"]

def c_assets_big():
    return c_assets() + ["G", "F"]

def c_bfreq():
    return [1,2,5,10,20,60,120,240]

def c_sfreq():
    return [2,20,240]

#deprecated, only there for deprecated dependency
class BFreq(enum.Enum):
    f1 = 1
    f2 = 2
    f5 = 5
    f10 = 10
    f20 = 20
    f60 = 60
    f120 = 120
    f240 = 240
    #f500 = 500

#deprecated, only there for deprecated dependency
class SFreq(enum.Enum):
    # f1 = 1
    f2 = 2
    f20 = 20
    f240 = 240

def c_G_queries():
    return {"G": ["on_asset == 'E'", "group in ['sw_industry1','sw_industry2','concept','market'] "]}

def c_I_queries():
    return {"I": ["ts_code in ['000001.SH','399001.SZ','399006.SZ']"]}

def c_group_score_weight():
    return {"area": 0.10,
            "market": 0.40,
            "sw_industry1": 0.20,
            "sw_industry2": 0.20,
            "state_company": 0.05,
            "is_hs": 0.05}  # "sw_industry3": 0.20,


def c_date_oth():
    return {"block_trade": _API_Tushare.my_block_trade,
            "holdertrade": _API_Tushare.my_holdertrade,
            "repurchase": _API_Tushare.my_repurchase,
            "share_float": _API_Tushare.my_share_float}


def c_assets_fina_function_dict():
    return {"fina_indicator": _API_Tushare.my_fina_indicator,
            "income": _API_Tushare.my_income,
            "balancesheet": _API_Tushare.my_balancesheet,
            "cashflow": _API_Tushare.my_cashflow}


def c_op():
    return {"plus": operator.add,
            "minus": operator.sub,
            "gt": operator.gt,
            "ge": operator.ge,
            "lt": operator.lt,
            "le": operator.le,
            "eq": operator.eq}


def c_candle():
    # array = [function, candle positive return use, candle negative return use]
    # e.g. "CDLDOJISTAR": [talib.CDLDOJISTAR, -100, 0] : for POSITIVE return WHEN -100 occurs. and 0 means not used for negative return
    return {"CDL2CROWS": [talib.CDL2CROWS, 0, 0],
            "CDL3BLACKCROWS": [talib.CDL3BLACKCROWS, 0, 0],
            "CDL3INSIDE": [talib.CDL3INSIDE, 0, 0],
            "CDL3LINESTRIKE": [talib.CDL3LINESTRIKE, 0, 0],
            "CDL3OUTSIDE": [talib.CDL3OUTSIDE, 0, 0],
            "CDL3STARSINSOUTH": [talib.CDL3STARSINSOUTH, 0, 0],
            "CDL3WHITESOLDIERS": [talib.CDL3WHITESOLDIERS, 0, 0],
            "CDLABANDONEDBABY": [talib.CDLABANDONEDBABY, 0, 0],
            "CDLADVANCEBLOCK": [talib.CDLADVANCEBLOCK, -100, 0],
            "CDLBELTHOLD": [talib.CDLBELTHOLD, 100, -100],
            "CDLBREAKAWAY": [talib.CDLBREAKAWAY, 0, 0],
            "CDLCLOSINGMARUBOZU": [talib.CDLCLOSINGMARUBOZU, 100, -100],
            "CDLCONCEALBABYSWALL": [talib.CDLCONCEALBABYSWALL, 0, 0],
            "CDLCOUNTERATTACK": [talib.CDLCOUNTERATTACK, 0, 0],
            "CDLDARKCLOUDCOVER": [talib.CDLDARKCLOUDCOVER, 0, 0],
            "CDLDOJI": [talib.CDLDOJI, 100, 0],
            "CDLDOJISTAR": [talib.CDLDOJISTAR, -100, 0],
            "CDLDRAGONFLYDOJI": [talib.CDLDRAGONFLYDOJI, 0, 0],
            "CDLENGULFING": [talib.CDLENGULFING, 100, -100],
            "CDLEVENINGDOJISTAR": [talib.CDLEVENINGDOJISTAR, 0, 0],
            "CDLEVENINGSTAR": [talib.CDLEVENINGSTAR, 0, 0],
            "CDLGAPSIDESIDEWHITE": [talib.CDLGAPSIDESIDEWHITE, 0, 0],
            "CDLGRAVESTONEDOJI": [talib.CDLGRAVESTONEDOJI, 0, 0],
            "CDLHAMMER": [talib.CDLHAMMER, 0, 0],
            "CDLHANGINGMAN": [talib.CDLHANGINGMAN, 0, -100],
            "CDLHARAMI": [talib.CDLHARAMI, 100, -100],
            "CDLHARAMICROSS": [talib.CDLHARAMICROSS, 0, -100],
            "CDLHIGHWAVE": [talib.CDLHIGHWAVE, 0, -100],
            "CDLHIKKAKE": [talib.CDLHIKKAKE, 100, 0],
            "CDLHIKKAKEMOD": [talib.CDLHIKKAKEMOD, 0, 0],
            "CDLHOMINGPIGEON": [talib.CDLHOMINGPIGEON, 0, 0],
            "CDLIDENTICAL3CROWS": [talib.CDLIDENTICAL3CROWS, 0, 0],
            "CDLINNECK": [talib.CDLINNECK, 0, 0],
            "CDLINVERTEDHAMMER": [talib.CDLINVERTEDHAMMER, 0, 0],
            "CDLKICKING": [talib.CDLKICKING, 0, 0],
            "CDLKICKINGBYLENGTH": [talib.CDLKICKINGBYLENGTH, 0, 0],
            "CDLLADDERBOTTOM": [talib.CDLLADDERBOTTOM, 0, 0],
            "CDLLONGLEGGEDDOJI": [talib.CDLLONGLEGGEDDOJI, 0, 0],
            "CDLLONGLINE": [talib.CDLLONGLINE, 100, -100],
            "CDLMARUBOZU": [talib.CDLMARUBOZU, 100, -100],
            "CDLMATCHINGLOW": [talib.CDLMATCHINGLOW, 0, 0],
            "CDLMATHOLD": [talib.CDLMATHOLD, 0, 0],
            "CDLMORNINGDOJISTAR": [talib.CDLMORNINGDOJISTAR, 0, 0],
            "CDLMORNINGSTAR": [talib.CDLMORNINGSTAR, 0, 0],
            "CDLONNECK": [talib.CDLONNECK, 0, 0],
            "CDLPIERCING": [talib.CDLPIERCING, 0, 0],
            "CDLRICKSHAWMAN": [talib.CDLRICKSHAWMAN, 0, 0],
            "CDLRISEFALL3METHODS": [talib.CDLRISEFALL3METHODS, 0, 0],
            "CDLSEPARATINGLINES": [talib.CDLSEPARATINGLINES, 0, 0],
            "CDLSHOOTINGSTAR": [talib.CDLSHOOTINGSTAR, -100, 0],
            "CDLSHORTLINE": [talib.CDLSHORTLINE, 100, -100],
            "CDLSPINNINGTOP": [talib.CDLSPINNINGTOP, 0, -100],
            "CDLSTALLEDPATTERN": [talib.CDLSTALLEDPATTERN, 0, 0],
            "CDLSTICKSANDWICH": [talib.CDLSTICKSANDWICH, 0, 0],
            "CDLTAKURI": [talib.CDLTAKURI, 0, 0],
            "CDLTASUKIGAP": [talib.CDLTASUKIGAP, 0, 0],
            "CDLTHRUSTING": [talib.CDLTHRUSTING, 0, -100],
            "CDLTRISTAR": [talib.CDLTRISTAR, 0, 0],
            "CDLUNIQUE3RIVER": [talib.CDLUNIQUE3RIVER, 0, 0],
            "CDLUPSIDEGAP2CROWS": [talib.CDLUPSIDEGAP2CROWS, 0, 0],
            "CDLXSIDEGAP3METHODS": [talib.CDLXSIDEGAP3METHODS, 0, 0],
            }


def c_d_groups(assets=c_assets(), a_ignore=[],market="CN"):
    import DB
    # E[0]=KEY, E[1][0]= LABEL 1 KEY, E[2][1]= LABEL 2 Instances,
    asset = {"asset": c_assets()}
    if "E" in assets:
        df_ts_code_E = DB.get_ts_code(["E"],market=market)

        a_columns=[x for x in df_ts_code_E.columns if x in ["sw_industry1","sw_industry2","sw_industry3","zj_industry1","jq_industry1","jq_industry2","area","market","is_hs","state_company","concept"]]
        # d_e = {"industry1": list(df_ts_code_E["industry1"].unique()),
        #        "industry2": list(df_ts_code_E["industry2"].unique()),
        #        "industry3": list(df_ts_code_E["industry3"].unique()),
        #        "concept": list(df_ts_code_concept["concept"].unique()),
        #        "area": list(df_ts_code_E["area"].unique()),
        #        "market": list(df_ts_code_E["market"].unique()),
        #        "is_hs": list(df_ts_code_E["is_hs"].unique()),
        #        "state_company": list(df_ts_code_E["state_company"].unique()),}

        d_e={}
        for column in a_columns:
            if column !="concept":
                d_e[column]=list(df_ts_code_E[column].unique())
            else:
                df_ts_code_concept = DB.get_ts_code(["concept"], market=market)
                d_e[column] = list(df_ts_code_concept[column].unique())

        asset = {**asset, **d_e}
    if "I" in assets:
        df_ts_code_I = DB.get_ts_code(["I"],market=market)
        a_columns = [x for x in df_ts_code_I.columns if x in ["category","publisher"]]

        # d_i = {"category": list(df_ts_code_I["category"].unique()),
        #        "publisher": list(df_ts_code_I["publisher"].unique()), }

        d_i = {}
        for column in a_columns:
            d_i[column] = list(df_ts_code_I[column].unique())

        asset = {**asset, **d_i}
    if "FD" in assets:
        df_ts_code_FD = DB.get_ts_code(["FD"],market=market)
        a_columns = [x for x in df_ts_code_FD.columns if x in ["fund_type", "invest_type","type","management","custodian"]]

        # d_fd = {"fund_type": list(df_ts_code_FD["fund_type"].unique()),
        #         "invest_type": list(df_ts_code_FD["invest_type"].unique()),
        #         "type": list(df_ts_code_FD["type"].unique()),
        #         "management": list(df_ts_code_FD["management"].unique()),
        #         "custodian": list(df_ts_code_FD["custodian"].unique()), }
        #

        d_fd = {}
        for column in a_columns:
            d_fd[column] = list(df_ts_code_FD[column].unique())

        asset = {**asset, **d_fd}
    return {key: value for key, value in asset.items() if key not in a_ignore}


def secondsToStr(elapsed=None):
    return strftime("%Y-%m-%d %H:%M:%S", localtime()) if elapsed is None else str(timedelta(seconds=elapsed))

@deco_wrap_line
def log(message, elapsed=None):
    print(secondsToStr(), '-', message, '-', "Time Used:", elapsed)

def endlog():
    sound("finished_all.mp3")
    log("END", secondsToStr(time.time() - start))
    time.sleep(2)


# skip rolling values that are already calculated and only treat nan values
# def fast_add_rolling(df, add_from="", add_to="", rolling_freq=5, func=pd.Series.mean):
#     nan_series = df.loc[df[add_to].isna(), add_to]  # check out all nan values
#     for index, value in nan_series.iteritems():  # iterarte over all nan value
#         get_rolling_frame = df[add_from][index - rolling_freq + 1:index + 1]  # get the custom made rolling object
#         df.at[index, add_to] = func(get_rolling_frame)  # calculate mean/std


@jit
def quick_rolling_prod(xs, n):
    cxs = np.cumprod(xs)
    nans = np.empty(n)
    nans[:] = np.nan
    nans[n - 1] = 1.
    a = np.concatenate((nans, cxs[:len(cxs) - n]))
    return cxs / a


@numba.njit  # try to use njit
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

def switch_ts_code(ts_code):
    dict_replace={".XSHE":".SZ", ".XSHG":".SH",".SZ":".XSHE",".SH":".XSHG"}
    for key,value in dict_replace.items():
        if key in ts_code:
            return re.sub(key,dict_replace[key],ts_code)

def switch_trade_date(trade_date):
    if "-" in str(trade_date):
        return str(trade_date).replace("-","")
    else:
        return str(str(trade_date)[0:4]+"-"+str(trade_date)[4:6]+"-"+str(trade_date)[6:8])

def timeseries_to_season(df):
    """converts a df time series to another df containing only the last day of season"""
    df_copy = df.copy()
    df_copy["trade_date_copy"] = df_copy.index
    df_copy["year"] = df_copy["trade_date_copy"].apply(lambda x: get_trade_date_datetime_y(x))  # can be way more efficient
    df_copy["month"] = df_copy["trade_date_copy"].apply(lambda x: get_trade_date_datetime_m(x))  # can be way more efficient

    a_index = []
    for year in df_copy["year"].unique():
        df_year = df_copy[df_copy["year"] == year]
        for month in [3, 6, 9, 12]:
            last_season_day = df_year[df_year["month"] == month].last_valid_index()
            if last_season_day is not None:
                a_index.append(last_season_day)
    return timeseries_helper(df=df,a_index=a_index)


def timeseries_to_week(df):
    """converts a df time series to another df containing only fridays"""
    df_copy = df.copy()
    df_copy["weekday"] = df_copy.index
    df_copy["weekday"] = df_copy["weekday"].apply(lambda x: get_trade_date_datetime_dayofweek(x))  # can be way more efficient
    df= df_copy[df_copy["weekday"] == "Friday"]
    return timeseries_helper(df=df,a_index=list(df.index))


def timeseries_to_month(df):
    """converts a df time series to another df containing only the last day of month"""
    df_copy = df.copy()
    df_copy["index_copy"] = df_copy.index
    df_copy["year"] = df_copy["index_copy"].apply(lambda x: get_trade_date_datetime_y(x))  # can be way more efficient
    df_copy["month"] = df_copy["index_copy"].apply(lambda x: get_trade_date_datetime_m(x))  # can be way more efficient

    a_index = []
    for year in df_copy["year"].unique():
        df_year = df_copy[df_copy["year"] == year]
        for month in range(1, 13):
            last_season_day = df_year[df_year["month"] == month].last_valid_index()
            if last_season_day is not None:
                a_index.append(last_season_day)
    return timeseries_helper(df=df,a_index=a_index)

def timeseries_to_year(df):
    """converts a df time series to another df containing only the last day of month"""
    df_copy = df.copy()
    df_copy["year"] = df_copy.index
    df_copy["year"] = df_copy["year"].apply(lambda x: get_trade_date_datetime_y(x))  # can be way more efficient

    a_index = []
    for year in df_copy["year"].unique():
        last_year_day = df_copy[df_copy["year"] == year].last_valid_index()
        a_index.append(last_year_day)
    return timeseries_helper(df=df,a_index=a_index)

def timeseries_helper(df,a_index):
    """converts index to time series and cleans up. Return only ohlc and pct_chg"""
    df_result=df.loc[a_index]
    df_result =ohlcpp(df_result)
    df_result["pct_chg"] = df_result["close"].pct_change() * 100
    return df_result

def custom_quantile(df, column, p_setting=[0,0.2,0.4,0.6,0.8,1], key_val=True):
    d_df = {}
    for low_quant, high_quant in custom_pairwise_overlap(p_setting):
        low_val, high_val = list(df[column].quantile([low_quant, high_quant]))

        key=f"{low_quant},{high_quant},{low_val},{high_val}" if key_val else f"{low_quant},{high_quant}"
        d_df[key] = df[df[column].between(low_val, high_val)]
    return d_df


def custom_expand(df, min_freq):
    d_result = {}
    for counter, expanding_index in enumerate(df.index):
        if counter >= min_freq:
            d_result[expanding_index] = df.loc[df.index[0]:expanding_index]
    return d_result


def custom_pairwise_noverlap(iterables):
    result = []
    for i, k in zip(iterables[0::2], iterables[1::2]):
        result.append((i, k))
    return result


def custom_pairwise_combination(a_array, n):
    """care about order: e.g. (0,1),(0,2)"""
    return list(itertools.combinations(a_array, n))

def custom_pairwise_permutation(a_array, n):
    """dont care about order:  e.g. (1,0),(0,1)"""
    return list(itertools.permutations(a_array, n))

def custom_pairwise_cartesian(a_array,n):
    """product with itself"""
    return list(itertools.product(a_array,repeat=n))

# for some reason the last one is always wrong
def custom_pairwise_overlap(iterables):
    return list(zip(iterables, iterables[1:] + iterables[:1]))[:-1]

def drange(start,end,step):
    return [x/100 for x in range(start,end,step)]


def print_iterables(d):
    if type(d)==dict:
        for key,value in d.items():
            print(key,value)
    else:
        for x in d:
            print(x)

def ohlcpp(df):
    return df[["ts_code","period","open","high","low","close","pct_chg"]]

"""past positive: how many past days are positive"""
def pp(s):
    return len(s[s>0])/len(s)

def df_between(df, start_date, end_date):
    df.index=df.index.astype(str)
    return df[(df.index >= start_date) & (df.index <= end_date)]

# def btest_quantile(series):
#     """this function should not exist. it helps in btest to eval quantil str in one line"""
#     array=list(series)
#     d_result={"left":array[0],"right":array[1]}
#     return d_result


# inside functions
class interrupt_class:  # static method inside class is required to break the loop because two threads
    interrput = False

def interrupt_user_input():
    input('Press a key \n')  # no need to store input, any key will trigger break
    interrupt_class.interrput = True

def interrupt_start():
    break_detector = threading.Thread(target=interrupt_user_input, daemon=True)
    break_detector.start()

def interrupt_confirmed():
    if interrupt_class.interrput:
        print("BREAK loop")
        sound("break.mp3")
        time.sleep(5)
        return True
    else:
        return False

if __name__ == '__main__':
    pass
    import DB
    df=DB.get_asset()
    print(df_between(df, "20150101", "20160101"))

else:  # IMPORTANT TO KEEP FOR SOUND AND TIME
    start = time.time()
    sound("start.mp3")
    atexit.register(endlog)
    log("START")


