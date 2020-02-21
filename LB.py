import operator
from multiprocessing import Process
import tushare as ts
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import smtplib
from email.message import EmailMessage
import os
from win32com.client import Dispatch
import traceback
import API_Tushare
import atexit
from time import time, strftime, localtime
import inspect
import time
import subprocess, os, platform
from datetime import timedelta
from playsound import playsound
from numba import jit
import numba
import enum

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


# decorator functions must be at top
def try_ignore(func):
    def this_function_will_never_be_seen(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except:
            result = None
        return result

    return this_function_will_never_be_seen


def wrap_line(func):
    def line(lines=50):
        print("=" * lines)

    def this_function_will_never_be_seen(*args, **kwargs):
        line()
        result = func(*args, **kwargs)
        line()
        return result

    return this_function_will_never_be_seen


def today():
    return str(datetime.now().date()).replace("-", "")


def plot_autocorrelation(series):
    from pandas.plotting import autocorrelation_plot
    from matplotlib import pyplot
    autocorrelation_plot(series)
    pyplot.show()


def standard_indi_name(ibase, deri, dict_variables):
    result = f"{ibase}.{deri}"
    variables = ""
    for key, enum_val in dict_variables.items():
        if key not in ["df", "ibase"]:
            try:
                variables = variables + f"{key}={enum_val.value},"
            except:
                variables = variables + f"{key}={enum_val},"
    result = result + f"({variables})"
    return result

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


def empty_asset_Tushare(asset="E"):
    if (asset == "E"):
        return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol"])
    else:
        return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol"])


def empty_date_Oth(query):
    if query == "holdertrade":
        return pd.DataFrame(columns=["ts_code", "ann_date", "holder_name", "holder_type", "in_de", "change_vol", "change_ratio", "after_share", "after_ratio", "avg_price", "total_share", "begin_date", "close_date"])
    elif query == "share_float":
        return pd.DataFrame(columns=["ts_code", "ann_date", "float_date", "float_share", "float_ratio", "holder_name", "share_type"])
    elif query == "repurchase":
        return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "proc", "exp_date", "vol", "amount", "high_limit", "low_limit"])
    elif query == "block_trade":
        return pd.DataFrame(columns=["ts_code", "trade_date", "price", "vol", "amount", "buyer", "seller"])


def empty_asset_E_D_Fun(query):
    if query == "balancesheet":
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


def emty_asset_E_W_pledge_stat():
    return pd.DataFrame(columns=["ts_code", "end_date", "pledge_count", "unrest_pledge", "rest_pledge", "total_share", "pledge_ratio"])


def empty_asset_E_top_holder():
    return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "holder_name", "hold_amount", "hold_ratio"])


def get_trade_date_datetime(trade_date):
    return datetime.strptime(str(trade_date), '%Y%m%d')


def get_trade_date_datetime_y(trade_date):
    date = get_trade_date_datetime(trade_date)
    return date.year


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
    date = get_trade_date_datetime(trade_date)
    return date.month


def get_trade_date_datetime_d(trade_date):
    date = get_trade_date_datetime(trade_date)
    return date.day


# D-W 1-7 in words
def get_trade_date_datetime_dayofweek(trade_date):
    date = get_trade_date_datetime(trade_date).strftime("%A")
    return date


# D-Y 1-365
def get_trade_date_datetime_dayofyear(trade_date):
    date = get_trade_date_datetime(trade_date).strftime("%j")
    return date


# W-Y 1-52
def get_trade_date_datetime_weekofyear(trade_date):
    date = get_trade_date_datetime(trade_date).strftime("%W")
    return date


def df_reverse_reindex(df):
    df = df.reindex(index=df.index[::-1])
    df = df.set_index(pd.Series(range(0, len(df.index))))
    return df


def df_reindex(df):
    return df.reset_index(drop=True, inplace=False)


def df_drop_duplicated_reindex(df, column_name):
    df[column_name] = df[column_name].astype(int)
    df = df.drop_duplicates(subset=column_name)
    df = df_reindex(df)
    return df


# skip rolling values that are already calculated and only treat nan values
def fast_add_rolling(df, add_from="", add_to="", rolling_freq=5, func=pd.Series.mean):
    nan_series = df.loc[df[add_to].isna(), add_to]  # check out all nan values
    for index, value in nan_series.iteritems():  # iterarte over all nan value
        get_rolling_frame = df[add_from][index - rolling_freq + 1:index + 1]  # get the custom made rolling object
        df.at[index, add_to] = func(get_rolling_frame)  # calculate mean/std


@jit
def quick_rolling_prod(xs, n):
    cxs = np.cumprod(xs)
    nans = np.empty(n)
    nans[:] = np.nan
    nans[n - 1] = 1.
    a = np.concatenate((nans, cxs[:len(cxs) - n]))
    return cxs / a


@try_ignore
def add_column(df, add_to, add_after, position):  # position 1 means 1 after add_after column. Position -1 means 1 before add_after column
    df.insert(df.columns.get_loc(add_after) + position, add_to, "", allow_duplicates=False)

def columns_remove(df, columns_array):
    for column in columns_array:
        try:
            df.drop(column, axis=1, inplace=True)
        except Exception as e:
            pass

def column_add_comp_chg(pct_chg_series):
    cun_pct_chg_series = 1 + (pct_chg_series / 100)
    return cun_pct_chg_series.cumprod()

def get_linear_regression_s(s_index, s_data):
    z = np.polyfit(s_index, s_data, 1)
    s_result = pd.Series(index=s_index, data=s_index * z[0] + z[1])
    return s_result

def get_linear_regression_rise(s_index, s_data):
    z = np.polyfit(s_index, s_data, 1)
    return z[0]


def calculate_beta(s1, s2):  # useful, otherwise s.corr mostly returns nan because std returns nan too often
    s1_name = s1.name
    s2_name = s2.name
    s1 = s1.copy()  # nessesary for some reason . dont delete it
    s2 = s2.copy()


    # calculate beta by only using the non na days = smallest amount of days where both s1 s2 are trading
    asset_all = pd.merge(s1, s2, how='inner', on=["trade_date"], suffixes=["", ""], sort=False)
    correl = asset_all[s1_name].corr(asset_all[s2_name], method="pearson")
    return correl

def open_file(filepath):
    filepath = "D:/GoogleDrive/私人/私人 Stock 2.0/" + filepath
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':  # Windows
        os.startfile(filepath)
    else:  # linux variants
        subprocess.call(('xdg-open', filepath))


@try_ignore
def close_file(filepath):
    filepath = "D:/GoogleDrive/私人/私人 Stock 2.0/" + filepath
    xl = Dispatch('Excel.Application')
    wb = xl.Workbooks.Open(filepath)
    wb.Close(True)

def groups_dict_to_string_iterable(dict_groups: dict):
    result = ""
    for key, dict_value in dict_groups.items():

        if type(dict_value) == list or type(dict_value) == dict:
            a_string_helper = []
            for x in dict_value:
                if callable(x):
                    a_string_helper.append(str(x.__name__))
                else:
                    a_string_helper.append(str(x))
            # a_string_helper = groups_dict_to_string_iterable  # Prevents error if bool is in dict_value array
            result = result + str(key) + ": [" + ', '.join(a_string_helper) + "], "
        elif type(dict_value) == str:
            result = result + str(key) + ": " + str(dict_value) + ", "
        elif type(dict_value) == int:
            result = result + str(key) + ": " + str(dict_value) + ", "
        elif type(dict_value) == float:
            result = result + str(key) + ": " + str(dict_value) + ", "
        elif type(dict_value) == bool:
            result = result + str(key) + ": " + str(dict_value) + ", "
        elif callable(dict_value):
            result = result + str(dict_value.__name__) + ": " + str(dict_value) + ", "
        else:
            result = result + str(key) + ": " + str(dict_value) + ", "
    return result


# returns only numeric volumns in a df
def get_numeric_df(df):
    return df.select_dtypes(include=[np.number])

def get_numeric_df_columns(df):
    return list(get_numeric_df(df).columns.values)

def shutdown_windows():
    os.system('shutdown -s')

def sound(file="error.mp3"):
    playsound("Media/Sound/" + file)


def a_path(path: str = ""):  #TODO use os path library instead of my own
    return [x for x in [path + ".csv", path + ".feather"]]


def handle_save_exception(e, path):
    if type(e) == FileNotFoundError:
        folder = "/".join(path.rsplit("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
    elif type(e) == PermissionError:
        close_file(path)
    else:
        traceback.print_exc()
    time.sleep(2)

def to_csv_feather(df, a_path, encoding='utf-8_sig', index=False, reset_index=True, drop=True, skip_feather=False, skip_csv=False):  # utf-8_sig
    if reset_index:
        df.reset_index(drop=drop, inplace=True)
    df = df.infer_objects()

    if not skip_csv:
        for _ in range(10):
            try:
                df.to_csv(a_path[0], index=index, encoding=encoding)
                break
            except Exception as e:
                handle_save_exception(e, a_path[0])

    if not skip_feather:
        try:
            df.to_feather(a_path[1])
        except Exception as e:
            handle_save_exception(e, a_path[1])


def to_excel(path_excel, dict_df):
    for i in range(0, 10):
        try:
            portfolio_writer = pd.ExcelWriter(path_excel, engine='xlsxwriter')
            for key, df in dict_df.items():
                df.to_excel(portfolio_writer, sheet_name=key, index=True, encoding='utf-8_sig')
            portfolio_writer.save()
            break
        except Exception as e:
            handle_save_exception(e, path_excel)


def line_print(text, lines=40):
    print("=" * lines)
    print(text)
    print("=" * lines)

def send_mail(trade_string="what to buy and sell"):
    sender_email = "sizhe.huang@guanyueinternational.com"
    receiver = "sizhe.huang@guanyueinternational.com"
    cc = "yang.qiong@guanyueinternational.com"
    password = "Ba22101964!"
    msg = EmailMessage()
    msg.set_content(trade_string)
    today = pd.datetime.now().date()
    msg['Subject'] = "Stock " + str(today.day) + "." + str(today.month) + "." + str(today.year)
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

def c_assets():
    return [e.value for e in Assets]

class Assets(enum.Enum):
    I = "I"
    E = "E"
    FD = "FD"


def c_SFreq():
    return [e.value for e in SFreq]  # return ["D", "W", "M", "Y", "S"]


class SFreq(enum.Enum):
    D = "D"
    # W="W"
    # S="S"
    # Y="Y"


def c_freq():
    return [e.value for e in Freq]

class Freq(enum.Enum):
    #f1 = 1
    f2 = 2
    f5 = 5
    f10 = 10
    f20 = 20
    f60 = 60
    f240 = 240

def c_rolling_freqs_fibonacci():
    return [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

def c_group_score_weight():
    return {"area": 0.10,
            "exchange": 0.40,
            "industry1": 0.20,
            "industry2": 0.20,
            "state_company": 0.05,
            "is_hs": 0.05}  # "industry3": 0.20,

def c_date_oth():
    return {"block_trade": API_Tushare.my_block_trade,
            "holdertrade": API_Tushare.my_holdertrade,
            "repurchase": API_Tushare.my_repurchase,
            "share_float": API_Tushare.my_share_float}

def c_assets_fina_function_dict():
    return {"fina_indicator": API_Tushare.my_fina_indicator,
            "income": API_Tushare.my_income,
            "balancesheet": API_Tushare.my_balancesheet,
            "cashflow": API_Tushare.my_cashflow}

def c_industry_level():
    return ['1', '2', '3']

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


def c_groups_dict(assets=c_assets(), a_ignore=[]):
    # E[0]=KEY, E[1][0]= LABEL 1 KEY, E[2][1]= LABEL 2 Instances,
    asset = {"asset": c_assets()}
    if "E" in assets:
        dict_e = {"industry1": ["建筑装饰", "纺织服装", "采掘", "汽车", "电气设备", "传媒", "机械设备", "钢铁", "银行", "轻工制造", "交通运输", "非银金融", "公用事业", "化工", "有色金属", "家用电器", "房地产", "综合", "农林牧渔", "建筑材料", "商业贸易", "通信", "计算机", "国防军工", "休闲服务", "食品饮料", "医药生物", "电子"],
                  "industry2": ["装修装饰", "园林工程", "其他轻工制造", "服装家纺", "采掘服务", "石油开采", "餐饮", "汽车零部件", "运输设备", "电机", "航空运输", "证券", "营销传播", "渔业", "金属制品", "玻璃制造", "基础建设", "文化传媒", "航运", "物流", "家用轻工", "房屋建设", "电源设备", "专业工程", "通用机械", "工业金属", "电气自动化设备", "水务", "其他采掘", "钢铁", "环保工程及服务", "银行", "专用设备",
                                "商业物业经营", "燃气",
                                "港口", "包装印刷", "高低压设备", "煤炭开采", "仪器仪表", "种植业", "视听器材", "专业零售", "互联网传媒", "船舶制造", "化学原料", "公交", "农产品加工", "其他建材", "塑料", "石油化工", "其他交运设备", "化学制品", "房地产开发", "高速公路", "汽车服务", "综合", "黄金", "白色家电", "动物保健", "橡胶", "航空装备", "造纸", "光学光电子", "食品加工", "纺织制造", "园区开发", "通信设备",
                                "医疗器械", "计算机设备",
                                "通信运营", "保险", "计算机应用", "旅游综合", "电力", "稀有金属", "贸易", "化学纤维", "多元金融", "中药", "其他电子", "一般零售", "化学制药", "金属非金属新材料", "景点", "畜禽养殖", "电子制造", "医药商业", "铁路运输", "酒店", "航天装备", "汽车整车", "医疗服务", "饲料", "饮料制造", "农业综合", "半导体", "生物制品", "元件", "林业", "地面兵装", "其他休闲服务", "水泥制造", "机场"],
                  "industry3": ["鞋帽", "女装", "铁路建设", "其他互联网服务", "休闲服装", "印刷包装机械", "粮食种植", "粮油加工", "农用机械", "装修装饰", "园林工程", "风电设备", "其他轻工制造", "其他采掘服务", "油气钻采服务", "有线电视网络", "毛纺", "其他服装", "石油开采", "路桥施工", "一般物业经营", "焦炭加工", "工控自动化", "化学工程", "餐饮", "影视动漫", "铝", "复合肥", "汽车零部件", "其他专业工程", "钢结构", "铁路设备",
                                "水产养殖",
                                "电机", "珠宝首饰", "冶金矿采化工设备", "航空运输", "证券", "营销服务", "平面媒体", "其他纺织", "环保设备", "其他基础建设", "重型机械", "金属制品", "普钢", "磨具磨料", "其他家用轻工", "纺织化学用品", "玻璃制造", "机械基础件", "炭黑", "家具", "新能源发电", "家电零部件", "航运", "物流", "氟化工及制冷剂", "线缆部件及其他", "储能设备", "机床工具", "房屋建设", "火电设备", "水务", "其他酒类",
                                "其他采掘", "男装",
                                "软饮料", "海洋捕捞", "乳品", "葡萄酒", "其它通用机械", "环保工程及服务", "其他种植业", "管材", "银行", "其他化学原料", "彩电", "燃气", "纯碱", "港口", "包装印刷", "制冷空调设备", "综合电力设备商", "其它电源设备", "民爆用品", "中压设备", "其他塑料制品", "楼宇设备", "铜", "其它专用机械", "涂料油漆油墨制造", "石油加工", "仪器仪表", "电网自动化", "计量仪表", "钨", "专业市场", "专业连锁",
                                "移动互联网服务",
                                "船舶制造", "磷化工及磷酸盐", "公交", "煤炭开采", "纺织服装设备", "互联网信息服务", "光伏设备", "内燃机", "氯碱", "铅锌", "其他建材", "低压设备", "聚氨酯", "其它视听器材", "其他交运设备", "耐火材料", "改性塑料", "其他化学制品", "农药", "食品综合", "无机盐", "房地产开发", "高压设备", "自然景点", "文娱用品", "高速公路", "汽车服务", "综合", "粘胶", "其他农产品加工", "黄金", "百货",
                                "动物保健", "显示器件",
                                "家纺", "其他纤维", "水利工程", "通信配套服务", "航空装备", "造纸", "轮胎", "终端设备", "LED", "金属新材料", "日用化学产品", "果蔬加工", "特钢", "其他稀有小金属", "火电", "园区开发", "医疗器械", "IT服务", "计算机设备", "洗衣机", "涤纶", "通信运营", "保险", "其他橡胶制品", "旅游综合", "光学元件", "贸易", "小家电", "电子零部件制造", "通信传输设备", "多元金融", "中药", "软件开发",
                                "冰箱", "其他电子",
                                "热电", "石油贸易", "化学制剂", "种子生产", "超市", "化学原料药", "乘用车", "畜禽养殖", "其他文化传媒", "非金属新材料", "医药商业", "铁路运输", "丝绸", "被动元件", "调味发酵品", "半导体材料", "酒店", "工程机械", "氮肥", "黄酒", "航天装备", "商用载客车", "医疗服务", "磷肥", "磁性材料", "饲料", "稀土", "农业综合", "集成电路", "啤酒", "钾肥", "肉制品", "生物制品", "合成革", "辅料",
                                "棉纺", "林业",
                                "多业态零售", "水电", "地面兵装", "其他休闲服务", "维纶", "印染", "印制电路板", "电子系统组装", "锂", "分立器件", "商用载货车", "玻纤", "国际工程承包", "水泥制造", "城轨建设", "空调", "燃机发电", "氨纶", "白酒", "人工景点", "机场"],
                  "area": ["海南", "西藏", "黑龙江", "青海", "江苏", "广西", "天津", "重庆", "吉林", "甘肃", "陕西", "浙江", "北京", "广东", "河南", "湖南", "辽宁", "上海", "新疆", "河北", "贵州", "山东", "深圳", "四川", "安徽", "湖北", "福建", "宁夏", "内蒙", "山西", "云南", "江西"],
                  "exchange": ["主板", "中小板", "创业板"],
                  "is_hs": ["N", "S", "H"],
                  "state_company": [True, False]}
        asset = {**asset, **dict_e}
    if "I" in assets:
        dict_i = {"category": ["三级行业指数", "四级行业指数", "行业指数", "综合指数", "其他", "一级行业指数", "主题指数", "成长指数", "价值指数", "二级行业指数", "基金指数", "规模指数", "策略指数", "债券指数"],
                  "publisher": ["上交所", "中证公司", "深交所"]}
        asset = {**asset, **dict_i}
    if "FD" in assets:
        dict_fd = {"fund_type": ["货币市场型", "商品型", "股票型", "混合型", "另类投资型", "债券型"],
                   "invest_type": ["白银期货型", "有色金属期货型", "主题型", "豆粕期货型", "货币型", "积极配置型", "偏股混合型", "成长型", "被动指数型", "增强指数型", "灵活配置型", "股票型", "黄金现货合约", "原油主题基金", "混合型", "债券型", "强化收益型", "稳定型"],
                   "type": ["契约型封闭式", "契约型开放式"],
                   "management": ["华泰证券资管", "兴业基金", "财通证券资管", "信达澳银基金", "中信建投基金", "国寿安保基金", "东吴基金", "泓德基金", "东海基金", "南华基金", "平安基金", "汇添富基金", "弘毅远方基金", "华宝基金", "九泰基金", "中海基金", "国联安基金", "诺德基金", "财通基金", "富国基金", "申万菱信基金", "国金基金", "广发基金", "大成基金", "中融基金", "天治基金", "西部利得基金", "新华基金", "融通基金", "浦银安盛基金",
                                  "中金基金",
                                  "浙商资管", "华安基金", "银华基金", "华夏基金", "鹏华基金", "银河基金", "汇安基金", "嘉实基金", "金鹰基金", "工银瑞信基金", "国泰基金", "方正富邦基金", "海富通基金", "招商基金", "华泰柏瑞基金", "安信基金", "南方基金", "诺安基金", "中信保诚基金", "博时基金", "易方达基金", "长盛基金", "圆信永丰基金", "交银施罗德基金", "泰达宏利基金", "建信基金", "国投瑞银基金", "中欧基金", "国海富兰克林基金", "天弘基金",
                                  "前海开源基金",
                                  "景顺长城基金", "泰信基金", "长信基金", "长城基金", "万家基金", "摩根士丹利华鑫基金", "中银基金", "东证资管", "兴全基金", "民生加银基金", "华富基金", "红土创新基金"],
                   "custodian": ["中信证券", "渤海银行", "招商证券", "国泰君安", "中信建投", "广发证券", "平安银行", "中金公司", "北京银行", "招商银行", "浦发银行", "中国工商银行", "中国银行", "中国建设银行", "海通证券", "浙商银行", "中国农业银行", "中国银河", "中国民生银行", "兴业银行", "兴业证券", "国信证券", "中信银行", "交通银行", "广发银行", "中国光大银行", "邮储银行", "宁波银行", "上海银行", "华夏银行"]}
        asset = {**asset, **dict_fd}
    asset = {key: value for key, value in asset.items() if key not in a_ignore}
    return asset

def c_setting_path():  # TODO create setting
    return "D:/GoogleDrive/私人/私人 Stock 2.0/"

def secondsToStr(elapsed=None):
    return strftime("%Y-%m-%d %H:%M:%S", localtime()) if elapsed is None else str(timedelta(seconds=elapsed))


@wrap_line
def log(message, elapsed=None):
    print(secondsToStr(), '-', message, '-', "Time Used:", elapsed)

def endlog():
    sound("finished_all.mp3")
    log("END", secondsToStr(time.time() - start))
    time.sleep(2)


if __name__ == '__main__':
    pass



else:  # IMPORTANT TO KEEP FOR SOUND AND TIME
    start = time.time()
    sound("start.mp3")
    atexit.register(endlog)
    log("START")
