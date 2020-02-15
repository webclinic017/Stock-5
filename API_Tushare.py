import tushare as ts
import pandas as pd
import time
import Util
import traceback

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def get(function, kwargs, df_if_empty=pd.DataFrame()):
    for _ in range(200):
        try:
            df = function(**kwargs)
            if df is None:
                message("NONE", function, kwargs)
                return df_if_empty
            elif df.empty:
                message("EMPTY", function, kwargs)
                return df_if_empty
            else:
                message("SUCCESS", function, kwargs)
                return df
        except Exception as e:
            message("ERROR", function, kwargs)
            print(e)
            traceback.print_exc()
            time.sleep(20)


# because some tushare function has no __name__ attribute which wi
def message(message, function, kwargs):
    try:
        print("Tushare", function.__name__, message, kwargs)
    except:
        print("Tushare", message, "BUT TUSHARE FUNCTION NAME NOT DISPLAYED!", kwargs)


def my_pro_bar(asset: str, ts_code: str, freq: str, start_date: str, end_date: str, adj="qfq", factors=[]):
    return get(function=ts.pro_bar, kwargs=locals(), df_if_empty=Util.empty_asset_Tushare())


def my_query(api_name="", ts_code="000001.SZ", start_date="00000000", end_date="00000000"):
    return get(function=pro.query, kwargs=locals())


def my_stockbasic(is_hs="", list_status="L", exchange="", fields='ts_code,name,area,list_date,is_hs'):
    return get(function=pro.stock_basic, kwargs=locals())


def my_pro_daily(trade_date="00000000"):
    return get(function=pro.daily, kwargs=locals())


def my_trade_cal(api_name="trade_cal", start_date="00000000", end_date="00000000"):
    return get(function=pro.query, kwargs=locals())


def my_holdertrade(ann_date, fields="ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio,after_share,after_ratio,avg_price,total_share,begin_date,close_date"):
    return get(function=pro.stk_holdertrade, kwargs=locals(), df_if_empty=Util.empty_date_Oth("holdertrade"))


def my_pledge_stat(ts_code="000001.SZ"):
    return get(function=pro.my_pledge_stat, kwargs=locals(), df_if_empty=Util.emty_asset_E_W_pledge_stat())


def my_cashflow(ts_code, start_date, end_date):
    return get(function=pro.cashflow, kwargs=locals(), df_if_empty=Util.empty_asset_E_D_Fun("cashflow"))


def my_fina_indicator(ts_code, start_date, end_date):
    return get(function=pro.fina_indicator, kwargs=locals(), df_if_empty=Util.empty_asset_E_D_Fun("fina_indicator"))


def my_balancesheet(ts_code, start_date, end_date):
    return get(function=pro.balancesheet, kwargs=locals(), df_if_empty=Util.empty_asset_E_D_Fun("balancesheet"))


def my_income(ts_code, start_date, end_date):
    return get(function=pro.income, kwargs=locals(), df_if_empty=Util.empty_date_Oth("income"))


def my_block_trade(trade_date):
    return get(function=pro.block_trade, kwargs=locals(), df_if_empty=Util.empty_date_Oth("block_trade"))


def my_share_float(ann_date):
    return get(function=pro.share_float, kwargs=locals())


def my_repurchase(ann_date):
    return get(function=pro.repurchase, kwargs=locals())


def my_index_classify(level="L1", src="SW"):
    return get(function=pro.index_classify, kwargs=locals())


def my_index_basic(market="SSE"):
    return get(function=pro.index_basic, kwargs=locals())


def my_index_member(index_code):
    return get(function=pro.index_member, kwargs=locals())


def my_fund_basic(market="E", fields="ts_code,name,fund_type,list_date,delist_date,issue_amount,m_fee,c_fee,benchmark,invest_type,type,market,custodian,management"):
    return get(function=pro.fund_basic, kwargs=locals())


if __name__ == '__main__':
    df = my_pro_bar(ts_code="000002.SZ", asset="E", freq="60min", start_date="2019-09-01", end_date="2019-09-01")
    print(df)
    pass
