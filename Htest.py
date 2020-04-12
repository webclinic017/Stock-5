import tushare as ts
import pandas as pd
import time
import DB
import LB
import traceback
from jqdatasdk import *
import _API_JQ
import pysnowball as ball

auth('13817373362', '373362')


def my_macro_run(query_content):
    macro.run_query(query(query_content))


def mymacro(macro_query):
    query_result = query(macro_query)
    df = macro.run_query(query_result)

    query_from = str(query_result).split("FROM")[1]
    query_from = query_from.replace(" ", "")
    query_from = query_from.replace('"', '')

    index_label = "stat_quarter"
    if "QUARTER" in query_from:
        index_label = "stat_quarter"
    elif "MONTH" in query_from:
        index_label = "stat_month"
    elif "YEAR" in query_from:
        index_label = "stat_year"
    else:
        columns = df.columns
        if "stat_quarter" in columns:
            index_label = "stat_quarter"
        elif "stat_month" in columns:
            index_label = "stat_month"
        elif "stat_year" in columns:
            index_label = "stat_year"
        elif "stat_date" in columns:
            index_label = "stat_date"
        else:
            index_label = "day"

    if "MAC_CPI_MONTH" == query_from:
        df = df[df["area_name"] == "全国"]

    print(f"index: {index_label}. query_:{query_from}")
    LB.convert_index(df, index_label)
    df.sort_values(index_label, inplace=True)
    return [query_from, index_label, df]
    # df.to_csv(f"jq/{query_from}.csv", index=False,encoding="utf-8_sig")


def main():
    # init stock market
    df_asset_E = DB.get_asset(ts_code="asset_E", asset="G")
    df_stock_market_all = DB.get_stock_market_all()
    df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sh = df_sh[["open", "high", "low", "close"]]
    # d_preload=DB.preload(asset="E")

    # winner loser difference

    # margin account, amount of people opening account.
    #margin account is useful. needs to be looked at closer

    # cash flow 资金流向

    # 举牌，大宗交易，大股东买卖

    # technical

    # fundamental

    # volatility/beta between stocks

    #volatility beta between other index

    #unstable period. Shorter period when volatile, Longer period when less volatile


    # leverage


    # global index Limited useful. Problem of overfitting. US stock is most dominant but also least related.
    # if we take msci world, we can see no relationship
    #basically global index is useless in the long run.
    a_global_index = ["800000.XHKG","INX","KS11","FTSE","RTS","MIB","GDAXI","N225","IBEX","FCHI","IBOV","MXX","GSPTSE"]
    for code in a_global_index:
        df = _API_JQ.break_jq_limit_helper_finance(code=code)
        df["day"]=df["day"].apply(LB.trade_date_switcher)
        df["day"]=df["day"].astype(int)
        df = pd.merge(df_sh, df, how="left", left_on="trade_date", right_on="day",suffixes=["_sh", "_F"], sort="False")
        if code == "INX":
            pass
            # df=df[df["day"]>20180101]
            # df["sh_pct_chg"]=df["close_sh"].pct_change()
            # df[f"{code}_pct_chg"]=df["close_F"].pct_change()
            # pearson=df["sh_pct_chg"].corr(df[f"{code}_pct_chg"].shift(1))
            # spearman=df["sh_pct_chg"].corr(df[f"{code}_pct_chg"].shift(1),method="spearman")
            # periods=len(df)
            #
            # #how china influences us
            # TT=len(df[(df["sh_pct_chg"]>0)& (df[f"{code}_pct_chg"]>0)])/periods
            # TF=len(df[(df["sh_pct_chg"]>0)& (df[f"{code}_pct_chg"]<0)])/periods
            # FT=len(df[(df["sh_pct_chg"]<0)& (df[f"{code}_pct_chg"]>0)])/periods
            # FF=len(df[(df["sh_pct_chg"]<0)& (df[f"{code}_pct_chg"]<0)])/periods
            #
            # # how US influences china
            # # TT = len(df[(df[f"{code}_pct_chg"].shift(1) > 0) & (df["sh_pct_chg"] > 0)]) / periods
            # # TF = len(df[(df[f"{code}_pct_chg"].shift(1) > 0) & (df["sh_pct_chg"] < 0)]) / periods
            # # FT = len(df[(df[f"{code}_pct_chg"].shift(1) < 0) & (df["sh_pct_chg"] > 0)]) / periods
            # # FF = len(df[(df[f"{code}_pct_chg"].shift(1) < 0) & (df["sh_pct_chg"] < 0)]) / periods
            # #
            # all=TT+TF+FT+FF
            # print("corr is ",pearson,spearman)
            # print(f"TT {TT}, TF {TF}, FT {FT}, FF {FF}")
            # print(f"TT+FF {(TT+FF)/all}. TF+FT {(TF+FT)/all}")
            # return

        df.to_csv(f"jq/{code}.csv", encoding='utf-8_sig')


    # commodity, Oil, food, gold, gas, copper
    a_F=["USOil","SOYF","NGAS","Copper","CORNF","WHEATF","XAUUSD","XAGUSD"]
    a_F=["USOil"]#only us oil useful. rest is crap
    for ts_code in a_F:
        df = DB.get_asset(ts_code=f"{ts_code}.FXCM", asset="F")
        df=pd.merge(df_sh,df, how="left", on="trade_date",suffixes=["_sh", "_F"],sort="False")
        df.to_csv(f"jq/{ts_code}.FXCM.csv", encoding='utf-8_sig')


    #Macro    https://www.joinquant.com/help/api/help?name=JQData#%E5%85%B3%E4%BA%8EJQData%E3%80%81jqdatasdk%E5%92%8Cjqdata
    a_macro_queries = [
        #useful
        macro.MAC_MONEY_SUPPLY_MONTH,
        macro.MAC_ENTERPRISE_BOOM_CONFIDENCE_IDX,
        macro.MAC_MANUFACTURING_PMI,
        macro.MAC_CPI_MONTH,

        #sometimes ueseful
        # macro.MAC_INDUSTRY_GROWTH,
        # macro.MAC_FIXED_INVESTMENT,
        # macro.MAC_FOREIGN_CAPITAL_MONTH,
        # macro.MAC_ECONOMIC_BOOM_IDX,
        # macro.MAC_INDUSTRY_ESTATE_INVEST_MONTH,
        # macro.MAC_INDUSTRY_ESTATE_FUND_SOURCE_MONTH,

        #absolutely useless
        # macro.MAC_FOREIGN_COOPERATE_YEAR,
        # macro.MAC_TRADE_VALUE_YEAR,
        # macro.MAC_GOLD_FOREIGN_RESERVE,
        # macro.MAC_SOCIAL_SCALE_FINANCE,
        # macro.MAC_FISCAL_TOTAL_YEAR,
        # macro.MAC_FISCAL_EXTERNAL_DEBT_YEAR,
        # macro.MAC_INDUSTRY_ESTATE_70CITY_INDEX_MONTH,
        # macro.MAC_LEND_RATE,
        # macro.MAC_CREDIT_BALANCE_YEAR,
        # macro.MAC_INDUSTRY_AGR_PRODUCT_IDX_QUARTER,
        # macro.MAC_SALE_RETAIL_MONTH,
        # macro.MAC_EMPLOY_YEAR,
        # macro.MAC_RESOURCES_WATER_SUPPLY_USE_YEAR,
        # macro.MAC_NONMANUFACTURING_PMI,
        # macro.MAC_REVENUE_EXPENSE_YEAR,
        # macro.MAC_ENGEL_COEFFICIENT_YEAR,
        # macro.MAC_RESIDENT_SAVING_DEPOSIT_YEAR,
        # macro.MAC_RURAL_NET_INCOME_YEAR,
        # macro.MAC_POPULATION_YEAR
    ]

    # make query, adjust df and safe
    a_all = []
    for macro_query in a_macro_queries:
        a_answer = mymacro(macro_query)
        name = a_answer[0]
        index_label = a_answer[1]
        df = a_answer[2]
        a_important_labels = list(df.columns)

        if "MAC_CPI_MONTH" == a_answer[0]:
            df = df[df["area_name"] == "全国"]
            a_important_labels = [index_label, "cpi_month"]
        if "MAC_ECONOMIC_BOOM_IDX" == a_answer[0]:
            a_important_labels = [index_label, "early_warning_idx", "leading_idx"]
        if "MAC_ENTERPRISE_BOOM_CONFIDENCE_IDX" == a_answer[0]:
            a_important_labels = [index_label, "boom_idx", "confidence_idx"]
        if "MAC_FIXED_INVESTMENT" == a_answer[0]:
            a_important_labels = [index_label, "real_estate_yoy"]
        if "MAC_FOREIGN_CAPITAL_MONTH" == a_answer[0]:
            a_important_labels = [index_label, "joint_num_acc_yoy", "value_acc_yoy", "foreign_value_acc_yoy"]
        if "MAC_INDUSTRY_ESTATE_FUND_SOURCE_MONTH" == a_answer[0]:
            a_important_labels = [index_label, "total_invest_yoy"]
        if "MAC_INDUSTRY_ESTATE_INVEST_MONTH" == a_answer[0]:
            a_important_labels = [index_label, "below90_house_yoy", "above144_house_yoy", "villa_flat_yoy"]
        if "MAC_INDUSTRY_GROWTH" == a_answer[0]:
            a_important_labels = [index_label, "growth_yoy", "foreign_acc"]
        if "MAC_MONEY_SUPPLY_MONTH" == a_answer[0]:
            a_important_labels = [index_label, "m1_yoy", "m2_yoy"]
        if "MAC_MONEY_SUPPLY_MONTH" == a_answer[0]:
            a_important_labels = [index_label, "m1_yoy", "m2_yoy"]

        df = df[a_important_labels]
        df[index_label] = df[index_label].astype(int)
        df = pd.merge(df_sh.copy().reset_index(), df, how='left', left_on=["trade_date"], right_on=[index_label], sort=False)
        df = df.set_index("trade_date")
        LB.columns_remove(df, [index_label])
        a_important_labels = [x for x in a_important_labels if x != index_label]
        for label in a_important_labels:
            df[label] = df[label].fillna(method="ffill")
        df.to_csv(f"jq/{name}.csv", index=True, encoding='utf-8_sig')
        a_all.append(df)

    df_all = pd.concat(a_all, axis=1)
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]
    df_helper = df_all[df_all.columns.difference(["open", "high", "low", "close", "id"])]
    df_all["mean"] = df_helper.mean(axis=1)
    df_all.to_csv("jq/all_macro.csv", encoding='utf-8_sig')

    # text analyzer
    # tested seems to be useful, but data is not big enough. So basically not very useful. CCTV_news is not only financial. Need a financial pure big data source

    # scrape all security report on one stock and compare
    #xueqiu data seem to be pretty accurate

    #quantile on stock trading time. should you buy at start, middle or end of trade day?
    #and then check which stock has gained the most momentum from social discussion


    #coutning index last high and low
    #count distance between last low and last high to determine next low
    #if start of month is bad, how likeliky is the whole month bad, test for 1,2,3,4,5 days

    #seasonality1
    #if first n month is good, then the whole year is good. This has high pearson 0.72, and high TT/FF rate if using index. So using index to predict this is pretty accurate.
    #A:monthofyear 1,2,4 月份有非常强的判断力。4月份因为是年报时间。month 1 and 2 have about 72%(I)/68%(E,G) of correct predicting the year. 如果整年好，12月份有80%的概率会涨。So if many stock have gained in first and 2nd month, this is a good sign for the whole market this year.
    #A:weekofmonth 1,2, have predictability about ~63% of correct predicting this month outcome
    #A:weekofmonth could also be wrong because each week NATURALLY is part of month and has some contribution to the months gain. regarding this 63% predictability is not that high.

    #intraday volatility
    #A: volatility highest at opening, lowest at close
    #A: meangain is lowest at begin, highest at end.
    #A: this finding is robust for all 3 major index.
    #A: This test was done with 15m data. First and last 15min are most significant.
    #A: While intraday_volatility seems to be consisten. intraday gain is different. 000001.SH loses most at first 15m and gainst most at last 15m. The other two index are mixed.
    #A: General advice: sell at last 15min or before 14:30 because high sharp ratio. Buy at begin because begin 15 min return is mostly negative

    # relationship between bond and stock market
    # A: little. bond market just goes up. If interest is lower, bond market goes higher??

    # correlation between us market gain and chinese market gain
    # A: the pearson relationship has increased over the last year. the more recent the higher relaitonship
    # A: the pearson since 2015 is 0.15-0.20 which is quite high
    # A: us influences China with pearson about 0.2. China influences us with pearson about 0.2. it is a circle of influence
    # A: us close has 57% of predicting chinese close to be positive or negative. 42% wrong.



    #support and resistance but using only max and min from last high or low point on
    #reduce complexity and time. Not checking the whole time period, but the last period

    #answer the question. if an asset is in uptrend/downtrend/cycle mode, how likely is it to break the mode.
    # is it likely to be broken by seasonality?
    # or by other events?

    #use us market industry to determine good industry for china

    #find stocks that have high beta in past 60 days. But stock_pct_chg - market_pct_chg is higher than market


    #how much does first/last 15 min of day predict today
    #A: alot. The pearson between 15min pct_chg and whole day pct_chg is ~0.55 which is insanely high
    #A: TT+FF rate is at 0.68%. So the first 15 mins hat 68% predict if today return is positive or negative.
    #A: the second highest pearson to predict the day is the first 30 mins after lunch break.
    #A: Basically. same founding in mont-to-year relationship can be found here.

    #how much does first/last 15 min of day predict tomorrow
    # A: weak predictability. around 53%. First 15min and last 15min have the highest predictability.

    #how much does rolling first/last 15 min of day predict tomorrow
    #A: rolling 5 mean of first 15mins -45mins have higher predictability of tomorrow. afternoon trade rolling mean have not so much predictability of tomorrow. But in general the predcitability is low. at best 0.51 TT+FF rate.


if __name__ == '__main__':
    # ball.set_token('xq_a_token=2ee68b782d6ac072e2a24d81406dd950aacaebe3;')
    # df=ball.report("SH600519")
    # print(df)
    main()
