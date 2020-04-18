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


def volatility():
    """one day pct_chg std
    Hypothesis: if one day pct_chg.std of all stocks is small market is stable

    """
    d_date = DB.preload(asset='E', on_asset=False)
    df_result = pd.DataFrame()
    for trade_date, df_date in d_date.items():
        print(trade_date)
        df_result.at[trade_date, "close"] = df_date["close"].mean()
        df_result.at[trade_date, "mean"] = df_date["pct_chg"].mean()
        df_result.at[trade_date, "std"] = df_date["pct_chg"].std()
        df_result.at[trade_date, "sharp"] = df_date["pct_chg"].mean() / df_date["pct_chg"].std()
    for i in [5, 10, 20, 60, 240]:
        df_result[f"std{i}"] = df_result["std"].rolling(i).mean()
    df_result.to_csv("volatilty.csv")


def main():
    # init stock market
    df_asset_E = DB.get_asset(ts_code="asset_E", asset="G")
    df_stock_market_all = DB.get_stock_market_all()
    df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sh = df_sh[["open", "high", "low", "close"]]


    # margin account/leverage/short, amount of people opening account.
    # A: margin account is useful. needs to be looked at closer

    # scrape all social security report 券商推荐 on one stock and compare
    #A: not done yet, but probably same as quexiu. It only tells about the attention and not about the correctness. big stock got more attenstion.

    #xueqiu data
    # A: xueqiu data seem to be pretty accurate. needs to be looked closer
    #A: In my test, I excluded the case that the new comments are on new IPO stocks. So all new discussion, comments, simulation trade, real trade are for long established stocks.
    #A: New follower between 0.7-0.8 is best. This means that not top viewed stocks are best.
    #A: new discussion between 0.5 is best. It is a very weird stat.
    #A: stock age is correlatedwith follower. Newer stock have less follower
    #A: the absolute stock sell return is not that high. So generally not very significant correlation.

    #A: interestingly, 0.1-0.3 new discussion seems to be good. This means stock with few attension will gain more.
    #A: it is hard to verify because xueqiu is new and data is only 2015-2019. The 2015 crazy time can distort the data. The gut feeling is that second lowest attetion and second highest attension have the best return.

    # volatility/beta between stocks
    # technical indicators
    # fundamental

    # Are good stock "close"/"pct_chg" different distributed?
    #A: Yes they are a bit different. Good stock are always at top which translates to being often at 0.8-1.0. Bad stock are vice versa.
    #A: The distribution actually reflects similar thing as e_max or gmean. So it does not reveal new information.
    #A: if close/pct_chg are normalized, then they are exactly the same.



    # A pattern recognition that detects buying signals if price is abv ma, and lower than last hig
    #A: This strategy would be good to use on big cap stock because their movement changes slower
    #TODO


    # unstable period. Shorter period when volatile, Longer period when less volatile
    # TODO

    #find a pattern recognition to buy good trend stocks at low price
    #1. good long term trend
    #2. long term trend not too high
    #3. price is not max
    #4. stop los if price falls

    #Distribution of "close"
    #A: short period e.g. 10 days are act as if there is no trend at all. If low then buy, high then sell.
    #A: longer period e.g. 500 have an uptrend move.
    #A: short period are staying longer at max/min
    #A: this further supports the idea. you want the shorterm price as low as possible. Long term trend as high as possible.


    #how many strategy beat asset_e?

    #. Find a way to balance between good momentum and low price.let current momentum not break and low point
    #A: after checking the chart of calc, the momentum gets broken alot. So the momentum change freq is very high. Hence, very hard to let momentum not break because it happens very often.
    #A: Ultimately, low price +patience wins.


    # use us market industry to determine good industry for china

    # find stocks that have high beta in past 60 days. But stock_pct_chg - market_pct_chg is higher than market

    # coutning index last high and low
    # count distance between last low and last high to determine next low
    # if start of month is bad, how likeliky is the whole month bad, test for 1,2,3,4,5 days


    # global index
    # A: Limited useful. Problem of overfitting. US stock is most dominant but also least related.
    # A: if we take msci world, we can see no relationship
    # a: basically global index is useless in the long run.
    a_global_index = ["800000.XHKG", "INX", "KS11", "FTSE", "RTS", "MIB", "GDAXI", "N225", "IBEX", "FCHI", "IBOV", "MXX", "GSPTSE"]
    for code in a_global_index:
        df = _API_JQ.break_jq_limit_helper_finance(code=code)
        df["day"] = df["day"].apply(LB.trade_date_switcher)
        df["day"] = df["day"].astype(int)
        df = pd.merge(df_sh, df, how="left", left_on="trade_date", right_on="day", suffixes=["_sh", "_F"], sort="False")
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
    # A: only us oil useful. rest is crap
    a_F = ["USOil", "SOYF", "NGAS", "Copper", "CORNF", "WHEATF", "XAUUSD", "XAGUSD"]
    a_F = ["USOil"]
    for ts_code in a_F:
        df = DB.get_asset(ts_code=f"{ts_code}.FXCM", asset="F")
        df = pd.merge(df_sh, df, how="left", on="trade_date", suffixes=["_sh", "_F"], sort="False")
        df.to_csv(f"jq/{ts_code}.FXCM.csv", encoding='utf-8_sig')

    # Macro economy data    https://www.joinquant.com/help/api/help?name=JQData#%E5%85%B3%E4%BA%8EJQData%E3%80%81jqdatasdk%E5%92%8Cjqdata
    # A: only CPI,PMI.Confidence, MONEY supply useful. rest is crap
    a_macro_queries = [
        # useful
        macro.MAC_MONEY_SUPPLY_MONTH,
        macro.MAC_ENTERPRISE_BOOM_CONFIDENCE_IDX,
        macro.MAC_MANUFACTURING_PMI,
        macro.MAC_CPI_MONTH,

        # sometimes ueseful
        # macro.MAC_INDUSTRY_GROWTH,
        # macro.MAC_FIXED_INVESTMENT,
        # macro.MAC_FOREIGN_CAPITAL_MONTH,
        # macro.MAC_ECONOMIC_BOOM_IDX,
        # macro.MAC_INDUSTRY_ESTATE_INVEST_MONTH,
        # macro.MAC_INDUSTRY_ESTATE_FUND_SOURCE_MONTH,

        # absolutely useless
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


    # seasonality1
    # if first n month is good, then the whole year is good. This has high pearson 0.72, and high TT/FF rate if using index. So using index to predict this is pretty accurate.
    # A:monthofyear 1,2,4 月份有非常强的判断力。4月份因为是年报时间。month 1 and 2 have about 72%(I)/68%(E,G) of correct predicting the year. 如果整年好，12月份有80%的概率会涨。So if many stock have gained in first and 2nd month, this is a good sign for the whole market this year.
    # A:weekofmonth 1,2, have predictability about ~63% of correct predicting this month outcome
    # A:weekofmonth could also be wrong because each week NATURALLY is part of month and has some contribution to the months gain. regarding this 63% predictability is not that high.

    # intraday volatility
    # A: volatility highest at opening, lowest at close
    # A: meangain is lowest at begin, highest at end.
    # A: this finding is robust for all 3 major index.
    # A: This test was done with 15m data. First and last 15min are most significant.
    # A: While intraday_volatility seems to be consisten. intraday gain is different. 000001.SH loses most at first 15m and gainst most at last 15m. The other two index are mixed.
    # A: General advice: sell at last 15min or before 14:30 because high sharp ratio. Buy at begin because begin 15 min return is mostly negative

    # relationship between bond and stock market
    # A: little. bond market just goes up. If interest is lower, bond market goes higher??

    # correlation between us market gain and chinese market gain
    # A: the pearson relationship has increased over the last year. the more recent the higher relaitonship
    # A: the pearson since 2015 is 0.15-0.20 which is quite high
    # A: us influences China with pearson about 0.2. China influences us with pearson about 0.2. it is a circle of influence
    # A: us close has 57% of predicting chinese close to be positive or negative. 42% wrong.

    # volatility between other index
    # A: since beta between sp500 and sh is very low. Their volatlity is hence useless.
    # A: the market is self directed, with less than 20% of times corresponding together in panic time.

    # how much does first/last 15 min of day predict today
    # A: alot. The pearson between 15min pct_chg and whole day pct_chg is ~0.55 which is insanely high
    # A: TT+FF rate is at 0.68%. So the first 15 mins hat 68% predict if today return is positive or negative.
    # A: the second highest pearson to predict the day is the first 30 mins after lunch break.
    # A: Basically. same founding in mont-to-year relationship can be found here.

    # how much does first/last 15 min of day predict tomorrow
    # A: weak predictability. around 53%. First 15min and last 15min have the highest predictability.

    # how much does rolling first/last 15 min of day predict tomorrow
    # A: rolling 5 mean of first 15mins -45mins have higher predictability of tomorrow. afternoon trade rolling mean have not so much predictability of tomorrow. But in general the predcitability is low. at best 0.51 TT+FF rate.

    # STD of pct_chg between all stocks on a day
    # A: hard to read since 2008 and 2015 are very different. high volatility comes either at near top or bottom. Definetly in crazy times. Market seems to be changing from 2008,2015,2023. So this indicator is not very useful.

    # winner loser difference
    # A: this is the same as std between all stocks. So no big indicator

    # cash flow 资金流向
    # A: not really manually tested, but it should be same as chart. So skipped.

    # 龙虎榜
    # A: It is just a ranking based on turnover and gain. I can do this myself

    # 举牌，大宗交易
    # A: Tested with logic: If someone sells, there is a buyer. So only the amount of trade can reflect the market condition. This is usually higher when market is in crazy times. Which is not a very useful indicator since crazy times can easly be determined.

    # text analyzer
    # A: tested seems to be useful, but data is not big enough. So basically not very useful. CCTV_news is not only financial. Need a financial pure big data source

    # answer the question. if an asset is in uptrend/downtrend/cycle mode, how likely is it to break the mode.
    # is it likely to be broken by seasonality?
    # or by other events?
    # A: I think this question is very hard to answer since the likelyhood of a trend being broken is random.

    # Technical analysis using connection of extrema values
    # A: tested, works good. It is relatively better than macd, but rare times macd can still be better.
    # A: In general less whipsaw than macd
    # A: but even here. A tradeoff between early signal, and accuracy must be made.

    # support and resistance but using only max and min from last high or low point on
    # reduce complexity and time. Not checking the whole time period, but the last period
    # A: this will not work as it is similar to extrema values test which I already performed.


    """
    Notes
    """

    # 见底和见顶部的象征其实一样。停留时间长。如何判断是底部还是顶部
    # do regression on the extrema and only use them if the p-value is good enough

    #maybe no automatic detection. instead do a simple filter, then human has to look at it.

    # even if outside factors like oil, employment affects the stock market. The stock market can remain unchanged and not react until a very long time.
    # so, even knowing the outside factors, still need to check technical price

    #If use longest trend streak, when it is very late to detect new trend. Lag would be too big. Even here, the trade off is real.


"""
My note
"""
    #Q: is it better to anticipate or wait for turning point to confirm?
    #A: both. It is a tradeoff. Sometimes wait is better. sometimes participate better
    #A: Hypothesis: Short term anticipate. Long term wait to confirm. Because short term swings very frequently. Long term swings very few. Waiting for confirm makes sense.


    #Bet against the trend or with the trend?
    #A: after you buy high price can go up, after you buy low price can go up.
    #A: so The past trend does not matter. What matter is future trend.
    #A: So the question is not relevant
    #A: the REAL question is. Is it more likely for a past uptrend to continue?
    #A: the answer to that is YES if it is a long term trend. NO if it is a short term trend.
    #A: short term (5,10 days) more likely to mean reverse. Long term (500,750 days) trend more likely to continue.
    #A: long term trend continues once confirmed. It does not switch that often. Shor term trend mean reverses very often.

    #Q: Is there are way to determin if the rise is a short term FIR or start of new long term trend?
    #A: No. It can go either way. But usually it is better to bet with mean. If it gained/lost too much, it will reverse to mean. Thats why buffet says patience pays off.

    #Q: combining two method will cause multiple causaion problem (or like this)
    #A: A method must be very significant to be able to use. Otherwise its wrong signals will interfere too much with other methods


"""
useful techniqes
1. MACD with EMA and SS
2. Bandpass filter (as oscilator) + macd (bandpass is always later than MACD) 
3. highpass as RSI 
4. A trendline on oscilator like bandpass seems to really good to indicate the next low or high. So use momentum to see the indicator if it is already deviating from price
5. close price to rsi but better: 1. better_rsi= (close - ss)/ss



IDEAS
0. See stock chart as wave
- Use methods from eletrical engineering
- use methods from quantum physic: superposition, heisenberg uncertainty. The smaller the days, the more random are the price jumps 

Use super position and heiserbergishe ungleichung on stocks


In uptrend: Buy until ATR drops
In downtrend: wait until ATR goes up and then buy. 


1.0 ATR is a pretty good volatility measure. When the price is at bottom. The True range is at highest because buyer and seller are very mixed. In an uptrend, The ATR should decrease
1.1 Most time dominant period is 15-40 which means this is the best period to do sell or buy. 

1. Average True Range to set threshhold
calculate average true range over past period
add a fraction of the average true range to nonlinear/ unstable period filters. 
by doing this, you allow the price to swing within the past true range. If it exceeds, then it is likely an special case. 

My thought:
ATR + STD = True Range

2. The general problem of all unstable period indicator is that they are too volatile and need a bigger frequency to compensate

2. Voting system
Use uncorrelated pairs: 
momentum: ma, instand_trend
Cycle:
oscilator: RSI





5. What JOHN Ehler did not incooperate:
volume, ANN, fundamentals, comparative best stock using same technique at one day, industry comparison


6. Relationships
the bigger the freq the more predictable, the more useless the data
The smoother your signal, the longer the lag (solveable by insta trendline, kalman filter)
the earlier you want to detect the turnpoint, the more sensitive the indicator needs to be, the more whiplas it creates. (Basically, you need to adjust the period based on probability for the next turning point signal)
the better the trend the easier to make money.
the more normalize, the morr sensitive, the more recent is the data,   《==》  the more whipsaw, the more noise.
In Up Trend move, buy and hold
In cycle mode buy and sell
in downtrend mode, Shortsell and hold
hyper overbought= long period + short period overbought
hyper udnerbought = long period + short period underbought

"""



if __name__ == '__main__':
    # ball.set_token('xq_a_token=2ee68b782d6ac072e2a24d81406dd950aacaebe3;')
    # df=ball.report("SH600519")
    # print(df)
    volatility()
