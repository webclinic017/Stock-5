import DB
import LB
from jqdatasdk import *
import _API_JQ
import UI
import numpy as np
import pandas as pd
import Alpha
import matplotlib.pyplot as plt

from _API_JQ import my_macro

auth('13817373362', '373362')

def unstable_period():
    """
    partly integrated
    idea: unstable period based on volatility
    """

    df= DB.get_asset(ts_code="000001.SH", asset="I")
    df=LB.df_between(df,"20000101",LB.today())

    #volatility measured in 3 ways:
    df=df.tail(len(df)-5)
    df=df.reset_index()

    #if standard deviation of last n days is high
    for freq in [120,240]:
        df[f"volatility1{freq}"]=df["pct_chg"].rolling(freq).std()
        df[f"helper1{freq}"] = pd.qcut(x=df[f"volatility1{freq}"], q=10, labels=False)


    df["volatility2"]=df["ivola"].pct_change().rolling(120).mean()


    #if market gains or loses too much, it is volatile
    for freq in [120,240]:
        df[f"volatility3{freq}"]=df[f"close.pgain(freq={freq})"]
        df[f"helper3{freq}"] = pd.qcut(x=df[f"volatility3{freq}"], q=11, labels=False)



    #if market beta is high, it is volatile
    for freq in [500]:
        df[f"{freq}d_high"]=df["close"].rolling(freq).max()
        df[f"{freq}d_low"]=df["close"].rolling(freq).min()
    # volatility decider by distance between last n period high and low
        df[f"helper4240"] = (df[f"{freq}d_high"] - df[f"{freq}d_low"]) / df[f"{freq}d_low"]


    #if only few people in the market wins, it is volatile

    for freq in [120,240]:
        df[f"helper{freq}"]=df[f"helper4240"]
        df[f"helper{freq}"] = pd.qcut(x=df[f"helper{freq}"], q=11, labels=False)

        df[f"rolling_freq{freq}"] = df[f"helper{freq}"].replace(to_replace={0: 360, 1: 280, 2: 220, 3: 180, 4: 140, 5: 100, 6: 60, 7: 40, 8:20, 9:10, 10:5})
        df[f"unstable_ma{freq}"] = np.nan
        for index in df.index:
            rolling_freq = df.at[index, f"rolling_freq{freq}"]
            df.at[index, f"unstable_ma{freq}"] = df.loc[index - rolling_freq:index, "close"].mean()



    df["helper120"]=df["helper120"]*1000
    df["helper3120"]=(df["helper3120"]-5).abs()
    UI.plot_chart(df, ["close", f"helper120", "helper4240"], {})




def market_volatility(start_date="20000101"):
    """
    partly integrated
    this function tries to create an rsi-like indicator to indicate how volatile the market is
    result:
    - all methods seems to behave similar, but still have minor difference
    - Basically, this is the same thing as trend vs cycle mode. If market has low volatility, it is flat, and it is in cycle mode. Else ,it is in trend mode.

    """


    df = DB.get_asset(ts_code="000002.SZ", asset="E")
    #df = LB.df_between(df, start_date, LB.today())

    df = df.tail(len(df) - 5)

    scale=df["close"].max()/10

    for freq in [500]:
        # method 1: volatility decider by distance between last n period high and low
        df[f"{freq}d_high"] = df["close"].rolling(freq).max()
        df[f"{freq}d_low"] = df["close"].rolling(freq).min()
        df[f"method1_{freq}"] = (df[f"{freq}d_high"] - df[f"{freq}d_low"]) / df[f"{freq}d_low"]
        df[f"method1_q{freq}"] = pd.qcut(x=df[f"method1_{freq}"], q=11, labels=False) * scale

        #method 2: ivola (be care ful because ivola has trend in it)
        df[f"method2_{freq}"]=df["ivola"].rolling(freq).max()-df["ivola"].rolling(freq).min()
        df[f"method2_q{freq}"] = pd.qcut(x=df[f"method2_{freq}"], q=11, labels=False)*scale

        #NOT GOOD ENOUGH
        #method 3: by purely the pct_changed over last n days
        df[f"method3_{freq}"]=df["close"].pct_change(freq)
        df[f"method3_q{freq}"] = pd.qcut(x=df[f"method3_{freq}"], q=11, labels=False)*scale

        # NOT GOOD ENOUGH
        # method 3: use slope
        df[f"method4_{freq}"] = Alpha.slope(df=df,abase="close",freq=int(freq),re=pd.Series.rolling,inplace=False)
        df[f"method4_q{freq}"] = pd.qcut(x=df[f"method4_{freq}"], q=11, labels=False) * scale


        # d_preload_date=DB.preload(asset="E",on_asset=False)
        # for trade_date, df_date in d_preload_date.items():
        #     print(trade_date)
        #
        #     #method 3: std of pgain120 of all stock on one day
        #     df.at[trade_date,f"method5_{freq}"]=df_date[f"close.pgain(freq={freq})"].std()
        # df[f"method5_q{freq}"] = pd.qcut(x=df[f"method5_{freq}"], q=11, labels=False) * scale

        #Method 6: high correlation with the market


        #chooses one of n methods to display
        x=Alpha.macd(df=df,abase="close",freq=120,freq2=240,type=4,inplace=True)
        df[x[0]]=df[f"method1_q{freq}"]/(df[x[0]]*1)
        a_methods=[f"method{x}_q{freq}" for x in [1]]
        UI.plot_chart(df, ["close", x[0]] + a_methods, {})

def main():
    # init stock market
    df_asset_E = DB.get_asset(ts_code="asset_E", asset="G")
    df_stock_market_all = DB.get_stock_market_all()
    df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sh = df_sh[["open", "high", "low", "close"]]

    # global index
    # A: Limited useful. Problem of overfitting. US stock is most dominant but also least related.
    # A: if we take msci world, we can see no relationship
    # a: basically global index is useless in the long run.
    a_global_index = ["800000.XHKG", "INX", "KS11", "FTSE", "RTS", "MIB", "GDAXI", "N225", "IBEX", "FCHI", "IBOV", "MXX", "GSPTSE"]
    for code in a_global_index:
        df = _API_JQ.break_jq_limit_helper_finance(code=code)
        df["day"] = df["day"].apply(LB.switch_trade_date)
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
        a_answer = my_macro(macro_query)
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




    # Q: qdii research
    # A: very hard to scrape all the data without an api
    #A: possible sources:
    #A: http://data.eastmoney.com/report/singlestock.jshtml?quoteid=0.000651&stockcode=000651
    #A: http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol=sz000651&t1=all&p=5
    #A: http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol=sz000651&t1=all&p=1
    #A: this url works but is very annoying to scrape
    # A: top 10% rank 2410
    # A: top 20% rank 2612
    # A: top 30% rank 2747
    # A: bot 30% rank 3711
    # A: bot 20% rank 3660
    # A: bot 10% rank 3555
    #higher than mean is 2766, lower than mean is 3500
    #A: Conclusion: YES! The more people write about it, the better the stock might be.
    #A: the pearson between final rank and qdii attention is -0.27 which is very high!
    #A: overall : it is highly correlated whatever qdii does


    #Q: QDII grade
    #A: it is significant that good stock have only positive rating. checked in bulllishness
    #A: pearson wise: it is -0.14, also highly correlated, but not as much as qdii_research


    #Q: shareholder class from jq
    #Q: for now, it doesnt work because jqdata error. But I can assume that institutional holding has a godo predictability.
    #A: TODO


    # Q: use global market beta to determine normal time and crazy time
    # A: In crazy time, correlation should be high. In normal time, correlation should be low.
    # A: TODO

    # Q:研究高送转
    # A: TODO check atest btest share_pct_ch

    # fundamental
    # TODO

    # Q:use us market industry to determine good industry for china
    # A: TODO

    # xueqiu data
    # A: xueqiu data seem to be pretty accurate. needs to be looked closer
    # A: In my test, I excluded the case that the new comments are on new IPO stocks. So all new discussion, comments, simulation trade, real trade are for long established stocks.
    # A: New follower between 0.7-0.8 is best. This means that not top viewed stocks are best.
    # A: new discussion between 0.5 is best. It is a very weird stat.
    # A: stock age is correlatedwith follower. Newer stock have less follower
    # A: the absolute stock sell return is not that high. So generally not very significant correlation.
    # A: interestingly, 0.1-0.3 new discussion seems to be good. This means stock with few attension will gain more.
    # A: it is hard to verify because xueqiu is new and data is only 2015-2019. The 2015 crazy time can distort the data. The gut feeling is that second lowest attetion and second highest attension have the best return.
    # A: TODO is useful, but need to be looked closer. e.g. scrap all xueqiu comment and analyze semantic


    #Q: find a 定投 strategy:
    #A: TODO


    # Q: Portfolio management, efficient frontier(highest risk and return combination):
    # in this case we consider risk = market risk = beta
    # A: since the beta is very high and can change very rapidly, past beta might be coincidence, past beta does not fully predict future. long term beta does better job
    # A: For short term investor, only consider short term beta
    # A: For long term investor, only consider long term beta
    # A: as we know, there is no real efficient frontier in A股 because all assets are highly market connected
    # A: Therefore choosing low market correlation automatically improves the portfolio, since the portfolio is mostly build using high beta assets
    # A: you can try to create portfolio using least beta, but it might cost some return.
    # A: Only very rare, there are two asset equally well, always choose the asset with least market correlation
    # A: TODO

    # Q: Portfolio management, kelley formula based on some estimation:
    # A: 1. The algorithm need to find something very predictive
    # A: 2. adjust the amount to buy calculated using kelly formula
    # A: But if something very predictive does not exist, kelly formula might be not effective, but still little useful
    # A: TODO

    # Q: Portfolio %: should be measured by kelly, or by market crazyness/condition?
    # A: If managed by kelly, then at normal/bad times, you would buy nothing, at good time, you would invest based on the estimation score
    # A: If managed by market condition, you would buy nothing at bad times, buy more a good crazy time.
    # A: Kelly uses marked condition to determin buy or not buy, kelly needs to be fine tuned. But they are basically the same

    # Q: volatility/beta between Groups (stocks are too many)
    # A: We can find some long term relationships, but short term relationships might be changed very quickly, so not always useful.
    # A: For Group
    # A: (白酒,保险)和创业板关联度最低,
    # A: (高送转)和主板关联度最低
    # A: (高送转)和中小板板关联度最低， (一带一路,采掘,煤炭，金属，航运)最高
    # A: possible assets that can be used because of negative beta:
    # A: These are mostly good stock 白马股
    """
    海天味业
    晨光文具
    牧原股份
    泰格医药
    歌尔股份
    恒立液压
    苏泊尔
    片仔癀
    新和成
    三花智控
    智飞生物
    大华股份
    华兰生物
    隆基股份
    涪陵榨菜
    华夏幸福
    中际旭创
    爱尔眼科
    立讯精密
    贵州茅台
    中公教育
    恒瑞医药
    正邦科技
    长春高新
    山东黄金
    恒生电子
    三一重工
    星宇股份
    格力电器
    东方雨虹
    五粮液
    伊利股份
    海螺水泥
    中航光电
    仁东控股
    纳指ETF
    白酒B
    食品B
    兴全轻资
    合润B
    博时主题
    纳指ETF
    兴全模式
    消费行业
    消费ETF
    消费进取
    东证睿丰
    国投产业
    兴全趋势
    景顺鼎益
    建信50B
    酒B
    优选LOF
    鹏华丰和
    中银中国
    诺安纯债

    """
    # A: NO Group have negative correlation with the market!: No real beta strategy possible G_G
    #A: Check out bullishness . Basically only FD I have checked before 消费and白酒
    #A: for E and EF just use bullishness instead, because some new IPO can have very less Beta
    #A: Conclusion: Hedging on A market is almost impossible due to high market correlation AND low amount of stocks that can be shorted
    #A: If long asset_e stocks for hedging, still face asset self beared volatility


    #Q: what is the general rule for reason and causation?
    #A: Reason -> causation. Reason must have at least as much state as causation
    #A: otherwise the mapping does not work
    #A; By this logic bond, oil, gold, or low volatility, low movement, low change data does not predict the market
    #A; A coin has only 2 states, a dice has 6. a coin can not describe the outcome of a dice.
    #A; But maybe the combination of low volatiliy can become causation.
    #A: Maybe 3 coins together can describe the dice (IF there is a relation anywhere)

    #Q: What are internal and external factors?
    #A: Internal factors are signals from the stockmarket: volume, price gain, rsi, ma ...
    #A: External factors are signals outside the stockmarket: oilprice, gold price, macro economy, pmi,
    #A: IN FUCKING THEORY, external factors should lead stockmarket. BUT in REALITY, stock market leads the economy.
    #A: Using external factors to predict stockmarket usually is more unreliable and has more delay than just looking at the price itself.
    #A: But some external factors can be used like fundamentals(which is not an external factor in real)

    #Q: Is there a time when volatility is low and bull market?
    #A: NO, this does not exist in A Stock. IN CN market, if there is bull market, volatiliy MUST be high


    #Q: use trend or detrended data?e.g. Volume
    #A: Trend data = abs data. Detrend data = Relative data = rolling.
    #A: In oder to predict, we often need both trend AND detrended. Because we need to compare current volume to 30 days ago AND since 2015.
    #A: Solution: multiple point sample. e.g. 12 days, 30days, 120 days.
    #A: Solution: Adaptive sampling points. Since last high or low


    #Q: is m0, m1, m2 useful
    #A: Yes, m1-m2.
    #A: But the problem is that there is no sign that to predict the m012, they are not periodic but manually conroled by the state
    #A: So the finding here only confirms the rise of stock market, it does not give hint in advance
    #A: https://www.legulegu.com/stockdata/broadmoney
    #A: it also does not explain the 2015 spike
    #A: it is more like stock market goes before m0 m1 m2


    # Q:unstable period. Shorter period when volatile, Longer period when less volatile
    # A: it is useful to some extend, but it seems that every method has its problem
    # A:unstable period creates new problem: difficult to explain
    # A: underlies the same problem of normal ma: whipsaw, bad when movement is flat, lag, unable to predict random movement
    # A: In general: not very useable

    #Q: stock vs bond. Buy stock if stock performed better and bond vice versa
    #A: This is actually idiotic since bond movement is very small, it makes not much sense to compare them
    #A: also bonds return are so small that one day in stock market can bring it back

    #A: Volatility/Gain adjusted macd
    #Q: since macd and some other indicator work best if market is volatile. Divide the macd signal by std or any other volatility. Then start use macd only if market volatility goes abv certain thresh
    #Q: Can also be used on RSI. RSI works best if volatility is big. Basically this is mapping rsi and usefulness of rsi into one indicator, same as gmean.
    #Q: But then it becomes similiar to an RSI, you can say abv some thresh I buy.
    #Q: MACD can be adjusted by using market_volatility


    #A: define crazy time and standard time by using cov and std. It works
    #Q: crazy time and standard time are continous. So the definition of such would naturally have a lag.
    #Q: you can define crazy time if std of past pct_chg exceeds some thresh hold, like an RSI
    #Q: Since it is continously, maybe using hard cut to binary is not a good idea. Instead use continous steps to define crazyness
    #Q: Check out function market_volatility. It actually works.

    # Q: Do high Dividends paying stock perform better in the long run?
    # A: Test done in bullisness by aggregating all past dividend yield ratio together as mean
    # A: top 10% rank 3468
    # A: top 20% rank 3372
    # A: top 30% rank 3323
    # A: bot 30% rank 3300
    # A: bot 20% rank 3310
    # A: bot 10% rank 3336
    # A: dividend almost had no impact on stock bullishness rank. So it no a good indicator. If it would be a good indicator, it would be too easy to detect.

    #Q: normalized highpass vs normalized close: are they the same?
    #A: Actually they are not the same. Normalized high pass describes the distance between price and its ma/lowpass
    #A: So a turning point would be that norm_close is at bottom and norm_highpass is at max. Because price is at low and there is a high willingness to buy
    #A: You can use highpass to see if a trend continues or is broken very fast
    #A: check out norm cj

    # Are good stock "close"/"pct_chg" different distributed?
    # A: Yes they are a bit different. Good stock are always at top which translates to being often at 0.8-1.0. Bad stock are vice versa.
    # A: The distribution actually reflects similar thing as e_max or gmean. So it does not reveal new information.
    # A: if close/pct_chg are normalized, then they are exactly the same.

    # Q: A pattern recognition that detects buying signals if price is abv ma, and lower than last hig
    # A: This strategy would be good to use on big cap stock because their movement changes slower
    # A: Tested with manual pattern recognition. It is bascially same as sfreq abv bfreq. The paradox is that you need to know future direction of the market in order to apply this strategy. So this strategy does not produce signal, but relies on other signal.
    # Algo: find a pattern recognition to buy good trend stocks at low price
    # 1. good long term trend
    # 2. long term trend not too high
    # 3. price is not max
    # 4. stop los if price falls

    # Q: Find a way to balance between good momentum and low price.let current momentum not break and low point
    # A: after checking the chart of calc, the momentum gets broken alot. So the momentum change freq is very high. Hence, very hard to let momentum not break because it happens very often.
    # A: Ultimately, low price +patience wins.

    # q: Distribution of "close"
    # A: short period e.g. 10 days are act as if there is no trend at all. If low then buy, high then sell.
    # A: longer period e.g. 500 have an uptrend move.
    # A: short period are staying longer at max/min
    # A: this further supports the idea. you want the shorterm price as low as possible. Long term trend as high as possible.

    # coutning index last high and low
    # count distance between last low and last high to determine next low

    # q: seasonality1: if first n month is good, then the whole year is good. This has high pearson 0.72, and high TT/FF rate if using index. So using index to predict this is pretty accurate.
    # A:monthofyear 1,2,4 月份有非常强的判断力。4月份因为是年报时间。month 1 and 2 have about 72%(I)/68%(E,G) of correct predicting the year. 如果整年好，12月份有80%的概率会涨。So if many stock have gained in first and 2nd month, this is a good sign for the whole market this year.
    # A:weekofmonth 1,2, have predictability about ~63% of correct predicting this month outcome
    # A:weekofmonth could also be wrong because each week NATURALLY is part of month and has some contribution to the months gain. regarding this 63% predictability is not that high.

    # q: intraday volatility
    # A: volatility highest at opening, lowest at close
    # A: meangain is lowest at begin, highest at end.
    # A: this finding is robust for all 3 major index.
    # A: This test was done with 15m data. First and last 15min are most significant.
    # A: While intraday_volatility seems to be consisten. intraday gain is different. 000001.SH loses most at first 15m and gainst most at last 15m. The other two index are mixed.
    # A: General advice: sell at last 15min or before 14:30 because high sharp ratio. Buy at begin because begin 15 min return is mostly negative

    # q: relationship between bond and stock market
    # A: little. bond market just goes up. If interest is lower, bond market goes higher??

    # margin account/leverage/short, amount of people opening account.
    # A: Margin is a very new tool introduced after 2010.So the observed period is rather short. But it seems that security sell, short selling indicator can predict the crazyness of the chart


    # q: correlation between us market gain and chinese market gain
    # A: the pearson relationship has increased over the last year. the more recent the higher relaitonship
    # A: the pearson since 2015 is 0.15-0.20 which is quite high
    # A: us influences China with pearson about 0.2. China influences us with pearson about 0.2. it is a circle of influence
    # A: us close has 57% of predicting chinese close to be positive or negative. 42% wrong.

    # q: volatility between other index
    # A: since beta between sp500 and sh is very low. Their volatlity is hence useless.
    # A: the market is self directed, with less than 20% of times corresponding together in panic time.

    # q: how much does first/last 15 min of day predict today
    # A: alot. The pearson between 15min pct_chg and whole day pct_chg is ~0.55 which is insanely high
    # A: TT+FF rate is at 0.68%. So the first 15 mins hat 68% predict if today return is positive or negative.
    # A: the second highest pearson to predict the day is the first 30 mins after lunch break.
    # A: Basically. same founding in mont-to-year relationship can be found here.

    # q: how much does first/last 15 min of day predict tomorrow
    # A: weak predictability. around 53%. First 15min and last 15min have the highest predictability.

    # q: how much does rolling first/last 15 min of day predict tomorrow
    # A: rolling 5 mean of first 15mins -45mins have higher predictability of tomorrow. afternoon trade rolling mean have not so much predictability of tomorrow. But in general the predcitability is low. at best 0.51 TT+FF rate.

    # q: STD of pct_chg between all stocks on a day
    # A: hard to read since 2008 and 2015 are very different. high volatility comes either at near top or bottom. Definetly in crazy times. Market seems to be changing from 2008,2015,2023. So this indicator is not very useful.

    # q: winner loser difference
    # A: this is the same as std between all stocks. So no big indicator

    # q: cash flow 资金流向
    # A: not really manually tested, but it should be same as chart. So skipped.

    # q: 龙虎榜
    # A: It is just a ranking based on turnover and gain. I can do this myself

    # q: 举牌，大宗交易
    # A: Tested with logic: If someone sells, there is a buyer. So only the amount of trade can reflect the market condition. This is usually higher when market is in crazy times. Which is not a very useful indicator since crazy times can easly be determined.

    # Q: text analyzer
    # A: tested seems to be useful, but data is not big enough. So basically not very useful. CCTV_news is not only financial. Need a financial pure big data source

    # Q: answer the question. if an asset is in uptrend/downtrend/cycle mode, how likely is it to break the mode.
    # Q: is it likely to be broken by seasonality?
    # Q: or by other events?
    # A: I think this question is very hard to answer since the likelyhood of a trend being broken is random.

    # Q: Technical analysis using connection of extrema values
    # A: tested, works good. It is relatively better than macd, but rare times macd can still be better.
    # A: In general less whipsaw than macd
    # A: but even here. A tradeoff between early signal, and accuracy must be made.

    # Q: support and resistance but using only max and min from last high or low point on. reduce complexity and time. Not checking the whole time period, but the last period
    # A: this will not work as it is similar to extrema values test which I already performed.

    # Q: is it better to anticipate or wait for turning point to confirm?
    # A: both. It is a tradeoff. Sometimes wait is better. sometimes participate better
    # A: Hypothesis: Short term anticipate. Long term wait to confirm. Because short term swings very frequently. Long term swings very few. Waiting for confirm makes sense.

    # Bet against the trend or with the trend?
    # A: after you buy high price can go up, after you buy low price can go up.
    # A: so The past trend does not matter. What matter is future trend.
    # A: So the question is not relevant
    # A: the REAL question is. Is it more likely for a past uptrend to continue?
    # A: the answer to that is YES if it is a long term trend. NO if it is a short term trend.
    # A: short term (5,10 days) more likely to mean reverse. Long term (500,750 days) trend more likely to continue.
    # A: long term trend continues once confirmed. It does not switch that often. Shor term trend mean reverses very often.

    # Q: Is there are way to determin if the rise is a short term FIR or start of new long term trend?
    # A: No. It can go either way. But usually it is better to bet with mean. If it gained/lost too much, it will reverse to mean. Thats why buffet says patience pays off.

    # Q: combining two method will cause multiple causaion problem (or like this)
    # A: A method must be very significant to be able to use. Otherwise its wrong signals will interfere too much with other methods

    # Q: Modle the 1/E game for stock market and general. If something happens that exceeds the max of everything before, then it is a signal? doesnt seem to apply for usoil for instance.
    # A: Using Logic deduction: for this to be useful, the price has to be randomly+independetly distributed.
    # A: we know in the long run, the price has trend. In short run, it is randomly distributed.
    # A: so maybe, use past 5 day pct_chg price. If today pct_chg is higher than all last 5 days, sell
    # A: if today close is higher than all last 5 days, sell.
    # A: This is just a randomwalk approach. After you trade, the price can still go higher or lower.
    # A: So maybe the 1/E analogy here is not the best

    #Q: where to get institution holding data?
    #A: no idea. It is possible to scrape all fund and check their holding. But this can only be done for top 10 holding stocks and each season due to publicatino periodicty.
    #A: It is in general not possible to get all fund portfolio data to see if they are buying more or not.
    #A: This makes institution data only useful on individual stock and not useful to predict the whole market
"""
summmary
the bigger the freq the more predictable, the more useless the data
The smoother your signal, the longer the lag (solveable by insta trendline, kalman filter)
the earlier you want to detect the turnpoint, the more sensitive the indicator needs to be, the more whiplas it creates. (Basically, you need to adjust the period based on probability for the next turning point signal)
the better the trend the easier to make money.
the more normalize, the more sensitive, the more recent is the data,   《==》  the more whipsaw, the more noise.
In Up Trend move, buy and hold
In cycle mode buy and sell
in downtrend mode, shortsell or dont buy at all
john ehler did not incoperate ANN, fundamental, industry, date
if combining indicators, they need to come from independed sources
The general problem of all unstable period indicator is that they are too volatile and need a bigger frequency to compensate
ATR is pretty good to determine volatility
most dominant period is 15-40.Does this mean the reverse happens are 15-40 days? Check again #TODO
Heisenberg: the earlier you know the signal, the less predictive it is
Newton: if there is no anti force, the current trend continues
Superposition: short term wave + long term wave interfere
Even if outside factors like oil, employment affects the stock market. The stock market can remain unchanged and not react until a very long time. So you need long freq macd to determine.
The market self contained drive seems to be stronger than anything else. It is just pure evaluation of a company, nothing else.
The reason why RSI or any other technical prediction doesnt work is because after a bottom, there could be another bottom. So in short term it is pretty random. You can only invest in long term, like decades
The idea to use independed/canonical sources and conjunct them together to create a more powerful indicator sounds good, but doesnt work. All indicator have FalseTrue or TrueFalse rate. They need to be extrem low to be able to conjunct. Paradox. you need good standalone indicator, to be able to combine them. But if you already have good standalone indicator, there is no need to improve anymore. 
E.g. Strat1 and Strat2 have both 70% to predict correct. But conjunct them together, only 0.7*0.7=0.49 times are correct.
Because if half says buy, half say sell, who do you listen? There is no objective judgement to rule out the other half. Only way to do is by voting. 
But it can happen that majority votes wrong.
AND voting of technical indicators are NOT independent. They are all the same derived from chart
Voting from fundamental/macro indicators are very slow(low frequency) and hence not compareable with technical indicator
The safes investment = the Lower the price the better, the higher the long term momentum the better

EVEN if there are two samples of indicator describing the market perfectly, combining them in one might be a bad idea some times because the combination does not describe anything good. Maybe it is better to leave them seperated and let human decide when and what to use.

Concept research problem: 高价股 was defined very late. Hence, by backtest and selecting stocks with high close, will not yield good results. This is because the concept somehow already exposed future
The problem of using extrema to predict future, is that after a minma, it can follow another minima and so on. It is random based on stock fundamental.
You can only calculated the probability based on past value if price will rebounce.
But this probability is also not accurate, since stock fundamentals change

Technical analysis are very similar. It seems that one can not extract new information, but can only extract existing information in another form.
Hence, technical analysis might be a overlooked area. Maybe we should focus search on pair trading instead since it is harder to research.

The whole market works like this. If market is at bottom, good stock will rise. If market is at top, ALL stock will go down. That is VERY IMPORTANT FINDING.
"""

if __name__ == '__main__':
    pass

    df= DB.get_asset(ts_code="000001.SH", asset="I")
    e1=Alpha.hhll(df=df, abase="close", freq=20, inplace=True, score=1,thresh=0.07)
    e2=Alpha.vola(df=df, abase=e1[0], freq=500, inplace=True)
    e3=Alpha.cj(df=df, abase="close", freq=240, inplace=True)
    df[e1[0]]=df[e1[0]]*600
    df[e2[0]]=df[e2[0]]*600
    df[e3]=df[e3]*2000


    UI.plot_chart(df, ["close", e3])
    #market_volatility()
    # ball.set_token('xq_a_token=2ee68b782d6ac072e2a24d81406dd950aacaebe3;')
    # df=ball.report("SH600519")
    # print(df)
