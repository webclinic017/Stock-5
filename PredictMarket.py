import cProfile
# set global variable flag
import Alpha
from Alpha import *
import numpy as np
import UI
from scipy.stats.mstats import gmean
from scipy.stats import gmean
import sys
import os
import matplotlib
import itertools
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import DB
import LB
import builtins
import Atest
from scipy.stats.mstats import gmean
from functools import partial


def a_ts_code_helper(index):
    if index == "sh":
        a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="主板"))
    elif index == "sz":
        a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="中小板"))
    elif index == "cy":
        a_ts_code = DB.get_ts_code(d_queries=LB.c_market_queries(market="创业板"))
    else:
        group = f"{index.split('_')[0]}_{index.split('_')[1]}"
        instance = index.split("_")[-1]

        df_group_lookup = DB.get_ts_code(a_asset=[group])
        a_ts_code = df_group_lookup[df_group_lookup[group] == instance]

    return a_ts_code

def pe_overview(df_result,d_preload):



    df_ts_code=DB.get_ts_code(a_asset=["E"])
    d_market_name=LB.c_index_name()

    #pe for index
    for indicator in ["pe_ttm"]:
        #initialize with 0
        for index in ["sh", "sz", "cy"]:
            df_result[f"{indicator}_a{index}"]=0
            df_result[f"{indicator}_a{index}_counter"]=0

        #loop over all stocks and sum to their related index
        for ts_code, df_asset in d_preload.items():
            print(f"CREATE PE VIEW WITH {ts_code}")
            for index in ["sh","sz","cy"]:
                market = df_ts_code.at[ts_code,"market"]
                if d_market_name[market]==index:
                    df_result[f"{indicator}_a{index}"]=df_result[f"{indicator}_a{index}"].add(df_asset[indicator].clip(0,200),fill_value=0)

                    df_asset["counter"] = 1
                    df_result[f"{indicator}_a{index}_counter"] = df_result[f"{indicator}_a{index}_counter"].add(df_asset["counter"], fill_value=0)

        #finalize
        for index in ["sh", "sz", "cy"]:
            df_result[f"{indicator}_a{index}"]=df_result[f"{indicator}_a{index}"]/df_result[f"{indicator}_a{index}_counter"]
            del df_result[f"{indicator}_a{index}_counter"]

    #pe for industry
    df_pe = df_result.copy()
    for indicator in ["pe_ttm"]:
        for group in LB.c_groups():
            for instance in df_ts_code[group].unique():
                print(f"calc pe for {group} {instance}")
                a_ts_code = df_ts_code[df_ts_code[group]==instance]

                df_pe[f"{indicator}_{group}_{instance}"]=0
                df_pe[f"{indicator}_{group}_{instance}_counter"]=0
                for ts_code, df_asset in d_preload.items():
                    if ts_code in a_ts_code.index: # member of the group_instance
                        df_asset["counter"]=1
                        df_pe[f"{indicator}_{group}_{instance}_counter"]=df_pe[f"{indicator}_{group}_{instance}_counter"].add(df_asset["counter"],fill_value=0)

                        df_pe[f"{indicator}_{group}_{instance}"]=df_pe[f"{indicator}_{group}_{instance}"].add(df_asset[indicator].clip(0,200),fill_value=0)

                df_pe[f"{indicator}_{group}_{instance}"]=df_pe[f"{indicator}_{group}_{instance}"]/ df_pe[f"{indicator}_{group}_{instance}_counter"]
                del df_pe[f"{indicator}_{group}_{instance}_counter"]

        df_pe.to_csv(f"Market/CN/PredictMarket/{indicator}.csv",encoding="utf-8_sig")



def validate(df_result, index="cy"):
    df_result[f"pct_chg_{index}"] = df_result[f"close_{index}"].pct_change()
    df_result[f"pct_chg_{index}"] = df_result[f"pct_chg_{index}"] + 1
    df_validate = df_result.copy()

    for column in df_validate.columns:
        col_min = df_validate[column].min()
        col_max = df_validate[column].max()

        if col_min in [0, 1] and col_max in [0, 1]:
            # already in portfolio form [0:1]
            pass
        elif col_min in [-1, 1] and col_max in [-1, 1]:
            # needs to be adjusted to [0:1]

            df_validate[column] = Alpha.norm(df=df_validate, abase=column, inplace=False, min=0, max=1)
        else:
            # not in standard form, not adjustable
            continue

        df_validate[column] = df_validate[column].shift(1)  # to make todays signal used for tomorrow.
        df_validate[f"pct_change_x_{column}"] = ((df_validate[f"pct_chg_{index}"] - 1) * df_validate[column]) + 1  # todays pct_chg x pquota
        df_validate[f"{column}_comp_chg"] = df_validate[f"pct_change_x_{column}"].cumprod()  # compound together

        # remove trash column
        del df_validate[f"pct_change_x_{column}"]

    return df_validate

def create_portfolio(df_result, df_stock_long_term, end_date):
    """a good portfolio must fullfill these conditions
    -reactive to the market(df_result)
    -good stock
    -diversity of industry
    -diversity of size
    -diversity of market

    rule:
    if CY is good, portfolio size more of :
    small stock, cy stocks, certain industries, high market related stocks

    """
    a_groups = LB.c_groups()
    psize=7

    cy_trend=df_result.at[end_date,"cy_trend"]
    cy_trend_pquota=df_result.at[end_date,"cy_trendmode_pquota"]
    df_portfolio=pd.DataFrame()


    df_stock_long_term=df_stock_long_term.sort_values(by="final_rank",ascending=True,inplace=False,)

    #find psize pairs of stocks with least industry overlay
    """
     algorithm:
     size =1 : trivial, find stock with highest rank
     size =2 : n². two for loop. for each stock check against each other stock
     pentality: -1 for each stock in the same group/ industry, 
     problem: two independent axis, how to discount?  
    
        if all industry are different, then no penality
        if all are same, then 100% penality
        solution: since we only do long term investment, the industry are fixed:
        we just choose the best 3 industry
        
    """







    df_portfolio.to_csv(f"Market/CN/PredictMarket/Portfolio/stock_all_time_until_{end_date}.csv", encoding="utf-8_sig")


def predict_stock_long( d_preload, end_date=LB.today(), a_freq = [20, 60, 120, 240]):
    """
    general compare concept:

    Agent = Asset and Groups:
    1. Single Stock (E, I, FD)
    2. Industry (G)
    3. Market, (Asset_E , but it is somehow bugged)

    Attribute = stats:
    1. technical stats (ma, overma)
    2. fundamental stats (roe, roi, pe, pb, growth)
    3. fund portfolio holding ()

    Long time or momentum = : to see
    1. long run: mean, std, abs val
    2. short run: momentum, higher than mean?

    Compare target: to see higher or lower
    1. Time comparison: Self past
    2. competition: Industry, Market

    Goal:
    Which industry has current momentum (Undervalued, price rising)
    Which Stock has current momentum (Undervalued, price rising, more funds are buying it than before)

    Which industry is long term speaking good
    Which stock is long term speaking good(Good technical stats, good fundamental stats, funds buying it very often)


    Relationship between long term and short term:
    1. Do I add them together, or do I ignore long term and see it as a cumsum or short term.

    Algorithm:
    1. Calcualte all stats for single stocks (Vs itself past)
    2. compound them into Industry or market
    3. Calculate all single stock against market (Long Term= fund holding , fundamental, technical) = Bullishness Expanding
    4. Calculate all single stock against market (Short term = who wins fun holding, fundamental, technical) = Bullishness Rolling

    5. Define which industry is relatively better = (How other stocks in the same industry are doing)

    #todo distinguish long vs short
    calculate stock score for all period since 3000


    note:
    fundamental time series part in df
    fundamental non time series part is NOT in df but seperated
    fundamental note:
    -This way of ranking biases overall big and good stocks, mostly big sh stocks because all stats are taking into account
    -good growth stocks usually have high growth, but other things are bad.


    technical note
        1. The higher the gmean, the higher the overall(gain + volatility) stats
        2. the higher the overma, the less volatile the stock.
        3. If a stocks gmean is ranked high, but overma is middle. this means the stock has some very strong and weak periods.
        4. pgain is a bad stats. it biases on high volatile stocks. E.g. A Stock gains on day 1 10times, and falls on day 2 90% back to the original. The pgain would be (1000% + 90%)/2 = 5.45. vs (120% + 120%)= 4.4.
        5. gmean alone defines all the stats we need.
        6. overma checks only the tendency of volatility. the higher overma, the less volatile.
        7. if market is good for sure for all, then bg high volatile stock, go after gmean high industry like 传媒，军工
        8. if market is normal or even bad, go for industry that can stay long over ma like 建筑材料，综合
        9. There are 2 Industry that are EXTREM good at BOTH. 食品饮料 and 医药

    """

    if os.path.isfile(f"Market/CN/PredictMarket/Stock/stock_all_time_until_{end_date}.csv"):
        return print(f"STOCK LONG TERM already done for {end_date}")


    # 0. INIT
    df_stock_long_term = DB.get_ts_code(a_asset=["E"])
    d_fun_indicator = LB.c_indicator_rank()

    # 1. SINGLE STOCK ALL TIME PREP
    for ts_code, df_asset in d_preload.items():
        print(f"STOCK ALL TIME PREP {end_date} {ts_code}")
        if df_asset.empty:
            continue

        # technical abvma = ability to stay over ma as long as possible
        for freq in a_freq:
            df_asset[f"abvma{freq}"] = Alpha.abv_ma(df=df_asset, abase="close", freq=freq, inplace=False)

        # technical freqhigh = ability to create 240d high
        for freq in a_freq:
            df_asset[f"rolling_max{freq}"] = df_asset["close"].rolling(freq).max()
            df_helper = df_asset.loc[df_asset[f"rolling_max{freq}"] == df_asset["close"]]
            df_asset[f"{freq}high"] = df_helper[f"rolling_max{freq}"]

        # technical freqlow = ability to avoid 240d low
        for freq in a_freq:
            df_asset[f"rolling_min{freq}"] = df_asset["close"].rolling(freq).min()
            df_helper = df_asset.loc[df_asset[f"rolling_min{freq}"] == df_asset["close"]]
            df_asset[f"{freq}low"] = df_helper[f"rolling_min{freq}"]

        # fundamentals are already there, no need to create



    # 2. SINGLE STOCK ALL TIME SUMMARY
    for ts_code, df_asset in d_preload.items():
        print(f"STOCK ALL TIME SUMMARY {end_date} {ts_code}")
        if df_asset.empty:
            continue

        # period
        df_stock_long_term.at[ts_code, "period"] = len(df_asset)

        # technical abvma
        for freq in a_freq:
            df_stock_long_term.at[ts_code, f"abvma{freq}_mean"] = df_asset[f"abvma{freq}"].mean()  # the higher the better

        # technical gmean
        df_asset["pct_change"] = 1 + df_asset["close"].pct_change()
        df_stock_long_term.at[ts_code, f"gmean_mean"] = gmean(df_asset["pct_change"].dropna())

        # technical freqhigh
        for freq in a_freq:
            df_stock_long_term.at[ts_code, f"{freq}high"] = df_asset[f"{freq}high"].clip(0, 1).sum() / len(df_asset)

        # technical freqlow
        for freq in a_freq:
            df_stock_long_term.at[ts_code, f"{freq}low"] = df_asset[f"{freq}low"].clip(0, 1).sum() / len(df_asset)

        # fundamental from time series
        for indicator in ["pe_ttm", "pb", "ps_ttm", "dv_ttm"]:
            df_stock_long_term.at[ts_code, f"{indicator}_mean"] = df_asset[f"{indicator}"].mean()

        # fundamental from NON time series
        df_fina = DB.get_asset(ts_code=ts_code, freq="fina_indicator", asset="E", )
        for indicator in d_fun_indicator.keys():
            df_stock_long_term.at[ts_code, f"{indicator}_mean"] = df_fina[indicator].mean()



    # 3. RANK SINGLE STOCK ONE TIME INSTITUTIONAL
    #df_fund_portfolio = Atest.asset_fund_portfolio().set_index("ts_code", inplace=False)
    df_fund_portfolio = DB.get(a_path=LB.a_path(r"Market\CN\ATest\fund_portfolio\all_time_statistic"), set_index="ts_code")
    current_year=end_date[0:4]
    current_month=end_date[4:6]
    current_season=0
    if current_month in ["01","02","03"]:
        current_season = 1
    elif current_month in ["04","05","06"]:
        current_season = 2
    elif current_month in ["07","08","09"]:
        current_season = 3
    elif current_month in ["10","11","12"]:
        current_season = 4
    column_name = f"{current_year}_{current_season}_rank"
    df_stock_long_term["stock_irank"]=df_fund_portfolio[column_name] # already ranked
    df_stock_long_term.loc[df_stock_long_term["period"].isna(),"stock_irank"]=df_stock_long_term["stock_irank"].max()


    # 4. RANK SINGLE STOCK ALL TIME TECHNICAL
    df_stock_long_term["stock_trank"] = builtins.sum([df_stock_long_term[f"abvma{freq}_mean"].rank(ascending=False) for freq in a_freq]) * 0.1 \
                                        + df_stock_long_term[f"gmean_mean"].rank(ascending=False) * 0.60 \
                                        + builtins.sum([df_stock_long_term[f"{freq}high"].rank(ascending=False) for freq in a_freq]) * 0.15 \
                                        + builtins.sum([df_stock_long_term[f"{freq}low"].rank(ascending=True) for freq in a_freq]) * 0.15 \


    # 5. RANK SINGLE STOCK ALL TIME FUNDAMENTAL
    d_fun = {**{"pe_ttm": True, "pb": True, "ps_ttm": True, "dv_ttm": False}, **d_fun_indicator}
    # check how many of these indicators are horizontally np.nan of a stock
    for ts_code in df_stock_long_term.index:
        df_stock_long_term.at[ts_code, "misssing_fun"] = df_stock_long_term[[f"{col}_mean" for col in d_fun]].loc[ts_code].isnull().sum()
    # if one stock misses some indicator, we just don't count them: fill_value =0
    df_stock_long_term["stock_frank"] = 0.0
    for indicator, ascending in d_fun.items():
        df_stock_long_term["stock_frank"] = df_stock_long_term["stock_frank"].add(df_stock_long_term[f"{indicator}_mean"].rank(ascending=ascending), fill_value=0)
    # the fundamental rank counts only categories where the stock HAS DATA. = all possible cols - missing cols
    df_stock_long_term["stock_frank"] = df_stock_long_term["stock_frank"] / (len(d_fun) - df_stock_long_term["misssing_fun"])


    # 6. RANK SINGLE STOCK ALL TIME TECHNICAL + FUNDAMENTAL + INSTITUTIONAL STATS
    df_stock_long_term["stock_tifrank"] = df_stock_long_term["stock_trank"] * 0.7 + df_stock_long_term["stock_frank"] * 0.2 + df_stock_long_term["stock_irank"] * 0.1


    # 7. RANK INDUSTRY ALL TIME TECHNICAL + FUNDAMENTAL + INSTITUTION
    a_group_results = []
    for group in LB.c_groups():
        df_grouped = df_stock_long_term.groupby(group, sort=False)
        df_grouped_result = df_grouped.mean()
        df_grouped_result["count"] = df_grouped.count()["stock_trank"]
        df_grouped_result["ts_code"] = df_grouped_result.index
        df_grouped_result["ts_code"] = f"{group}_" + df_grouped_result["ts_code"]
        df_grouped_result["group"] = group
        df_grouped_result = df_grouped_result.set_index("ts_code", inplace=False, drop=True)
        a_group_results += [df_grouped_result]
    df_industry_long_term = pd.concat(a_group_results, sort=False)
    df_industry_long_term.to_csv(f"Market/CN/PredictMarket/Industry/industry_all_time_until_{end_date}.csv", encoding="utf-8_sig")


    # 8. UPDATE ALL TIME INDUSTRY STATS BACK TO SINGLE STOCK
    for ts_code in df_stock_long_term.index:
        for group in LB.c_groups():
            instance = df_stock_long_term.at[ts_code, group]
            if instance == None:
                continue
            # map each group with their t, r, tr rank
            for tf in ["t","i", "f","tif"]:
                df_stock_long_term.at[ts_code, f"{group}_{tf}rank"] = df_industry_long_term.at[f"{group}_{instance}", f"stock_{tf}rank"]
    # summarize all group of one stock together into one industry overall rank
    for tf in ["t", "i","f", "tif"]:
        df_stock_long_term[f"industry_{tf}rank"] = builtins.sum([df_stock_long_term[f"{group}_{tf}rank"] for group in LB.c_groups()])


    # 9. CREATE SINGLE STOCK ALL TIME FINAL RANK
    df_stock_long_term[f"final_rank"] = df_stock_long_term[f"stock_tifrank"] * 0.9 + df_stock_long_term[f"industry_tifrank"] * 0.10


    df_stock_long_term.to_csv(f"Market/CN/PredictMarket/Stock/stock_all_time_until_{end_date}.csv", encoding="utf-8_sig")
    #todo add std into consideration of fundamentals rank
    #create stats for each stock: offensive= ability to gain, maximal high
    #defensive = year, volatility, minimal low,  year,
    #market condition: size, sh or cy market, good industry
    #general stats= good industry
    return df_stock_long_term

def predict_stock_short(df_result,d_preload):

    """
    indicator that stock could be good in short run
    1. Stocks beats index (tried using that, no good result, short term seems too random)
    2. Stock beats industry
    3. Stock has low market correlation (The same applies to this. in short term it is too random. In long term it has no short term usage)
    4. Stock has vol and is not too crazyly high
    5. Stock industry as general is rising (same as beats index will not work because short term is too random)
    6. Stock industry as general has vol (even if volume can be detected, reverse happens too quick)
    """

    df_overview=DB.get_ts_code(a_asset=["E"])# ts_code series
    df_result_helper=df_result.copy() # time series

    a_freq= [20,40]
    a_index=["sh","cy"]
    a_used_ts_code=[]
    if True:
        for index in a_index:
            for freq in a_freq:
                df_result_helper[f"pct_chg{freq}_{index}"]= 1 + df_result_helper[f"close_{index}"].pct_change(freq)


        for ts_code, df_asset in d_preload.items():
            #create past pct_change on the go
            print(f"STOCK SHORT TERM {ts_code}")
            if df_asset.empty:
                continue

            a_used_ts_code+=[ts_code]

            #check what stock has gained in past freq
            for freq in a_freq:
                df_result_helper[f"pct_chg{freq}_{ts_code}"]= 1 + df_asset[f"close"].pct_change(freq)

            #check if stock has gained more than three index
            for index in a_index:
                df_result_helper[f"{ts_code}_to_{index}"]=np.nan
                for freq in a_freq:
                    df_result_helper[f"{ts_code}_to_{index}_{freq}"]=df_result_helper[f"pct_chg{freq}_{ts_code}"]/df_result_helper[f"pct_chg{freq}_{index}"]
                    df_result_helper[f"{ts_code}_to_{index}"]=df_result_helper[f"{ts_code}_to_{index}"].add(df_result_helper[f"{ts_code}_to_{index}_{freq}"],fill_value=0)
                df_result_helper[f"{ts_code}_to_{index}"] =df_result_helper[f"{ts_code}_to_{index}"]/len(a_freq)

            #combine all 3 index score together
            df_result_helper[f"{ts_code}_to_index"]=np.nan
            for index in a_index:
                df_result_helper[f"{ts_code}_to_index"]=df_result_helper[f"{ts_code}_to_index"].add(df_result_helper[f"{ts_code}_to_{index}"],fill_value=0)
            df_result_helper[f"{ts_code}_to_index"]=df_result_helper[f"{ts_code}_to_index"]/len(a_index)



        #only relevant colums
        a_relevant_col = [f"{ts_code}_to_index" for ts_code in a_used_ts_code]
        df_result_helper = df_result_helper[a_relevant_col]

        #rename f"{ts_code}_to_index" to f"ts_code"
        df_result_helper=df_result_helper.transpose()
        df_result_helper["ts_code"]=df_result_helper.index
        df_result_helper["ts_code"]=df_result_helper["ts_code"].astype(str).str.slice(0,9)
        df_result_helper=df_result_helper.set_index("ts_code", drop=True)

        #a_path = LB.a_path(f"Market/CN/PredictMarket/Stock_Short_term/overview")
        #LB.to_csv_feather(df=df_result_helper, a_path=a_path)
    else:
        df_result_helper=pd.read_csv("Market/CN/PredictMarket/Stock_Short_term/overview",encoding="utf-8_sig").set_index("trade_date")


    #for each day create ts_code view and check which industy has most momentum
    a_racing=[]
    for trade_date in df_result.tail(500).index[::5]:
        print(f"create date view {trade_date}")
        df_date_view=df_overview.copy()
        df_date_view[f"{trade_date}_mom"]=df_result_helper[trade_date]
        d_df=DB.to_excel_with_static_data(df_ts_code=df_date_view,add_static=False, sort=[f"{trade_date}_mom",False], a_asset=["E"],path=f"Market/CN/PredictMarket/Stock_Short_Term/{trade_date}.xlsx")
        df_group=d_df["sw_industry1"]
        df_group[f"{trade_date}rank"]=df_group[f"{trade_date}_mom"].rank(ascending=False)
        a_racing+=[df_group[f"{trade_date}rank"]]


    df_racing=pd.DataFrame(a_racing)
    a_path = LB.a_path(f"Market/CN/PredictMarket/Stock_Short_term/racing")
    LB.to_csv_feather(df=df_racing, a_path=a_path)



def predict_industry(df_result_copy, d_preload):
    # Step 2: Industry Score
    df_ts_code_G = DB.get_ts_code(a_asset=["G"], d_queries=LB.c_G_queries_small_groups())

    # Step 2.1: Industry Long Time Score
    df_longtime_score = Atest.asset_bullishness(df_ts_code=df_ts_code_G)
    df_longtime_score = df_longtime_score[df_longtime_score["period"] > 2000]
    df_longtime_score.sort_values(by="final_position", ascending=True, inplace=True)

    """df_longtime_score.to_csv("temp.csv", encoding="utf-8_sig")
    df_longtime_score = pd.read_csv("temp.csv")
    df_longtime_score = df_longtime_score.set_index("ts_code")"""

    # Step 2.2: Industry Short Time Score
    fields = ["close", "vol"]
    result_col = []
    for ts_code in df_ts_code_G.index:
        print("calculate short time score", ts_code)
        # create asset aligned with sh and cy index
        try:
            df_asset = DB.get_asset(ts_code=ts_code, asset="G", a_columns=fields)
        except:
            continue

        # check for duplicated axis
        duplicate = df_asset[df_asset.index.duplicated()]
        if not duplicate.empty:
            print(ts_code, " has duplicated bug, check out G creation")
            continue

        # calculate trendmode pquota
        df_asset = df_asset.rename(columns={key: f"{key}_{ts_code}" for key in fields})
        df_asset = pd.merge(df_result_copy, df_asset, how='left', on="trade_date", suffixes=["", ""], sort=False)
        predict_trendmode(df_result=df_asset, index=ts_code, d_preload=d_preload)

        # add asset result to result table
        df_result_copy[f"{ts_code}_trendmode_pquota"] = df_asset[f"{ts_code}_trendmode_pquota"]
        result_col += [f"{ts_code}_trendmode_pquota"]

    # general market condition
    for column in result_col:
        df_result_copy["market_trend"] = df_result_copy["market_trend"].add(df_result_copy[column])
    df_result_copy["market_trend"] = df_result_copy["market_trend"] / len(result_col)

    # rank the industry short score. Rank the bigger the better
    d_score_short = {}
    for column in result_col:
        d_score_short[column] = df_result_copy[column].iat[-1]

    df_final_industry_rank = pd.Series(d_score_short, name="ts_code_trendmode_pquota")
    df_final_industry_rank = df_final_industry_rank.to_frame()
    df_final_industry_rank["short_score"] = df_final_industry_rank["ts_code_trendmode_pquota"].rank(ascending=False)  # the higher pquota the better

    # rank the industry long score
    print("df long score is ", df_longtime_score)
    d_score_long = {}
    for column in result_col:
        ts_code = column[:-17]  # 17 because the name is "_trendmode_pquota"
        if ts_code in df_longtime_score.index:
            d_score_long[column] = df_longtime_score.at[ts_code, "final_position"]
            print("correct", column)
        else:
            print("oopse ts_code wrong or something. or substring removal wrong?", ts_code)
    s_long_score = pd.Series(d_score_long, name="ts_code_trendmode_pquota")

    # Step 2.3: Industry Current Final Score
    df_final_industry_rank["long_score"] = s_long_score
    df_final_industry_rank["final_score"] = df_final_industry_rank["long_score"] * 0.7 + df_final_industry_rank["short_score"] * 0.3

    return df_final_industry_rank


def predict_trendmode(df_result, d_preload, debug=0, index="cy"):
    """1. RULE BASED: write direclty on _trendmode_pquota column
     base pquota is used by counting the days since bull or bear. The longer it goes on the more crazy it becomes"""
    df_result[f"{index}_trendmode_pquota"] = 0.0
    df_result[f"{index}_trendmode_pquota_days_counter"] = 0

    trend_duration = 0
    last_trend = 0
    for trade_date, today_trend in zip(df_result.index, df_result["cy_trend"]):
        if today_trend == 1 and last_trend in [0, 1]:  # continue bull
            trend_duration += 1
        elif today_trend == 1 and last_trend in [0, -1]:  # bear becomes bull
            trend_duration = 0
            last_trend = 1
        elif today_trend == -1 and last_trend in [0, 1]:  # bull becomes bear
            trend_duration = 0
            last_trend = -1
        elif today_trend == -1 and last_trend in [0, -1]:  # continue bear
            trend_duration += 1
        else:  # not initialized
            pass

        df_result.at[trade_date, f"{index}_trendmode_pquota_days_counter"] = trend_duration

    # assign base portfolio size: bottom time 0% max time 60%
    # note one year is 240, but minus -20 days to start because the signal detects turning point with delay

    # bull
    df_result.loc[(df_result["cy_trend"] == 1) & (df_result[f"{index}_trendmode_pquota_days_counter"] < 220), f"{index}_trendmode_pquota"] = 0.5
    df_result.loc[(df_result["cy_trend"] == 1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(220, 460)), f"{index}_trendmode_pquota"] = 0.6
    df_result.loc[(df_result["cy_trend"] == 1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(460, 700)), f"{index}_trendmode_pquota"] = 0.7
    df_result.loc[(df_result["cy_trend"] == 1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(700, 1000)), f"{index}_trendmode_pquota"] = 0.8

    # bear
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"] < 220), f"{index}_trendmode_pquota"] = 0.2
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(220, 460)), f"{index}_trendmode_pquota"] = 0.1
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(460, 700)), f"{index}_trendmode_pquota"] = 0.0
    df_result.loc[(df_result["cy_trend"] == -1) & (df_result[f"{index}_trendmode_pquota_days_counter"].between(700, 1000)), f"{index}_trendmode_pquota"] = 0.0

    del df_result[f"{index}_trendmode_pquota_days_counter"]

    """2. EVENT BASED: OVERMA: ON INDEX RELEVANT STOCKS(NOT ALL): write direclty on overma_name column
    use Overma to define buy and sell signals"""
    overma_name = step4(df_result=df_result, d_preload=d_preload, a_freq_close=[60, 120], a_freq_overma=[60, 120], index=index)

    """3. EVENT BASED: ROLLING NORM: ON INDEX RELEVANT STOCKS(NOT ALL): writes on _trendmode_pquota_fol column
    to see if all stocks are too high or not to generate trade signals
    """
    a_ts_code = a_ts_code_helper(index=index)
    a_ts_code = list(a_ts_code.index)

    df_result[f"{index}_trendmode_pquota_fol"] = 0.0
    df_result[f"{index}_trendmode_pquota_fol_counter"] = 0

    for ts_code, df_asset in d_preload.items():
        if ts_code in a_ts_code:
            kwargs = {"df": df_asset, "abase": "close", "inplace": True, "freq_obj": [60, 120]}  # 80,100,120
            func = Alpha.fol_rolling_norm
            fol_name = Alpha.alpha_name(kwargs, func.__name__)
            if fol_name in df_asset.columns:
                print(f"{index} predict_trendmode ALREADY HAS fol ", ts_code)
            else:
                print(f"{index} predict_trendmode calculate fol ", ts_code)
                fol_name = Alpha.fol_rolling_norm(**kwargs)

            df_asset["counter_helper"] = 1
            df_result[f"{index}_trendmode_pquota_fol"] = df_result[f"{index}_trendmode_pquota_fol"].add(df_asset[fol_name], fill_value=0)
            df_result[f"{index}_trendmode_pquota_fol_counter"] = df_result[f"{index}_trendmode_pquota_fol_counter"].add(df_asset["counter_helper"], fill_value=0)

    df_result[f"{index}_trendmode_pquota_fol"] = df_result[f"{index}_trendmode_pquota_fol"] / df_result[f"{index}_trendmode_pquota_fol_counter"]
    df_result[f"{index}_trendmode_pquota_fol"] = Alpha.norm(df=df_result, abase=f"{index}_trendmode_pquota_fol", inplace=False, min=-1, max=1)  # normalize to range [-1:1]
    del df_result[f"{index}_trendmode_pquota_fol_counter"]

    """4 VOLUME on trende mode does not give any signals, so leave it out for now
    """

    """5 PUT ALL TOGEHTER"""
    df_result[f"{index}_trendmode_pquota"] = df_result[f"{index}_trendmode_pquota"].add(df_result[overma_name] * 0.15, fill_value=0)
    df_result[f"{index}_trendmode_pquota"] = df_result[f"{index}_trendmode_pquota"].subtract(df_result[f"{index}_trendmode_pquota_fol"] * 0.15, fill_value=0)

    # special treatment for crazy time, portfolio to be 1
    df_result.loc[(df_result["sh_volatility"] > 0.35) & (df_result["cy_trend"] == 1), f"{index}_trendmode_pquota"] = 1

    # special treatnebt for crazy time, if the price has fall off more than 12% from local max, then start sell
    df_result["rolling_max_60"] = df_result[f"close_{index}"].rolling(60).max()
    df_result["off_rolling_max_60"] = df_result[f"close_{index}"] / df_result["rolling_max_60"]
    df_result.loc[(df_result["sh_volatility"] > 0.35) & (df_result["cy_trend"] == 1) & (df_result["off_rolling_max_60"] < 0.88), f"{index}_trendmode_pquota"] = 0

    # remove waste columns
    del df_result[overma_name]
    del df_result[f"{index}_trendmode_pquota_fol"]
    del df_result["rolling_max_60"]
    del df_result["off_rolling_max_60"]

    # cut final result to be between 0 and 1
    df_result[f"{index}_trendmode_pquota"].clip(0, 1, inplace=True)


def predict_cyclemode(df_result, d_preload, debug=0, index="sh"):
    def normaltime_signal(df_result, index="sh"):
        divideby = 0
        for counter in range(-50, 50):
            if counter not in [0]:  # because thats macd
                if f"{index}_r{counter}:buy_sell" in df_result.columns:
                    df_result[f"{index}_r:buy_sell"] = df_result[f"{index}_r:buy_sell"].add(df_result[f"{index}_r{counter}:buy_sell"], fill_value=0)
                    divideby += 1
        df_result[f"{index}_r:buy_sell"] = df_result[f"{index}_r:buy_sell"] / divideby

    def alltime_signal(df_result, index="sh"):
        # when switching between normal time strategy and crazy time strategy, there is no way to gradually switch. You either choose one or the other because crazy time is very volatile. In this time. I choose macd for crazy time.

        df_result[f"{index}_ra:buy_sell"] = 0.0
        for divideby, thresh in enumerate([0.35]):
            df_result[f"{index}_ra:buy_sell{thresh}"] = 0.0
            df_result.loc[df_result["sh_volatility"] <= thresh, f"{index}_ra:buy_sell{thresh}"] = df_result[f"{index}_r:buy_sell"]  # normal time
            df_result.loc[df_result["sh_volatility"] > thresh, f"{index}_ra:buy_sell{thresh}"] = df_result[f"{index}_r0:buy_sell"]  # crazy time
            df_result[f"{index}_ra:buy_sell"] += df_result[f"{index}_ra:buy_sell{thresh}"]
            del df_result[f"{index}_ra:buy_sell{thresh}"]
        df_result[f"{index}_ra:buy_sell"] = df_result[f"{index}_ra:buy_sell"] / (divideby + 1)

    """
    APPROACH: divide and conquer: choose most simple case for all variables
    1. Long period instead of short period
    2. Group of stocks(index, industry ETF) instead of individual stocks
    3. Only Buy Signals or Sell signals instead of both
    4. Overlay technique: Multiple freq results instead of one. If there is a variable like freq or threshhold, instead of using one, use ALL of them and then combine them into one result


    STEPS:
    3. (todo industry) check if index volume is consistent with past gain for index
    4. (todo idustry) calculate how many stocks are overma
    5. (done) check if top n stocks are doing well or not
    9. (todo currently no way access) calculate how many institution are holding stocks  http://datapc.eastmoney.com/emdatacenter/JGCC/Index?color=w&type=
    11. (todo finished partly) Price and vol volatility
    12. (todo with all stocks and industry) check volatiliy adj macd
    
    not working well 
    17. Fundamental values not to predict shorterm market. Only useable to select stocks




    RESULT INTERPRETATION:
    1. magnitude is 1 to -1. The magnitude represents chance/likelihood/confidence that the market will rise or fall in future.
    2. the magnitude can be mapped to portfolio size. The higher the chance, the more money to bet in.
    3. the results mostly shows BEST buy can sell moments. If a buy moment goes back to 0 couple days after a 0.3 buy moment signal, this means the best buy moment is not anymore.
    4. the moments where signal shows 0, there are many ways to interpret:
        -last buy_sell signal counts.
        -don't do anything
        -take a smaller freq result
    5. One good way to interpret is buy when signal shows above 0.2., buy 20% portfolio. Hold until signal shows -0.2 or more and sell Everything.
    6. Whenever sell occurs, always sell everthing. In crazy time, this must happen very quickly. In normal times, the signal is not that accurate, you can have a bit more time to sell at once or bit by bit.
    7. Whenever buy occurs, the amplitude means the confidence. This can be 1:1 mapped to portfolio size.
    8. Future bull market can be detected if signals show 0.2 or more. Future bear market is vice versa. The bull or bear market goes as long as the opposing signal appears. Then the trend switches


    KNOWLEDGE SUMMARY - normal vs crazy time:
    occurence: crazy time only occurs 10% of the time. 3 years in 30 years.
    volatility: normal time low volatiliy. Crazy time high volatility.
    strategy: normal time buy low, sell high = against trend. crazy time you buy with trend.
    freq: normal time use 60-240 as freq. crazy time use 20-60 as freq.
    MACD crazy time: MACD good on crazy time because it buys with trend AND it is able to detect trend swap very good.
    MACD normal time: bad because there is not significant trend to detect. too much whipsaw. AND you should buy against the trend.
    Turning point: normal time anticipate BUT with tiny confirmation. Crazy time wait for turning point, also with Tiny confirmation.
    Volume: crazy time high, normal time low
    overma: crazy time high, normal time low
    => in crazy time, use different strategy and higher freq
    => if UP trend is SURE, then buy low sell high. if trend is not sure, wait for confirmation to buy
    
    The mistake in my previous research was to seek and define bull and bear market. When in reality. One must first define crazy and normal time.

    PORTFOLIO:
    Crazy and normal time can both have 100% portfolio. You can not choose how market gives 60% return or 600%. You can only choose your portfolio size, buy or not buy. Don't miss even if market returns 20%. 
    => The final signal is craziness adjusted portfolio size. This means that crazy time and normal time signals CAN BE COMPARED AGAINST. They are both on scala -1 to 1 to make comparison consistent.
    => This also makes portfolio decisions easier. You can directly convert the confidence into portfolio size.
    => sh stocks most time in cycle mode. cy Stocks most time in trend mode.

    DEBUG:
    level 0 = only show combined final result like r4:buy
    level 1 = level 0 + index based like r4:sh_buy
    level 2 = level 1 + different freq like r4:sh_close120.buy
    level 3 = level 2 + junk
    
    
    Design TODO
    MANUAL CORRECTION
    combination of two theories
    a pquote tester that varifies the result
    tweak accuracy
    strength of a trend, to check how strong the turning point must be to turn over the trend
    variable frequency. At high volatile time, use smaller freq
    find a better and more reliable time classifier to replace the hard coded sh version
    idea to use cummulative signals during normal time. e.g. signals from last 5 to 10 days together combined show me how strong sell should be instead of one single.
    add the idea that in bear market, holing period is short and in bull, holding period is long
    
    Technical TODO
    naming conventions
    manage efficiency of the code, less redundant
    maybe interface to see more clear the final result
    """

    # START PREDICT
    print(f"START PREDICT ---> {index} <---")
    print()

    df_result[f"{index}_r:buy_sell"] = 0.0
    df_result[f"{index}_ra:buy_sell"] = 0.0

    # 0 MACD  (on single index) CRAZY
    step0(df_result=df_result, index=index, debug=debug)

    # 3 VOLUME (on single index) NORMAL
    step3(df_result=df_result, index=index, debug=debug)

    # 4 OVERMA (on all stocks) NORMAL
    step4(df_result=df_result, index=index, d_preload=d_preload, debug=debug)

    # Combine NORMAL TIME buy and sell signal into one.
    normaltime_signal(df_result, index=index)

    # Add CRAZY TIME signal into the normal time signal = > all time signal.
    alltime_signal(df_result, index=index)

    # OPTIONAL: smooth the result to have less whipsaw
    # df_result["ra:buy_sell"]=Alpha.zlema(df=df_result, abase="ra:buy_sell", freq=5, inplace=False ,gain=0)

    # portfolio strategies
    to_cyclemode_pquota(df_result=df_result, abase=f"{index}_ra:buy_sell", index=index)

    # check only after year 2000
    df_result = LB.trade_date_to_calender(df=df_result, add=["year"])
    df_result = df_result[df_result["year"] >= 2000]

    # remove waste columns
    del df_result["year"]
    del df_result[f"{index}_r:buy_sell"]
    del df_result[f"{index}_r0:buy_sell"]
    del df_result[f"{index}_r3:buy_sell"]
    del df_result[f"{index}_r4:buy_sell"]

    # return is needed because a new df_result has been created
    return df_result


def to_cyclemode_pquota(df_result, abase, index="sh"):
    """
    This portfolio strategy is simple: buy when > 0.2. Sell when <0.2
    buy until sell signal occurs


    """
    df_result[f"{index}_cyclemode_pquota"] = 0.0
    portfolio = 0.0

    for trade_date in df_result.index:
        # loop over each day
        signal = df_result.at[trade_date, abase]
        if signal > 0:
            portfolio = builtins.max(portfolio, signal)
        elif signal < 0:
            portfolio = 0.0  # reset portfolio to 0
        elif signal == 0:
            # variation 1: no nothing and use previous high as guideline
            # variation 2: interpret it as sell signal if previous signal was buy. interpret as buy if previous signal was sell.
            # variation 3: use a low freq strategy to take a deeper look into it
            pass

        # assign value at end of day
        df_result.at[trade_date, f"{index}_cyclemode_pquota"] = portfolio


def step0(df_result, debug=0, index="sh"):
    """MACD"""

    result_name = f"{index}_r0:buy_sell"

    # create all macd
    a_results_col = []
    for sfreq in [60, 120, 180, 240]:
        for bfreq in [180, 240, 300, 360]:
            if sfreq < bfreq:
                print(f"{index}: step0 sfreq{sfreq} bfreq{bfreq}")
                a_cols = macd(df=df_result, abase=f"close_{index}", freq=sfreq, freq2=bfreq, inplace=True, type=4, score=1)
                a_results_col += [a_cols[0]]

                # delete unessesary columns such as macd dea, diff
                if debug < 2:
                    for counter in range(1, len(a_cols)):  # start from 1 because 0 is reserved for result col
                        del df_result[a_cols[counter]]

    # add all macd results together
    df_result[result_name] = 0.0
    for counter, result_col in enumerate(a_results_col):
        df_result[result_name] = df_result[result_name].add(df_result[result_col], fill_value=0)
        if debug < 2:
            del df_result[result_col]

    # normalize
    df_result[result_name] = df_result[result_name] / (counter + 1)

    # calculate overlay freq volatility: adjust the result with volatility (because macd works best on high volatile time)
    # df_result["r0:buy_sell"] = df_result["r0:buy_sell"] * df_result["volatility"]


def step3(df_result, index="sh", debug=0):
    """volume

    volume is best used to predict start of crazy time. in normal time, there is not so much information in volume.
    """

    def step3_single(df_result, index, freq_close=240, freq_vol=360, debug=0):
        """
        This can detect 3 signals:
        1. high volume and high gain -> likely to reverse to bear
        2. low volume and high gain -> even more likely to reverse to bear
        3. high volume and low gain -> likely to reverse to bull
        """

        vol_name = f"vol_{index}"
        close_name = f"close_{index}"
        result_name = f"r3:{index}_vol{freq_vol}_close{freq_close}"

        # normalize volume and close first with rolling 240 days
        norm_vol_name = Alpha.rollingnorm(df=df_result, abase=vol_name, freq=freq_vol, inplace=True)
        norm_close_name = Alpha.rollingnorm(df=df_result, abase=close_name, freq=freq_close, inplace=True)

        # 1. Sell Signal: filter only days where vol > 0.7 and close > 0.6
        df_helper = df_result.loc[(df_result[norm_vol_name] > 0.7) & (df_result[norm_close_name] > 0.6)]
        sell_signal1 = df_helper[norm_vol_name] + df_helper[norm_close_name]  # higher price, higher volume the more clear the signal

        # 2. Sell Signal: filter only days where vol < 0.5 and close > 0.8
        df_helper = df_result.loc[(df_result[norm_vol_name] < 0.4) & (df_result[norm_close_name] > 0.80)]
        sell_signal2 = (1 - df_helper[norm_vol_name]) + df_helper[norm_close_name]  # higher price, lower volume the more clear the signal

        # combine both type of sell signals
        df_result[f"{result_name}_sell"] = sell_signal1.add(sell_signal2, fill_value=0)

        # 3. Buy Signal: filter only days where vol > 0.6 and close < 0.4
        df_helper = df_result.loc[(df_result[norm_vol_name] > 0.7) & (df_result[norm_close_name] < 0.4)]
        buy_signal = df_helper[norm_vol_name] + (1 - df_helper[norm_close_name])  # higher volume, lower price the more clear the signal
        df_result[f"{result_name}_buy"] = buy_signal

        # 4. Delete unessesary columns produced
        if debug < 3:
            del df_result[norm_vol_name]
            del df_result[norm_close_name]

        return [f"{result_name}_buy", f"{result_name}_sell"]

    result_name = f"{index}_r3:buy_sell"

    # loop over all frequency
    df_result[f"{index}_r3:buy"] = 0.0
    df_result[f"{index}_r3:sell"] = 0.0
    result_list = []
    counter = 0
    for freq_close in [240, 500]:
        for freq_vol in [120, 500]:
            print(f"{index}: step3 close{freq_close} vol{freq_vol}...")
            counter += 1
            buy_sell_label = step3_single(df_result=df_result, freq_close=freq_close, freq_vol=freq_vol, index=index, debug=debug)
            result_list = result_list + [buy_sell_label]

    # combine all frequecies into one result for ONE index
    for buy_freq_signal, sell_freq_signal in result_list:
        df_result[f"{index}_r3:buy"] = df_result[f"{index}_r3:buy"].add(df_result[buy_freq_signal], fill_value=0)
        df_result[f"{index}_r3:sell"] = df_result[f"{index}_r3:sell"].add(df_result[sell_freq_signal], fill_value=0)
        if debug < 2:
            del df_result[buy_freq_signal]
            del df_result[sell_freq_signal]

    # normalize the result
    df_result[f"{index}_r3:buy"] = df_result[f"{index}_r3:buy"] / (counter * 2)
    df_result[f"{index}_r3:sell"] = df_result[f"{index}_r3:sell"] / (counter * 2)

    # combine buy and sell
    df_result[result_name] = df_result[f"{index}_r3:buy"].add(df_result[f"{index}_r3:sell"] * (-1), fill_value=0)

    if debug < 3:
        del df_result[f"{index}_r3:buy"]
        del df_result[f"{index}_r3:sell"]
    return


def step4(df_result, d_preload, index="sh", a_ts_code=[], a_freq_close=[240, 500], a_freq_overma=[120, 500], debug=0, ):
    """Overma"""

    def step4_single(df_result, d_preload, a_ts_code, freq_close=240, freq_overma=240, index="sh", debug=0):
        """calculate how many stocks are overma generally very useful

        for period in [500,240,120]:
            1. General overma
            2. Index overma
            3. Industry  overma
            4. Size overma
        """

        # 1. General ALL STOCK overma
        # 1.1 normalize overma series
        if f"{index}_overma{freq_overma}" not in df_result.columns:
            df_result[f"{index}_overma{freq_overma}"] = 0.0
            df_result[f"{index}_counter{freq_overma}"] = 0.0

            for ts_code, df_asset in d_preload.items():
                if ts_code in a_ts_code:
                    # calculate if stocks is over its ma
                    df_asset[f"{index}_ma{freq_overma}"] = df_asset["close"].rolling(freq_overma).mean()
                    df_asset[f"{index}_overma{freq_overma}"] = (df_asset["close"] >= df_asset[f"{index}_ma{freq_overma}"]).astype(int)
                    df_asset[f"{index}_counter{freq_overma}"] = 1

                    df_result[f"{index}_overma{freq_overma}"] = df_result[f"{index}_overma{freq_overma}"].add(df_asset[f"{index}_overma{freq_overma}"], fill_value=0)
                    # counter to see how many stocks are available
                    df_result[f"{index}_counter{freq_overma}"] = df_result[f"{index}_counter{freq_overma}"].add(df_asset[f"{index}_counter{freq_overma}"], fill_value=0)

            # finally: calculate the percentage of stocks overma
            df_result[f"{index}_overma{freq_overma}"] = df_result[f"{index}_overma{freq_overma}"] / df_result[f"{index}_counter{freq_overma}"]

        # 1.2 normalize close series
        norm_close_name = Alpha.rollingnorm(df=df_result, freq=freq_close, abase=f"close_{index}", inplace=True)

        # 1.3 generate  Buy Signal: price < 0.25 and overma < 0.25
        df_helper = df_result.loc[(df_result[f"{index}_overma{freq_overma}"] < 0.25) & (df_result[norm_close_name] < 0.25)]
        df_result[f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_buy"] = (1 - df_helper[f"{index}_overma{freq_overma}"]) + (1 - df_helper[norm_close_name])  # the lower the price, the lower overma, the better

        # 1.4 generate  Sell Signal: price > 0.75 and overma > 0.75
        df_helper = df_result.loc[(df_result[f"{index}_overma{freq_overma}"] > 0.75) & (df_result[norm_close_name] > 0.75)]
        df_result[f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_sell"] = df_helper[f"{index}_overma{freq_overma}"] + df_helper[norm_close_name]  # the lower the price, the lower overma, the better

        # 1.5 delete unessary columns
        if debug < 3:
            del df_result[f"{index}_overma{freq_overma}"]
            del df_result[f"{index}_counter{freq_overma}"]
            del df_result[norm_close_name]

        return [f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_buy", f"{index}_r4:overma{freq_overma}_close{freq_close}_{index}_sell"]

    # generate matching list of ts_code for index to be used for overma later
    a_ts_code = a_ts_code_helper(index=index)
    a_ts_code = list(a_ts_code.index)

    df_result[f"{index}_r4:buy"] = 0.0
    df_result[f"{index}_r4:sell"] = 0.0

    # loop over all frequency
    result_list = []
    counter = 0
    for freq_close in a_freq_close:
        for freq_overma in a_freq_overma:
            print(f"{index}: step4 close{freq_close} overma{freq_overma}...")
            buy_sell_label = step4_single(df_result=df_result, d_preload=d_preload, a_ts_code=a_ts_code, freq_close=freq_close, freq_overma=freq_overma, index=index, debug=debug)
            result_list = result_list + [buy_sell_label]
            counter += 1

    # combine all frequecies into one result for ONE index
    for buy_signal, sell_signal in result_list:
        df_result[f"{index}_r4:buy"] = df_result[f"{index}_r4:buy"].add(df_result[buy_signal], fill_value=0)
        df_result[f"{index}_r4:sell"] = df_result[f"{index}_r4:sell"].add(df_result[sell_signal], fill_value=0)
        if debug < 2:
            del df_result[buy_signal]
            del df_result[sell_signal]

    # normalize the result
    df_result[f"{index}_r4:buy"] = df_result[f"{index}_r4:buy"] / (counter * 2)  # why times 2 actually
    df_result[f"{index}_r4:sell"] = df_result[f"{index}_r4:sell"] / (counter * 2)

    # combine buy and sell
    df_result[f"{index}_r4:buy_sell"] = df_result[f"{index}_r4:buy"].add(df_result[f"{index}_r4:sell"] * (-1), fill_value=0)

    # debug
    if debug < 3:
        del df_result[f"{index}_r4:buy"]
        del df_result[f"{index}_r4:sell"]

    return f"{index}_r4:buy_sell"


def step5(df_result, d_preload, debug=0):
    """check if top n stocks (low beta stocks stocks) are doing well or not
    If even they are bad, then the whole stock market is just bad for sure

    algorith:
    1. define top n stocks using fundamentals and technicals
    2. check if they are doing well in last freq D: 5, 20, 60

    （1. cheat, use shortage to manually define these 50 stocks)
    """

    def step5_single(df_result, debug=0):
        # 2. Generate step 5 buy sell signal using custom defined rules
        # works worse than v2 with macd
        r5_freq_buy_result = []
        r5_freq_sell_result = []
        df_result["r5:buy"] = 0.0
        df_result["r5:sell"] = 0.0  # step5 does not produce any sell signal

        for freq in [120, 240, 500]:
            print(f"all: step5 close{freq}...")
            # rolling norm
            topn_close_name = Alpha.rollingnorm(df=df_result, freq=freq, abase="r5:topn_index", inplace=True)

            # is max
            df_result["topn_emax"] = df_result["r5:topn_index"].expanding().max()
            is_top_pct = Alpha.ismax(df=df_result, abase="r5:topn_index", emax="topn_emax", inplace=True, q=0.85, score=1)

            # 2.1 Buy if past normalized return is < 0.2
            df_helper = df_result.loc[(df_result[topn_close_name] < 0.20)]
            df_result[f"r5:topn_close{freq}_buy"] = 1 - df_helper[topn_close_name]
            r5_freq_buy_result += [f"r5:topn_close{freq}_buy"]

            # 2.2 Sell if they are not at top 15% and there is no buy signal = bear but not bear enough
            df_helper = df_result[(df_result[is_top_pct] == -1) & (df_result[f"r5:topn_close{freq}_buy"].isna())]
            df_helper["sell_helper"] = 1
            df_result[f"r5:topn_close{freq}_sell"] = df_helper["sell_helper"]
            r5_freq_sell_result += [f"r5:topn_close{freq}_sell"]

            if debug < 2:
                del df_result[topn_close_name]
                del df_result["topn_emax"]
                del df_result[is_top_pct]

        # combine all freq into one
        counter = 0
        for freq_result in r5_freq_buy_result:
            df_result["r5:buy"] = df_result["r5:buy"].add(df_result[freq_result], fill_value=0)
            counter += 1
            if debug < 1: del df_result[freq_result]

        counter = 0
        for freq_result in r5_freq_sell_result:
            df_result["r5:sell"] = df_result["r5:sell"].add(df_result[freq_result], fill_value=0)
            counter += 1
            if debug < 1: del df_result[freq_result]

        df_result["r5:buy"] = df_result["r5:buy"] / counter
        df_result["r5:sell"] = df_result["r5:sell"] / counter

        # for now exclude sell result
        # df_result["r5:sell"] = 0.0

        # combine buy and sell
        df_result["r5:buy_sell"] = df_result[f"r5:buy"].add(df_result[f"r5:sell"] * (-1), fill_value=0)

        # adjust with volatility

        if debug < 2:
            del df_result["r5:buy"]
            del df_result["r5:sell"]
            # del df_result["r5:topn_index"]
        return

    def step5_single_v2(df_result, debug=0):
        # 2. Generate step 5 buy sell signal using macd. Because MACD buys on uptrend, sell on downtrend. goes very well with good stocks that are uptrend most of the time.

        # create all macd
        a_results_col = []
        for sfreq in [120, 180, 240]:
            for bfreq in [180, 240, 300, 360, 500]:
                if sfreq < bfreq:
                    print(f"all: step5 sfreq{sfreq} bfreq{bfreq}")
                    a_cols = macd(df=df_result, abase=f"r5:topn_index", freq=sfreq, freq2=bfreq, inplace=True, type=4, score=1)
                    a_results_col += [a_cols[0]]
                    if debug < 2:
                        for counter in range(1, len(a_cols)):
                            del df_result[a_cols[counter]]

        # add all macd results together
        df_result["r5:buy_sell"] = 0.0
        for counter, result_col in enumerate(a_results_col):
            df_result["r5:buy_sell"] = df_result["r5:buy_sell"].add(df_result[result_col], fill_value=0)
            if debug < 2:
                del df_result[result_col]

        # normalize
        df_result["r5:buy_sell"] = df_result["r5:buy_sell"] / counter

        # adjust with sh_index volatility
        df_result["r5:buy_sell"] = df_result["r5:buy_sell"] * df_result["sh_volatility"]

        return

    # 1. Generate top n index
    """
    贵州茅台
    泸州老窖
    伊利股份
    招商银行
    海螺水泥
    恒瑞医药
    云南白药
    苏泊尔
    格力电器
    """

    a_ts_codes = ["600519.SH", "000568.SZ", "600887.SH", "600036.SH", "600585.SH", "600272.SH", "000538.SZ", "002032.SZ", "000651.SZ"]

    df_result["step5_counter"] = 0.0
    df_result["step5_topn_pct_chg"] = 0.0

    for ts_code, df_asset in d_preload.items():
        if ts_code in a_ts_codes:
            # add counter together
            df_asset["step5_counter"] = 1
            df_result["step5_counter"] = df_result["step5_counter"].add(df_asset["step5_counter"], fill_value=0)

            # add gain together
            df_result["step5_topn_pct_chg"] = df_result["step5_topn_pct_chg"].add(df_asset["pct_chg"], fill_value=0)

    df_result["step5_topn_pct_chg"] = df_result["step5_topn_pct_chg"] / df_result["step5_counter"]
    df_result["r5:topn_index"] = Alpha.comp_chg(df=df_result, abase="step5_topn_pct_chg", inplace=False, start=100)

    if debug < 2:
        del df_result["step5_counter"]
        del df_result["step5_topn_pct_chg"]

    step5_single_v2(df_result=df_result, debug=debug)


def cy_mode(df_result, abase="close"):
    """
    this function detects in what mode/phase the cy stock is
    """

    # add all freq of rolling norm together
    df_result["fol_close_norm"] = 0.0
    a_del_cols = []
    counter = 0
    for freq in range(10, 510, 10):
        print(f"freq is {freq}")
        name = Alpha.rollingnorm(df=df_result, abase=abase, freq=freq, inplace=True)
        df_result["fol_close_norm"] = df_result["fol_close_norm"] + df_result[name]
        counter += 1
        a_del_cols += [name]
    df_result["fol_close_norm"] = df_result["fol_close_norm"] / counter

    # produce bull or bear market. 1 means bull, -1 means bear.
    bull_bear = 0.0
    for trade_date in df_result.index:

        # loop over each day
        signal = df_result.at[trade_date, "fol_close_norm"]
        if signal > 0.8:  # bull
            bull_bear = 1
        elif signal < 0.2:
            bull_bear = -1  # reset portfolio to 0
        else:
            # variation 1: no nothing and use previous high as guideline
            # variation 2: interpret it as sell signal if previous signal was buy. interpret as buy if previous signal was sell.
            # variation 3: use a low freq strategy to take a deeper look into it
            pass

        # assign value at end of day
        df_result.at[trade_date, "bull_bear"] = bull_bear

    df_result.drop(a_del_cols, axis=1, inplace=True)
    # df_result.to_csv("egal.csv")


def all(withupdate=False):
    """
    Goal: Predict market, generate concrete detailed buy and sell signals
    1. When to buy/sell/: If market is good => predict macro market bull or bear
    2. How much to buy/sell: If market is micro market overbought or underbought
    3. What to buy/sell: Check stocks, etfs, industries, concepts


    todo:
    interface
    industry PE
    portfolio
    validation
    overall industry volume created by individual stock using sh_cyclemode_method or cy_trendmode
    refactoring
    improve FUN_Valuation

    PE vs FUN and TEC RANK. how to discount pe and FUN AND TECHNICAL RANK
    """

    # 0: UPDATE DATA and INIT
    if withupdate: DB.update_all_in_one_cn_v2(night_shift=True)
    df_result = DB.get_stock_market_overview()
    df_result["sh_volatility"] = Alpha.detect_cycle(df=df_result, abase=f"close_sh", inplace=False)  # use sh index to calculate volatility no matter what
    df_result["cy_trend"] = Alpha.detect_bull(df=df_result, abase=f"close_cy", inplace=False)  # use cy index to detect macro trends
    df_result["market_trend"] = 0.0  # reserved for usage later
    df_result_copy = df_result.copy()
    d_preload_E = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")
    #d_preload_FD = DB.preload(asset="FD", freq="D", on_asset=True, step=1, market="CN")


    #0.1 ADD PE, PB of all stock view
    pe_overview(df_result=df_result,d_preload=d_preload_E)

    return

    # 1. STOCK LONG TERM SCORE
    if True:
        for trade_date in df_result.index[::240]:
            #create stock rank for a given date
            d_preload_E_until = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN", query_df=f"trade_date <= {trade_date}")
            df_long_term=predict_stock_long( d_preload=d_preload_E_until, end_date=trade_date)

            #create portfolio strategy for a given date
            create_portfolio(df_result=df_result,df_stock_long_term=df_long_term, end_date=trade_date)

    return


    # 2. STOCK SHORT TERM
    # 2.1: first predict market
    if False:
        predict_trendmode(df_result=df_result, index="cy", d_preload=d_preload_E)
        df_result=predict_cyclemode(df_result=df_result, index="sh", d_preload=d_preload_E)
        df_result["market_pquota"]=df_result["cy_trendmode_pquota"]+df_result["sh_cyclemode_pquota"]

        #delete volume as they will not be used later
        for cn_market in ["sh","sz","cy"]:
            if f"vol_{cn_market}"in df_result.columns:
                del df_result[f"vol_{cn_market}"]
        a_path = LB.a_path(f"Market/CN/PredictMarket/Predict_Market")
        LB.to_csv_feather(df=df_result, a_path=a_path)
        df_result = df_result.set_index("index")


    # 2. 2: second predict industry
    if False:
        df_final_industry_rank = predict_industry(df_result_copy=df_result_copy, d_preload=d_preload_E)
        a_path = LB.a_path(f"Market/CN/PredictMarket/Industry_Score")
        LB.to_csv_feather(df=df_final_industry_rank, a_path=a_path)
    else:
        df_final_industry_rank = pd.read_csv("Market/CN/PredictMarket/Industry_Score.csv")
    df_final_industry_rank = df_final_industry_rank.set_index("index")


    # 2. 3: third calc stock short term
    predict_stock_short(df_result=df_result,d_preload=d_preload_E)


    # 4: validate PREDICTION
    if False:
        df_validate = validate(df_result=df_result, index="cy")
        a_path = LB.a_path(f"Market/CN/PredictMarket/Validate")
        LB.to_csv_feather(df=df_validate, a_path=a_path)






def _deprecated_seasonal_effect(df_result, debug=0):
    """
    currently no use of seasonal effect because they are too periodic.
    Seasonal effect are interesting, but deviation are too big.
    Hence it makes the stats useless

    1. overlay of chinese month of year effect
    2. overlay of first month prediction effect
    3. overlay of day of month effect
    """
    # PART 1
    # init
    df_trade_date = DB.get_trade_date()

    df_result["year"] = df_trade_date["year"]
    df_result["month"] = df_trade_date["month"]
    df_result["day"] = df_trade_date["day"]
    df_result["weekofyear"] = df_trade_date["weekofyear"]
    df_result["dayofweek"] = df_trade_date["dayofweek"]
    df_result["r8:buy_sell"] = 0.0

    # overlay of all divisions are NOT IN USE
    for division in ["month", "weekofyear"]:
        # overlay of seasonal effect
        df_division = DB.get(a_path=LB.a_path(f"Market/CN/ATest/seasonal_stock/{division}"), set_index=division)
        df_result[division] = df_result[division].astype(int)
        df_result[division] = df_result[division].replace(df_division["pct_chg"].to_dict())
        df_result[division] = df_result[division].astype(float)
        # df_result["r8:buy_sell"]+=df_result[division]

    # PART 2
    df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sh = LB.trade_date_to_calender(df_sh)
    # overlay of chinese new year effect(compare ny gain against others. If strong then the whole year is strong)
    # in order to give a more real prediction, we conduct the prediction step by step from the past

    df_sh_helper = df_sh[df_sh["month"] == 2]
    df_result = df_sh_helper.groupby("year").mean()
    df_result.to_csv("test.csv")
    # todo unfinished because I feel it will not be better than other existing signals
    # overlay of first month (compare first month gain against others. If strong then the whole year is strong)

    # overlay first and last week of year


def _deprecated_rsi(df_result, debug=0):
    """
    rsi freq: this step is to check if different freq combination of rsi would make a better rsi signal
    """

    df_result["r10:buy"] = 0
    df_result["r10:sell"] = 0
    for counter, freq in enumerate([20, 40, 60, 80, 100, 120, 180, 240, 300, 360]):
        rsi_name = Alpha.rsi(df=df_result, abase="close_sh", freq=freq, inplace=True)

        # create buy signal
        df_helper = df_result.loc[(df_result[rsi_name] < 50)]
        df_result[f"r10:close_sh{freq}_buy"] = df_helper[rsi_name]

        # create sell signal
        df_helper = df_result.loc[(df_result[rsi_name] > 50)]
        df_result[f"r10:close_sh{freq}_sell"] = df_helper[rsi_name]

        df_result["r10:buy"] = df_result["r10:buy"].add(df_result[f"r10:close_sh{freq}_buy"], fill_value=0)
        df_result["r10:sell"] = df_result["r10:sell"].add(df_result[f"r10:close_sh{freq}_sell"], fill_value=0)

        if debug < 1:
            del df_result[rsi_name]

    df_result["r10:buy"] = df_result["r10:buy"] / (counter + 1)
    df_result["r10:sell"] = df_result["r10:sell"] / (counter + 1)


def _deprecated_volatility(df_result, debug=0, index="sh"):
    """
    Different kind of volatility: against past, intraday, against other stock

    1. check time with volatility against itself in the past = rolling
    2. check time with volatility against others now
    3. check time with intraday volatility
    4. check time with low volatility and uptrend (This does not exist in A Stock)

    method 1: 1. calculate price_std with freq overlay. 2. calculate together with close rolling norm
    method 2: 1. calculate price_std and rolling norm overlay together in one step
    Note: this method tried using BOTH method and the result is okish, all signals have almost the same threshhold which is bad. Therefore I conclude that this method is not that much useful.
    Note: all other steps like 3,4,5 are using method 2 and got good result
    """
    # 1. Check volatility AGAINST PAST
    # 1.1 check time with PRICE volatility AGAINST PAST
    # result -> can predict crazy and normal time

    # normalize price
    df_result[f"r11:buy"] = 0.0
    df_result[f"r11:sell"] = 0.0
    divideby = 1
    for freq in [120, 240, 500]:
        print(f"step11 {index} close{freq}", )
        # normalize close
        norm_close_name = Alpha.rollingnorm(df=df_result, abase=f"close_{index}", freq=freq, inplace=True)

        # calcualte close std
        df_result[f"close_std{freq}"] = df_result[f"close_{index}"].rolling(freq).std()

        # normalize result(dirty way, should not be like that because it knows future)
        norm_close_std_name = Alpha.norm(df=df_result, abase=f"close_std{freq}", inplace=True)

        # generate buy signal: volatiliy is low and past price is low = Buy
        # volatility < 0.2, past gain < 0.4: buy. indicates turning point
        df_helper = df_result[(df_result[norm_close_name] < 0.3) & (df_result[norm_close_std_name] < 0.2)]
        df_result[f"r11:norm{freq}_buy"] = (1 - df_helper[norm_close_name]) + (1 - df_helper[norm_close_std_name])
        df_result[f"r11:buy"] = df_result[f"r11:buy"].add(df_result[f"r11:norm{freq}_buy"], fill_value=0)

        # generate Sell signal: volatiliy is low and past price is high = Sell
        # volatility < 0.2, past gain > 0.8: buy. indicates turning point
        df_helper = df_result[(df_result[norm_close_name] > 0.7) & (df_result[norm_close_std_name] < 0.2)]
        df_result[f"r11:norm{freq}_sell"] = (df_helper[norm_close_name]) + (1 - df_helper[norm_close_std_name])
        df_result[f"r11:sell"] = df_result[f"r11:sell"].add(df_result[f"r11:norm{freq}_sell"], fill_value=0)

        # increment divideby
        divideby += 1

        # debug
        if debug < 2:
            del df_result[norm_close_name]
            del df_result[f"close_std{freq}"]
            del df_result[norm_close_std_name]
            del df_result[f"r11:norm{freq}_buy"]
            del df_result[f"r11:norm{freq}_sell"]

    # normalize
    df_result[f"r11:buy"] = df_result[f"r11:buy"] / (divideby * 2)
    df_result[f"r11:sell"] = df_result[f"r11:sell"] / (divideby * 2)

    # generate sell signal: volatiliy is low and past price is high = Sell

    # 1.2 check time with VOL volatility AGAINST PAST
    # Result: ok but not as good as price std (because volume is not mean normalized?)
    """
    func_vol_partial = partial(func, abase="vol_sh")
    LB.frequency_ovelay(df=df_result, func=func_vol_partial, a_freqs=[[20, 40, 60, 120, 240]], a_names=["sh_vol_std", "vol_sh"], debug=debug)
    df_result.loc[df_result["year"] < 2000, "sh_vol_std"] = 0.0
    """

    # 2. Volatility against other stock

    # 3. Volatility intraday


def _deprecated_support_resistance():
    """
    two things  to be done with support and resistance:
    1. calculate actual support and resistance
    A: This can be looked on

    2. produce signals using minmax. IF current low is higher than last low and current high is higher than last high.
    A: Problem is that this method is slower than macd freq overlay. IT also does not support freq overlay that good. In addition, it produces more whipsaw. Also, baidu index data start from 2010, google trend does not work for CN stocks. In general. data is hard to get and same as index volume.
    A: We just use macd freq overlay instead of this.

    """


def _deprecated_search_engine():
    """
    using baidu, weibo, google trend data to predict.

    the problem :
    1. They are simultaneously occuring the stock market volume. So they don't predict in ADVANCE but at the same time. This makes them less useful as I can just watch the volume.
    2. They are sort of external signals. Some times high search volume does not mean bull but bear. And some times high search volume does not move the market at all.

    :return:
    """


def _deprecated_fundamentals():
    """
    Fundamentals
    This is a big topic, lets break it down to think what indicators could be useful

    Problem:
    1. fundamental indicators are lagging, the market knows first and bed before the market is even bull
    2. fundamental are publicit not very frequently.
    3. fundamentals can be faked
    4. Insider trades even pushes

    Use:
    1. Use fundamentals in a very very long term to see if a stock is stable or not
    2. the keyword is stability. The more stable the fundamental data, the better predictable the future in very very long run.
    3. But using the method in 2. will lose chance to many opportunistic investments
    4. use fundamentals also limits the stock to be very old. New stocks can not be viewed


    Note that we want to find the extrem values first as they are easier to find and have the most predictive power.

    1. PE
    2. PB
    3.

    :return:
    """


def _deprecated_cummulative_volume_vs_sh():
    """
    create a sumarized time series to see the overall vol of the complete market

    RESULT: The summarized volume is Exactly same as the sh index volume

    """

    # load in all the stocks
    d_preload = DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")

    df_market = DB.get_asset(ts_code="000001.SH", asset="I", freq="D", market="CN")
    df_market = df_market[["close", "vol", "amount"]]
    df_market["agg_abs_amount"] = 0.0
    df_market["agg_rel_amount"] = 0.0
    df_market["agg_stocks"] = 0.0

    print("lengh of preload", len(d_preload))

    for ts_code, df_asset in d_preload.items():
        print(ts_code)

        df_asset["count_helper"] = 1.0
        df_market["agg_stocks"] = df_market["agg_stocks"].add(df_asset["count_helper"], fill_value=0)
        df_market["agg_abs_amount"] = df_market["agg_abs_amount"].add(df_asset["amount"], fill_value=0)

    df_market["agg_rel_amount"] = df_market["agg_abs_amount"] / df_market["agg_stocks"]
    a_path = LB.a_path(f"Market/CN/PredictMarket/Market")
    LB.to_csv_feather(df=df_market, a_path=a_path)


if __name__ == '__main__':
    #DB.update_all_in_one_cn_v2(until=10,night_shift=False)
    all(withupdate=False)

"""

strategy analysis:
macd: buy bull, sell bear. good at volatile time. either buy or not buy
rollingnorm: buy bull sell bear. gradiently give signals.

CY Model:
continous phase of these 4 states: like in the buddha 金刚经
-normal uptrend
-turning point (= crazy time) = crazy up and crazy down
-normal downtrend
-turning point (= silent turning point)

Most important things to find in the first place:
1. start of crazy time
2. end of crazy time

"""
