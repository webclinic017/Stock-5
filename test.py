import DB
import LB
import pandas as pd
import numpy as np
import UI


def asset_beat_index(start_date="00000000",end_date=LB.today()):
    """checks how many stock beats their index
    1. normalize all index to the day certain asset is IPOd
    2. Check if index or asset is better until today

    Amazing Result:

    53%主板beat index
    60%中小板beat index
    30%创业板beat index

    only 30% beat industry1
    only 30% beat industry2
    only 30% beat industry3

    many stock who beat the index, in the past n years, dont beat index in the next future years
    in earlier years like 2000 to 2005, 2/3 of stock who were good in the past stay good.
    now like 2015 to 2020, half of them become bad. This means it is hard now to pickup stock that stays good

    The pct% of stocks beating index,industry1,industry2,industry3 is increasing over time
    in 2000-2005 it was 7%
    in 2005-2010 it was 17%
    in 2010-2015 it was 29%
    in 2015-2020 it was 22%
    of course, this number is distored by new IPOS and crazy period timing

    TAKEAWAY:
    Stocks which perform better than index,industry1-3 might be a good stock. some of them are temporal good.
    But a good stock, always perform better than index.
    =shrinks down the pool
    =step 2: from this pool. take the best industry
    BUT in theory, past does not predict future. Past good stock does not remain good, but is only more likely to remain good.

    """

    #init
    df_ts_code=DB.get_ts_code()
    df_industry1_code=DB.get_ts_code(a_asset=[f"industry1"])
    df_industry2_code=DB.get_ts_code(a_asset=[f"industry2"])
    df_industry3_code=DB.get_ts_code(a_asset=[f"industry3"])

    #preload
    d_index=DB.preload(asset="I", step=1, d_queries_ts_code=LB.c_index_queries())
    d_e=DB.preload(step=1)
    df_result=pd.DataFrame()

    for ts_code, df_asset in d_e.items():

        print(ts_code)
        #compare against exchange
        exchange=df_ts_code.at[ts_code,"exchange"]
        if exchange=="创业板":
            compare="399006.SZ"
        elif exchange=="中小板":
            compare ="399001.SZ"
        elif exchange=="主板":
            compare="000001.SH"
        df_exchange = d_index[compare].copy()
        df_exchange=LB.df_between(df_exchange,start_date,end_date)

        #compare against industry
        try:
            industry1 = df_industry1_code.at[ts_code, "industry1"]
            industry2 = df_industry2_code.at[ts_code, "industry2"]
            industry3 = df_industry3_code.at[ts_code, "industry3"]
        except Exception as e:
            print(ts_code,"skipped",e)
            continue

        df_industry1= DB.get_asset(ts_code=f"industry1_{industry1}", asset="G")
        df_industry2= DB.get_asset(ts_code=f"industry2_{industry2}", asset="G")
        df_industry3= DB.get_asset(ts_code=f"industry3_{industry3}", asset="G")
        df_industry1 = LB.df_between(df_industry1, start_date, end_date)
        df_industry2 = LB.df_between(df_industry2, start_date, end_date)
        df_industry3 = LB.df_between(df_industry3, start_date, end_date)

        df_result.at[ts_code,"in1"]=industry1
        df_result.at[ts_code,"in2"]=industry2
        df_result.at[ts_code,"in3"]=industry3
        df_asset["gmean_norm"]=df_asset["close"].pct_change()
        df_result.at[ts_code,"gmean"]=df_asset["gmean_norm"].mean()/df_asset["gmean_norm"].std()

        #run and evaluate
        for key,df_compare in {"index":df_exchange, "industry1":df_industry1,"industry2":df_industry2,"industry3":df_industry3}.items():
            df_asset_slim=LB.ohlcpp(df_asset).reset_index()
            df_index_slim=LB.ohlcpp(df_compare).reset_index()

            df_asset_slim["trade_date"]=df_asset_slim["trade_date"].astype(int)
            df_index_slim["trade_date"]=df_index_slim["trade_date"].astype(int)

            df_slim=pd.merge(df_asset_slim,df_index_slim,on="trade_date",how="inner",suffixes=[f"_{ts_code}",f"_{compare}"],sort=False)

            if df_slim.empty:
                continue

            for code in [ts_code,compare]:
                df_slim[f"norm_{code}"]=df_slim[f"close_{code}"]/df_slim.at[0,f"close_{code}"]
                df_slim[f"norm_pct_{code}"]=df_slim[f"norm_{code}"].pct_change()
            #result=norm_ts_code/norm_compare

            df_result.at[ts_code,f"{key}_period"]=period=len(df_slim)-1
            df_result.at[ts_code,f"{key}_asset_vs_index_gain"]= df_slim.at[period, f"norm_{ts_code}"] / df_slim.at[period, f"norm_{compare}"]
            df_result.at[ts_code,f"{key}_asset_vs_index_sharp"]= (df_slim[f"norm_pct_{ts_code}"].mean() / df_slim[f"norm_pct_{ts_code}"].std()) / df_slim[f"norm_pct_{compare}"].mean() / df_slim[f"norm_pct_{compare}"].std()
            df_result.at[ts_code,f"{key}_asset_vs_index_gmean"]= (df_slim[f"norm_pct_{ts_code}"].mean() / df_slim[f"norm_pct_{ts_code}"].std())

            #TODO  beat industry, concept
            df_result.at[ts_code,"index"]=compare


    for key in ["index", "industry1", "industry2", "industry3"]:
        df_result[f"beat_{key}"]=(df_result[f"{key}_asset_vs_index_gain"]>1).astype(int)
    df_result.loc[ (df_result[f"beat_index"]==1) & (df_result[f"beat_industry1"]==1) & (df_result[f"beat_industry2"]==1) & (df_result[f"beat_industry3"]==1), "beat_all"]=1
    df_result.index.name="ts_code"

    DB.to_excel_with_static_data(df_ts_code=df_result,path=f"Market/CN/ATest/Beat_Index/result_{start_date}_{end_date}.xlsx")
    # a_path=LB.a_path("Market/CN/ATest/Beat_Index/result")
    # LB.to_csv_feather(df=df_result,a_path=a_path)

def year_beat_index():
    """
    compares index,industry1-3 vs asset on a yearly basis
    it seems that only 2007-2009, 2015 more than 50% stock beat index
    This means, small stocks are only good at crazy time
    Over all years. Only 30 to 40% beat the index
    Mostly when time is crazy, bad stock gain more, and start to beat index in that year.
    But this happens very rarely. So beating index is highly correlated with crazy time.
    The % of stocks beating index is not directly predictive as policical directions are disturbing this indicator
    """

    # init
    df_ts_code = DB.get_ts_code()
    df_industry1_code = DB.get_ts_code(a_asset=[f"industry1"])
    df_industry2_code = DB.get_ts_code(a_asset=[f"industry2"])
    df_industry3_code = DB.get_ts_code(a_asset=[f"industry3"])

    # preload index
    d_index = DB.preload(asset="I", step=1, d_queries_ts_code=LB.c_index_queries())
    d_index_y={}
    for key,df in d_index.items():
        df_year=LB.timeseries_to_year(df)
        df_year["year"]=df_year.index
        df_year["year"]=df_year["year"].apply(LB.get_trade_date_datetime_y).astype(int)
        d_index_y[key]=df_year.set_index("year")

    #preload aset
    d_e = DB.preload(step=1)
    d_e_y={}
    for ts_code, df_asset in d_e.items():
        df_asset_y=LB.timeseries_to_year(df_asset)
        df_asset_y["year"] = df_asset_y.index
        df_asset_y["year"] = df_asset_y["year"].apply(LB.get_trade_date_datetime_y).astype(int)
        d_e_y[ts_code] = df_asset_y.set_index("year")

    for ts_compare,df_compare in d_index_y.items():

        for year in df_compare.index:
            exists_counter=0
            better_than_index_counter=0
            compare_pct_chg=df_compare.at[year,"pct_chg"]

            for ts_code, df_asset in d_e_y.items():
                print(ts_compare, year,ts_code)
                if year in df_asset.index:
                    exists_counter+=1

                    if df_asset.at[year,"pct_chg"]>compare_pct_chg:
                        better_than_index_counter+=1


            try:
                df_compare.at[year,"pct_better"]=better_than_index_counter/exists_counter
            except:
                pass
        df_compare.to_csv(f"Market/CN/Atest/Beat_Index/year_result_{ts_compare}.csv")






def beat_index_evaluator():
    a_years=["20000101","20050101","20100101","20150101","20200101"]
    for end_date in a_years:
         asset_beat_index("0000101",end_date)


    # for old,new in LB.custom_pairwise_overlap(a_years):
    #     df_old=pd.read_excel(f"Market/CN/ATest/Beat_Index/result_00000000_{old}.xlsx")
    #     df_new=pd.read_excel(f"Market/CN/ATest/Beat_Index/result_00000000_{new}.xlsx")
    #     len_df_old=len(df_old)
    #     len_df_new=len(df_new)
    #     df_old=df_old[ (df_old["beat_index"]==1) & (df_old["beat_industry1"]==1) &(df_old["beat_industry2"]==1) &(df_old["beat_industry3"]==1) ]
    #     df_new=df_new[ (df_new["beat_index"]==1) & (df_new["beat_industry1"]==1) &(df_new["beat_industry2"]==1) &(df_new["beat_industry3"]==1) ]
    #
    #
    #
    #     #new stock that now become good
    #     diff_new=df_new[~df_new.index.isin(df_old.index)]#
    #
    #     for counter, (key, row) in enumerate(diff_new.iterrows()):
    #         #print(counter, old, new, key, "new and not in 5 years ago", row["name"])
    #         pass
    #
    #
    #     # old stock that become bad=stock in previous year but not now
    #     diff_old=df_old[~df_old.index.isin(df_new.index)]
    #     for counter, (key,row) in enumerate(diff_old.iterrows()):
    #         #print(counter,old,new,key,"5 years ago good, not now",row["name"])
    #         pass
    #
    #     #5 years ago good, now also good
    #     diff_both = df_old[df_old.index.isin(df_new.index)]
    #     for counter, (key, row) in enumerate(diff_both.iterrows()):
    #         print(counter, old, new, key, "both good", row["name"])
    #         pass
    #
    #     print("both good",len(df_old),len(diff_both))
    #     print("past good, now bad",len(df_old),len(diff_old))
    #     print("past bad, now good",len(df_new),len(diff_new))
    #     print("% old beat index since ipo",len(df_old)/len_df_old)
    #     print("% new beat index since ipo",len(df_new)/len_df_new)
    #     print()
    #     print()
    #     print()
    #     print()



def stock_market_abv_ma():
    """
    This strategy tests if using abv ma 5-240 is a good strategy

    Result:
    using abv_ma on any freq does not yield significant result
    all results are pretty the same. not very useful

    """
    df=DB.get_stock_market_all()
    df["tomorrow"]=df["open.fgain1"].shift(-1)

    for freq in [5,20,60,240]:
        df_filtered=df[df[f"abv_ma{freq}"]<=0.8]
        gain=df_filtered["tomorrow"].mean()
        print(freq,"und 0.8 :",gain)

    for freq in [5,20,60,240]:
        df_filtered=df[df[f"abv_ma{freq}"]>0.2]
        gain=df_filtered["tomorrow"].mean()
        print(freq,"abv 0.2 :",gain)


def pattern_bull():
    """
    this function tries to identify the pattern when a stock is retreated from a good long term bullishness trend
    1. Find long term bull trend
    2. find short term retreat

    optional:
    1. the asset is still abv ma
    2. last ext

    Basically:
    - Buy if 60 ma is close to 240 ma
    - And hope that it will bounce back


    Result:
    -This works ONLY if YOU are 100% SURE about the uptrend
    -
    -Sometimes it bounces back
    -Somestimes it doesnt
    -It totlly depends on stocks fundamental and self strength
    -This strategy itself is natural and only works on bullish stock retreat
    - This is a good example to show that technical analysis alone is not enough. Fundamental can compliment here a lot.
    - For 600519, in 3000 days, only 7 times bottom happens. Which means, 12 years, 7 times real bottom.
    """

    df= DB.get_asset(ts_code="600519.SH")

    df["signal"]=0
    print(df.abv_ma240)
    import Alpha
    df["lp"]=Alpha.lowpass(df=df,abase="close",freq=10)

    df[f"e_min60"]=df["lp"].rolling(60).min()
    df["ma120"]=df["lp"].rolling(60).mean()
    df["ma240"]=df["lp"].rolling(240).mean()
    df["abv_ma120"]= (df["lp"]>df["ma120"]).astype(int)

    df[f"e_gain60"]=df["pct_chg"].rolling(60).mean()

    df.loc[
            (df["ma240"] / df["ma120"]).between(0.95, 1.05)
            &(df["e_gain60"]>0)
             ,"signal" ]=1
    df["signal"]=df["signal"]*df["e_max"]

    UI.plot_chart(df, ["lp", "signal", "ma240", "ma60", "e_max"])






"1.test how good stocks at their maximum perform"
"2.use indicator to see how many stocks are at their maximum"
"3.simplyfy the gathering process each day to make more frequent update"


