
import DB
import LB
import pandas as pd


def asset_beat_index():
    """checks how many stock beats their index
    1. normalize all index to the day certain asset is IPOd
    2. Check if index or asset is better until today

    Amazing Result:
    53%主板beat index
    60%中小板beat index
    30%创业板beat index
    """

    #init
    df_ts_code=DB.get_ts_code()
    d_index=DB.preload(asset="I",step=1,d_queries_ts_code=LB.c_I_queries())
    d_e=DB.preload(step=1)
    df_result=pd.DataFrame()

    for ts_code, df_asset in d_e.items():
        print(ts_code)
        exchange=df_ts_code.at[ts_code,"exchange"]


        if exchange=="创业板":
            compare="399006.SZ"
        elif exchange=="中小板":
            compare ="399001.SZ"
        elif exchange=="主板":
            compare="000001.SH"

        l1=df_ts_code.at[ts_code,"industry1"]
        l2=df_ts_code.at[ts_code,"industry2"]
        l3=df_ts_code.at[ts_code,"industry3"]


        df_compare=d_index[compare].copy()

        df_asset_slim=LB.ohlcpp(df_asset).reset_index()
        df_index_slim=LB.ohlcpp(df_compare).reset_index()

        df_slim=pd.merge(df_asset_slim,df_index_slim,on="trade_date",how="inner",suffixes=[f"_{ts_code}",f"_{compare}"],sort=False)

        for code in [ts_code,compare]:
            df_slim[f"norm_{code}"]=df_slim[f"close_{code}"]/df_slim.at[0,f"close_{code}"]
            df_slim[f"norm_pct_{code}"]=df_slim[f"norm_{code}"].pct_change()
        #result=norm_ts_code/norm_compare

        df_result.at[ts_code,"period"]=period=len(df_slim)-1
        df_result.at[ts_code,"asset_vs_index_gain"]= df_slim.at[period, f"norm_{ts_code}"] / df_slim.at[period, f"norm_{compare}"]
        df_result.at[ts_code,"asset_vs_index_sharp"]= (df_slim[f"norm_pct_{ts_code}"].mean() / df_slim[f"norm_pct_{ts_code}"].std()) / df_slim[f"norm_pct_{compare}"].mean() / df_slim[f"norm_pct_{compare}"].std()
        df_result.at[ts_code,"asset_vs_index_gmean"]= (df_slim[f"norm_pct_{ts_code}"].mean() / df_slim[f"norm_pct_{ts_code}"].std())


        #TODO sharp ratio, geomean, beat industry, concept
        df_result.at[ts_code,"index"]=compare
    df_result["beat_index"]=(df_result["asset/index"]>1).astype(int)
    df_result.index.name="ts_code"

    DB.to_excel_with_static_data(df_ts_code=df_result,path="Market/CN/ATest/Beat_Index/result.xlsx")
    # a_path=LB.a_path("Market/CN/ATest/Beat_Index/result")
    # LB.to_csv_feather(df=df_result,a_path=a_path)



asset_beat_index()