import pandas as pd
import time
import os.path
import numpy as np
import Util
import DB
import os
import Indicator_Create
import itertools

pd.options.mode.chained_assignment = None  # default='warn'


# evaluate a columns agains fgain in various aspects
def eval_fgain(df, ts_code, column, dict_fgain_mean_detail):
    dict_ts_code_mean = {}
    for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
        # general ts_code pgain
        df_fgain_mean.at[ts_code, fgain] = dict_ts_code_mean[fgain] = df[fgain].mean()
        # general ts_code pearson with fgain
        df_fgain_mean.at[ts_code, f"{fgain}_pearson"] = df[column].corr(df[fgain], method="pearson")

    # evaluate after percentile
    p_setting = [(0, 0.18), (0.18, 0.5), (0.5, 0.82), (0.82, 1)]
    for low_quant, high_quant in p_setting:
        low_val, high_val = list(df[column].quantile([low_quant, high_quant]))
        df_percentile = df[df[column].between(low_val, high_val)]
        for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
            df_fgain_mean.at[ts_code, f"{fgain}_p{low_quant, high_quant}"] = df_percentile[fgain].mean() / dict_ts_code_mean[fgain]

    # evaluate after occurence bins
    try:
        o_setting = 4
        s_occurence = df[column].value_counts(bins=o_setting)
        for (index, value), counter in zip(s_occurence.iteritems(), range(0, o_setting)):
            df_occ = df[df[column].between(index.left, index.right)]
            for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_o{counter}"] = df_occ[fgain].mean() / dict_ts_code_mean[fgain]
    except Exception as e:
        print("Occurence did not work")

    # evaluate after probability/ occurence TODO

    # evaluate after seasonality
    for trend_freq in Util.c_rolling_freq():
        for trend in [1, 0]:
            for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_trend{trend_freq}{trend}"] = df.loc[df[f"trend{trend_freq}"] == trend, fgain].mean() / dict_ts_code_mean[fgain]


# Bruteforce all: Indicator X Derivation X Derivation variables  for all ts_code through all time
def bruteforce_create_derivative():
    setting = {
        "target": "asset",  # date
        "step": 1000,  # 1 or any other integer
        "group_result": False,
        "path_general": "Market/CN/Bruteforce/result/",
        "path_result": "Market/CN/Bruteforce/",
        # "path_general": "E:/Bruteforce/result/",
        # "path_result": "E:/Bruteforce/",
        "big_update": False
    }

    dict_df_asset = DB.preload(load=setting["target"], step=setting["step"], query="trade_date > 20050101")
    dict_df_summary = {f"fgain{freq}": pd.DataFrame() for freq in Util.c_rolling_freq()}
    e_ibase = Indicator_Create.IBase
    e_ideri = Indicator_Create.IDeri

    len_e_ibase = len([x for x in e_ibase])
    len_e_ideri = len([x for x in e_ideri])

    # for each possible base ibase
    for ibase, ibase_counter in zip(e_ibase, range(1, len([x for x in e_ibase]) + 1)):
        ibase_name = ibase.value

        # for each possible derivative function
        for ideri, ideri_counter in zip(e_ideri, range(1, len([x for x in e_ideri]) + 1)):
            deri_name = ideri.value
            deri_function = Indicator_Create.get_deri_func(ideri)
            settings_explode = Indicator_Create.function_all_combinations(deri_function)
            len_setting_explode = len(settings_explode)

            # for each possible way to create the derivative function
            for one_setting, setting_counter in zip(settings_explode, range(1, len(settings_explode) + 1)):

                # CHECK IF OVERRIDE OR NOT if small update, then continue if file exists
                if not setting["big_update"]:
                    for key in Util.c_rolling_freq():
                        path = setting["path_general"] + f"{ibase_name}/{deri_name}/" + Util.standard_indi_name(ibase=ibase_name, deri=deri_name, dict_variables=one_setting) + f"_fgain{key}.csv"
                        if not os.path.exists(path):
                            Util.line_print(f"SMALL UPDATE: File NOT EXIST. DO. -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{deri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")
                            break  # go into the ibase
                    else:
                        Util.line_print(f"SMALL UPDATE: File Exists. Skip. -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{deri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")
                        continue  # go to next ibase
                else:
                    Util.line_print(f"BIG UPDATE: -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{deri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")

                # create sample
                path = setting["path_general"] + f"{ibase_name}/{deri_name}/" + Util.standard_indi_name(ibase=ibase_name, deri=deri_name, dict_variables=one_setting) + f"_sample"
                df_sample = dict_df_asset["000001.SZ"][[ibase_name]]
                deri_function(df=df_sample, ibase=ibase.value, **one_setting)
                Util.to_csv_feather(df=df_sample, index=True, reset_index=False, a_path=Util.a_path(path), skip_feather=True)

                # Initialize ALL ts_code and fgain result for THIS COLUMN, THIS DERIVATION, THIS SETTING
                dict_fgain_mean_detail = {f"fgain{freq}": pd.DataFrame() for freq in Util.c_rolling_freq()}

                # for each possible asset
                print(f"START: ibase={ibase.value} ideri={ideri.value} setting=" + str({**one_setting}))
                for ts_code, df_asset in dict_df_asset.items():
                    # create new derived indicator ON THE FLY
                    df_asset_copy = df_asset.copy()  # maybe can be saved, but is risky. Copy is safer but slower
                    deri_column = deri_function(df=df_asset_copy, ibase=ibase.value, **one_setting)

                    # evaluate new derived indicator
                    eval_fgain(df=df_asset_copy, ts_code=ts_code, column=deri_column, dict_fgain_mean_detail=dict_fgain_mean_detail)

                # save evaluated results
                for key, df in dict_fgain_mean_detail.items():
                    df.index.name = "ts_code"
                    path = setting["path_general"] + f"{ibase_name}/{deri_name}/" + Util.standard_indi_name(ibase=ibase_name, deri=deri_name, dict_variables=one_setting) + f"_{key}"
                    # DB.ts_code_series_to_excel(df_ts_code=df, path=path, sort=[key, False], asset="E", group_result=setting["group_result"])
                    Util.to_csv_feather(df=df, index=True, reset_index=False, a_path=Util.a_path(path), skip_feather=True)

        # save Summary after one columns all derivation is finished
        # row_summary = df.mean()
        # row_summary["bruteforce"] = f"{ibase}_corr_{key}"
        # dict_df_summary[key] = dict_df_summary[key].append(row_summary, sort=False, ignore_index=True)
        # Util.to_csv_feather(df=dict_df_summary[key], skip_feather=True, a_path=Util.a_path(setting["path_result"] + f"{key}_summary"))


if __name__ == '__main__':
    bruteforce_create_derivative()
