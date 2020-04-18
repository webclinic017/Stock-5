import os
import os.path
import pandas as pd
import DB
import Alpha
import LB
import random
from scipy.stats.mstats import gmean
import traceback

pd.options.mode.chained_assignment = None  # default='warn'

"""
brute force is only useful when double combining
But since this would yield exploding complexity, it should not be considered
so manual auto


KEEP THIS FILE AS DEPRECATED SNAPSHOT FROM LAST STAND 
"""


# evaluate a columns agains fgain in various aspects
def bruteforce_eval_fgain(df, ts_code, column, d_fgain_gmean_detail):
    d_ts_code_mean = {}
    df[column] = df[column].astype(float)
    try:
        std = df[column].std()
        nd_pct = len(df[df[column].isna() | df[column] != 0.0]) / len(df)
        # autocorr2 = df[column].autocorr(2)
        # autocorr20 = df[column].autocorr(20)
        # autocorr240 = df[column].autocorr(240)
        # skew = df[column].skew()
        # kurt = df[column].kurt()

        for fgain, df_fgain_mean in d_fgain_gmean_detail.items():
            # general ts_code pgain
            df_fgain_mean.at[ts_code, fgain] = d_ts_code_mean[fgain] = gmean(df[fgain])
            # general ts_code pearson with fgain
            df_fgain_mean.at[ts_code, f"{fgain}_pearson"] = df[column].corr(df[fgain], method="pearson")
            # other overview
            df_fgain_mean.at[ts_code, f"std"] = std
            df_fgain_mean.at[ts_code, f"nd_pct"] = nd_pct
    except Exception as e:
        print("error", e)
        print("wtf.should not happend TODO")
        return

    # evaluate after Quantile
    try:
        p_setting = [(0, 0.18), (0.18, 0.5), (0.5, 0.82), (0.82, 1)]
        for low_quant, high_quant in p_setting:
            low_val, high_val = list(df[column].quantile([low_quant, high_quant]))
            df_percentile = df[df[column].between(low_val, high_val)]

            for fgain, df_fgain_mean in d_fgain_gmean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_p{low_quant, high_quant}"] = gmean(df_percentile[fgain]) / d_ts_code_mean[fgain]
    except Exception as e:
        print("Quantile did not work")

    # evaluate after occurence bins. Bins are able to catch 1 and 0 values, while quantile can not distinct between them
    try:
        o_setting = 4
        s_occurence = df[column].value_counts(bins=o_setting)
        for counter, (index, value) in enumerate(s_occurence.iteritems()):
            df_occ = df[df[column].between(index.left, index.right)]
            for fgain, df_fgain_mean in d_fgain_gmean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_o{counter}"] = gmean(df_occ[fgain]) / d_ts_code_mean[fgain]
    except Exception as e:
        print("Occurence did not work")

    # evaluate after seasonality
    # for trend_freq in LB.c_bfreq():
    #     for trend in [1, 0]:
    #         for fgain, df_fgain_mean in d_fgain_mean_detail.items():
    #             df_fgain_mean.at[ts_code, f"{fgain}_trend{trend_freq}{trend}"] = df.loc[df[f"trend{trend_freq}"] == trend, fgain].mean() / d_ts_code_mean[fgain]


setting = {
    "target": "asset",  # date
    "preload_step": 5,  # 1 or any other integer
    "sample_size": 500,  # 1 or any other integer
    "group_result": False,
    "path_general": "Market/CN/Bruteforce/result/",
    "path_result": "Market/CN/Bruteforce/",
    "big_update": False
}


def bruteforce_summary(folderPath, summarypath):
    # init summary
    d_fgain = {f"fgain{x}": pd.DataFrame() for x in LB.c_bfreq()}

    # iterate over all results
    for subFolderRoot, foldersWithinSubFolder, files in os.walk(folderPath, topdown=False):
        for fileName in files:
            print("summarize ...", fileName)

            ibase, rest = fileName.split(".", 1)

            d_args = {}
            try:
                deri, rest = rest.split("(", 1)
                args, rest = rest.split(")", 1)
                for arg_pair in args.split(","):
                    key = arg_pair.split("=")[0]
                    if key != "":
                        print("arg_pair", arg_pair)
                        value = arg_pair.split("=")[1]
                        d_args[key] = value
                fgain = rest.split("_", 1)[1].split(".")[0]
            except:
                # if the function does not have args and ()
                deri, rest = rest.split("_", 1)
                fgain = rest.split(".")[0]

            if fgain != "sample":
                df = pd.read_csv(f"{subFolderRoot}/{fileName}")
                # mean
                df_mean = df.mean()
                df_mean["ibase"] = ibase
                df_mean["deri"] = deri
                df_mean["args"] = d_args
                d_fgain[fgain] = d_fgain[fgain].append(df_mean, sort=False, ignore_index=True)

    for key, df in d_fgain.items():
        a_path = LB.a_path(f"{summarypath}{key}")
        LB.to_csv_feather(df=df, a_path=a_path, skip_feather=True, index_relevant=False)


# Bruteforce all: Indicator X Derivation X Derivation variables  for all ts_code through all time
def bruteforce_iterate():
    d_df_asset = DB.preload(asset=setting["target"], step=setting["preload_step"], query_df="trade_date > 20050101")

    e_ibase = Alpha.ABase
    e_ideri = Alpha.ADeri

    len_e_ibase = len([x for x in e_ibase])
    len_e_ideri = len([x for x in e_ideri])

    # for each possible base ibase
    for ibase_counter, ibase in enumerate(e_ibase, start=1):
        ibase_name = ibase.value

        # for each possible derivative function
        for ideri_counter, ideri in enumerate(e_ideri, start=1):
            ideri_name = ideri.value
            if ideri_name == "create":
                deri_function = Alpha.get_func(ibase_name)
            else:
                deri_function = Alpha.get_func(ideri_name)
            print("deri is,", ideri_name, deri_function.__name__)
            settings_explode = Alpha.function_all_combinations(deri_function)

            print("all combinations", settings_explode)
            len_setting_explode = len(settings_explode)

            # for each possible way to create the derivative function
            for setting_counter, one_setting in enumerate(settings_explode, start=1):

                # CHECK IF OVERRIDE OR NOT if small update, then continue if file exists
                if not setting["big_update"]:
                    for key in LB.c_bfreq():
                        path = f"{setting['path_general']}{ibase_name}/{ideri_name}/" + LB.indi_name(abase=ibase_name, deri=ideri_name, d_variables=one_setting) + f"_fgain{key}.csv"
                        if not os.path.exists(path):
                            LB.line_print(f"SMALL UPDATE: File NOT EXIST. DO. -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{ideri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")
                            break  # go into the ibase
                    else:
                        LB.line_print(f"SMALL UPDATE: File Exists. Skip. -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{ideri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")
                        continue  # go to next ibase
                else:
                    LB.line_print(f"BIG UPDATE: -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{ideri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")

                # create sample
                path = f"{setting['path_general']}{ibase_name}/{ideri_name}/" + LB.indi_name(abase=ibase_name, deri=ideri_name, d_variables=one_setting) + f"_sample"
                df_sample = d_df_asset["000001.SZ"].copy()
                print("one setting is", {**one_setting})
                deri_function(df=df_sample, ibase=ibase.value, **one_setting)
                LB.to_csv_feather(df=df_sample, a_path=LB.a_path(path), index_relevant=False, skip_feather=True)

                # Initialize ALL ts_code and fgain result for THIS COLUMN, THIS DERIVATION, THIS SETTING
                d_fgain_mean_detail = {f"fgain{freq}": pd.DataFrame() for freq in LB.c_bfreq()}

                # for each possible asset
                print(f"START: ibase={ibase.value} ideri={ideri.value} setting=" + str({**one_setting}))
                d_df_asset_sample = {x: d_df_asset[x] for x in random.sample([x for x in d_df_asset.keys()], setting["sample_size"])}
                for ts_code, df_asset in d_df_asset_sample.items():
                    # create new derived indicator ON THE FLY and Evaluate
                    print("ts_code", ts_code)
                    df_asset_copy = df_asset.copy()  # maybe can be saved, but is risky. Copy is safer but slower
                    deri_column = deri_function(df=df_asset_copy, ibase=ibase.value, **one_setting)
                    bruteforce_eval_fgain(df=df_asset_copy, ts_code=ts_code, column=deri_column, d_fgain_gmean_detail=d_fgain_mean_detail)

                # save evaluated results
                for key, df in d_fgain_mean_detail.items():
                    df.index.name = "ts_code"
                    path = f"{setting['path_general']}{ibase_name}/{ideri_name}/" + LB.indi_name(abase=ibase_name, deri=ideri_name, d_variables=one_setting) + f"_{key}"
                    # DB.ts_code_series_to_excel(df_ts_code=df, path=path, sort=[key, False], asset="E", group_result=setting["group_result"])
                    LB.to_csv_feather(df=df, a_path=LB.a_path(path), index_relevant=True, skip_feather=True)


if __name__ == '__main__':
    # TODO generate test cases
    # need to use geomean for eval function
    df = DB.get_asset()

    bruteforce_iterate()
    bruteforce_summary(setting["path_general"], setting["path_result"])
