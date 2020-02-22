import os
import os.path
import pandas as pd
import DB
import Indicator_Create
import LB

pd.options.mode.chained_assignment = None  # default='warn'


# evaluate a columns agains fgain in various aspects
def bruteforce_eval_fgain(df, ts_code, column, dict_fgain_mean_detail):
    dict_ts_code_mean = {}
    try:
        for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
            # general ts_code pgain
            df_fgain_mean.at[ts_code, fgain] = dict_ts_code_mean[fgain] = df[fgain].mean()
            # general ts_code pearson with fgain
            df_fgain_mean.at[ts_code, f"{fgain}_pearson"] = df[column].corr(df[fgain], method="pearson")
    except:
        print("wtf.should not happend TODO")
        return

    # evaluate after Quantile
    try:
        p_setting = [(0, 0.18), (0.18, 0.5), (0.5, 0.82), (0.82, 1)]
        for low_quant, high_quant in p_setting:
            low_val, high_val = list(df[column].quantile([low_quant, high_quant]))
            df_percentile = df[df[column].between(low_val, high_val)]
            for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_p{low_quant, high_quant}"] = df_percentile[fgain].mean() / dict_ts_code_mean[fgain]
    except:
        print("Quantile did not work")

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
    for trend_freq in LB.c_bfreq():
        for trend in [1, 0]:
            for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_trend{trend_freq}{trend}"] = df.loc[df[f"trend{trend_freq}"] == trend, fgain].mean() / dict_ts_code_mean[fgain]


setting = {
    "target": "asset",  # date
    "step": 100,  # 1 or any other integer
    "group_result": False,
    "path_general": "Market/CN/Bruteforce/result/",
    "path_result": "Market/CN/Bruteforce/",
    # "path_general": "E:/Bruteforce/result/",
    # "path_result": "E:/Bruteforce/",
    "big_update": False
}


def bruteforce_summary(folderPath, summarypath):
    # init summary
    dict_fgain = {f"fgain{x}": pd.DataFrame() for x in LB.c_bfreq()}

    # iterate over all results
    for subFolderRoot, foldersWithinSubFolder, files in os.walk(folderPath, topdown=False):
        for fileName in files:
            print("summarize ...", fileName)

            ibase, rest = fileName.split(".", 1)

            dict_args = {}
            try:
                deri, rest = rest.split("(", 1)
                args, rest = rest.split(")", 1)
                for arg_pair in args.split(","):
                    key = arg_pair.split("=")[0]
                    if key != "":
                        print("arg_pair", arg_pair)
                        value = arg_pair.split("=")[1]
                        dict_args[key] = value
                fgain = rest.split("_", 1)[1].split(".")[0]
            except:
                # if the function does not have args and ()
                deri, rest = rest.split("_", 1)
                fgain = rest.split(".")[0]

            if fgain != "sample":
                df = pd.read_csv(subFolderRoot + "/" + fileName)
                # mean
                df_mean = df.mean()
                df_mean["ibase"] = ibase
                df_mean["deri"] = deri
                df_mean["args"] = dict_args
                dict_fgain[fgain] = dict_fgain[fgain].append(df_mean, sort=False, ignore_index=True)

    for key, df in dict_fgain.items():
        a_path = LB.a_path(summarypath + f"{key}")
        LB.to_csv_feather(df=df, a_path=a_path, skip_feather=True)

        # print(os.path.join(subFolderRoot, fileName))



# Bruteforce all: Indicator X Derivation X Derivation variables  for all ts_code through all time
def bruteforce_iterate():
    dict_df_asset = DB.preload(load=setting["target"], step=setting["step"], query="trade_date > 20050101")

    e_ibase = Indicator_Create.IBase
    e_ideri = Indicator_Create.IDeri

    len_e_ibase = len([x for x in e_ibase])
    len_e_ideri = len([x for x in e_ideri])

    # for each possible base ibase
    for ibase, ibase_counter in zip(e_ibase, range(1, len([x for x in e_ibase]) + 1)):
        ibase_name = ibase.value

        # for each possible derivative function
        for ideri, ideri_counter in zip(e_ideri, range(1, len([x for x in e_ideri]) + 1)):
            ideri_name = ideri.value
            if ideri_name == "create":
                print("names", ibase_name, ideri_name)
                deri_function = Indicator_Create.get_func(ibase_name)
            else:
                deri_function = Indicator_Create.get_func(ideri_name)
            print("deri is,", ideri_name, deri_function.__name__)
            settings_explode = Indicator_Create.function_all_combinations(deri_function)

            print("all combinations", settings_explode)
            len_setting_explode = len(settings_explode)

            # for each possible way to create the derivative function
            for one_setting, setting_counter in zip(settings_explode, range(1, len(settings_explode) + 1)):

                # CHECK IF OVERRIDE OR NOT if small update, then continue if file exists
                if not setting["big_update"]:
                    for key in LB.c_bfreq():
                        path = setting["path_general"] + f"{ibase_name}/{ideri_name}/" + LB.standard_indi_name(ibase=ibase_name, deri=ideri_name, dict_variables=one_setting) + f"_fgain{key}.csv"
                        if not os.path.exists(path):
                            LB.line_print(f"SMALL UPDATE: File NOT EXIST. DO. -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{ideri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")
                            break  # go into the ibase
                    else:
                        LB.line_print(f"SMALL UPDATE: File Exists. Skip. -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{ideri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")
                        continue  # go to next ibase
                else:
                    LB.line_print(f"BIG UPDATE: -> {ibase}:{ibase_counter}/{len_e_ibase}. Deri.{ideri_name}:{ideri_counter}/{len_e_ideri}. setting:{setting_counter}/{len_setting_explode}")

                # create sample
                path = setting["path_general"] + f"{ibase_name}/{ideri_name}/" + LB.standard_indi_name(ibase=ibase_name, deri=ideri_name, dict_variables=one_setting) + f"_sample"
                df_sample = dict_df_asset["000001.SZ"].copy()
                print("one setting is", {**one_setting})
                deri_function(df=df_sample, ibase=ibase.value, **one_setting)
                LB.to_csv_feather(df=df_sample, index=True, reset_index=False, a_path=LB.a_path(path), skip_feather=True)

                # Initialize ALL ts_code and fgain result for THIS COLUMN, THIS DERIVATION, THIS SETTING
                dict_fgain_mean_detail = {f"fgain{freq}": pd.DataFrame() for freq in LB.c_bfreq()}

                # for each possible asset
                print(f"START: ibase={ibase.value} ideri={ideri.value} setting=" + str({**one_setting}))
                for ts_code, df_asset in dict_df_asset.items():
                    # create new derived indicator ON THE FLY
                    df_asset_copy = df_asset.copy()  # maybe can be saved, but is risky. Copy is safer but slower
                    print("ts_code", ts_code)
                    if df_asset.empty:
                        print("empty", ts_code)
                    deri_column = deri_function(df=df_asset_copy, ibase=ibase.value, **one_setting)


                    # evaluate new derived indicator
                    bruteforce_eval_fgain(df=df_asset_copy, ts_code=ts_code, column=deri_column, dict_fgain_mean_detail=dict_fgain_mean_detail)

                # save evaluated results
                for key, df in dict_fgain_mean_detail.items():
                    df.index.name = "ts_code"
                    path = setting["path_general"] + f"{ibase_name}/{ideri_name}/" + LB.standard_indi_name(ibase=ibase_name, deri=ideri_name, dict_variables=one_setting) + f"_{key}"
                    # DB.ts_code_series_to_excel(df_ts_code=df, path=path, sort=[key, False], asset="E", group_result=setting["group_result"])
                    LB.to_csv_feather(df=df, index=True, reset_index=False, a_path=LB.a_path(path), skip_feather=True)

        # save Summary after one columns all derivation is finished
        # row_summary = df.mean()
        # row_summary["bruteforce"] = f"{ibase}_corr_{key}"
        # dict_df_summary[key] = dict_df_summary[key].append(row_summary, sort=False, ignore_index=True)
        # Util.to_csv_feather(df=dict_df_summary[key], skip_feather=True, a_path=Util.a_path(setting["path_result"] + f"{key}_summary"))


if __name__ == '__main__':
    # TODO generate test file
    # USE Sound theory harmonic to find the best frequency
    # import Indicator_Create
    df = DB.get_asset()
    # df=df[["open","close","pct_chg"]]
    # lol=Indicator_Create.open
    # name=Indicator_Create.trend(df=df,ibase="close")
    # df.to_csv("test.csv")
    # print("before main")
    # import Indicator_Create
    # lol=Indicator_Create.open
    # fu=lol(df=df,ibase="close")

    bruteforce_iterate()
    bruteforce_summary(setting["path_general"], setting["path_result"])
