import scrapy
import requests
import DB
import pandas as pd
import LB

def qdii_research():
    """券商研报 from sina finance
    scrapy method
    """
    def scrape(url):
        sel = scrapy.Selector(text=requests.get(url).text)
        s_titles = sel.xpath('//td[@class="tal f14"]/a/@title').extract()
        s_date = sel.xpath('//td[@class="tal f14"]/following-sibling::td[2]/text()').extract()
        s_qdii = sel.xpath('//td[@class="tal f14"]/following-sibling::td[3]/a/div/span/text()').extract()
        s_person = sel.xpath('//td[@class="tal f14"]/following-sibling::td[4]/div/span/text()').extract()

        a_result = []
        for counter, (title, date, qdii, person) in enumerate(zip(s_titles, s_date, s_qdii, s_person)):
            a_result.append([title, date, qdii, person])
        return a_result


    #1. Create a list of all urls
    df_ts_code=DB.get_ts_code(a_asset=["E"])
    for ts_code in df_ts_code.index:
        #transform into finance.sina code
        ts_code_sina=ts_code[-2:].lower()+ts_code[0:6]

        ts_result=[]
        page_count = 1
        while True:
            print(ts_code,page_count)
            url = f"http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol={ts_code_sina}&t1=all&p={page_count}"
            a_result = scrape(url=url)
            if a_result:
                ts_result+=a_result
                page_count +=  1
            else:
                break

        #post processing
        df_result=pd.DataFrame(data=ts_result,columns=["title","date","qdii","person"])
        df_result["date"]=df_result["date"].str.replace("-","")
        df_result=LB.df_reverse_reindex(df_result)
        df_result["ts_code"]=ts_code
        df_result=df_result.set_index("date",drop=True)
        a_path=LB.a_path(f"Market/CN/Asset/E/qdii_research/{ts_code}")
        LB.to_csv_feather(df_result,a_path=a_path,skip_csv=True)



def qdii_grade():
    """券商评估 from sina finance
    direct pandas method
    """

    #1. Create a list of all urls
    df_ts_code=DB.get_ts_code(a_asset=["E"])
    for ts_code in df_ts_code.index:

        a_df_result=[]
        page_count = 1
        while True:
            print(ts_code,page_count)
            url = f"http://stock.finance.sina.com.cn/stock/go.php/vIR_StockSearch/key/{ts_code[0:6]}.phtml?p={page_count}"
            df = pd.read_html(url,flavor=['bs4'])

            #pre processing the df
            df = df[0]
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])

            if len(df)>5:
                a_df_result.append(df)
                page_count += 1
            else:
                print("finished",ts_code)
                break

        #post processing
        columns=['ts_code', 'name',"target","grade","qdii",	"person",	"industry","date","summary","latest_close","latest_pct_chg"	,"favorite","thread"]
        if a_df_result:
            df_result=pd.concat(a_df_result,sort=False,ignore_index=True)
            df_result = LB.df_reverse_reindex(df_result)
            df_result.columns=columns   # df_result["date"]=df_result["date"].str.replace("-","")

        else:
            df_result=pd.DataFrame(columns=columns)

        df_result["ts_code"] = ts_code
        df_result = df_result.set_index("date", drop=True)
        a_path=LB.a_path(f"Market/CN/Asset/E/qdii_grade/{ts_code}")
        LB.to_csv_feather(df_result,a_path=a_path,skip_csv=True)



if __name__ == "__main__":
    #df=pd.read_html("https://finviz.com/screener.ashx?v=111&o=ticker&r=0",flavor=['bs4'])
    df=pd.read_csv("egal.csv")
    print(df)
    # df=df[0]
    # df.columns = df.iloc[0]
    # df=df.drop(df.index[0])
    # df["股票代码"]=df["股票代码"].astype(str)
    # df.to_csv("test.csv",encoding="utf-8_sig")
