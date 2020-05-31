import scrapy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import DB
import LB
import os
# from snownlp import SnowNLP
import time
driver = webdriver.Firefox()


def scrap_eastmoney_comment(url, a_path, ts_code):
    driver.get(url)
    a_result = []

    try:
        counter = 0
        while True:
            # wait till al ajax are loaded
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".all.hs_list")))
            print("loaded")
            """
            # load html bloc
            html = driver.page_source
            soup = BeautifulSoup(html, "lxml")
            all_title = soup.select(".articleh.normal_post")
            print("html works")

            # scrap each article
            for title in all_title:
                try:
                    read=title.select(".l1.a1")[0].text
                    comment=title.select(".l2.a2")[0].text
                    h3 = title.select(".l3.a3 a")[0].text
                    href = title.select(".l3.a3 a")[0]["href"]
                    username = title.select(".l4.a4 a font")[0].text
                    date_source = title.select(".l5.a5")[0].text

                    d_result = {"read":read,"comment":comment,"h3": h3, "href": href, "user": username, "date": date_source}
                    a_result.append(pd.Series(d_result))
                    print(ts_code, counter, d_result)
                except Exception as e:
                    print(e)
"""
            # click the next page
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".pagernums a:last-of-type"))).click()
            counter += 1

    except Exception as e:
        print(e)
    df = pd.DataFrame(a_result)
    LB.to_csv_feather(df=df, a_path=a_path, skip_feather=True)
    print("Finished at page", counter)


def scrap_xueqiu_comment(url, a_path, ts_code):
    driver.get(url)
    a_result = []
    # select only 讨论 tab
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".stock-timeline-tabs a:nth-of-type(2)"))).click()

    try:
        counter = 0
        while True:
            # wait till al ajax are loaded
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".pagination__next")))

            try:
                # scrap this page
                # load html bloc
                html = driver.page_source
                soup = BeautifulSoup(html, "lxml")
                all_title = soup.select(".stock-timeline .status-list .timeline__item")

                # scrap each article
                for title in all_title:
                    try:
                        username = title.select(".timeline__item__info div .user-name")
                        username = username[0].text
                        date_source = title.select(".timeline__item__info .date-and-source")
                        date_source = date_source[0].text.split('·')[0]

                        # check if this is a normal comment or article related to this stock
                        h3 = title.select(".timeline__item__bd h3")
                        if h3:
                            # is an article
                            content = title.select(".content.content--description div")
                            href = title.select(".timeline__item__content a")
                            href = href[0]["href"]
                            h3 = h3[0].text
                        else:
                            # not an article
                            content = title.select(".content.content--description div")
                            href = np.nan
                            h3 = np.nan
                        d_result = {"content": content[0].text, "h3": h3, "href": href, "user": username, "date": date_source}
                        a_result.append(pd.Series(d_result))
                        print(ts_code, counter, d_result)
                    except Exception as e:
                        print(e)

                #click the next page
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".pagination__next"))).click()
                counter += 1
            except:
                try:  # second try to set the pagination visible
                    driver.execute_script('document.getElementsByClassName("pagination")[0].style.display = "block";')
                    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".pagination__next"))).click()
                    print("you almost got me")
                except Exception as e:
                    print("end of page")
                    print(e)
                    break
    except Exception as e:
        print(e)
    df = pd.DataFrame(a_result)
    LB.to_csv_feather(df=df, a_path=a_path, skip_feather=True)
    print("Finished at page", counter)


def initiator(website="eastmoney"):
    df_ts_code = DB.get_ts_code(a_asset=["E"])
    for ts_code in df_ts_code.index:
        print(ts_code)
        tail = str(ts_code)[7:9]
        head = str(ts_code)[0:6]

        if website == "xueqiu":
            url = f"https://xueqiu.com/S/{tail}{head}"
            a_path = LB.a_path(f"Market/CN/Asset/E/D_Xueqiu_scrap/{ts_code}")
            if not os.path.isfile(a_path[0]):
                scrap_xueqiu_comment(url, a_path, ts_code)
        elif website == "eastmoney":
            url = f"http://guba.eastmoney.com/list,{head},f_1.html"
            a_path = LB.a_path(f"Market/CN/Asset/E/D_Eastmoney_scrap/{ts_code}")
            if not os.path.isfile(a_path[0]):
                scrap_eastmoney_comment(url, a_path, ts_code)
    driver.close()



if __name__ == "__main__":
    initiator()
