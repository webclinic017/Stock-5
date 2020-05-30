import scrapy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

driver = webdriver.Firefox()


def scrap(url):
    print("start ", url)
    driver.get(url)
    html = driver.page_source
    try:
        soup = BeautifulSoup(html, "lxml")
        # 获得有小区信息的panel
        status_list = soup.find_all('div', class_="status-list")[0]
        all_title = soup.select(".stock-timeline .status-list .timeline__item")

        import time
        counter = 0
        while True:
            try:

                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".pagination__next"))).click()
                # driver.find_element_by_css_selector(".pagination__next").click()
                print("Navigating to Next Page")
                # time.sleep(2)

                print(counter)
                counter += 1
            except Exception  as e:
                print(e)
                break

        for title in all_title:
            try:
                username = title.select(".timeline__item__info div .user-name")
                date_source = title.select(".timeline__item__info .date-and-source")

                # check if this is a normal comment or article related to this stock
                h3 = title.select(".timeline__item__bd h3")
                if h3:
                    # is an article
                    content = title.select(".content.content--description div")
                    href = title.select(".timeline__item__content a")
                    print(href[0]["href"])
                    print(h3[0].text)
                else:
                    # not an article
                    content = title.select(".content.content--description div")

                print(username[0].text, date_source[0].text.split('·')[0], content[0].text)
            except Exception as e:
                print(e)

    except Exception as e:
        pass
    print("Finished")


def initiator():
    url = "https://xueqiu.com/S/SZ399006"
    scrap(url)
    driver.close()


if __name__ == "__main__":
    initiator()
