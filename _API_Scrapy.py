import scrapy
import requests
import xlrd
import xlsxwriter
import time


def start():
    url = (time.strftime("%Y.%m.%d-%H%M%S."))
    readworkbook = xlrd.open_workbook('quanshang.xlsx')
    readworksheet = readworkbook.sheet_by_index(0)

    print('START')

    for i in range(readworksheet.nrows):
        url = readworksheet.cell_value(i, 0)
        print("url:", url)

        sel = scrapy.Selector(text=requests.get(url).text)

        # main = sel.xpath('//div[@class="main"]/table/tbody').extract()
        s_titles = sel.xpath('//td[@class="tal f14"]/a/@title').extract()
        s_date = sel.xpath('//td[@class="tal f14"]/following-sibling::td[2]/text()').extract()
        s_qdii = sel.xpath('//td[@class="tal f14"]/following-sibling::td[3]/a/div/span/text()').extract()
        s_person = sel.xpath('//td[@class="tal f14"]/following-sibling::td[4]/div/span/text()').extract()

        for counter, (title, date, qdii, person) in enumerate(zip(s_titles, s_date, s_qdii, s_person)):
            print(counter, title, date, qdii, person)


if __name__ == "__main__":
    start()