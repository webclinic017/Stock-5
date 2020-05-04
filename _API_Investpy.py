import investpy

"""the question is, where is the csv file generated?"""

def my_get_indices(country=None):
    return investpy.get_indices(country=country)


def my_get_stocks(country=None):
    return investpy.get_stocks(country=country)


def my_get_funds(country=None):
    return investpy.get_funds(country=country)


def my_stock_historical_data(stock, country, from_date, to_date):
    return investpy.get_stock_historical_data(stock=stock,
                                              country=country,
                                              from_date=from_date,
                                              to_date=to_date)


def my_fund_historical_data(fund, country, from_date, to_date):
    return investpy.get_fund_historical_data(fund=fund,
                                             country=country,
                                             from_date=from_date,
                                             to_date=to_date)


def my_index_historical_data(index, country, from_date, to_date):
    return investpy.get_index_historical_data(index=index,
                                              country=country,
                                              from_date=from_date,
                                              to_date=to_date)
