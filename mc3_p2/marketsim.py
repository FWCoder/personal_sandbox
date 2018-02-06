"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # TODO: Your code here
    
    date_range = pd.date_range(start_date, end_date)
    orders_df = pd.read_csv(orders_file)
    
    companies = []
    for symbol in orders_df["Symbol"]:
        if symbol not in companies:
            companies.append(symbol)

    columns = companies

    stock_values = get_data(companies, date_range, False)
    stock_values = stock_values.dropna()
    order_dates = stock_values.index.tolist()
    initData = np.zeros((len(order_dates), len(columns)))
    stockNumb = pd.DataFrame(initData, index=order_dates, columns=columns)
    portvals = pd.Series(np.zeros(len(order_dates)), order_dates)
    cashvals = pd.Series(np.zeros(len(order_dates)), order_dates)
    currentStockNumb = pd.Series(np.zeros(len(companies)), companies)

    index = 0
    current_cash = 0

    
    
    while index < len(order_dates):
        current_date = order_dates[index].strftime("%Y-%m-%d") 

        if(index == 0):
            current_cash = start_val
        order_index = 0
        while order_index < len(orders_df.index):
            current_order = orders_df.ix[order_index]
            order_date = current_order['Date']
            if(current_date == order_date):
                temp_stock_num = currentStockNumb.copy()
                temp_cash = current_cash
                current_stock_order = current_order["Order"]
                current_stock_name = current_order["Symbol"]
                current_stock_exchange = current_order["Shares"]
                if(current_stock_order == "BUY"):
                    current_stock_numb = currentStockNumb[current_stock_name] +  current_stock_exchange
                    temp_cash -= stock_values.loc[current_date][current_stock_name]*current_stock_exchange
                else:    
                    current_stock_numb = currentStockNumb[current_stock_name] -  current_stock_exchange
                    temp_cash += stock_values.loc[current_date][current_stock_name]*current_stock_exchange
                temp_stock_num[current_stock_name] = current_stock_numb

                # Calculate leverage
                long_val = 0
                short_val = 0
                for company in companies:
                    if(temp_stock_num[company] > 0):
                        long_val += temp_stock_num[company]*stock_values.loc[current_date][company]
                    else:    
                        short_val += temp_stock_num[company]*stock_values.loc[current_date][company]

                leverage_ratio = (long_val + abs(short_val))/(long_val - abs(short_val)+temp_cash)
                
                #print leverage_ratio
                if(leverage_ratio < 2):        
                    current_cash = temp_cash
                    currentStockNumb = temp_stock_num.copy()

            order_index += 1

        
        for company in companies:
            stockNumb.loc[current_date][company] = currentStockNumb[company] 

        cashvals[current_date] = current_cash 

        index += 1

    for date in order_dates:
        total_stock_values = 0
        for company in companies:
            total_stock_values += stockNumb.loc[date][company] * stock_values.loc[date][company]
        portvals[date] = total_stock_values + cashvals[date]

    # print stockNumb
    # print cashvals
    # print portvals
    return portvals




def test_run():
    """Driver function."""

    # Define input parameters

    start_date = '2011-01-05'
    end_date = '2011-01-20'
    orders_file = os.path.join("orders", "orders-short.csv")

    # start_date = '2011-01-10'
    # end_date = '2011-12-20'
    # orders_file = os.path.join("orders", "orders.csv")

    # start_date = '2011-01-14'
    # end_date = '2011-12-14'
    # orders_file = os.path.join("orders", "orders2.csv")

    start_val = 1000000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
