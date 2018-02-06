"""MC2-P2: Technical Analysis Strategies."""
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

from numpy import convolve

# import from the previous projects
from util import get_data, plot_data 
from marketsim import compute_portvals
from portfolio.analysis import get_portfolio_value, get_portfolio_stats

filename = "orders.txt"

def plot_bollinger_data(df, symbols, window, title="Bollinger Bands", xlabel="Date", ylabel="Stock Prices"):
    if os.path.exists(filename):
        os.remove(filename);
    file = open(filename, 'w+')
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    long_entry = False
    short_entry = False
    index = 0
    currentCash = 10000
    currentLongStock = 0
    currentShortStock = 0
    file.write("Date,Symbol,Order,Shares\n")
    while index < len(df.index):
        
        if(index > window):
            if(df[symbols][index - 1] < df['Lower Band'][index - 1] and df[symbols][index] > df['Lower Band'][index] and not long_entry):
                plt.axvline(df.index[index], color='g')
                currentCash -= 100*df[symbols][index]
                currentLongStock += 100
                file.write(str(df.index[index].date())+","+symbols+",BUY,100\n")
                long_entry = True
            elif(df[symbols][index - 1] < df['SMA'][index - 1] and df[symbols][index] > df['SMA'][index] and long_entry):
                plt.axvline(df.index[index], color='k')
                currentCash += currentLongStock*df[symbols][index]
                file.write(str(df.index[index].date())+","+symbols+",SELL,"+str(currentLongStock)+"\n")
                currentLongStock = 0
                long_entry = False
            elif(df[symbols][index - 1] > df['Upper Band'][index - 1] and df[symbols][index] < df['Upper Band'][index] and not short_entry):
                plt.axvline(str(df.index[index]), color='r')
                currentCash += 100*df[symbols][index]
                file.write(str(df.index[index].date())+","+symbols+",SELL,100\n")
                currentShortStock += 100
                short_entry = True
            elif(df[symbols][index - 1] > df['SMA'][index - 1] and df[symbols][index] < df['SMA'][index] and short_entry):
                plt.axvline(df.index[index], color='k')
                currentCash -= currentShortStock*df[symbols][index]
                file.write(str(df.index[index].date())+","+symbols+",BUY,"+str(currentShortStock)+"\n")
                currentShortStock = 0
                short_entry = False

        index += 1
    file.close()
    plt.savefig('bollinger_strategy.png')

def plot_normalized_data(df, title="Daily portfolio values", xlabel="Date", ylabel="Normalized prices"):
    df = df / df.ix[0,:] 
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('bollinger_strategy_back_test.png')

def build_bollinger_band(symbols, start_date, end_date, window, start_val):
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([symbols], dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    SPY_prices = prices_all['SPY'] 
    sma = pd.rolling_mean(prices, window)

    sma_std = pd.rolling_std(prices, window)

    upper_band = sma + 2*sma_std
    lower_band = sma - 2*sma_std
    bollinger_df = pd.DataFrame(prices)
    bollinger_df['SMA'] = sma
    bollinger_df['Upper Band'] = upper_band
    bollinger_df['Lower Band'] = lower_band
    plot_bollinger_data(bollinger_df, symbols, window)

    # Process orders
    portvals = compute_portvals(start_date, end_date, filename, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    df_portfolio = pd.DataFrame(SPY_prices)
    df_portfolio['Portfolio'] = portvals

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

     # Simulate a $SPX-only reference portfolio to get stats
    prices_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPY = prices_SPY[['SPY']]  # remove SPY
    portvals_SPY = get_portfolio_value(prices_SPY, [1.0])
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(portvals_SPY)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY: {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY: {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    plot_normalized_data(df_portfolio)
    

def test_run():
    start_date = '2007-12-31' # Dec 31, 2007 
    end_date = '2009-12-31' # Dec 31, 2009
    build_bollinger_band('IBM', start_date, end_date, 20, 10000)   

if __name__ == "__main__":
    test_run()