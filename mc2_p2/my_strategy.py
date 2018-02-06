"""MC2-P2: Technical Analysis Strategies."""
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import date

from numpy import convolve

# import from the previous projects
from util import get_data, plot_data 
from marketsim import compute_portvals
from portfolio.analysis import get_portfolio_value, get_portfolio_stats

filename = "orders.csv"

def plot_macd_strategy(df, df_hist, price, symbols, title="MACD Strategy", xlabel="Date", ylabel="Stock Prices"):
    if os.path.exists(filename):
        os.remove(filename);
    file = open(filename, 'w+')
    
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fillcolor = 'darkslategrey'
    ax.fill_between(df.index, df_hist, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    long_position = False
    short_position = False
    index = 1
    currentCash = 10000
    currentLongStock = 0
    currentShortStock = 0
    file.write("Date,Symbol,Order,Shares\n")
    plt.axhline(0, color='k')
    while index < len(df.index):

        if(df['MACD'][index - 1] < df['Signal'][index - 1] and df['MACD'][index] >= df['Signal'][index] and not long_position):
        
            plt.axvline(df.index[index], color='g')
            currentLongStock += 100
            file.write(str(df.index[index].date())+","+symbols+",BUY,100\n")
            long_position = True
        elif(df['MACD'][index - 1] > df['Signal'][index - 1] and df['MACD'][index] <= df['Signal'][index] and long_position):
        
            plt.axvline(df.index[index], color='k')
            file.write(str(df.index[index].date())+","+symbols+",SELL,"+str(currentLongStock)+"\n")
            currentLongStock = 0
            long_position = False
        
        # elif(df_hist[index-1] > df_hist[index] and not short_position):
        #     plt.axvline(df.index[index], color='k')
        #     file.write(str(df.index[index].date())+","+symbols+",SELL,100\n")
        #     currentShortStock += 100
        #     short_position=True
        
        # elif(df_hist[index-1] > df_hist[index] and short_position):
        #     plt.axvline(df.index[index], color='k')
        #     file.write(str(df.index[index].date())+","+symbols+",BUY,"+str(currentLongStock)+"\n")
        #     currentShortStock = 0
        #     short_position=False
         
        
        index += 1

    file.close()
    #plt.show()
    plt.savefig('my_strategy.png')

def plot_normalized_data(df, title="Daily portfolio values", xlabel="Date", ylabel="Normalized prices"):
    df = df / df.ix[0,:] 
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('my_strategy_back_test.png')

def build_MACD(symbols, start_date, end_date, n_fast, n_slow, start_val):
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([symbols], dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    SPY_prices = prices_all['SPY']


    ema_fast = pd.ewma(prices, span=n_fast, min_periods = n_slow - 1)
    
    ema_slow = pd.ewma(prices, span=n_slow, min_periods = n_slow - 1)
    
    macd = ema_fast - ema_slow
    macd_signal = pd.ewma(macd, span=9, min_periods = 8)

    macd_hist  = macd - macd_signal
    macd_df = pd.DataFrame(macd, columns=['MACD'])
    macd_df['Signal'] = macd_signal

    macd_df.dropna()
    
    plot_macd_strategy(macd_df, macd_hist, prices, symbols)

    # Process orders
    portvals = compute_portvals(start_date, end_date, filename, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    df_portfolio = pd.DataFrame(SPY_prices)
    df_portfolio['Bollinger Strategy'] = compute_portvals(start_date, end_date, 'orders.txt', start_val)
    df_portfolio['MACD'] = portvals

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
    build_MACD('IBM', start_date, end_date, 12, 26, 10000)   

if __name__ == "__main__":
    test_run()