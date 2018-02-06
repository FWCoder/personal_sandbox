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

filename = "stochastic_orders.csv"

def plot_stochastic_oscillator_strategy(df, df_hist, price, symbols, title="MACD Strategy", xlabel="Date", ylabel="Stock Prices"):
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

        #if(df['MACD'][index - 1] < df['Signal'][index - 1] and df['MACD'][index] > df['Signal'][index] ):
        if(df_hist[index-1] < df_hist[index] and not long_position):
            plt.axvline(df.index[index], color='g')
            currentLongStock += 100
            file.write(str(df.index[index].date())+","+symbols+",BUY,100\n")
            long_position = True
        #elif(df['MACD'][index - 1] > df['Signal'][index - 1] and df['MACD'][index] < df['Signal'][index]):
        elif(df_hist[index-1] > df_hist[index] and long_position):
            plt.axvline(df.index[index], color='k')
            file.write(str(df.index[index].date())+","+symbols+",SELL,"+str(currentLongStock)+"\n")
            currentLongStock = 0
            long_position = False
        #elif(df_hist[index-1] < df_hist[index] and not short_entry):
        # elif(df['MACD'][index - 1] < df['MACD'][index - 1] and prices[symbols][index-1] > prices[symbols][index] and not short_entry):
        #     plt.axvline(df.index[index], color='r')
        #     file.write(str(df.index[index].date())+","+symbols+",BUY,100\n")
        #     currentShortStock = 100
        #     short_entry = True
        # #elif(df_hist[index-1] > df_hist[index] and short_entry):
        # elif(df['MACD'][index - 1] > df['MACD'][index - 1] and prices[symbols][index-1] < prices[symbols][index] and short_entry):
        #     plt.axvline(df.index[index], color='k')
        #     file.write(str(df.index[index].date())+","+symbols+",SELL,"+str(currentShortStock)+"\n")
        #     currentShortStock = 0
        #     short_entry = False
        
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

def build_stochastic_oscillator(symbols, start_date, end_date, windows, start_val):
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

    df_portfolio = pd.DataFrame(SPY_prices)
    df_portfolio['Bollinger Strategy'] = compute_portvals(start_date, end_date, 'orders.txt', start_val)
    df_portfolio['MACD'] = compute_portvals(start_date, end_date, filename, start_val)
    
    plot_normalized_data(df_portfolio)
    

def test_run():
    start_date = '2007-12-31' # Dec 31, 2007 
    end_date = '2009-12-31' # Dec 31, 2009
    build_MACD('IBM', start_date, end_date, 12, 26, 10000)   

if __name__ == "__main__":
    test_run()