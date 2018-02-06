"""MC3-P2: Build a Trading Learner"""

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
    # plot_bollinger_data(bollinger_df, symbols, window)
    bb_value = (prices - sma)/(2*sma_std)
    return bb_value

def generate_5_day_returns(prices):
	index = 0
	return_df = pd.zeros(len(prices.index))
	return return_df;


def run():
	symbol = 'IBM'
    start_date = '2007-12-31' # Dec 31, 2007 
    end_date = '2009-12-31' # Dec 31, 2009
    bb_value = build_bollinger_band(symbol, start_date, end_date, 20, 10000)   
    predicted_y = generate_5_day_returns(prices)
    print predicted_y

if __name__ == "__main__":
    run()