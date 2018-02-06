"""MC2_P2: Technical Analysis Strategies"""
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import date

from numpy import convolve


from util import get_data, plot_data 
from marketsim import compute_portvals

myFile = "orders.csv"

def displayMACD(df, symbols, title="MACD Strategy", xlabel="Date", ylabel="Stock Prices"):
    if os.path.exists(myFile):
        os.remove(myFile);
    file = open(myFile, 'w+')

    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    longEnt = False
    shortEnt = False

    index = 1
    currentLong = 0
    currentShort = 0
    isBuy = False
    isSell = False
    file.write("Date,Symbol,Order,Shares\n")
    while index < len(df.index):

        if(df['MACD'][index - 1] < df['Signal'][index - 1] and df['MACD'][index] > df['Signal'][index]):
            plt.axvline(df.index[index], color='g')
            #currentLong += 100
            file.write(str(df.index[index].date())+","+symbols+",BUY," + str(100) +"\n")
            #currentShort = 0
            # isBuy = True
            # isSell = False
            #long_entry = True
            print "BUY"

        if(df['MACD'][index - 1] > df['Signal'][index - 1] and df['MACD'][index] < df['Signal'][index]):
            plt.axvline(df.index[index], color='k')
            file.write(str(df.index[index].date())+","+symbols+",SELL,"+str(currentLong)+"\n")
            currentLong = 0
            # isBuy = False
            # isSell = True
            print "SELL"
        
            
        # if(isSell):
        #     file.write(str(df.index[index].date())+","+symbols+",SELL,"+str(100)+"\n")
        #     currentShort += 100
        #     print "SELL"

        # if(isBuy):
        #     file.write(str(df.index[index].date())+","+symbols+",BUY,"+str(100)+"\n")
        #     currentLong += 100
        #     print "BUY"
        
            #long_entry = False

        index += 1
    print df
    file.close()
    #plt.show()
    plt.savefig('my_strategy.png')

def plot_normalized_data(df, title="Daily portfolio values", xlabel="Date", ylabel="Normalized prices"):
    df = df / df.ix[0,:] 
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('my_strat_backtesting.png')

def createMACD(symbols, start_date, end_date, n_fast, n_slow, start_val):
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([symbols], dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    SPY_prices = prices_all['SPY']


    emaSlow = pd.ewma(prices, span=n_slow, min_periods = n_slow - 1)
    emaFast = pd.ewma(prices, span=n_fast, min_periods = n_slow - 1)

    macdInitial = emaFast - emaSlow

    macdSignal = pd.ewma(macdInitial, span=9, min_periods = 8)
    macdHist  = macdInitial - macdSignal
    macdDF = pd.DataFrame(macdInitial, columns=['MACD'])
    macdDF['Signal'] = macdSignal

    macdDF = macdDF.dropna()

    
    displayMACD(macdDF, symbols)

    df_portfolio = pd.DataFrame(SPY_prices)
    df_portfolio['Bollinger Bounds'] = compute_portvals(start_date, end_date, 'orders.txt', start_val)
    df_portfolio['My Strategy'] = compute_portvals(start_date, end_date, myFile, start_val)
    plot_normalized_data(df_portfolio)
    

def test_run():
    start_date = '2007-12-31' # Dec 31, 2007 
    end_date = '2009-12-31' # Dec 31, 2009
    createMACD('IBM', start_date, end_date, 12, 26, 10000)   

if __name__ == "__main__":
    test_run()