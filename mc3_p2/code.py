"""MC3-P2: Build a Trading Learner"""

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

# import from the previous projects
import LinRegLearner as lrl
from util import get_data, plot_data 
from marketsim import compute_portvals
from portfolio.analysis import get_portfolio_value, get_portfolio_stats


def run():

	symbol = 'ML4T-399'
	start_date = '2007-12-31' # Dec 31, 2007 
	end_date = '2009-12-31' # Dec 31, 2009
	order_file = "orders.txt"
	test_start_date = '2009-12-31' # Dec 31, 2009
	test_end_date = '2010-12-31' # Dec 31, 2010
	window_size = 12
	starting_value = 10000
	
	dates = pd.date_range(start_date, end_date)
	prices_all = get_data([symbol], dates)  # automatically adds SPY
	prices = prices_all[symbol]  # only portfolio symbols

	train_bb_value = build_bollinger_band(symbol, start_date, end_date, window_size)   
	df_train = pd.DataFrame(train_bb_value['BB'])
	df_train['Prices'] = pd.DataFrame(prices)
	df_train['Momentum'] = build_momentum(symbol, start_date, end_date, window_size)
	df_train['Volatility'] = build_volatility(symbol, start_date, end_date, window_size)
	trainX = np.array(df_train[['BB', 'Momentum', 'Volatility']])[window_size:-5]
	
	test_bb_value = build_bollinger_band(symbol, test_start_date, test_end_date, window_size)  
	df_test = pd.DataFrame(test_bb_value['BB']) 
	df_test['Momentum'] = build_momentum(symbol, test_start_date, test_end_date, window_size)
	df_test['Volatility'] = build_volatility(symbol, test_start_date, test_end_date, window_size)
	testX = np.array(df_test[['BB', 'Momentum', 'Volatility']])[window_size:-5]

	df_train['5-days-return'] = generate_5_day_returns(symbol, start_date, end_date)
	trainY = np.array(df_train['5-days-return'])[window_size:-5]

	learner = lrl.LinRegLearner()
	learner.addEvidence(trainX, trainY)
	predY = learner.query(trainX)
	
	dates = df_train.index[window_size:-5]
	print "ML4T In Sample Testing"
	print
	plot_predicted_data(df_train["Prices"], dates, trainY, predY, 'ML4T_TrainingChart.png','Trading Y/ Predicted Y')
	create_policy(symbol, df_train["Prices"], dates, trainY, predY, 'ml4t_in_sample_policy.txt', window_size, "ml4t-in-sample-order.png", "ML4T In Sample Data With Entries/Exit")
	apply_market_simmulator(symbol, start_date, end_date, 'ml4t_in_sample_policy.txt', starting_value, 'ml4t-in-sample-backtest.png', 'ML4T In Sample Backtest')

	outSamplePredY = learner.query(testX)
	testingDates = df_test.index[window_size:-5]
	print
	print "ML4T Out Sample Testing"
	print
	create_policy_out_sample(symbol, testingDates, outSamplePredY, 'ml4t_out_sample_policy.txt', window_size, "ml4t-out-sample-order.png", "ML4T Out Sample Data With Entries/Exit")
	apply_market_simmulator(symbol, test_start_date, test_end_date, 'ml4t_out_sample_policy.txt', starting_value, 'ml4t-out-sample-backtest.png', 'ML4T Out Sample Backtest')

	symbol = 'IBM'
	
	dates = pd.date_range(start_date, end_date)
	prices_all = get_data([symbol], dates)  # automatically adds SPY
	prices = prices_all[symbol]  # only portfolio symbols

	train_bb_value = build_bollinger_band(symbol, start_date, end_date, window_size)   
	df_train = pd.DataFrame(train_bb_value['BB'])
	df_train['Prices'] = pd.DataFrame(prices)
	df_train['Momentum'] = build_momentum(symbol, start_date, end_date, window_size)
	df_train['Volatility'] = build_volatility(symbol, start_date, end_date, window_size)
	trainX = np.array(df_train[['BB', 'Momentum', 'Volatility']])[window_size:-5]
	
	test_bb_value = build_bollinger_band(symbol, test_start_date, test_end_date, window_size)  
	df_test = pd.DataFrame(test_bb_value['BB']) 
	df_test['Momentum'] = build_momentum(symbol, test_start_date, test_end_date, window_size)
	df_test['Volatility'] = build_volatility(symbol, test_start_date, test_end_date, window_size)
	testX = np.array(df_test[['BB', 'Momentum', 'Volatility']])[window_size:-5]

	df_train['5-days-return'] = generate_5_day_returns(symbol, start_date, end_date)
	trainY = np.array(df_train['5-days-return'])[window_size:-5]

	learner = lrl.LinRegLearner()
	learner.addEvidence(trainX, trainY)
	predY = learner.query(trainX)
	
	dates = df_train.index[window_size:-5]
	print "======================================================"
	print "IBM In Sample Testing"
	print
	create_policy(symbol, df_train["Prices"], dates, trainY, predY, 'ibm_in_sample_policy.txt', window_size, "ibm-in-sample-order.png", "IBM In Sample Data With Entries/Exit")
	apply_market_simmulator(symbol, start_date, end_date, 'ibm_in_sample_policy.txt', starting_value, 'ibm-in-sample-backtest.png', 'IBM In Sample Backtest')

	outSamplePredY = learner.query(testX)
	testingDates = df_test.index[window_size:-5]
	
	print
	print "IBM Out Sample Testing"
	print
	create_policy_out_sample(symbol, testingDates, outSamplePredY, 'ibm_out_sample_policy.txt', window_size, "ibm-out-sample-order.png", "IBM Out Sample Data With Entries/Exit")
	apply_market_simmulator(symbol, test_start_date, test_end_date, 'ibm_out_sample_policy.txt', starting_value, 'ibm-out-sample-backtest.png', 'IBM Out Sample Backtest')

def create_policy_out_sample(symbol, dates, predY, filename, window, chartFileName="apply_policy.png", title="Training Data with Entries/Exits", xlabel="Date", ylabel="Value"):
	if os.path.exists(filename):
		os.remove(filename);
	file = open(filename, 'w+')
	file.write("Date,Symbol,Order,Shares\n")
	
	df = pd.DataFrame(predY, index=dates, columns=['Predicted Y'])
	ax = df.plot(title = title, linewidth=2)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	longPosition = False
	currentStockNumber = 0
	dayOnPosition = 0
	shortPosition = False
	t = 0	
	while t < (len(predY)):
		if(predY[t] > 0.01):
			if(shortPosition):
				shortPosition = False
				plt.axvline(df.index[t], color='k')
				buyingStockNumb = -currentStockNumber
				currentStockNumb = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,"+str(buyingStockNumb)+"\n") # Buy 100 stocks	
				dayOnPosition = 0
			else:
				if(not longPosition):
					longPosition = True
					plt.axvline(df.index[t], color='g')
				dayOnPosition = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,100\n") # Buy 100 stocks
				currentStockNumber += 100

		elif(predY[t] < -0.01):
			if(longPosition):
				longPosition = False
				plt.axvline(df.index[t], color='k')
				dayOnPosition = 0
				sellingStockNumb = currentStockNumber
				currentStockNumb = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,"+str(sellingStockNumb)+"\n") # Buy 100 stocks	
			else:
				if(not shortPosition):
					shortPosition = True
					plt.axvline(df.index[t], color='r')
				dayOnPosition = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,100\n") # Buy 100 stocks
				currentStockNumber -= 100
		else:
			if(dayOnPosition > 5):
				if(shortPosition):
					shortPosition = False
					plt.axvline(df.index[t], color='k')
					buyingStockNumb = -currentStockNumber
					currentStockNumb = 0
					file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,"+str(buyingStockNumb)+"\n") # Buy 100 stocks	
					dayOnPosition = 0
				elif(longPosition):
					longPosition = False
					plt.axvline(df.index[t], color='k')
					dayOnPosition = 0
					sellingStockNumb = currentStockNumber
					currentStockNumb = 0
					file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,"+str(sellingStockNumb)+"\n") # Sell 100 stocks	
			else:
				if(shortPosition or longPosition):
					dayOnPosition += 1

		if(longPosition or shortPosition):
			dayOnPosition += 1

		t += 1
	file.close()
	plt.savefig(chartFileName)


def create_policy(symbol, prices, dates, trainY, predY, filename, window, chartFileName="apply_policy.png", title="Training Data with Entries/Exits", xlabel="Date", ylabel="Value"):
	if os.path.exists(filename):
		os.remove(filename);
	file = open(filename, 'w+')
	file.write("Date,Symbol,Order,Shares\n")
	
	df = pd.DataFrame(prices)
	df['Training Y'] = pd.DataFrame(trainY, index=dates)
	df['Predicted Y'] = pd.DataFrame(predY, index=dates)
	ax = df.plot(title = title, linewidth=2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	longPosition = False
	currentStockNumber = 0
	dayOnPosition = 0
	shortPosition = False
	t = 0	
	while t < (len(predY)):
		if(predY[t] > 0.01):
			if(shortPosition):
				shortPosition = False
				plt.axvline(df.index[t], color='k')
				buyingStockNumb = -currentStockNumber
				currentStockNumb = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,"+str(buyingStockNumb)+"\n") # Buy 100 stocks	
				dayOnPosition = 0
			else:
				if(not longPosition):
					longPosition = True
					plt.axvline(df.index[t], color='g')
				dayOnPosition = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,100\n") # Buy 100 stocks
				currentStockNumber += 100

		elif(predY[t] < -0.01):
			if(longPosition):
				longPosition = False
				plt.axvline(df.index[t], color='k')
				dayOnPosition = 0
				sellingStockNumb = currentStockNumber
				currentStockNumb = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,"+str(sellingStockNumb)+"\n") # Buy 100 stocks	
			else:
				if(not shortPosition):
					shortPosition = True
					plt.axvline(df.index[t], color='r')
				dayOnPosition = 0
				file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,100\n") # Buy 100 stocks
				currentStockNumber -= 100
		else:
			if(dayOnPosition > 5):
				if(shortPosition):
					shortPosition = False
					plt.axvline(df.index[t], color='k')
					buyingStockNumb = -currentStockNumber
					currentStockNumb = 0
					file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,"+str(buyingStockNumb)+"\n") # Buy 100 stocks	
					dayOnPosition = 0
				elif(longPosition):
					longPosition = False
					plt.axvline(df.index[t], color='k')
					dayOnPosition = 0
					sellingStockNumb = currentStockNumber
					currentStockNumb = 0
					file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,"+str(sellingStockNumb)+"\n") # Sell 100 stocks	
			else:
				if(shortPosition or longPosition):
					dayOnPosition += 1

		# if(predY[t] > 0.01):
		# 	file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,100\n") # Buy 100 stocks
		# elif(predY[t] < 0.01):
		# 	file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,100\n")

		# if(predY[t] > 0.01 and not longPosition):
		# 	longPosition = True
		# 	file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,100\n") # Buy 100 stocks
		# elif(predY[t] < -0.01 and not shortPosition):
		# 	shortPosition = True
		# 	file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,100\n")
		# elif(shortPosition and dayOnPosition > 5):
		# 	shortPosition = False
		# 	file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",BUY,100\n")
		# 	dayOnPosition = 0
		# elif(longPosition and dayOnPosition > 5):
		# 	longPosition = False
		# 	file.write(dates[t].strftime('%Y-%m-%d')+","+symbol+",SELL,100\n")
		# 	dayOnPosition = 0

		if(longPosition or shortPosition):
			dayOnPosition += 1

		t += 1
	file.close()
	plt.savefig(chartFileName)

def plot_predicted_data(prices, dates, trainY, predY, fileName, title, xlabel="Date", ylabel="Value"):
	
	df = pd.DataFrame(prices)
	df['Training Y'] = pd.DataFrame(trainY, index=dates)
	df['Predicted Y'] = pd.DataFrame(predY, index=dates)
	ax = df.plot(title = title, linewidth=2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
	plt.xlim(dates[0], dates[len(dates)/2])
	plt.ylim(0, max(trainY)+1/2)
	plt.savefig(fileName)

def plot_normalized_data(df, chartName='backtesting.png', title="Data Out Sample Backtest", xlabel="Date", ylabel="Prices"):
    df = df / df.ix[0,:] 
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(chartName)

def apply_market_simmulator(symbol, start_date, end_date, filename, start_val, chartName='backtesting.png', title="Data Out Sample Backtest"):
	# Process orders
	dates = pd.date_range(start_date, end_date)
	prices_all = get_data([symbol], dates)  # automatically adds SPY
	prices = prices_all[symbol]  # only portfolio symbols
	SPY_prices = prices_all['SPY']
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
	plot_normalized_data(df_portfolio, chartName, title)


def build_bollinger_band(symbols, start_date, end_date, window):
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([symbols], dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    sma = pd.rolling_mean(prices, window)
    sma_std = pd.rolling_std(prices, window)
    bb_value = pd.DataFrame(prices)
    bb_value['BB'] = (prices - sma)/(2*sma_std)
    return bb_value

def build_momentum(symbol, start_date, end_date, N):
	dates = pd.date_range(start_date, end_date)
	prices_all = get_data([symbol], dates)  # automatically adds SPY
	prices = prices_all[symbol]  # only portfolio symbols
	momentum = (prices-prices.shift(N))/prices.shift(N)

	return momentum

def build_volatility(symbol, start_date, end_date, window):
	dates = pd.date_range(start_date, end_date)
	prices_all = get_data([symbol], dates)  # automatically adds SPY
	prices = prices_all[symbol]  # only portfolio symbols
	daily_return = (prices/prices.shift(1)) - 1
	daily_return_std = pd.rolling_std(daily_return, window)
	return daily_return_std
    

def generate_5_day_returns(symbol, start_date, end_date):
	dates = pd.date_range(start_date, end_date)
	prices_all = get_data([symbol], dates)
	prices = prices_all[symbol]  # only portfolio symbols
	return_df = (prices.shift(5) - prices)/prices
	return return_df


if __name__ == "__main__":
    run()