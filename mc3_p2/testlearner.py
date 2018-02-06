"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl
import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":
    inf = open('Data/ripple.csv')
    #inf = open('Data/simple.csv')
    #inf = open('Data/3_groups.csv')
    #inf = open('Data/best4linreg.csv')
    #inf = open('Data/best4KNN.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # plt.plot(data, 'ro')
    # plt.title("Best Data Set for KNN Learner")
    # plt.savefig('best4KNN.png')
    

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print "Linear Regression Learner"
    # create a learner and train it
    learner = lrl.LinRegLearner() # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    df_pred = pd.DataFrame(testY, columns=['Data'])
    df_pred['Linear'] = predY
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    # plt.scatter(predY, testY)
    # plt.title("How the linear model can predict the data")
    # # ax = df_pred.scatter(title="Best Data Set for KNN Graphing")
    # plt.savefig('test1.png')

    print "\nKNN Learner"
    learner = knn.KNNLearner(k=3)
    learner.addEvidence(trainX, trainY)
#    Y = learner.query(testX)
#    print len(Y)

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
    df_pred['KNN'] = predY

    # plt.scatter(predY, testY)
    # plt.title("How the KNN can predict the Ripple data")
    # # ax = df_pred.scatter(title="Best Data Set for KNN Graphing")
    # plt.savefig('test2.png')

    print "\nBag Learner"
    learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":1}, bags = 25, boost = False)
    learner.addEvidence(trainX, trainY)

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

