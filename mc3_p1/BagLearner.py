"""
Bootstrap Aggregating Learner.
"""

import numpy as np
import random
import KNNLearner as knn

class BagLearner(object):

    def __init__(self, learner = knn.KNNLearner, kwargs = {"k":3}, bags = 20, boost = False):
        self.type = learner
        self.kwargs = kwargs 
        self.bags = bags
        self.boost = boost
        self.learnerObjs = []

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        weights = np.ones(len(dataX))/len(dataX)
        for index in range(0, self.bags):
            learnerObject = self.type(**self.kwargs)
            new_dataX, new_dataY = self.randomDataGenerator(dataX, dataY)
            learnerObject.addEvidence(new_dataX, new_dataY)
            predY = learnerObject.query(dataX)
            errors = dataY - predY
            
            self.learnerObjs.append(learnerObject)
            

    def randomDataGenerator(self,dataX, dataY):
        new_dataX = []
        new_dataY = []
        for index in range(0, len(dataX)):
            randomIndex = random.randint(0, len(dataX)-1)
            new_dataX.append(dataX[randomIndex])
            new_dataY.append(dataY[randomIndex])
        return new_dataX, new_dataY

        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        point_results = []
        for learner in self.learnerObjs:
            point_results.append(learner.query(points))
        return np.mean(point_results, axis=0)



if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
