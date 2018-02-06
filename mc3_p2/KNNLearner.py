"""
KNN Learner.
"""

import numpy as np
import math

class KNNLearner(object):

    def __init__(self, k = 3):
        self.k_coef = k
        self.dataX = []
        self.dataY = []

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # Add data to X data set
        for dataPoint in dataX:
            self.dataX.append(dataPoint)

        # Add data to Y data set
        for dataPoint in dataY:
            self.dataY.append(dataPoint)

        
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = []
        for point in points:
            distances = []
            for i in range(len(self.dataX)):
                dataPointX = self.dataX[i]
                distance = self.getDistance(point, dataPointX)
                distances.append([i, distance, self.dataY[i]])

            sorted_list = sorted(distances, key=lambda distance_tuple: distance_tuple[1])
            nearest_neighbors = sorted_list[0:self.k_coef]
            result.append(np.mean([neighbor[2] for neighbor in nearest_neighbors]))

        return result
        

    def getDistance(self, dataSet1, dataSet2):
        distance = 0
        for x in range(len(dataSet1)):
            distance += pow((dataSet1[x] - dataSet2[x]), 2)
        return math.sqrt(distance)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
