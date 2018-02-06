import numpy as np


class KNNLearner(object):

    # Constructor
    def __init__(self, k=3):
        self.k = k
        self.Xtrain = []
        self.Ytrain = []

    # Add Data Point to Set
    def addEvidence(self, Xtrain, Ytrain):
        for dataPoint in Xtrain:
            self.Xtrain.append(dataPoint)
        for dataPoint in Ytrain:
            self.Ytrain.append(dataPoint)

    # Perform Query
    def query(self, Xtest):
        result = []

        # for each query point, produce an estimate
        for query_point in Xtest:
            euclidean_distance_tuples = []
            # for each data point in set, produce tuple of euclidean distance
            for i in range(self.Xtrain.__len__()):
                evidence_point = self.Xtrain[i]

                euclidean_distance = self.calculate_euclidean_distance(query_point, evidence_point)

                # append tuple [Index, Distance, Y]
                euclidean_distance_tuples.append([i, euclidean_distance, self.Ytrain[i]])

            # then sort the tuples by euclidean distance
            sorted_list = sorted(euclidean_distance_tuples, key=lambda distance_tuple: distance_tuple[1])

            # get the closest "k" points
            nearest_neighbors = sorted_list[0:self.k]

            # take their mean and append to result array
            result.append(np.mean([neighbor[2] for neighbor in nearest_neighbors]))

        return result

    # HELPER FUNCTIONS
    def calculate_euclidean_distance(self, point1, point2):
        distance_squared = 0
        for i in range(point1.__len__()):
            distance_squared += (point1[i] - point2[i]) ** 2
        return distance_squared ** 0.5


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
