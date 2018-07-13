import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k

    def fit(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        """
        self.dim = data.shape[-1]
        self._initialize_means(data)
        # TODO: Initialize Mixtures, then run EM algorithm until it converges.

    def _initialize_means(self, data):
        # TODO: Initialize cluster centers
        pass

    def predict(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        :return: labels of each datapoint and it's mean
                 0 <= labels[i] <= k - 1
        """
        pass


def KMeansPlusPlus(KMeans):
    def _initialize_means(self, data):
        # TODO: Initialize cluster centers using K Means++ algorithm
        pass
