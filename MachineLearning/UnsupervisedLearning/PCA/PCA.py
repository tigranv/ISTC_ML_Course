import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k
        self.location_ = None
        self.matrix_ = None

    def fit(self, data):
        """
        finds best params for X = Mu + A * Lambda
        :param data: data of shape (number of samples, number of features)
        HINT! use SVD
        """
        pass

    def transform(self, data):
        """
        for given data returns Lambdas
        x_i = mu + A dot lambda_i
        where mu is location_, A is matrix_ and lambdas are projection of x_i
        on linear space from A's rows as basis
        :param data: data of shape (number of samples, number of features)
        """
        # Lemma: x is vector and A dot A.T == I, then x's coordinates in Linear Space(A's rows as basis)
        # is A dot x
        return np.ones((len(data), self.k))
