import numpy as np


class GMM:
    def __init__(self, k):
        self.k = k
        self.means = []
        self.covariances = []
        self.pis = []
        self.gammas = []

    def fit(self, data):
        """
        :params data: np.array of shape (..., dim)
                                  where dim is number of dimensions of point
        """
        self._initialize_params(data)
        while True:  # TODO: fix
            self._E_step(data)
            self._M_step(data)

    def _initialize_params(self, data):
        # TODO: initialize means, covariances, pis
        pass

    def _E_step(data):
        # TODO: find gammas from means, covariances, pis
        self.gammas = []

    def _M_step(data):
        # TODO: find means, covariances, pis from gammas
        pass

    def predict(self, data):
        """
        :param data: np.array of shape (..., dim)
        :return: np.array of shape (...) without dims
                         each element is integer from 0 to k-1
        """
        pass

    def get_means(self):
        return self.means.copy()

    def get_covariances(self):
        return self.covariances.copy()

    def get_pis(self):
        return self.pis.copy()
