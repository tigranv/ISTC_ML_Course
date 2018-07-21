class K_NN:
    def __init__(self, k):
        """
        :param k: number of nearest neighbours
        """
        self.k = k

    def fit(self, data):
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """
        # TODO: preprocessing
        pass

    def predict(self, data):
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
        data = np.array(data)
        shp = data.shape
        if len(data.shape) == 1:
            data = data.reshape([1] + list(data.shape))
        # TODO: predict
        prediction = np.array([0])
        return prediction.reshape(shp[:-1])
