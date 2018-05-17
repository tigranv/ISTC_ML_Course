from sklearn.preprocessing import PolynomialFeatures
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
from sklearn.model_selection import KFold
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
import matplotlib.pyplot as plt
import numpy as np

def fit_linear_regression(X, Y, lbd = 0):
	X = np.array(X)
	b = np.dot(X.T, X)
	b += lbd * np.eye(b.shape[0])
	b = np.linalg.inv(b)
	b = np.dot(b, X.T)
	b = np.dot(b, Y)
	return b

def makeX(data_list, degree):
	poly = PolynomialFeatures(degree)
	return poly.fit_transform(data_list)

def fit_polynomial_regression(data_list, response_list, degree=2, lbd = 0):
	return fit_linear_regression(makeX(data_list, degree), response_list, lbd)

def mean_square_loss(X, Y, b):
	X = np.array(X)
	Loss = np.array(Y) - (X.dot(b))
	Loss = Loss.dot(Loss)
	return Loss/X.shape[0]

##############################################################################################

def cross_validation(X, Y, lbd = 0, degree = 1, splits = 5):
	kf = KFold(n_splits = splits)
	ans = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		b = fit_linear_regression(X_train, Y_train, lbd)
		ans += mean_square_loss(X_test, Y_test, b)
	return ans/splits

###############################################################################################




def find_lbd(train_X, train_Y, degree = 5, lbd_range = 0.4, point_num = 50):
	dg = degree
	mn = np.inf
	lbd = 0

	c = point_num
	for i in np.arange(c, dtype = np.float64)*lbd_range/c:
		bb = fit_polynomial_regression(train_X, train_Y, lbd = i, degree = dg)
		yy = makeX(train_X, degree).dot(bb)
		loss = cross_validation(makeX(train_X, degree = 5), train_Y, lbd = i, degree = dg, splits=5)
		if loss < mn:
			mn, lbd = loss, i
	return lbd

def plot_msl_lbt(train_X, train_Y, degree = 5, lbd_range = 0.4, point_num = 50):
	dg = degree
	loss = []

	c = point_num
	for i in np.arange(c, dtype = np.float64)*lbd_range/c:
		bb = fit_polynomial_regression(train_X, train_Y, lbd = i, degree = dg)
		yy = makeX(train_X, degree).dot(bb)
		loss.append( cross_validation(makeX(train_X, degree = 5), train_Y, lbd = i, degree = dg, splits=5) )
	plt.plot(np.arange(c)*lbd_range/c, np.array(loss), color = 'r')
	plt.show()

if __name__ == "__main__":
	b = np.random.rand(makeX(np.ones((1000, 4)), 4).shape[1])
	train_X = np.random.rand(1000, 4)
	train_Y = makeX(train_X, 4).dot(b) + np.random.normal(0, 0.1, 1000)
	print('lambda for min cross_validation = ', find_lbd(train_X, train_Y, degree = 4, lbd_range = 0.4, point_num = 200))
	plot_msl_lbt(train_X, train_Y, degree = 4, lbd_range = 0.4, point_num = 200)
