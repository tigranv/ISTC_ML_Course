import random
import numpy

def generate_data():

    random.seed(42)
    numpy.random.seed(42)

    y = []
    for ii in range(100):
        y.append( random.randint(20,65) )
    x = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in y]

    return y, x
### need massage list into a 2d numpy array to get it to work in LinearRegression
    #ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    #net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    #from sklearn.model_selection import train_test_split
    #ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    #return ages_train, ages_test, net_worths_train, net_worths_test
