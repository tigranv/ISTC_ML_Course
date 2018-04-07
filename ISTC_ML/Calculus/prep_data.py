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
