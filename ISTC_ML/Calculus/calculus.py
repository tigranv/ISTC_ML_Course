import math
import sys
from prep_data import generate_data
import matplotlib.pyplot as plt
import numpy as np

def h(area, theta):
    return [theta[1] * area[i] + theta[0] for i in range(len(area))]

area, price = generate_data()

predictions = h(area, [0.5, 7])

plt.clf()
plt.scatter(area, price, color="b", label="housing price data")
plt.plot(area, predictions, color="black")
plt.legend(loc=2)
plt.xlabel("area")
plt.ylabel("price")
plt.show()

def Loss(theta, price, area):
    return (1 / (2 * len(area))) * sum([math.pow(h - y, 2) for h, y in zip(h(area, theta), price)])


loss_list = [Loss([1, i], price, area) for i in np.arange(0.0, 10.0, 0.1)]

plt.clf()
plt.plot(np.arange(0.0, 10.0, 0.1), loss_list, color="black")
plt.legend(loc=2)
plt.xlabel("i")
plt.ylabel("loss")
plt.show()

def grad_zero(theta, price, area):
    return (1 / len(area)) * sum([h - y for h, y in zip(h(area, theta), price)])
def grad_one(theta, price, area):
    return (1 / len(area)) * sum([(h - y) * area[i] for h, y, i in zip(h(area, theta), price, range(len(area)))])

def gradient_descent(th, price, area, a):
    l = Loss(th, price, area)
    count = 0
    list = []

    for i in range(500):
        print(l)
        list.append(l)
        temp0 = th[0] - a * grad_zero(th, price, area)
        temp1 = th[1] - a * grad_one(th, price, area)
        th[0] = temp0
        th[1] = temp1
        
        l = Loss(th, price, area)
    print(th)
    plt.clf()
    plt.plot(np.arange(i+1), list, color="red")
    plt.legend(loc=2)
    plt.xlabel("i")
    plt.ylabel("loss")
    plt.show()


gradient_descent([10, 100], price, area, 0.001)



    