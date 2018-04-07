import math
import sys
from prep_data import generate_data
import matplotlib.pyplot as plt

def f(area, a, b):
    pred = []
    for i in range(len(area)):
        pred.append(a * area[i] + b)

    return pred 

area, price  = generate_data()

predictions =  f(area, 7, 0.5)

plt.clf()
plt.scatter(area, price, color="b", label="housing price data")
plt.plot(area, predictions, color="black")
plt.legend(loc=2)
plt.xlabel("area")
plt.ylabel("price")
plt.show()

print()

loss = sum([math.pow(x - y, 2) for x, y in zip(predictions, price)])

print(loss)

def LossF(a, b, price, area):
    predictions =  f(area, a, b)
    return sum([math.pow(x - y, 2) for x, y in zip(predictions, price)])

print(LossF(1, 1, price, area))

loss_list = []

for i in range(10):
    loss_list.append(LossF(i, 1, price, area))


plt.clf()
plt.plot(range(10), loss_list, color="black")
plt.legend(loc=2)
plt.xlabel("i")
plt.ylabel("loss")
plt.show()

    