from turtle import color
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

LEARNING_RATE = 0.1
EPOCHS = 25

data = np.genfromtxt('data.csv', delimiter=',')
c = np.empty(data[:, 2].shape, dtype=str)
c[data[:, 2] == 0] = 'red'
c[data[:, 2] == 1] = 'blue'
print(c)
plt.scatter(data[:, 0], data[:, 1], color=c)
plt.show()