import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from perceptron import Perceptron

def draw_decision_surface(weights, bias, points):
    x = [np.min(points[:, 0] - 1, np.max(points[:, 0]) + 1)]
    y = [np.min(points[:, 1] - 1, np.max(points[:, 1]) + 1)]

    

LEARNING_RATE = 0.01
EPOCHS = 25

data = np.genfromtxt('data.csv', delimiter=',')
c = np.empty(data[:, 2].shape, dtype=str)
c[data[:, 2] == 0] = 'red'
c[data[:, 2] == 1] = 'blue'
print(c)
plt.scatter(data[:, 0], data[:, 1], color=c)
plt.show()



# perceptron = Perceptron(
#     LEARNING_RATE, 
#     data.shape[1] - 1, 
#     EPOCHS
# )

# print()

# training_set = list(zip(data[:,np.array([True, True, False])], data[:, 2]))

# print()

# log = perceptron.fit(training_set)
# for i in log:
#     print(i)