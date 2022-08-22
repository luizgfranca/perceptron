import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import loss

from perceptron import Perceptron

def draw_decision_surface(weights, bias, points):
    x = [np.min(points[:, 0] - 1, np.max(points[:, 0]) + 1)]
    y = [np.min(points[:, 1] - 1, np.max(points[:, 1]) + 1)]

    # (y - y0) = m (x - x0)

    

LEARNING_RATE = 0.01
EPOCHS = 250

data = np.genfromtxt('data.csv', delimiter=',')
c = np.empty(data[:, 2].shape, dtype=str)
c[data[:, 2] == 0] = 'red'
c[data[:, 2] == 1] = 'blue'
print(c)
plt.scatter(data[:, 0], data[:, 1], color=c)
#plt.show()

perceptron = Perceptron(
    LEARNING_RATE, 
    data.shape[1] - 1, 
    EPOCHS
)

print()

training_set = list(zip(data[:,np.array([True, True, False])], data[:, 2]))

print()

log = perceptron.fit(training_set)
log_g = np.array(log)

softmax_perceptron = Perceptron(
    LEARNING_RATE, 
    data.shape[1] - 1, 
    EPOCHS,
    loss.sigmoid
)

print()

inputs = data[:,np.array([True, True, False])]
training_set = list(zip(inputs, data[:, 2]))

print()

log = softmax_perceptron.fit(training_set)
log_softmax = np.array(log)
for i in log:
    print(i)

softmax_perceptron.test(inputs, log_softmax[:, 3])
perceptron.test(inputs, log_g[:, 3])