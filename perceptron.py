from tkinter.messagebox import NO
import numpy as np
from random import seed, random

class Perceptron:
    def __init__(self, learning_rate, number_of_inputs, max_epochs, seed = None):
        self.learning_rate = learning_rate
        
        if(seed != None):
            np.random.seed(seed)
            seed(seed)
        
        self.weights = np.random.rand(number_of_inputs)
        self.bias = random()
        print(f"generated initial calibration {self.weights}; {self.bias}")
        self.max_epochs = max_epochs

    def eval(self, inputs):
        return np.sum(inputs * self.weights) + self.bias

    def g(self, r):
        if(r >= 0): return 0
        else: return 1

    def step(self, inputs):
        return self.g(self.eval(inputs))

    def fit(self, training_set):
        print(f'max_epochs={self.max_epochs}')

        training_log = []

        for i in range(self.max_epochs):
            print(f"epoch {i}")

            is_fitted = True
            
            for X, y in training_set:
                #print(f"training item {X} -> {y}")
                y_res = self.step(X)
                #print(f"evaluation {y_res}")

                training_log.append([i, X, y, y_res, self.weights, self.bias])

                # prediction = 1; objective 0
                if y_res > y: 
                    self.weights += (self.learning_rate * X)
                    self.bias += self.bias + self.learning_rate
                    #print(f"new model calibration {self.weights}; {self.bias}")
                    is_fitted = False
                    continue

                # prediction = 0; objective 1
                if y_res < y: 
                    self.weights -= (self.learning_rate * X)
                    self.bias -= self.bias + self.learning_rate
                    #print(f"new model calibration {self.weights}; {self.bias}")
                    is_fitted = False
                    continue
            
            if is_fitted == True:
                break

        return training_log

    def test(self, test_params, test_results):
        accurate_responses = 0

        for X, y in zip(test_params, test_results):
            y_pred = self.step(X)
            if y_pred == y:
                accurate_responses += 1

            print(f"imputs {X} : predicted={y_pred}, actual={y}")