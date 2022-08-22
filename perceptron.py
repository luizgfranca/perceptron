from tkinter.messagebox import NO
import numpy as np
from random import seed, random

class Perceptron:
    def __init__(self, learning_rate, number_of_inputs, max_epochs, loss_function = None, seed = None):
        self.learning_rate = learning_rate
        
        if(seed != None):
            np.random.seed(seed)
            seed(seed)
        
        self.weights = np.random.rand(number_of_inputs)
        self.bias = random()
        print(f"generated initial calibration {self.weights}; {self.bias}")
        self.max_epochs = max_epochs
        self.loss_function = self.g if loss_function == None else loss_function

    def eval(self, inputs):
        return np.sum(inputs * self.weights) + self.bias

    def g(self, r):
        if(r >= 0): return 0
        else: return 1

    def step(self, inputs):
        return self.loss_function(self.eval(inputs))

    def fit(self, training_set):
        print(f'max_epochs={self.max_epochs}')

        training_log = []

        for i in range(self.max_epochs):
            print(f"epoch {i}")
            not_fitted = 0
            is_fitted = True
            
            for X, y in training_set:
                #print(f"training item {X} -> {y}")
                y_res = self.step(X)
                #print(f"evaluation {y_res}")

                training_log.append(np.array([i, X, y, y_res, self.weights, self.bias]))

                # prediction = 1; objective 0
                if y_res > y: 
                    self.weights += (self.learning_rate * X)
                    self.bias += self.bias + self.learning_rate
                    #print(f"new model calibration {self.weights}; {self.bias}")
                    is_fitted = False
                    not_fitted += 1
                    continue

                # prediction = 0; objective 1
                if y_res < y: 
                    self.weights -= (self.learning_rate * X)
                    self.bias -= self.bias + self.learning_rate
                    #print(f"new model calibration {self.weights}; {self.bias}")
                    is_fitted = False
                    not_fitted += 1
                    continue
            
            print(f'not fitted: {not_fitted}')
            
            if is_fitted == True:
                break

        return training_log

    def test(self, test_params, test_results, verbose=False):
        accurate_responses = 0
        mean_error = self.step(test_params[0]) - test_results[0]
        
        for X, y in zip(test_params, test_results):
            y_pred = self.step(X)

            if(verbose):
                print(f"inputs {X} : predicted={y_pred}, actual={y}")
                print(f'curr mean error: {mean_error}; curr error: {abs(y_pred - y)}')

            mean_error = ( mean_error + abs(y_pred - y) ) / 2

        print(f'mean error: {mean_error}')