from tkinter.messagebox import NO
import numpy as np
from random import seed, random

class Perceptron:
    def __init__(self, learning_rate, number_of_inputs, seed = None):
        self.learning_rate = learning_rate
        
        if(seed != None):
            np.random.seed(seed)
            seed(seed)
        
        self.weights = np.random.rand(number_of_inputs)
        self.bias = random()


    