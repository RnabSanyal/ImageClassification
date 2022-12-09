import numpy as np
from pprint import pprint

class Perceptron:

    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    
    def setWeights(self):
        #print(self.all_labels, self.features)
        self.weights = np.zeros((self.all_labels, self.features + 1)) # bias term

    
    def fit(self, trainingData, trainingLabels):

        self.all_labels = len(np.unique(trainingLabels))
        examples, self.features = trainingData.shape
        self.setWeights()
        
        # adding extra column for bias
        trainingData = np.hstack([np.ones((examples, 1)), trainingData])
        
        for iteration in range(self.max_iterations):

            y_hat = np.argmax(np.matmul(self.weights, trainingData.T), axis=0)

            update = 0
            for i in range(examples):
                
                if trainingLabels[i] != y_hat[i]:
                    self.weights[trainingLabels[i]] += trainingData[i]
                    self.weights[y_hat[i]] -= trainingData[i]
                    update = 1
            
            if update == 0:
                break
    
    def predict(self, X):

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.argmax(np.matmul(self.weights, X.T), axis=0)