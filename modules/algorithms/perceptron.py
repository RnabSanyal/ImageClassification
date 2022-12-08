import util
import os
import numpy as np
from pprint import pprint

class Perceptron:

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = list(trainingData[0].keys())

        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")

            for i in range(len(trainingData)):
                predicted_label = self.classify(trainingData[i])[0]
                if predicted_label != trainingLabels[i]: #if the predcited value isn't same as actual value
                    self.weights[trainingLabels[i]] += trainingData[i]  # increase the weights of actual value
                    self.weights[predicted_label] -= trainingData[i]  # decrease the weights of the predicted value

    def classify(self, data):
        guesses = []
        for info in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * info
            guesses.append(vectors.argMax())
        return guesses