import os
import numpy as np
from pprint import pprint

class Executor:


    def __init__(self, data_source = 'facedata', featureExtractor=None, algorithm=None, metric = 'acc'):

        self.data_source = data_source
        
        if metric in ['acc', 'err']:
            self.metric = metric
        else:
            raise Exception("Invalid Metric")

        self.featureExtractor = featureExtractor

        if algorithm:
            self.algorithm = algorithm
        
        self.prepareData()
        self.execute()


    def prepareData(self):

        # read data files and convert to numpy arrays
        # return 3 raw data sets: train, validation, test
        # images dim: 28 * 28; faces dim: 70 * 60

        if self.data_source == 'facedata':
            rows = 70
        elif self.data_source == 'digitdata':
            rows = 28

        # called from the root directory
        path = os.path.join('./data', self.data_source)

        for file_name in os.listdir(path):

            file_path = os.path.join(path, file_name)
            if 'train' in file_name:
                if 'label' in file_name:
                    self.Y_train = self.parseLabels(file_path)
                else:
                    self.X_train = self.parseData(file_path, rows)
            elif 'test' in file_name:
                if 'label' in file_name:
                    self.Y_test = self.parseLabels(file_path)
                else:
                    self.X_test = self.parseData(file_path, rows)
            elif 'validation' in file_name:
                if 'label' in file_name:
                    self.Y_val = self.parseLabels(file_path)
                else:
                    self.X_val = self.parseData(file_path, rows)
        
        if self.featureExtractor:
            
            self.X_train = self.featureExtractor.extract(self.X_train)
            self.X_val = self.featureExtractor.extract(self.X_val)
            self.X_test = self.featureExtractor.extract(self.X_test)
        
        # flatten data arrays for training
        self.X_train = Executor.flatten(self.X_train)
        self.X_val = Executor.flatten(self.X_val)
        self.X_test = Executor.flatten(self.X_test)


    def parseData(self, file_path, rows):
        
        data_set = []
        file_handle = open(file_path)
        lines = file_handle.readlines()

        image = []

        for (i, line) in enumerate(lines):
            
            image.append([0 if x == ' ' else 1 for x in line[:-1]])
            if (i+1)%rows == 0: #number of rows for image reached
                data_set.append(image)
                image = []
        
        return np.array(data_set)


    def parseLabels(self, file_path):

        file_handle = open(file_path)
        lines = file_handle.readlines()
        return np.array([int(line[:-1]) for line in lines])


    def execute(self):

        # Run training with different subsets of the data, 10%, 20%, ..., 100%
        # Randomly sample data for each subset 5 times 
        # calcualte mean and std of accuracy on train, validation and test

        if not self.algorithm:
            raise Exception("Learning Algorithm not provided")

        no_imgs, _ = self.X_train.shape

        self.train_mean_met = []
        self.train_std_met = []
        self.val_mean_met = []
        self.val_std_met = []
        self.test_mean_met = []
        self.test_std_met = []

        increment = int(no_imgs/10)

        for i in range(1,11):

            idx = np.random.randint(0, no_imgs, size=(increment*i))
            train_met = []
            val_met = []
            test_met = []

            for i in range(5):

                X = self.X_train[idx, :]
                Y = self.Y_train[idx]

                self.algorithm.fit(X, Y)

                train_met.append(Executor.calculate_metric(self.algorithm, X, Y, self.metric))
                val_met.append(Executor.calculate_metric(self.algorithm, self.X_val, self.Y_val, self.metric))
                test_met.append(Executor.calculate_metric(self.algorithm, self.X_test, self.Y_test, self.metric))
            
            self.train_mean_met.append(np.mean(train_met))
            self.train_std_met.append(np.std(train_met))

            self.val_mean_met.append(np.mean(val_met))
            self.val_std_met.append(np.std(val_met))

            self.test_mean_met.append(np.mean(test_met))
            self.test_std_met.append(np.std(test_met))


    @staticmethod
    def calculate_metric(algorithm, X, Y_true, metric = 'acc'):
        
        Y_pred = algorithm.predict(X)

        acc = np.sum(Y_pred == Y_true)/len(Y_pred)

        if metric == 'acc':
            return acc
        elif metric == 'err':
            return 1 - acc


    @staticmethod
    def flatten(X):

        # return flattened 2D matrix
        no_imgs, img_rows, img_cols = X.shape
        return X.reshape(no_imgs, img_rows * img_cols)