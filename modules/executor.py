import os
import numpy as np
from pprint import pprint

class Executor:


    def __init__(self, data_source = 'facedata', feautureExtractor=None):

        self.data_source = data_source
        self.prepareData()


    def prepareData(self):

        # read data files and convert to numpy arrays
        # return 3 raw data sets: train, validation, test
        # images dim: 28 * 28; faces dim: 70 & 60

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