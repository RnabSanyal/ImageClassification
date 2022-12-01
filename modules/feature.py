import numpy as np

class FeatureExtractor:

    # configure feature extractor object
    # functionality: Divide 2D arrays into grids 
    #                and calculate percentage of 
    #                1s in each grid. Can be 
    #                configured to round_off into 
    #                tiles.
    #
    # input:
    # grid_structure: (rows in grid, columns in grid)
    # round_off: no. of decimal places to round off
    #
    # note:
    # grid must divide the image dimensions 


    def __int__(self, grid_structure = (4,4), round_off = False):
            
            self.grid_structure = grid_structure
            self.round_off = round_off


    def extract(self, X):
        
        # performs pooling, calculating percentage for each grid
        # input:
        # X: image matrix containing 1s and 0s
        #    expected shape: (# images, rows, cols)

        no_imgs, img_rows, img_cols = X.shape
        grid_rows, grid_cols = self.grid_structure

        if img_rows % grid_rows or img_cols % grid_cols:
            raise Exception("Grid size incompatible with image dimensions")
        
        # reshape x to run calculations on axes directly
        X_reshaped = \
            X.reshape(no_imgs, img_rows/grid_rows, grid_rows, img_cols/grid_cols, grid_cols)
        
        X_features = X_reshaped.sum(axis = (2,4)) / (grid_rows * grid_cols)

        if self.round_off:
            X_features = np.round(X_features, self.round_off)
        
        return X_features