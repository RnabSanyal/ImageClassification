import numpy as np

class NaiveBayes:

    def __init__(self, k):

        self.k = k
        self.likelihoods = None


    def fit(self, X, Y):

        self.X = X
        self.Y = Y

        # initialize list of priors
        self.y_probs = []
        for i in np.unique(self.Y):

            self.y_probs.append(np.sum(self.Y == i)/len(self.Y))
        
        # creating likelihood matrix
        # force round X to 2
        X = np.round(X,2)
        likelihoods = []

        for i in np.unique(self.Y):

            idx = self.Y == i

            feature_counts = np.zeros((X.shape[1], 101)) # no. of features * possible values (from 0% to 100%)

            # subset of examples corresponding to current label
            X_sub = X[idx]

            columns = X.shape[1]

            # incrementing counts for values of features present
            for feature in range(columns):
                values = (X_sub[:, feature] * 100).astype(int)
                for value in values:
                    feature_counts[feature, value] += 1
            
            # probability with laplace smoothing
            feature_probabilities = (feature_counts + self.k) / (sum(idx) +  (self.k * columns))
            
            likelihoods.append(feature_probabilities)
        
        self.likelihoods = np.array(likelihoods)
        

    def predict(self, X_pred):

        y_hat = []
        X_pred = np.round(X_pred,2)
        for x_i in X_pred:

            probs = []

            for i, y_prob in enumerate(self.y_probs):
                
                likelihood_sum = 0
                for feature, feature_val in enumerate(x_i):
                    try:
                        likelihood_sum += np.log(self.likelihoods[i, feature, int(feature_val*100)])
                    except Exception as e:
                        print(e)
                        print(feature_val)
                        return 
                
                probs.append(np.log(y_prob) + likelihood_sum)
            
            y_hat.append(np.argmax(probs))
        
        return np.array(y_hat)
