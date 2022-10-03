import numpy as np


class MyLogisticRegression:
        
    def __init__(self):
        self.theta = None
    

    def add_bias(self, X):
        # Create a vector of size |X| (= number of samples) with all values being 1
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        # Return new data matrix with the 1-vector stack "in front of" X
        return np.hstack([ones, X])

    
    def calc_loss(self, y, y_pred):
        # Calculate and return the Cross Entropy loss (binary classification)
        return (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()

    
    def calc_h(self, X):
        
        h = None
        
        #########################################################################################
        ### Your code starts here ###############################################################        
        
        h = 1 / (1 + np.exp(-(np.dot(X, self.theta.reshape(-1,1)))))
        
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return h
        

    def calc_gradient(self, X, y, h):
        
        grad = None
        
        #########################################################################################
        ### Your code starts here ###############################################################
        
        grad = (2/X.shape[0]) * np.dot(X.T, (h - np.reshape(y,(len(y),1)))).reshape(-1, )
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return grad

    
    
    
    def fit(self, X, y, lr=0.001, num_iter=100, verbose=False):

        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        # weights initialization
        self.theta = np.random.rand(X.shape[1])

        for i in range(num_iter):

            #########################################################################################
            ### Your code starts here ###############################################################      

            h = self.calc_h(X)
            self.theta = self.theta - lr * self.calc_gradient(
                X, y, h
            )
            
            
            ### Your code ends here #################################################################
            #########################################################################################        
            
            # Print loss every 10% of the iterations
            if verbose == True:
                if(i % (num_iter/10) == 0):
                    print('Loss: {:.6f} \t {:.0f}%'.format(self.calc_loss(y, h), (i / (num_iter/100))))

        # Print final loss
        if verbose == True:
            print('Loss: {:.6f} \t 100%'.format(self.calc_loss(y, h)))
    
        return self
    
    
    def predict(self, X, threshold=0.5):
        
        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        y_pred = None

        #########################################################################################
        ### Your code starts here ###############################################################
        
        y_hat = self.calc_h(X)
    
        # if y_hat < 0.5 --> round up to 1
        y_pred = [1 if i > 0.5 else 0 for i in y_hat]
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return y_pred