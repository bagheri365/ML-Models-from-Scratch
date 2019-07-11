# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Multiple Linear Regression (MLR) Model from Scratch
# Author: Alireza Bagheri
# GitHub: https://github.com/alireza365
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np; np.random.seed(123)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Data_example():
	# ---------------------------------------------------------
    # Data Example
    n_samples = 1000           # Number of data samples
    n_features = 2             # Number of features
    # ---------------------------------------------------------
    x = np.random.uniform(0.0, 1.0, (n_samples, n_features))
    X = np.hstack((np.ones((n_samples, 1)), x)) 
    # ---------------------------------------------------------
    mu, sigma = 0, 2           # Mean and standard deviation
    noise = np.random.normal(mu, sigma, (n_samples, 1))
    # ---------------------------------------------------------
    beta = np.array([30, -10, 70])
    beta = beta.reshape(len(beta), 1)
    # ---------------------------------------------------------
    y = X.dot(beta) + noise    # Actual y
    # ---------------------------------------------------------
    return x, y
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def MLR_with_Analytical_Solution(x, y):
    # ---------------------------------------------------------
    n_samples = np.size(x, axis = 0)
    X = np.hstack((np.ones((n_samples, 1)), x)) 
    
    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    print('-'*50)
    print('MLR with Analytical Solution')
    print('Parameters:\n', beta_hat)
    # ---------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def MLR_with_Gradient_Descent(x, y):
    # Define MLR class
    class Multiple_Linear_Regression:
        # ----------------------------------------------------------
        def __init__(self, lr = 0.1, itr = 20):
                                # Initialization
            self.lr = lr        # Learning rate
            self.itr = itr      # Number of iterations (epochs)
        # ----------------------------------------------------------       
        def predict(self, x = [], flag = False):
            if flag==False:
                self.y_pred = self.X.dot(self.beta)
            else:
                x = np.insert(x,0,1,axis=1)
                self.y_pred = x.dot(self.beta)
            return self.y_pred
        # ----------------------------------------------------------        
        def cost_func(self):
            self.cost = sum((self.y - self.y_pred)**2)/len(self.y)
            return self.cost
        # ----------------------------------------------------------
        def gradient_descent(self):
            self.dbeta = 2/np.size(self.X, axis = 0)*(self.X.T.dot(self.X).dot(self.beta) - self.X.T.dot(self.y))      
        # ----------------------------------------------------------
        def update_params(self): 
            self.beta = self.beta - self.lr * self.dbeta
        # ----------------------------------------------------------
        def fit(self, x, y):       
            self.X = np.insert(x,0,1,axis=1)
            self.y = y
            
            n_samples, n_features = np.shape(x)
        
            # Initialize beta to 0
            self.beta = np.zeros((n_features + 1, 1))
            
            costs = []    
            while(self.itr+1):
                self.predict()
                self.cost_func()
                costs.append(self.cost)
                self.gradient_descent()
                self.update_params()
                self.itr -= 1
            return costs
        # ---------------------------------------------------------
        def R2_Score(self, x, y):
            ss_res = sum((self.predict(x, flag = True) - y)**2)
            ss_tot = sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot
            return r2     
    # -------------------------------------------------------------        
    lr = 0.1        # Learning rate
    n_itr = 400     # Number of epochs
    
    obj = Multiple_Linear_Regression(lr, n_itr)
    J = obj.fit(x, y)
    
    print('-'*50)
    print('MLR with Gradient Descent')
    print('Coefficients: \n', obj.beta)
    
    print('r2 score: \n', obj.R2_Score(x, y))
    # -------------------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.plot(range(len(J)), J,  '-b')
    plt.title('Multivariable Linear Regression')
    plt.xlabel('Iterations')
    plt.ylabel('Cost function')
    plt.grid()
    # -------------------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def MLR_with_Scikit_Learn(x, y):
    # ------------------------------------------------------------- 
    # Model Intialization
    reg = LinearRegression()
    
    # Data Fitting
    reg = reg.fit(x, y)
    
    # Y Prediction
    #Y_pred = reg.predict(x)
    print('-'*50)
    print('MLR with scikit-learn')
    print('Intercept: \n', reg.intercept_)
    print('Coefficients: \n', reg.coef_)
    print('r2 score: \n', reg.score(x, y))
    # ------------------------------------------------------------- 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    # -------------------------------------------------------------------------
    # Generate data
    x, y = Data_example()
    # -------------------------------------------------------------------------
    # MLR with Analytical Solution
    
    MLR_with_Analytical_Solution(x, y)
    # -------------------------------------------------------------------------
    # MLR with Gradient Descent 
    
    MLR_with_Gradient_Descent(x, y)
    # -------------------------------------------------------------------------
    # MLR with Scikit-Learn 
    
    MLR_with_Scikit_Learn(x, y) 
# -----------------------------------------------------------------------------
if __name__ == "__main__": main()
# -----------------------------------------------------------------------------