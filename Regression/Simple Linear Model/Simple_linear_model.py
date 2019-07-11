# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Simple Linear Regression (SLR) Model from Scratch
# Author: Alireza Bagheri
# GitHub: https://github.com/alireza365
# License: MIT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np; np.random.seed(123)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Data_example():	
    # Data Example
    n = 100              # Number of data samples
    x = np.random.uniform(-5.0, 5.0, n)
    
    
    mu, sigma = 0, 2     # Mean and standard deviation
    noise = np.random.normal(mu, sigma, n)
    
    y = 3*x - 2 + noise  # Data y    
    
    return x, y
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def SLR_with_Analytical_Solution(x, y):
    
    def mean(x): # np.mean(x)   
        return sum(x)/len(x)  

    def variance(x): # np.var(x)  
        return sum((i - mean(x))**2 for i in x)/len(x) 
    
    def covariance(x, y): # np.cov(x, y, rowvar=False, bias=True)[0][1]
        return sum((i - mean(x))*(j - mean(y)) for i, j in zip(x, y))/len(x) 
    
    
    a = covariance(x,y)/variance(x)
    b = mean(y) - a*mean(x) 
    # -------------------------------------------------------------------------
    print('-'*50)
    print('SLR with analytical solution\n')
    # Print the coefficients
    print('Coefficient:', a, ', Intercept:', b)
    # -------------------------------------------------------------------------
    # Plot data and the the fitted line
    y_pred = a * x + b
    
    plt.figure(figsize=(8,6))
    plt.plot(x, y, 'ob', label = "Actual")
    plt.plot(x, y_pred, 'r', label = "Predicted")
    plt.ylabel('y', rotation = 0, fontsize = 14)
    plt.xlabel('x', fontsize = 14)
    plt.legend(loc='upper left')
    plt.title('SLR with Analytical Solution')
    plt.grid()
    plt.show()   
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def SLR_with_Gradient_Descent(x, y):
    # Define SLR class
    class Simple_Linear_Regression:
        # ----------------------------------------------------------
        def __init__(self, lr = 0.1, itr = 20):
                                # Initialization
            self.lr = lr        # Learning rate
            self.itr = itr      # Number of iterations (epochs)
            self.a = -8         # Coefficient (Slope)
            self.b = -8         # Intercept
        # ----------------------------------------------------------       
        def predict(self, X = [], flag = False):
            if flag==False:
                self.y_pred = self.a*self.x + self.b
            else:
                self.y_pred = self.a*X + self.b
            return self.y_pred
        # ----------------------------------------------------------        
        def cost_func(self):
            self.cost = sum((self.y - self.y_pred)**2)/len(self.y)
            return self.cost
        # ----------------------------------------------------------
        def gradient_descent(self):
            self.da, self.db = 0, 0
            for x_, y_ in zip(self.x, self.y):
                e = y_ - (self.a*x_ + self.b)
                self.da += -2/len(self.x)*x_*e
                self.db += -2/len(self.x)*e
        # ----------------------------------------------------------
        def update_params(self): 
            self.a = self.a - self.lr * self.da
            self.b = self.b - self.lr * self.db
            return self.a, self.b
        # ----------------------------------------------------------
        def fit(self, x, y):     
            self.x = x
            self.y = y
            
            costs = []
            params = []
            
            while(self.itr+1):
                self.predict()
                self.cost_func()
                costs.append(self.cost)
                
                self.gradient_descent()
                params.append([self.a, self.b])
                self.update_params()
                self.itr -= 1
            return costs, params
        # ---------------------------------------------------------
        def R2_Score(self, x, y):
            ss_res = sum((self.predict(x, flag = True) - y)**2)
            ss_tot = sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot
            return r2     
    # -------------------------------------------------------------------------
    lr = 0.1   # Learning rate
    n_itr = 20 # Number of iterations (epochs)
    
    obj = Simple_Linear_Regression(lr, n_itr)
    J, params = obj.fit(x, y)
    print('-'*50)
    print('SLR with Gradient Descent')
    print('Coefficient:', obj.a, ', Intercept:', obj.b)
    print('r2 score:', obj.R2_Score(x, y))
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(range(len(J)), J,  '-xb')
    plt.title('SLR with Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost function')
    plt.xticks(range(0,n_itr+1,2))
    plt.grid()
    
    y_pred = obj.a* x + obj.b
    #y_pred = obj.predict(x, True)
    
    plt.subplot(1,2,2)
    plt.plot(x, y, 'ob', label = "Actual")
    plt.plot(x, y_pred, 'r', label = "Predicted")
    plt.ylabel('y', fontsize= 14, rotation=0)
    plt.xlabel('x', fontsize= 14)
    plt.legend(loc='best')
    plt.title('SLR with Gradient Descent')
    plt.grid()
    plt.show() 
    # -------------------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def SLR_with_Scikit_Learn(x, y):
    # Model Intialization
    reg = LinearRegression()
    
    # Data Fitting
    x = x.reshape(-1, 1)
    reg = reg.fit(x, y)
    
    # Y Prediction
    # Y_pred = reg.predict(x)
    print('-'*50)
    print('SLR with scikit-learn')
    print('Coefficient:', reg.coef_, ', Intercept:', reg.intercept_)
    print('r2 score:', reg.score(x, y))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    # -------------------------------------------------------------------------
    # Generate data
    x, y = Data_example()
    # -------------------------------------------------------------------------
    # SLR with Analytical Solution

    SLR_with_Analytical_Solution(x, y)
    # -------------------------------------------------------------------------
    # SLR with Gradient Descent 
    
    SLR_with_Gradient_Descent(x, y)
    # -------------------------------------------------------------------------
    # SLR with Scikit-Learn 
    
    SLR_with_Scikit_Learn(x, y) 
# -----------------------------------------------------------------------------
if __name__ == "__main__": main()
# -----------------------------------------------------------------------------
