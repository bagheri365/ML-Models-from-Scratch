# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Principal Component Analysis (PCA) from Scratch
# Author: Alireza Bagheri
# GitHub: https://github.com/alireza365
# License: MIT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Data_example():
    # ---------------------------------------------------------  
    iris = load_iris()
    X = iris['data']
    #y = iris['target']
    
    n_samples, n_features = X.shape
    
    print('Number of samples:', n_samples)
    print('Number of features:', n_features)
    print('-'*50)    
    # ---------------------------------------------------------
    return X
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components   
    
    def fit(self, X): 
        
        # Standardize data 
        X = X.copy()
        self.mean = np.mean(X, axis = 0)
        self.scale = np.std(X, axis = 0)
        X_std = (X - self.mean) / self.scale
            
        # Eigendecomposition of covariance matrix       
        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat) 
            
        # Adjusting the eigenvectors that are largest in absolute value to be positive    
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs*signs[np.newaxis,:]
        eig_vecs = eig_vecs.T
           
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i,:]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
            
        self.components = eig_vecs_sorted[:self.n_components,:]
            
        # Explained variance ratio
        self.explained_variance_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
            
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)
    
        return self
    
    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
            
        return X_proj 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    # -------------------------------------------------------------------------
    # Generate data
    X = Data_example()
    # -------------------------------------------------------------------------
    # PCA from Scratch
    
    n_components = 2
    my_pca = MyPCA(n_components).fit(X)
    X_proj = my_pca.transform(X) # Apply dimensionality reduction to X
    
    print('Components from scratch:\n', my_pca.components)
    print('Explained variance ratio from scratch:\n', my_pca.explained_variance_ratio)
    print('Cumulative explained variance from scratch:\n', my_pca.cum_explained_variance)
    print('Transformed data shape from scratch:', X_proj.shape)    
    print('-'*50)
    # -------------------------------------------------------------------------
    # PCA from Scikit-learn
    
    X_std = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components).fit(X_std)
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    X_pca = pca.transform(X_std) 
    
    print('Components:\n', pca.components_)
    print('Explained variance ratio:\n', pca.explained_variance_ratio_)
    print('Cumulative explained variance:\n', cum_explained_variance)
    print('Transformed data shape:', X_pca.shape)    
# -----------------------------------------------------------------------------
if __name__ == "__main__": main()