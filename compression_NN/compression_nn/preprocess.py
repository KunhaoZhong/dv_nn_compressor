import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class ZCAWhitening:
    def __init__(self, epsilon=1e-6):
        """
        Initialize ZCA whitening preprocessor
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.epsilon = epsilon
        self.mean = None
        self.zca_matrix = None
        
    def fit(self, X):
        """
        Compute ZCA whitening matrix using training data
        
        Args:
            X: Input data of shape (n_samples, n_features)
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)
        
        # Compute SVD
        U, S, V = np.linalg.svd(cov)
        
        # Compute ZCA whitening matrix
        self.zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + self.epsilon)), U.T))
        
    def transform(self, X):
        """
        Apply ZCA whitening to input data
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Whitened data as torch.Tensor
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        X_centered = X - self.mean
        X_whitened = np.dot(X_centered, self.zca_matrix.T)
        return torch.from_numpy(X_whitened).float()
    
    def fit_transform(self, X):
        """
        Fit and apply ZCA whitening to input data
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Whitened data as torch.Tensor
        """
        self.fit(X)
        return self.transform(X)

def prepare_data_for_training(train_x, train_y, val_x, val_y, test_x, test_y, batch_size, whitening=True):
    """
    Prepare data for training with ZCA whitening only on training and validation sets
    
    Returns:
        train_loader, val_loader, test_loader, zca_transformer
    """
    if whitening:
        # Initialize ZCA whitening
        zca = ZCAWhitening()
        
        # Ensure all inputs are float32
        train_x = train_x.astype(np.float32)
        val_x = val_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        val_y = val_y.astype(np.float32)
        test_y = test_y.astype(np.float32)

        # Fit and transform training data
        train_x_whitened = zca.fit_transform(train_x)

        # Transform validation data
        val_x_whitened = zca.transform(val_x)

        # Create datasets (test data remains untransformed)
        train_dataset = TensorDataset(train_x_whitened, torch.from_numpy(train_y))
        val_dataset = TensorDataset(val_x_whitened, torch.from_numpy(val_y))
        test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader, zca
    
    else:
        # Ensure all inputs are float32
        train_x = train_x.astype(np.float32)
        val_x = val_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        val_y = val_y.astype(np.float32)
        test_y = test_y.astype(np.float32)

        # Create datasets (test data remains untransformed)
        train_dataset = TensorDataset(train_x, torch.from_numpy(train_y))
        val_dataset = TensorDataset(val_x, torch.from_numpy(val_y))
        test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader, -100
    
    

class MinMaxNormalizer:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
    
    def fit(self, data):
        """Compute min and max values for each feature"""
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
    
    def transform(self, data):
        """Apply min-max normalization"""
        return (data - self.min_vals) / (self.max_vals - self.min_vals)
    
    def inverse_transform(self, normalized_data):
        """Transform normalized data back to original scale"""
        return normalized_data * (self.max_vals - self.min_vals) + self.min_vals
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)