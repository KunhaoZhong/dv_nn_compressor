import torch
import pickle
import os
from copy import deepcopy
from .preprocess import ZCAWhitening
from .nn_modules import RegressionNN, RegressionNN_ResNet, RegressionNN_Transformer, RegressionTrainer

class CompressionNN:
    def __init__(self, input_dim, output_dim, test_id=None, hidden_dims=[512, 256, 128, 64], 
                 dropout_rate=0.2, activation=torch.nn.ReLU(), output_act=None, arch_name='MLP', bn_affine=True):
        
        if arch_name =='MLP':
            self.model = RegressionNN(input_dim, output_dim, hidden_dims, 
                                    dropout_rate, activation, output_act, bn_affine)
            
        elif arch_name =='ResMLP':
            self.model = RegressionNN_ResNet(input_dim, output_dim, hidden_dims, 
                                    dropout_rate, activation, output_act, bn_affine)
        elif arch_name =='Transformer':
            print('Transformers Architectures: (note the difference with MLP in hidden-dims)')
            d_model=hidden_dims[0]
            num_heads=hidden_dims[1]
            num_layers=hidden_dims[2]
            d_ff=hidden_dims[3]
            print('d_model (Model dimension) = ', d_model)
            print('num_heads (Number of attention heads) = ', num_heads)
            print('num_layers (Number of transformer blocks) = ', num_layers)
            print('d_ff (Feed-forward network dimension) = ', d_ff)
            self.model = RegressionNN_Transformer(input_dim, output_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, dropout_rate=dropout_rate, activation=activation, output_act=output_act)
        else:
            raise NotImplementedError("NN-model not implemented")
        self.zca = ZCAWhitening()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_id = test_id or 'default'
        self.output_act = output_act
        self.arch_name = arch_name
        self.bn_affine = bn_affine
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
    def get_model_config(self):
        """Safely extract model configuration"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'output_act': self.output_act,
            'arch_name': self.arch_name,
            'bn_affine': self.bn_affine,
        }
        
    def save(self, path=None):
        """Save model and preprocessing information"""
        if path is None:
            path = f'models/model_{self.test_id}.pkl'
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        # Move model to CPU and get state dict
        self.model.cpu()
        model_state = self.model.state_dict()
        
        # Create a clean copy of ZCA parameters
        zca_mean = self.zca.mean.copy() if self.zca.mean is not None else None
        zca_matrix = self.zca.zca_matrix.copy() if self.zca.zca_matrix is not None else None
        
        # Prepare save dictionary with only necessary data
        save_dict = {
            'model_state_dict': model_state,
            'zca_mean': zca_mean,
            'zca_matrix': zca_matrix,
            'test_id': self.test_id,
            'model_config': self.get_model_config()
        }
        
        # Save using pickle
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            
        print(f"Model and preprocessing saved to {path}")
        
        # Move model back to original device
        self.model.to(self.device)
        
    def fit(self, train_loader, val_loader, epochs=100, early_stopping_patience=10,
            save_dir='models', loss_fun=torch.nn.MSELoss()):
        """Train model and save results"""
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'model_{self.test_id}.pkl')
        best_model_path = os.path.join(save_dir, f'best_model_{self.test_id}.pt')
        
        # Initialize trainer
        trainer = RegressionTrainer(
            model=self.model,
            criterion = loss_fun,
            device=self.device
        )
        
        # Train the model
        train_losses, val_losses = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            model_save_path=best_model_path
        )
        
        # Convert losses to plain Python lists
        train_losses = [float(loss) for loss in train_losses]
        val_losses = [float(loss) for loss in val_losses]
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_id': self.test_id,
            'hyperparameters': self.get_model_config()
        }
        
        history_path = os.path.join(save_dir, f'history_{self.test_id}.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Save model and preprocessing
        self.save(save_path)
        
        return train_losses, val_losses
    
    @classmethod
    def load(cls, test_id, model_dir='models', device=None):
        """Load saved model and preprocessing"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        path = os.path.join(model_dir, f'model_{test_id}.pkl')
            
        # Load using pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create new instance
        instance = cls(
            input_dim=save_dict['model_config']['input_dim'],
            output_dim=save_dict['model_config']['output_dim'],
            test_id=save_dict['test_id'],
            hidden_dims=save_dict['model_config']['hidden_dims'],
            dropout_rate=save_dict['model_config']['dropout_rate'],
            output_act=save_dict['model_config']['output_act'],
            arch_name = save_dict['model_config']['arch_name'],
            bn_affine = save_dict['model_config']['bn_affine'],
            
        )
        
        # Load model state
        instance.model.load_state_dict(save_dict['model_state_dict'])
        
        # Load preprocessing state
        instance.zca.mean = save_dict['zca_mean']
        instance.zca.zca_matrix = save_dict['zca_matrix']
        
        # Set device
        instance.device = device
        instance.model = instance.model.to(device)
        
        return instance
    
    def predict(self, X):
        """Make predictions on raw input data"""
        self.model.eval()
        
        # print('KZ TESTING',type(X))
        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Apply ZCA transformation
        X_transformed = self.zca.transform(X)
        
        # Convert to tensor and move to device
        if not isinstance(X_transformed, torch.Tensor):
            X_tensor = torch.from_numpy(X_transformed).float().to(self.device)
        else:
            X_tensor = X_transformed
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()

    def get_last_hidden_layer(self, X):
        """Get only the last layer latent representation"""
        self.model.eval()

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Apply ZCA transformation
        X_transformed = self.zca.transform(X)
        
        # Convert to tensor and move to device
        if not isinstance(X_transformed, torch.Tensor):
            X_tensor = torch.from_numpy(X_transformed).float().to(self.device)
        else:
            X_tensor = X_transformed
        
        # Make predictions
        with torch.no_grad():
            last_latent = self.model(X_tensor, path='last_layer')
        
        return last_latent.cpu().numpy()
