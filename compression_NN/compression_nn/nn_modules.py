import torch
import torch.nn as nn
import tqdm
import math

class RegressionNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128, 64], 
                 dropout_rate=0.2, activation=nn.ReLU(), output_act=None, bn_affine=True):
        """
        Neural Network for regression with batch normalization
        
        Args:
            input_dim: Number of input features
            output_dim: Number of outputs (1 or 2)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function to use
        """
        super(RegressionNN, self).__init__()
        
        # Create lists to hold layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, affine=bn_affine))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self.activation = activation
        self.output_act = output_act
        
    def forward_no_last(self, x):
        """
        Forward pass until the output layer; i.e get the last layer latent representation
        """
        for layer, batch_norm, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = batch_norm(x)
            x = self.activation(x)
            x = dropout(x)
        
        # NO output layer
        return x
        
        
    def forward(self, x, path='full'):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if path=='full':
            # Hidden layers with batch norm, activation, and dropout
            for layer, batch_norm, dropout in zip(self.layers, self.batch_norms, self.dropouts):
                x = layer(x)
                x = batch_norm(x)
                x = self.activation(x)
                x = dropout(x)

            # Output layer (no activation for regression)
            x = self.output_layer(x)
            if self.output_act is not None:
                x = self.output_act(x)
            return x
        elif path=='last_layer':
            return self.forward_no_last(x)
        
        else:
            raise NotImplementedError("Forward pass path doesn't exist")    
    
class RegressionNN_ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128, 64], 
                 dropout_rate=0.2, activation=nn.ReLU(), output_act=None, bn_affine=True):
        super(RegressionNN_ResNet, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Main layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, affine=bn_affine))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Residual connection (if dimensions match)
            if prev_dim == hidden_dim:
                self.residual_layers.append(nn.Identity())
            else:
                self.residual_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = activation
        self.output_act = output_act
        
    def forward_no_last(self,x):
        """
        Forward pass until the output layer; i.e get the last layer latent representation
        """
        for layer, batch_norm, dropout, residual in zip(
            self.layers, self.batch_norms, self.dropouts, self.residual_layers):

            # Store input for residual connection
            identity = x

            # Main path
            x = layer(x)
            x = batch_norm(x)
            x = self.activation(x)
            x = dropout(x)

            # Add residual connection
            x = x + residual(identity)
        
        # NO output layer
        return x
    
    def forward(self, x, path='full'):
        if path=='full':
            for layer, batch_norm, dropout, residual in zip(
                self.layers, self.batch_norms, self.dropouts, self.residual_layers):

                # Store input for residual connection
                identity = x

                # Main path
                x = layer(x)
                x = batch_norm(x)
                x = self.activation(x)
                x = dropout(x)

                # Add residual connection
                x = x + residual(identity)

            x = self.output_layer(x)
            if self.output_act is not None:
                x = self.output_act(x)
            return x
        elif path=='last_layer':
            return self.forward_no_last(x)
        else:
            raise NotImplementedError("Forward pass path doesn't exist")
            
            
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
                or [seq_length, d_model]
        Returns:
            Tensor of same shape as input
        """
        # Handle both 2D and 3D inputs
        if len(x.size()) == 2:
            # Input is [seq_length, d_model]
            return x + self.pe.squeeze(0)[:x.size(0)]
        else:
            # Input is [batch_size, seq_length, d_model]
            # Ensure pe is properly broadcast regardless of batch size
            return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        # Multi-head attention
        attention_output, _ = self.attention(x, mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)
        
        return x

class RegressionNN_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, num_heads=8, num_layers=4,
                 d_ff=2048, dropout_rate=0.1, activation=nn.ReLU(), output_act=None):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation
        self.output_act = output_act
        
    def forward_no_last(self, x):
        """
        Forward pass until the output layer; i.e get the last layer latent representation
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        x = self.activation(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
         # Global average pooling over sequence length
            if len(x.size()) > 2:
                x = torch.mean(x, dim=1)
        
        return x
    
    def forward(self, x, path='full'):
        if path == 'full':
            # Get transformer output
            x = self.forward_no_last(x)
                        
            # Output projection
            x = self.output_layer(x)
            if self.output_act is not None:
                x = self.output_act(x)
                
            return x
            
        elif path == 'last_layer':
            return self.forward_no_last(x)
        else:
            raise NotImplementedError("Forward pass path doesn't exist")


class RegressionTrainer:
    def __init__(self, model, criterion=nn.MSELoss(), optimizer=None, device='cuda'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters())
        # Add scheduler here
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',           # reduce LR when validation loss stops decreasing
            factor=0.3,          # multiply LR by this factor
            patience=15,          # number of epochs to wait before reducing LR
            min_lr=1e-6         # don't reduce LR below this value
        )
        self.device = device
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    def train(self, train_loader, val_loader, epochs, early_stopping_patience, 
              model_save_path=None):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in tqdm.tqdm(range(epochs)):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch < epochs//2 and epoch % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Training Loss: {train_loss:.6f}')
                print(f'Validation Loss: {val_loss:.6f}')
            
            elif epoch > epochs//2 and epoch % 15 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Training Loss: {train_loss:.7f}')
                print(f'Validation Loss: {val_loss:.7f}')

            # Add scheduler step here
            self.scheduler.step(val_loss)
            
            # KZ NOTE: early-stopping start after half training; I feel this is better but other 
            # types of early stopping can be better; see e.g. separable_bk
            if epoch > epochs//2:
            # if epoch > 50:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if model_save_path:
                        torch.save(self.model.state_dict(), model_save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        print(f'Best validation loss is {best_val_loss}')
                        break
        
        return train_losses, val_losses