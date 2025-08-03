import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionNeuralNet:
    """
    Linear Regression implemented from scratch using Neural Network approach
    Supports both batch and online (stochastic) learning
    """
    
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 1000, 
                 tolerance: float = 1e-6, random_seed: int = 42):
        """
        Initialize the Linear Regression Neural Network
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance
            random_seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.random_seed = random_seed
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.epoch_history = []
        
        # Set random seed
        p.random.seed(random_seed)n
    
    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights and bias"""
        # Xavier/Glorot initialization
        self.weights = np.random.normal(0, np.sqrt(2.0 / n_features), n_features)
        self.bias = 0.0
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias # calculating y_pred
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE cost
        """
        m = len(y_true) # no. of samples
        return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)  #cost function -> half MSE
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias
        
        Args:
            X: Input features
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Tuple of (weight_gradients, bias_gradient)
        """
        m = len(y_true)
        error = y_pred - y_true
        
        dW = (1 / m) * np.dot(X.T, error) # transpose times error as these are matrices
        db = (1 / m) * np.sum(error) # wrt b, d(y_pred-y_true)/db = 1
        
        return dW, db
    
    def fit_batch(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train the model using batch gradient descent
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            verbose: Whether to print training progress
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        self.cost_history = []
        self.epoch_history = []
        
        prev_cost = float('inf')
        
        for epoch in range(self.max_epochs):
            # Forward pass
            y_pred = self._forward_pass(X)
            
            # Compute cost
            current_cost = self._compute_cost(y, y_pred)
            self.cost_history.append(current_cost)
            self.epoch_history.append(epoch)
            
            # Compute gradients
            dW, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if abs(prev_cost - current_cost) < self.tolerance:
                if verbose:
                    print(f"Converged at epoch {epoch} with cost {current_cost:.6f}")
                break
            
            prev_cost = current_cost
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {current_cost:.6f}")
        
        if verbose:
            print(f"Training completed. Final cost: {current_cost:.6f}")
    
    def fit_online(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train the model using online (stochastic) gradient descent
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            verbose: Whether to print training progress
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        self.cost_history = []
        self.epoch_history = []
        
        for epoch in range(self.max_epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            # Process one sample at a time
            for i in range(n_samples):
                # Get single sample
                xi = X_shuffled[i:i+1]  # Keep 2D shape
                yi = y_shuffled[i:i+1]
                
                # Forward pass
                y_pred_i = self._forward_pass(xi)
                
                # Compute cost for this sample
                sample_cost = self._compute_cost(yi, y_pred_i)
                epoch_cost += sample_cost
                
                # Compute gradients
                dW, db = self._compute_gradients(xi, yi, y_pred_i)
                
                # Update parameters
                self.weights -= self.learning_rate * dW
                self.bias -= self.learning_rate * db
            
            # Average cost for the epoch
            avg_epoch_cost = epoch_cost / n_samples
            self.cost_history.append(avg_epoch_cost)
            self.epoch_history.append(epoch)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Average Cost = {avg_epoch_cost:.6f}")
        
        if verbose:
            print(f"Online training completed. Final average cost: {avg_epoch_cost:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self._forward_pass(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score
        
        Args:
            X: Features
            y: True values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Mean Squared Error
        
        Args:
            X: Features
            y: True values
            
        Returns:
            MSE
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def mean_absolute_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            X: Features
            y: True values
            
        Returns:
            MAE
        """
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

def load_and_preprocess_data(filepath: str, target_col: str, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and preprocess data
    
    Args:
        filepath: Path to CSV file
        target_col: Name of target column
        feature_cols: List of feature column names (if None, use all except target)
        
    Returns:
        Tuple of (X, y, dataframe)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Handle feature selection
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y, df

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using standardization (z-score normalization)
    
    Args:
        X: Features array
        
    Returns:
        Tuple of (normalized_X, mean, std)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        random_seed: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_seed)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def plot_training_curves(models_dict: dict, title: str = "Training Curves Comparison"):
    """
    Plot training curves for multiple models
    
    Args:
        models_dict: Dictionary with model names as keys and model objects as values
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for name, model in models_dict.items():
        plt.plot(model.epoch_history, model.cost_history, label=f'{name}', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Cost (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often helps visualize convergence
    plt.show()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual"):
    """
    Plot predictions vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} - Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_data(df: pd.DataFrame, target_col: str, feature_cols: List[str]):
    """
    Perform exploratory data analysis
    
    Args:
        df: DataFrame
        target_col: Target column name
        feature_cols: Feature column names
    """
    print("=== EXPLORATORY DATA ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: {target_col}")
    print(f"Feature variables: {feature_cols}")
    print("\nDataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Correlation analysis
    print(f"\nCorrelation with {target_col}:")
    correlations = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    print(correlations)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Correlation heatmap
    # plt.subplot(2, 3, 1)
    # correlation_matrix = df[feature_cols + [target_col]].corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    # plt.title('Correlation Matrix')
    
    # Target distribution
    plt.subplot(2, 3, 2)
    plt.hist(df[target_col], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.title(f'{target_col} Distribution')
    
    # Feature distributions
    for i, feature in enumerate(feature_cols[:4]):  # Show first 4 features
        plt.subplot(2, 3, 3 + i)
        plt.scatter(df[feature], df[target_col], alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f'{feature} vs {target_col}')
    
    plt.tight_layout()
    plt.show()

# ===============================
# MAIN IMPLEMENTATION
# ===============================

def main():
    print("=== LINEAR REGRESSION FROM SCRATCH - NEURAL NETWORK APPROACH ===\n")
    
    # Load and preprocess housing data
    print("Loading Housing Dataset...")
    X_housing, y_housing, df_housing = load_and_preprocess_data(
        'Housing.csv', 
        'price', 
        ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    )
    
    # Analyze data
    analyze_data(df_housing, 'price', ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
    
    # Normalize features
    X_housing_norm, mean_housing, std_housing = normalize_features(X_housing)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_housing_norm, y_housing, test_size=0.2, random_seed=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Initialize models
    batch_model = LinearRegressionNeuralNet(learning_rate=0.01, max_epochs=1000, tolerance=1e-6)
    online_model = LinearRegressionNeuralNet(learning_rate=0.001, max_epochs=500, tolerance=1e-6)
    
    print("\n=== BATCH LEARNING ===")
    batch_model.fit_batch(X_train, y_train, verbose=True)
    
    print("\n=== ONLINE LEARNING ===")
    online_model.fit_online(X_train, y_train, verbose=True)
    
    # Evaluate models
    print("\n=== MODEL EVALUATION ===")
    
    # Batch model evaluation
    batch_train_mse = batch_model.mean_squared_error(X_train, y_train)
    batch_test_mse = batch_model.mean_squared_error(X_test, y_test)
    batch_train_r2 = batch_model.score(X_train, y_train)
    batch_test_r2 = batch_model.score(X_test, y_test)
    
    print("Batch Learning Results:")
    print(f"  Training MSE: {batch_train_mse:.2f}")
    print(f"  Test MSE: {batch_test_mse:.2f}")
    print(f"  Training R²: {batch_train_r2:.4f}")
    print(f"  Test R²: {batch_test_r2:.4f}")
    
    # Online model evaluation
    online_train_mse = online_model.mean_squared_error(X_train, y_train)
    online_test_mse = online_model.mean_squared_error(X_test, y_test)
    online_train_r2 = online_model.score(X_train, y_train)
    online_test_r2 = online_model.score(X_test, y_test)
    
    print("\nOnline Learning Results:")
    print(f"  Training MSE: {online_train_mse:.2f}")
    print(f"  Test MSE: {online_test_mse:.2f}")
    print(f"  Training R²: {online_train_r2:.4f}")
    print(f"  Test R²: {online_test_r2:.4f}")
    
    # Plot training curves
    models_dict = {
        'Batch Learning': batch_model,
        'Online Learning': online_model
    }
    plot_training_curves(models_dict, "Training Curves: Batch vs Online Learning")
    
    # Plot predictions
    batch_pred = batch_model.predict(X_test)
    online_pred = online_model.predict(X_test)
    
    plot_predictions(y_test, batch_pred, "Batch Learning")
    plot_predictions(y_test, online_pred, "Online Learning")
    
    # Print model parameters
    print("\n=== MODEL PARAMETERS ===")
    feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    print("Batch Model:")
    print(f"  Bias: {batch_model.bias:.4f}")
    for i, feature in enumerate(feature_names):
        print(f"  Weight[{feature}]: {batch_model.weights[i]:.4f}")
    
    print("\nOnline Model:")
    print(f"  Bias: {online_model.bias:.4f}")
    for i, feature in enumerate(feature_names):
        print(f"  Weight[{feature}]: {online_model.weights[i]:.4f}")
    
    # Make sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    sample_idx = [0, 1, 2]
    for idx in sample_idx:
        actual = y_test[idx]
        batch_pred_val = batch_pred[idx]
        online_pred_val = online_pred[idx]
        
        print(f"Sample {idx + 1}:")
        print(f"  Actual: {actual:.0f}")
        print(f"  Batch Prediction: {batch_pred_val:.0f} (Error: {abs(actual - batch_pred_val):.0f})")
        print(f"  Online Prediction: {online_pred_val:.0f} (Error: {abs(actual - online_pred_val):.0f})")
        print()
    
    # Test with advertising dataset
    print("\n=== TESTING WITH ADVERTISING DATASET ===")
    X_ad, y_ad, df_ad = load_and_preprocess_data('advertising.csv', 'Sales', ['TV', 'Radio', 'Newspaper'])
    
    # Quick analysis
    print(f"Advertising dataset shape: {df_ad.shape}")
    print("Correlation with Sales:")
    correlations_ad = df_ad.corr()['Sales'].abs().sort_values(ascending=False)
    print(correlations_ad)
    
    # Normalize and split
    X_ad_norm, _, _ = normalize_features(X_ad)
    X_train_ad, X_test_ad, y_train_ad, y_test_ad = train_test_split(X_ad_norm, y_ad, test_size=0.2)
    
    # Train simple model on advertising data
    ad_model = LinearRegressionNeuralNet(learning_rate=0.01, max_epochs=1000)
    ad_model.fit_batch(X_train_ad, y_train_ad, verbose=False)
    
    ad_test_r2 = ad_model.score(X_test_ad, y_test_ad)
    print(f"Advertising dataset R² score: {ad_test_r2:.4f}")

if __name__ == "__main__":
    main()