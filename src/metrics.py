import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def compute_regression_metrics(y_true, y_pred):
    """
    Computes standard regression evaluation metrics for human-readable assessment.
    
    Args:
        y_true (list or np.array): Ground truth target values.
        y_pred (list or np.array): Predicted mean values from the model.
        
    Returns:
        dict: A dictionary containing RMSE, MAE, R2, and Pearson correlation.
    """
    # Ensure inputs are standard numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Root Mean Squared Error (RMSE) - Penalizes large errors heavily
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # 2. Mean Absolute Error (MAE) - Average absolute difference
    mae = mean_absolute_error(y_true, y_pred)
    
    # 3. R-squared (R2) - Proportion of variance explained by the model (Max: 1.0)
    r2 = r2_score(y_true, y_pred)
    
    # 4. Pearson Correlation Coefficient (r) - Linear correlation (-1.0 to 1.0)
    # Added a safety check to prevent division by zero if predictions are constant
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        pearson_corr, _ = pearsonr(y_true, y_pred)
    else:
        pearson_corr = 0.0
        
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Pearson": float(pearson_corr)
    }