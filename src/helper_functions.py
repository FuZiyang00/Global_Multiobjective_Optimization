import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classifier and display:
    - Precision, Recall, Accuracy, and AUC
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

def confusion_matrix_plot(y_true, y_pred):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def discretize_multiple_features(source_df, target_df, feature_cols, n_bins=10):
    """
    Discretize multiple features at once to avoid fragmentation.
    """
    # Dictionary to store all new discretized columns
    new_columns = {}
    columns_to_drop = []
    
    for feature_col in feature_cols:
        if feature_col not in source_df.columns or feature_col not in target_df.columns:
            continue
            
        new_col_name = f"{feature_col}_PERCENTILE"
        
        try:
            # Calculate bin edges from source
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = source_df[feature_col].quantile(quantiles).values
            bin_edges = np.unique(np.sort(bin_edges))
            
            if len(bin_edges) < 2:
                print(f"Warning: {feature_col} has insufficient unique values")
                continue
            
            # Discretize and store in dictionary
            new_columns[new_col_name] = pd.cut(target_df[feature_col], 
                                             bins=bin_edges, 
                                             labels=False, 
                                             include_lowest=True,
                                             duplicates='drop')
            
            columns_to_drop.append(feature_col)
            print(f"Prepared discretization for {feature_col}")
            
        except Exception as e:
            print(f"Error with {feature_col}: {e}")
    
    # Create DataFrame from all new columns at once
    if new_columns:
        new_cols_df = pd.DataFrame(new_columns, index=target_df.index)
        
        # Drop original columns and concat new ones
        target_cleaned = target_df.drop(columns=columns_to_drop)
        result = pd.concat([target_cleaned, new_cols_df], axis=1)
        
        return result
    else:
        return target_df

