import numpy as np
from helpers.helpers import *


def audit_missing_values(X, feature_names):
    """Compute NaN counts and percentages across all features and observations."""
    n_samples, n_features = X.shape
    nan_mask = np.isnan(X)
    col_nan_counts = nan_mask.sum(axis=0)
    col_nan_pcts = col_nan_counts / n_samples * 100.0

    row_nan_counts = nan_mask.sum(axis=1)
    row_nan_pcts = row_nan_counts / n_features * 100.0

    return {
        "col_nan_counts": col_nan_counts,
        "col_nan_pcts": col_nan_pcts,
        "row_nan_counts": row_nan_counts,
        "row_nan_pcts": row_nan_pcts,
    }
    # maybe print a pretty table with feature names?


def remove_features_with_missing_values(X, feature_names, threshold=0.8):
    """Remove features with more than threshold fraction of missing values.
    Still have to decide on a threshold that's feasible"""
    n_samples, n_features = X.shape
    nan_mask = np.isnan(X)
    col_nan_counts = nan_mask.sum(axis=0)
    col_nan_pcts = col_nan_counts / n_samples * 100.0

    features_to_remove = []
    for j in range(n_features):
        if col_nan_pcts[j] > threshold * 100.0:
            features_to_remove.append(j)

    X_cleaned = np.delete(X, features_to_remove, axis=1)
    feature_names_cleaned = np.delete(feature_names, features_to_remove)

    return X_cleaned, feature_names_cleaned, features_to_remove 

def audit_constants(X, feature_names, consistency_threshold=0.95):
    """Scans X for constant colums and near-constant columns (only considering valid observations)

    Calculates consistency on valid (non-NaN) observations.
    """
    n_samples, n_features = X.shape
    constant_features = []  # 100% constant
    all_nan_features = []   # 100% NaN
    near_constant_features = []  # > consistency_threshold

    for j in range(n_features):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        n_valid = len(valid)

        if n_valid == 0:
            all_nan_features.append((feature_names[j], j, 100.0))
            continue

        unique_vals, counts = np.unique(valid, return_counts=True)
        max_count = counts.max()
        dominant_val = unique_vals[counts.argmax()]
        consistency_ratio = max_count / n_valid

        if len(unique_vals) == 1:
            constant_features.append({
                "name": feature_names[j],
                "index": j,
                "val": dominant_val,
                "n_valid": n_valid,
                "pct_of_valid": 100.0,
                "pct_of_total": max_count / n_samples * 100.0,
            })
        elif consistency_ratio >= consistency_threshold:
            near_constant_features.append({
                "name": feature_names[j],
                "index": j,
                "dominant_val": dominant_val,
                "n_valid": n_valid,
                "consistency_pct": consistency_ratio * 100.0,
                "n_unique": len(unique_vals),
            })

    # Sort near-constant features descending by consistency
    near_constant_features.sort(key=lambda x: x["consistency_pct"], reverse=True)

    return all_nan_features, constant_features, near_constant_features