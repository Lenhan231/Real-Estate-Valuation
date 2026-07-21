"""
Simple target encoding - property type only.
Hierarchical encoding (district, locality) cannot be used because
these columns are dropped during preprocessing.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class OOFTargetEncoder:
    """
    Compute target encodings using out-of-fold strategy.
    Currently only encodes property_type due to preprocessing constraints.

    Safe: Each fold's statistics exclude the validation set.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.encodings = {}

    def fit(self, X, y):
        """Fit encoders on full training data for production."""
        self.encodings = self._compute_encodings(X, y)
        return self

    def transform(self, X):
        """Apply encodings to data."""
        X_encoded = X.copy()

        # Property type median price
        if 'property_type_nha_mat_tien' in X.columns:
            X_encoded['property_type_median_price'] = (
                X['property_type_nha_mat_tien'] * self.encodings.get('nha_mat_tien_median', 0) +
                X.get('property_type_nha_trong_hem', 0) * self.encodings.get('nha_trong_hem_median', 0)
            )

        return X_encoded

    def fit_transform_cv(self, X, y, n_splits=None):
        """
        Fit and transform using out-of-fold strategy.
        Returns array with encodings computed from training fold only.

        Safe for training: no information leakage.
        """
        if n_splits is None:
            n_splits = self.n_splits

        X_encoded = X.copy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Initialize new columns
        X_encoded['property_type_median_price'] = 0.0

        # For each fold
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            # Reset indices
            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            X_val.reset_index(drop=True, inplace=True)

            # Compute encodings from training fold only
            encodings_fold = self._compute_encodings(X_train, y_train)

            # Property type encoding
            if 'property_type_nha_mat_tien' in X_val.columns:
                encoded_vals = (
                    X_val.get('property_type_nha_mat_tien', 0) * encodings_fold.get('nha_mat_tien_median', 0) +
                    X_val.get('property_type_nha_trong_hem', 0) * encodings_fold.get('nha_trong_hem_median', 0)
                ).values
                X_encoded.iloc[val_idx, X_encoded.columns.get_loc('property_type_median_price')] = encoded_vals

        return X_encoded

    def _compute_encodings(self, X, y):
        """Compute target encodings."""
        encodings = {}

        # Global median
        encodings['global_median'] = float(y.median())

        # Property type median
        if 'property_type_nha_mat_tien' in X.columns:
            mask = X['property_type_nha_mat_tien'] == 1
            encodings['nha_mat_tien_median'] = float(y[mask].median()) if mask.sum() > 0 else encodings['global_median']
        else:
            encodings['nha_mat_tien_median'] = encodings['global_median']

        if 'property_type_nha_trong_hem' in X.columns:
            mask = X['property_type_nha_trong_hem'] == 1
            encodings['nha_trong_hem_median'] = float(y[mask].median()) if mask.sum() > 0 else encodings['global_median']
        else:
            encodings['nha_trong_hem_median'] = encodings['global_median']

        return encodings
