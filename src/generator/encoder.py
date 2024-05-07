import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing_extensions import Self

NAN_INT = 7535805


class LabelEncoder(BaseEstimator):
    """Label Encoder that groups infrequent values into one label.

    Attributes:
        min_obs (int): minimum number of observation to assign a label.
        label_encoders (list of dict): label encoders for columns
        label_maxes (list of int): maximum of labels for columns
    """

    def __init__(self, min_obs: int = 10):
        """Initialize the OneHotEncoder class object.

        Args:
            min_obs (int): minimum number of observation to assign a label.
        """

        self.min_obs = min_obs
        self.is_fitted = False

    def __repr__(self):
        return ("LabelEncoder(min_obs={})").format(self.min_obs)

    def _get_label_encoder_and_max(self, x: pd.Series) -> tuple[dict, int]:
        """Return a mapping from values and its maximum of a column to integer labels.

        Args:
            x (pandas.Series): a categorical column to encode.

        Returns:
            (tuple):
                - (dict): mapping from values of features to integers
                - (int): maximum label
        """

        # NaN cannot be used as a key for dict. Impute it with a random
        # integer.
        label_count = x.fillna(NAN_INT).value_counts()
        n_uniq = label_count.shape[0]

        label_count = label_count[label_count >= self.min_obs]
        n_uniq_new = label_count.shape[0]

        # If every label appears more than min_obs, new label starts from 0.
        # Otherwise, new label starts from 1 and 0 is used for all old labels
        # that appear less than min_obs.
        offset = 0 if n_uniq == n_uniq_new else 1

        label_encoder = pd.Series(np.arange(n_uniq_new) + offset, index=label_count.index)
        max_label = label_encoder.max()
        label_encoder = label_encoder.to_dict()

        return label_encoder, max_label

    def _transform_col(self, x: pd.Series, i: int) -> pd.Series:
        """Encode one categorical column into labels.

        Args:
            x (pandas.Series): a categorical column to encode
            i (int): column index

        Returns:
            (pandas.Series): a column with labels.
        """
        return x.fillna(NAN_INT).map(self.label_encoders[i]).fillna(0).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            (
                self.label_encoders[i],
                self.label_maxes[i],
            ) = self._get_label_encoder_and_max(X[col])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns into label encoded columns

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): label encoded columns
        """

        assert self.is_fitted, "fit() or fit_transform() must be called before transform()."

        X = X.copy()
        for i, col in enumerate(X.columns):
            X.loc[:, col] = self._transform_col(X[col], i)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Encode categorical columns into label encoded columns

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): label encoded columns
        """

        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        X = X.copy()
        for i, col in enumerate(X.columns):
            (
                self.label_encoders[i],
                self.label_maxes[i],
            ) = self._get_label_encoder_and_max(X[col])

            X.loc[:, col] = X[col].fillna(NAN_INT).map(self.label_encoders[i]).fillna(0).astype(int)

        self.is_fitted = True
        return X
