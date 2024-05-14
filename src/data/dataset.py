from pathlib import Path

import joblib
import pandas as pd
import torch
from category_encoders import HashingEncoder
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler

from generator import FeatureEngineering


class DataStorage:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _categorize_train_features(self, train_x: pd.DataFrame, train_y: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for train data
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """
        hashing_enc = HashingEncoder(cols=[*self.cfg.generator.cat_features], n_components=100).fit(train_x, train_y)
        X_train_hashing = hashing_enc.transform(train_x)
        joblib.dump(hashing_enc, Path(self.cfg.data.meta) / "hash_encoder.pkl")
        train_x = train_x[[*self.cfg.generator.num_features]].join(X_train_hashing)

        return train_x

    def _categorize_test_features(self, test_x: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for test data
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        hashing_enc = joblib.load(Path(self.cfg.data.meta) / "hash_encoder.pkl")
        test_hashing = hashing_enc.transform(test_x)
        test_x = test_x[[*self.cfg.generator.num_features]].join(test_hashing)

        return test_x

    def _numerical_train_scaling(self, train: pd.DataFrame) -> pd.DataFrame:
        """Numerical scaling
        Args:
            config: config
            train: dataframe
            test: dataframe
        Returns:
            dataframe
        """
        scaler = MinMaxScaler()
        train[self.cfg.generator.num_features] = scaler.fit_transform(train[self.cfg.generator.num_features])
        joblib.dump(scaler, Path(self.cfg.data.meta) / "minmax_scaler.pkl")

        return train

    def _numerical_test_scaling(self, test: pd.DataFrame) -> pd.DataFrame:
        """Numerical scaling
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """
        scaler = joblib.load(Path(self.cfg.data.meta) / "minmax_scaler.pkl")
        test[self.cfg.generator.num_features] = scaler.transform(test[self.cfg.generator.num_features])

        return test

    def load_train_dataset(self) -> pd.DataFrame:
        train = pd.read_parquet(Path(self.cfg.data.path) / f"{self.cfg.data.train}.parquet")

        feature_engineering = FeatureEngineering(self.cfg)
        train = feature_engineering.add_hash_features(train)
        train = feature_engineering.convert_categorical_features(train)
        train = feature_engineering.combine_features(train)
        train = feature_engineering.reduce_mem_usage(train)

        train_x = train.drop(columns=[*self.cfg.generator.drop_features, self.cfg.data.target])
        train_y = train[self.cfg.data.target]

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        test = pd.read_parquet(Path(self.cfg.data.path) / "test.parquet")

        feature_engineering = FeatureEngineering(self.cfg)
        test = feature_engineering.add_hash_features(test)
        test = feature_engineering.convert_categorical_features(test)
        test = feature_engineering.combine_features(test)
        test = feature_engineering.reduce_mem_usage(test)

        test_x = test.drop(columns=self.cfg.generator.drop_features)

        return test_x

    def load_train_dataloader(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[torch.Tensor]:
        # Convert to tensor
        X_train_sparse = torch.tensor(X_train[self.cfg.generator.sparse_features].values, dtype=torch.long).to(
            self.cfg.models.device
        )
        X_train_dense = torch.tensor(X_train[self.cfg.generator.dense_features].values, dtype=torch.float).to(
            self.cfg.models.device
        )
        y_train = torch.tensor(y_train.values, dtype=torch.float).unsqueeze(1).to(self.cfg.models.device)

        return X_train_sparse, X_train_dense, y_train
