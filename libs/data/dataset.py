from pathlib import Path

import joblib
import pandas as pd
from category_encoders import CountEncoder
from omegaconf import DictConfig
from sklearn.preprocessing import QuantileTransformer

from generator import FeatureEngineering, LabelEncoder


class DataStorage:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _categorize_train_features(self, train_x: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for train data
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """
        le = LabelEncoder()
        train_x[[*self.cfg.generator.cat_features]] = le.fit_transform(train_x[[*self.cfg.generator.cat_features]])
        joblib.dump(le, Path(self.cfg.data.meta) / "label_encoder.pkl")

        return train_x

    def _categorize_test_features(self, test_x: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for test data
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        le = joblib.load(Path(self.cfg.data.meta) / "label_encoder.pkl")
        test_x[[*self.cfg.generator.cat_features]] = le.transform(test_x[[*self.cfg.generator.cat_features]])

        return test_x

    def _count_train_features(self, train_x: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for train data
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """
        cnt = CountEncoder()
        train_enc = cnt.fit_transform(train_x[[*self.cfg.generator.cat_features]])
        train_x = train_x.join(train_enc.add_suffix("_count"))
        joblib.dump(cnt, Path(self.cfg.data.meta) / "count_encoder.pkl")

        return train_x

    def _count_test_features(self, test_x: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for test data
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        cnt = joblib.load(Path(self.cfg.data.meta) / "count_encoder.pkl")
        test_enc = cnt.transform(test_x[[*self.cfg.generator.cat_features]])
        test_x = test_x.join(test_enc.add_suffix("_count"))

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
        scaler = QuantileTransformer(n_quantiles=100, output_distribution="normal")
        train[[*self.cfg.generator.num_features]] = scaler.fit_transform(train[[*self.cfg.generator.num_features]])
        joblib.dump(scaler, Path(self.cfg.data.meta) / "rankgauss.pkl")

        return train

    def _numerical_test_scaling(self, test: pd.DataFrame) -> pd.DataFrame:
        """Numerical scaling
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """
        scaler = joblib.load(Path(self.cfg.data.meta) / "rankgauss.pkl")
        test[[*self.cfg.generator.num_features]] = scaler.transform(test[[*self.cfg.generator.num_features]])

        return test

    def load_train_dataset(self) -> pd.DataFrame:
        train = pd.read_parquet(Path(self.cfg.data.path) / f"{self.cfg.data.train}.parquet")

        feature_engineering = FeatureEngineering(self.cfg)

        if self.cfg.models.name == "lightgbm":
            train = self._categorize_train_features(train)
            train = feature_engineering.convert_categorical_features(train)
            train = self._count_train_features(train)

        elif self.cfg.models.name == "catboost":
            train = self._categorize_train_features(train)
            train = feature_engineering.reduce_mem_usage(train)
            train = self._count_train_features(train)

        else:
            train = self._categorize_train_features(train)
            train = self._numerical_train_scaling(train)
            train = train.fillna(0)

        train = feature_engineering.reduce_mem_usage(train)

        train_x = train.drop(columns=[*self.cfg.generator.drop_features, self.cfg.data.target])
        train_y = train[self.cfg.data.target]

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        test = pd.read_parquet(Path(self.cfg.data.path) / "test.parquet")

        feature_engineering = FeatureEngineering(self.cfg)

        if self.cfg.models.name == "lightgbm":
            test = self._categorize_test_features(test)
            test = feature_engineering.convert_categorical_features(test)
            test = self._count_test_features(test)

        elif self.cfg.models.name == "catboost":
            test = self._categorize_test_features(test)
            test = self._count_test_features(test)

        else:
            test = self._categorize_test_features(test)
            test = self._numerical_test_scaling(test)
            test = test.fillna(0)

        test = feature_engineering.reduce_mem_usage(test)

        test_x = test.drop(columns=self.cfg.generator.drop_features)

        return test_x
