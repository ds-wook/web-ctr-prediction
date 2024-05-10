from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from generator import FeatureEngineering, LabelEncoder


class DataStorage:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _categorize_train_features(self, train: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for train data
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """

        le = LabelEncoder(100)
        train[[*self.cfg.generator.hash_features]] = le.fit_transform(train[[*self.cfg.generator.hash_features]])
        joblib.dump(le, Path(self.cfg.data.meta) / "label_encoder.pkl")

        return train

    def _categorize_test_features(self, test: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for test data
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        le = joblib.load(Path(self.cfg.data.meta) / "label_encoder.pkl")
        test[[*self.cfg.generator.hash_features]] = le.transform(test[[*self.cfg.generator.hash_features]].astype(str))

        return test

    def load_train_dataset(self) -> pd.DataFrame:
        train = pd.read_parquet(Path(self.cfg.data.path) / f"{self.cfg.data.train}.parquet")

        feature_engineering = FeatureEngineering(self.cfg)
        # train = self._categorize_train_features(train)
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
        # test = self._categorize_test_features(test)
        test = feature_engineering.add_hash_features(test)
        test = feature_engineering.convert_categorical_features(test)
        test = feature_engineering.combine_features(test)
        test = feature_engineering.reduce_mem_usage(test)

        test_x = test.drop(columns=self.cfg.generator.drop_features)

        return test_x


def sampling_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    pfile = pq.ParquetFile(Path(cfg.data.path) / "train.parquet")

    train = pd.DataFrame()
    negative = pd.DataFrame()
    positive = pd.DataFrame()
    chunksize = 10**7

    for chunk in tqdm(pfile.iter_batches(batch_size=chunksize), desc="Sampling data", leave=False):
        chunk = chunk.to_pandas()
        positive_sample = chunk[chunk["Click"] == 1]
        negative_sample = chunk[chunk["Click"] == 0]
        negative = pd.concat([negative, negative_sample], axis=0, ignore_index=True)
        positive = pd.concat([positive, positive_sample], axis=0, ignore_index=True)

    negative_sample = negative.sample(frac=cfg.data.sampling, replace=False, random_state=cfg.data.seed)
    train = pd.concat([train, negative_sample, positive], axis=0, ignore_index=True)
    del negative, positive

    return train


def negative_sampling_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    pfile = pq.ParquetFile(Path(cfg.data.path) / "train.parquet")

    train = pd.DataFrame()
    chunksize = 10**7

    for chunk in tqdm(pfile.iter_batches(batch_size=chunksize), desc="Sampling data", leave=False):
        chunk = chunk.to_pandas()
        # chunk = chunk.sample(frac=cfg.data.sampling, replace=False, random_state=602)
        neg_samp = chunk[chunk["Click"] == 0].sample(n=len(chunk[chunk["Click"] == 1]), random_state=cfg.data.seed)
        train = pd.concat([train, neg_samp, chunk[chunk["Click"] == 1]], axis=0)

    return train
