from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from generator import FeatureEngineering


class DataStorage:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_train_data(self) -> pd.DataFrame:
        train = pd.read_csv(Path(self.cfg.data.path) / "train_sample.csv")
        train = FeatureEngineering(self.cfg).generate_features(train)
        train_x = train[self.cfg.data.select_features]
        train_y = train[self.cfg.data.target]

        return train_x, train_y

    def load_test_data(self) -> pd.DataFrame:
        test = pd.read_csv(Path(self.cfg.data.path) / "test.csv")
        test = FeatureEngineering(self.cfg).generate_features(test)
        test_x = test[self.cfg.data.select_features]

        return test_x


def sampling_train(cfg: DictConfig) -> pd.DataFrame:
    pfile = pq.ParquetFile(Path(cfg.data.path) / "train.parquet")

    train = pd.DataFrame()
    chunksize = 10**7

    for chunk in tqdm(pfile.iter_batches(batch_size=chunksize), desc="Sampling data", leave=False):
        chunk = chunk.to_pandas()
        train = pd.concat([train, chunk.sample(frac=cfg.sampling)], axis=0, ignore_index=True)

    return train
