import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


class FeatureEngineering:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def add_hash_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adding to each value the name of its column and applying a hash function
        with tqdm(total=len(self.cfg.generator.cat_features), desc="Hashing features") as pbar:
            for col in self.cfg.generator.cat_features:
                df[col] = df[col].fillna("NaN")
                df[col] = df[col].astype(str)
                df.loc[:, col] = df.loc[:, col].apply(lambda x: hash(x) % 10**6)
                pbar.update(1)

        return df

    def convert_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        with tqdm(total=len(self.cfg.generator.cat_features), desc="Convert features") as pbar:
            # Convert to category type
            for col in self.cfg.generator.cat_features:
                df[col] = df[col].astype("category")
                pbar.update(1)

        return df

    def combine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["F14_18"] = df["F14"] * df["F18"]
        df["F18_36"] = df["F18"] * df["F36"]
        df["F27_29"] = df["F27"] * df["F29"]

        return df

    def reduce_mem_usage(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)

                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2

        if verbose:
            print(
                f"Mem. usage decreased to {end_mem:5.2f} Mb "
                + f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
            )

        return df
