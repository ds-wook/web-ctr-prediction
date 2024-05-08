from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from tqdm import tqdm

from data import DataStorage
from models import ModelResult, bulid_model


def inference_models(result: list[ModelResult], test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(result.models)
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(result.models.values(), total=folds, desc="Predicting models"):
        preds += (
            model.predict(xgb.DMatrix(test_x)) / folds
            if isinstance(model, xgb.Booster)
            else (
                model.predict(test_x) / folds
                if isinstance(model, lgb.Booster)
                else model.predict_proba(test_x)[:, 1] / folds
            )
        )

    return preds


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        data_storage = DataStorage(cfg)

        # load dataset
        train_x, train_y = data_storage.load_train_data()

        # choose trainer
        trainer = bulid_model(cfg)

        # train model
        trainer.run_cv_training(train_x, train_y)

        # save model
        trainer.save_model(Path(cfg.models.path))

        # load model
        result = joblib.load(Path(cfg.models.path) / f"{cfg.models.results}.pkl")

        # load dataset
        test_x = data_storage.load_test_data()
        submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

        # predict
        preds = inference_models(result, test_x)
        submit[cfg.data.target] = preds
        submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
