from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import rankdata
from tqdm import tqdm


def ensemble_predictions(predictions, weights, type_="linear"):
    assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)

    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res

    elif type_ == "geometric":
        numerator = np.average([np.log(p) for p in predictions], weights=weights, axis=0)
        res = np.exp(numerator / sum(weights))
        return res

    elif type_ == "rank":
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)

    elif type_ == "sigmoid":
        logit_values = np.log(predictions / (1 - predictions))
        result = np.average(logit_values, weights=weights, axis=0)
        return 1 / (1 + np.exp(-result))

    return res


def calculate_sigmoid_preds(values: list[np.ndarray]) -> np.ndarray:
    """
    Calculate the sigmoid result of the ensemble predictions.
    :param values: list of predictions
    :param weight: list of weights
    :return: ensemble prediction
    """
    values = np.array(values)

    logit_values = np.log(values / (1 - values))
    result = np.mean(logit_values, axis=0)

    return 1 / (1 + np.exp(-result))


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.3.1")
def _main(cfg: DictConfig):
    # Load submission file
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

    # Load predictions and calculate ranks
    preds = [
        pd.read_csv(Path(cfg.output.path) / f"{pred}.csv")[cfg.data.target].to_numpy()
        for pred in tqdm(cfg.preds, desc="Loading predictions", colour="red", total=len(cfg.preds))
    ]

    # Calculate average predictions
    submit[cfg.data.target] = calculate_sigmoid_preds(preds)

    # Save the ensembled submission
    submit.to_csv(Path(cfg.output.path) / f"{cfg.output.name}.csv", index=False)


if __name__ == "__main__":
    _main()
