from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


def calculate_sigmoid_preds(values: list[np.ndarray], weight: list[float]) -> np.ndarray:
    """
    Calculate the sigmoid result of the ensemble predictions.
    :param values: list of predictions
    :param weight: list of weights
    :return: ensemble prediction
    """
    values = np.array(values)
    weight = np.array(weight)

    logit_values = np.log(values / (1 - values))
    result = np.dot(weight, logit_values)

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

    # Calculate average predictions with equal weights
    submit[cfg.data.target] = calculate_sigmoid_preds(preds, [*cfg.weights])

    # Save the ensembled submission
    submit.to_csv(Path(cfg.output.path) / f"{cfg.output.name}.csv", index=False)


if __name__ == "__main__":
    _main()
