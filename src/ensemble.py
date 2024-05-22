from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import rankdata


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.3.1")
def _main(cfg: DictConfig):
    # Load submission file
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

    # Load predictions and calculate ranks
    preds = [
        rankdata(pd.read_csv(Path(cfg.output.path) / f"{pred}.csv")[cfg.data.target].to_numpy()) / len(submit)
        for pred in cfg.preds
    ]

    # Calculate average predictions
    submit[cfg.data.target] = np.average(preds, axis=0)

    # Save thae ensembled submission
    submit.to_csv(Path(cfg.output.path) / f"{cfg.output.name}.csv", index=False)


if __name__ == "__main__":
    _main()
