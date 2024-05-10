from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import rankdata


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.3.1")
def _main(cfg: DictConfig):
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")
    preds = [
        pd.read_csv(Path(cfg.output.path) / f"{cfg.preds[i]}.csv")[cfg.data.target].to_numpy()
        for i in range(len([*cfg.preds]))
    ]
    weights = [1 / len(preds) for _ in range(len(preds))]

    preds = [rankdata(pred) / pred.shape[0] for pred in preds]
    submit[cfg.data.target] = np.average(preds, weights=weights, axis=0)

    submit.to_csv(Path(cfg.output.path) / f"{cfg.output.name}.csv", index=False)


if __name__ == "__main__":
    _main()
