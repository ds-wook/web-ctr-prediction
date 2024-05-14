from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from engine import train_step, valid_step
from models import DLRM


def run_trainer(cfg: DictConfig, train_loader: torch.utils.data.DataLoader) -> None:

    # Define the model, loss function and optimizer
    model = DLRM(26, 13, 9530, 128, [256, 128], [128, 128], self_interaction=True)
    model.to(cfg.models.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.models.lr)

    for epoch in range(cfg.models.epochs + 1):
        for i, (x_sparse, x_dense, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = model(x_sparse, x_dense)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
