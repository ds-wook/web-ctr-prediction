# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_parquet("../input/web-ctr-prediction/train.parquet")

# %%
print(train.shape)

# %%
train["Click"].value_counts(normalize=True)

# %%
sns.countplot(data=train, x="Click")

# %%
train.head()
# %%
train["F38"].unique()
# %%
sns.histplot(data=train, x="F38", bins=100)
plt.show()
# %%
train["F01"].value_counts()
# %%
sns.histplot(data=train, x="F04", bins=100)
plt.show()
# %%
sns.histplot(data=train, x="F06", bins=100)
plt.show()
# %%
train["F04_log"] = np.log1p(train["F04"])
sns.histplot(data=train, x="F04_log", bins=100)
plt.show()


# %%
from dataclasses import dataclass


@dataclass
class Config:
    preds = []

    class data:
        path = "input/web-ctr-prediction/"
        meta = "res/meta/"
        submit = "sample_submit"
        train = "train_sample"
        test = "test"
        submit = "sample_submission"
        n_splits = 5
        seed = 1119
        sampling = 0.3

    class generator:
        cat_features = []
        num_features = []
        drop_features = ["ID"]
        sparse_features = []
        dense_features = []

    class models:
        name = "lightgbm"
        params = {}
        path = "res/models/"
        results = "5fold-ctr-lightgbm-0.4-seed1119"
        early_stopping_rounds = 100
        num_boost_round = 1000
        verbose_eval = 100
        seed = 1119
        device = "cuda:0"
        l2_reg_linear = 0.0001
        l2_reg_embedding = 0.0001
        dnn_activation = "prelu"
        lr = 0.0001
        verbose = 1
        patience = 2
        mode = "max"
        batch_size = 4096
        epochs = 10

    class output:
        path = "output"
        submission = "sample_submission"
        name = "sigmoid-ensemble-final-4models-2sample"


cfg = Config()


# %%
