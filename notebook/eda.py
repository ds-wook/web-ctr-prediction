# %%
import numpy as np
import pandas as pd

train = pd.read_parquet("../input/web-ctr-prediction/train_sample.parquet")

train.head()
# %%
train.shape
# %%
train = pd.read_parquet("../input/web-ctr-prediction/train_sample_baseline.parquet")

train.head()
# %%
train.dtypes[train.dtypes == "object"].index
# %%
train.loc[:, "F01"] = train.loc[:, "F01"].apply(lambda x: "F01" + str(x))
train["F01"].head()
# %%
train.loc[:, "F01"] = train.loc[:, "F01"].apply(lambda x: hash(x) % 10**6)
train["F01"].head()
# %%
import pandas as pd

test = pd.read_csv("../input/web-ctr-prediction/test.csv")

# %%
test.to_parquet("../input/web-ctr-prediction/test.parquet")

# %%
