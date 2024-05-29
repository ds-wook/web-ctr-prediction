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
