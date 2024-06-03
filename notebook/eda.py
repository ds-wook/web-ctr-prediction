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
