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
plt.show()
# %%
train.head()
# %%
train.info()

# %%
cat_features = train.dtypes[train.dtypes == "object"].index.tolist()
cat_features
# %%
num_features = train.dtypes[train.dtypes != "object"].index.tolist()
# %%
num_features
# %%
sns.histplot(data=train, x="F04", bins=100)
plt.show()
# %%
for col in num_features:
    sns.histplot(data=train, x=col, bins=100)
    plt.show()

# %%
