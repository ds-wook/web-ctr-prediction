# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_parquet("../input/web-ctr-prediction/train_sample_baseline.parquet")

# %%
print(train.shape)

# %%
train["Click"].value_counts(normalize=True)

# %%
num_features = train.select_dtypes("int64").columns.to_list() + train.select_dtypes("float64").columns.to_list()
print(num_features)
# %%
train[train[num_features] < 0].sum()
# %%
sns.heatmap(train[num_features].corr())
plt.show()

# %%
from scipy.stats import boxcox

# train 데이터셋에 대해 특성 F01에 Box-Cox 변환 적용
train["F01_boxcox_transformed"], _ = boxcox(train["F01"])  # 1을 더해주는 이유는 0 값이 존재할 경우에 대비한 것입니다.


# %%
sns.histplot(data=train, x="F06")

# %%
