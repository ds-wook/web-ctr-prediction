# web-ctr-prediction
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

This repository is the 1st solution of [web ctr competition](https://dacon.io/competitions/official/236258/overview/description).


## Setting
- CPU: i7-11799K core 8
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3090 Ti


## Requirements

By default, `hydra-core==1.3.0` was added to the requirements given by the competition.
For `pytorch`, refer to the link at https://pytorch.org/get-started/previous-versions/ and reinstall it with the right version of `pytorch` for your environment.

You can install a library where you can run the file by typing:

```sh
$ conda env create --file environment.yaml
```

## Run code

Code execution for the new model is as follows:

Running the learning code shell.

   ```sh
    $ python -m scripts.covert_to_parquet
    $ sh scripts/shell/sampling_dataset.sh
    $ sh scripts/shell/lgb_experiment.sh
    $ sh scripts/shell/cb_experiment.sh
    $ sh scripts/shell/xdeepfm_experiment.sh
    $ sh scripts/shell/fibinet_experiment.sh
    $ python -m scripts.ensemble
   ```

   Examples are as follows.

   ```sh
    MODEL_NAME="lightgbm"
    SAMPLING=0.45

    for seed in 517 1119
    do
        python -m scripts.train \
            data.train=train_sample_${SAMPLING}_seed${seed} \
            models=${MODEL_NAME} \
            models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}

        python -m scripts.predict \
            models=${MODEL_NAME} \
            models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed} \
            output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}
    done
   ```

## Summary
![competition-model](https://github.com/ds-wook/web-ctr-prediction/assets/46340424/21f6f58c-1844-4d6b-a915-3afcacdca4a2)


Simple is better than complex

## Negative Sampling
Negative sampling is very important in recommendation systems. This method is very effective when it is not possible to train on large volumes of data.
In my experiment, I used seeds 414 and 602 for a 40% negative sample, and seeds 517 and 1119 for a 45% negative sample.

## Features
#### Label Encoder
I encoded the Label of each categorical dataset and trained them together, referring to the [kaggler](https://github.com/jeongyoonlee/Kaggler) code.


#### Count features
I encoded the frequency of occurrence of each categorical dataset and trained them.

#### Gauss Rank
![gauss rank](https://github.com/ds-wook/web-ctr-prediction/assets/46340424/4d9ce6bc-8d6c-41f4-b001-298bb4538265)

Routine to rank a set of given ensemble forecasts according to their "value".
This method normally distributes the distribution of each numerical data, resulting in better performance for the model. Experimental results show higher performance than ``MinMaxScaler``.

### Model
Considering the characteristics of tabular data, we devised a strategy to train GBDT models and NN models, and then ensemble them.

#### GBDT
+ LightGBM
    + With count features
    + StratifiedKfold: 5

+ CatBoost
    + Use GPU
    + Not used cat_features parameter
    + With count features
    + StratifiedKFold: 5

#### Deep CTR

+ xDeepFM
    + With Gauss Rank
    + StratifiedKFold: 5

+ FiBiNET
    + With Gauss Rank
    + StratifiedKFold: 5
    + Long training and inferencing time

### Ensemble
#### Sigmoid Ensemble 
I used the concept of log-odds from logistic regression to construct an ensemble:
$$\sigma(𝑥)=\frac{1}{1 + e^{-x}}$$  
$$\sigma^{-1}(x)= \log(\frac{x}{1-x})$$  
$$\hat{y}=\sigma(\frac{1}{n}\sum_i^n \sigma^{-1}(x_i))=\sigma(\mathbb{E}[\sigma^{-1}(X)])$$  

+ It seems to perform better than other ensembles (Rank, Voting).
+ Since the prediction values are probabilities, we used the logit function and its inverse to perform bagging for the ensemble.


## Benchmark
+ Each model result

|Model|cv|public-lb|private-lb|
|-----|--|---------|----------|
|LightGBM-0.45 sampling|**0.7850**|0.7863|0.7866|
|FiBiNET-0.45 sampling|0.7833|0.7861|0.7862|
|xDeepFM-0.45 sampling|0.7819|**0.7866**|**0.7867**|
|wide&deep-0.45 sampling|0.7807|0.7835|0.7837|
|AutoInt-0.45 sampling|0.7813|0.7846|0.7848|
|CatBoost-0.45 sampling|0.7765|0.7773|0.7778|

+ Ensemble result

|Method|public-lb|private-lb|
|------|---------|----------|
|Rank Ensemble|0.7889|-|
|Average Ensemble|0.7892|-|
|Weighted average Ensemble|0.7891|-|
|Sigmoid Ensemble|**0.7903**|**0.7905**|


## Doesn't Work
+ Day Cross validation
+ Day feature
+ Catboost with cat_features parameter
+ XGBoost with GPU
+ Hash features: need more RAM
+ DeepFM
+ LightGBM DART
  
## Reference
+ [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://lightgbm.readthedocs.io/en/stable/)
+ [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)
+ [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433)
+ [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170)
+ [CatBoost is a high-performance open source library for gradient boosting on decision trees](https://catboost.ai/)
+ [Efficient Click-Through Rate Prediction for Developing Countries via Tabular Learning](https://arxiv.org/pdf/2104.07553)
+ [Label Encoder](https://github.com/jeongyoonlee/Kaggler/blob/master/kaggler/preprocessing/categorical.py)
+ [Gauss Rank](https://github.com/aldente0630/gauss-rank-scaler)
+ [Sigmoid Ensemble](https://www.kaggle.com/competitions/amex-default-prediction/discussion/329103)
