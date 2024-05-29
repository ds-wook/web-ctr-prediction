# web-ctr-prediction
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
web ctr competition

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
   $ python scripts/covert_to_parquet.py
   $ sh scripts/run_lgb_experiment.sh
   $ sh scripts/run_cb_experiment.sh
   $ sh scripts/run_wdl_experiment.sh
   $ sh scripts/run_xdeepfm_experiment.sh
   $ sh scripts/run_fibinet_experiment.sh
   $ python src/ensemble.py
   ```

   Examples are as follows.

   ```sh
    MODEL_NAME="lightgbm"
    SAMPLING=0.45

    for seed in 517 1119
    do
        python src/train.py \
            data.train=train_sample_${SAMPLING}_seed$seed \
            models=${MODEL_NAME} \
            models.results=5fold-ctr-${MODEL_NAME}-count-${SAMPLING}-seed$seed

        python src/predict.py \
            models=${MODEL_NAME} \
            models.results=5fold-ctr-${MODEL_NAME}-count-${SAMPLING}-seed$seed \
            output.name=5fold-ctr-${MODEL_NAME}-count-${SAMPLING}-seed$seed
    done

   ```

## Summary
![model summary](https://github.com/ds-wook/web-ctr-prediction/assets/46340424/e0d915dd-fad1-4bc9-959b-90025c15043e)

Simple is better than complex

## Negative Sampling
Negative sampling is very important in recommendation systems. This method is very effective when it is not possible to train on large volumes of data.

## Features
#### Label Encoder
![kaggler](https://github.com/jeongyoonlee/Kaggler)

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
+ Wide & Deep
    + With Gauss Rank
    + StratifiedKFold: 5  

+ xDeepFM
    + With Gauss Rank
    + StratifiedKFold: 5

+ FiBiNET
    + With Gauss Rank
    + StratifiedKFold: 5

### Ensemble
#### Sigmoid Ensemble 
I used the concept of log-odds from logistic regression to construct an ensemble:
$$𝑓(𝑥)=\frac{1}{1 + e^{-x}}$$
$$f^{-1}(x)= \log(\frac{x}{1-x})$$
$$\hat{y}=f(\frac{1}{n}\sum_i^n f^{-1}(x_i))=f(\mathbb{E}[f^{-1}(X)])$$

+ It seems to perform better than other ensembles (Rank, Voting).
+ Since the prediction values are probabilities, we used the logit function and its inverse to perform bagging for the ensemble.


## Benchmark
+ Each model result

|Model|cv|public-lb|private-lb|
|-----|--|---------|----------|
|lightgbm-0.45|**0.7849**|**0.7863**|-|
|catboost-0.45|0.7755|0.7766|-|
|wide&deep-0.45|0.7812|0.7840|-|
|DeepFM-0.45|0.7791|0.7801|-|
|xDeepFM-0.45|0.7819|0.7801|-|
|AutoInt-0.45|0.7801|-|-|
|FiBiNET-0.45|0.7819|-|-|

+ Ensemble result

|Method|public-lb|private-lb|
|------|---------|----------|
|Rank Ensemble|0.7889|-|
|Average Ensemble|0.7892|-|
|Weighted average Ensemble|0.7891|-|
|Sigmoid Ensemble|**0.7893**|-|


## Doesn't Work
+ Day Cross validation
+ Day feature
+ Catboost with cat_features parameter
+ XGBoost with GPU
+ Hash features: need more RAM

## Reference
+ [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://lightgbm.readthedocs.io/en/stable/)
+ [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)
+ [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247)
+ [CatBoost is a high-performance open source library for gradient boosting on decision trees](https://catboost.ai/)
+ [Efficient Click-Through Rate Prediction for Developing Countries via Tabular Learning](https://arxiv.org/pdf/2104.07553)
+ [Label Encoder](https://github.com/jeongyoonlee/Kaggler/blob/master/kaggler/preprocessing/categorical.py)
+ [Hash Embeddings for Efficient Word Representations](https://proceedings.neurips.cc/paper/2017/file/f0f6ba4b5e0000340312d33c212c3ae8-Paper.pdf)
+ [Sigmoid Ensemble](https://www.kaggle.com/competitions/amex-default-prediction/discussion/329103)
