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
   $ sh scripts/run_wdl_experiment.sh
   $ sh scripts/run_deepfm_experiment.sh
   $ python src/ensemble.py
   ```

   Examples are as follows.

   ```sh
    export PYTHONHASHSEED=0

    MODEL_NAME="lightgbm"
    SAMPLING=0.4
    SEED=42

    python src/sampling.py \
        data.seed=${SEED} \
        data.sampling=${SAMPLING} \
        data.train=train_day_sample_${SAMPLING}_seed${SEED} \

    python src/train.py \
        data.train=train_day_sample_${SAMPLING}_seed${SEED} \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-day-seed${SEED}

    python src/predict.py \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-day-seed${SEED} \
        output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-day-seed${SEED}
   ```


## Benchmark
||cv|public-lb|private-lb|
|-----|--|---------|----------|
|5fold-lightgbm-0.4|0.7828|0.7848|-|
|5fold-catboost-0.4|0.7753|-|-|
|5fold-wide&deep-0.4|0.7806|-|-|
|5fold-deepFM-0.4|0.7790|-|-|
|5fold-lightgbm-0.45|0.7833|0.7853|-|
|5fold-catboost-0.45|0.7755|0.7766|-|
|5fold-wide&deep-0.45|0.7812|0.7840|-|
|5fold-deepFM-0.45|0.7791|0.7801|-|


## Summary
![model summary](https://github.com/ds-wook/web-ctr-prediction/assets/46340424/442e1804-6dd1-45cb-a9af-35ce6f6d3100)

Simple is better than complex

## Negative Sampling
Negative sampling is very important in recommendation systems. This method is very effective when it is not possible to train on large volumes of data.

## Features
#### Hash feature
![hash features](https://github.com/ds-wook/web-ctr-prediction/assets/46340424/0d7826bb-7754-4c46-b668-3bb44fbd595c)


#### Gauss Rank
[TBD]
Routine to rank a set of given ensemble forecasts according to their "value"

### Model
Due to the nature of Tabular data, we devised a strategy to train GBDT models and NN models and then ensemble them.

#### GBDT
+ LightGBM
    + With hash feature
    + StratifiedKfold: 5

+ CatBoost
    + Use GPU
    + Not used cat_features parameter
    + StratifiedKFold: 5

#### Deep CTR
+ Wide & Deep
    + With Gauss Rank
    + StratifiedKFold: 5  

+ DeepFM
    + With Gauss Rank
    + StratifiedKFold: 5

### Ensemble
#### Rank Ensemble 
+ Use 18 models
+ Ensemble the results by changing the data sampling seed during training.
+ Applied a ranking methodology to optimize AUC.

## Doesn't Work
+ Day Cross validation
+ Day feature
+ Catboost with cat_features parameter
+ XGBoost with GPU


## Reference
+ [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://lightgbm.readthedocs.io/en/stable/)
+ [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)
+ [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247)
+ [CatBoost is a high-performance open source library for gradient boosting on decision trees](https://catboost.ai/)
+ [Efficient Click-Through Rate Prediction for Developing Countries via Tabular Learning](https://arxiv.org/pdf/2104.07553)
+ [Hash Embeddings for Efficient Word Representations](https://proceedings.neurips.cc/paper/2017/file/f0f6ba4b5e0000340312d33c212c3ae8-Paper.pdf)
+ [Rank Ensemble](https://www.kaggle.com/code/finlay/amex-rank-ensemble)
