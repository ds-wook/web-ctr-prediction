# web-ctr-prediction
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
웹 광고 클릭률 예측 AI 경진대회


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
|5fold-lightgbm-0.45|0.7833|0.7848|-|
|5fold-catboost-0.45|0.7755|-|-|
|5fold-wide&deep-0.45|0.7812|0.7840|-|
|5fold-deepFM-0.45|0.7791|0.7848|-|


## Summary
[TBD]

## Negative Sampling
추천분야에서 negative sampling은 매우 중요합니다. 대용량의 데이터를 학습 할수 없는 상황이라면 이러한 방법은 매우 효과적입니다.

## Features
#### Hash feature
[TBD]

#### Gauss Rank
[TBD]

### Model
정형 데이터의 특성상 GBDT 모델과 NN 모델을 학습해서 앙상블 하자는 전략을 세웠습니다.

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
+ Data Sampling seed를 바꿔가며 학습한 결과를 앙상블함
+ AUC를 최적화하기위해 Rank 방법론 적용

## Doesn't Work
+ Day Cross validation
+ Day feature
+ Catboost with cat_features parameter
+ XGBoost with GPU


## Reference
- []
- [Ensemble](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/276138)