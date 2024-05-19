# web-ctr-prediction
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
웹 광고 클릭률 예측 AI 경진대회


## Setting
- CPU: i7-11799K core 8
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3090 Ti

## Cross Validation
+ Stratified Group KFold

## Ensemble Strategy
+ Rank Ensemble

## Model Process
+ Boosting is All you need
+ CTR Model


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
|5fold-lightgbm-0.35|0.7825|-|-|
|5fold-catboost-0.35|-|-|-|


## Submit

## Doesn't Work
+ meta feature: mean features
+ forcasting model: NBeat is not performance


## Reference

- [Model](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [Loss](https://www.sciencedirect.com/science/article/pii/S0169207021001679)
- [Ensemble](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/276138)