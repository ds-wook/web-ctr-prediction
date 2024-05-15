export PYTHONHASHSEED=0

MODEL_NAME="wdl"
SAMPLING=0.4
SEED=414

python src/train.py \
    data.train=train_sample_${SAMPLING}_seed${SEED} \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED}

python src/predict.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED}