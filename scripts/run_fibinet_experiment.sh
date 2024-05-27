export PYTHONHASHSEED=0

MODEL_NAME="fibinet"
SAMPLING=0.45
SEED=1119

python src/train.py \
    data.train=train_sample_${SAMPLING}_seed${SEED} \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED}

python src/predict.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED}