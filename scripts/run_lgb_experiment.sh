export PYTHONHASHSEED=0

MODEL_NAME="lightgbm"
SAMPLING=0.45
SEED=42

python src/sampling.py \
    data.seed=${SEED} \
    data.sampling=${SAMPLING} \
    data.train=train_sample_${SAMPLING}_seed${SEED} \

python src/train.py \
    data.train=train_sample_${SAMPLING}_seed${SEED} \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED}

python src/predict.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${SEED}