export PYTHONHASHSEED=0

MODEL_NAME="lightgbm"
SAMPLING_NUMBER=0.35
SAMPING="postive_sampling"
SEED=1119

python src/sampling.py \
    mode=${SAMPING} \
    data.sampling=${SAMPLING_NUMBER} \
    data.seed=${SEED}

python src/train.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}-seed${SEED}

python src/predict.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}-seed${SEED} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}-seed${SEED}