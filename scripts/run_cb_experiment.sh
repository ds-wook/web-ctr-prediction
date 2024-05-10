export PYTHONHASHSEED=0

MODEL_NAME="catboost"
SAMPLING_NUMBER=0.35
SAMPING="negative_sampling"
TRAIN="train_sample_65"

python src/sampling.py \
    mode=${SAMPING} \
    data.sampling=${SAMPLING_NUMBER}

python src/train.py \
    data.train=${TRAIN} \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}

python src/predict.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}