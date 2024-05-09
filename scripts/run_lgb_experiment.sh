export PYTHONHASHSEED=0

MODEL_NAME="lightgbm"
SAMPLING_NUMBER=0.35
SAMPING="postive_sampling"

# python src/sampling.py \
#     mode=${SAMPING} \
#     data.sampling=${SAMPLING_NUMBER}

python src/train.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}

python src/predict.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER}