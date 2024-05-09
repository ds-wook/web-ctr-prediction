export PYTHONHASHSEED=0

MODEL_NAME="lightgbm"
SAMPLING_NUMBER=0.4
SAMPING="postive_sampling"      
learing_rate=0.3

python src/sampling.py \
    mode=${SAMPING} \
    data.sampling=${SAMPLING_NUMBER}

python src/train.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}

python src/predict.py \
    models=${MODEL_NAME} \
    output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING_NUMBER} \