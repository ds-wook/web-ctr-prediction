export PYTHONHASHSEED=0

MODEL_NAME="lightgbm"
SAMPLING_NUMBER=0.05

python src/sampling.py \
    sampling=${SAMPLING_NUMBER}

python src/train.py \
    models=${MODEL_NAME} \
    models.results=5fold-ctr-${MODEL_NAME}

python src/predict.py \
    models=${MODEL_NAME}