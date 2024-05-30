MODEL_NAME="lightgbm"
SAMPLING=0.4

for seed in 414 602
do
    python src/train.py \
        data.train=train_sample_${SAMPLING}_seed${seed} \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}

    python src/predict.py \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed} \
        output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}
done