MODEL_NAME="wdl"
SAMPLING=0.45

for seed in 517 1119
do
    python -m scripts.train \
        data.train=train_sample_${SAMPLING}_seed${seed} \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}

    python  -m scripts.predict \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed} \
        output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}
done