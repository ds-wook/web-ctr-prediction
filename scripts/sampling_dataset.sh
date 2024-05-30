for sampling in 0.5
do
    for seed in 517 1119
    do
        python src/sampling.py \
            data.seed=${seed} \
            data.sampling=${sampling} \
            data.train=train_sample_${sampling}_seed${seed}
    done
done