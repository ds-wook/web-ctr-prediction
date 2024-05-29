for sampling in 0.4 0.45 0.5
do
    for seed in 42 517 1119
    do
        python src/sampling.py \
            data.seed=$seed \
            data.sampling=$sampling \
            data.train=train_sample_$sampling_seed$seed
    done
done