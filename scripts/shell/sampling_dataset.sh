for sampling in 0.4 0.45
do
    for seed in 517 1119
    do
        python -m.sampling \
            data.seed=${seed} \
            data.sampling=${sampling} \
            data.train=train_sample_${sampling}_seed${seed}
    done
done
