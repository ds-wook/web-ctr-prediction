python -m scripts.covert_to_parquet
sh scripts/shell/sampling_dataset.sh
sh scripts/shell/lgb_experiment.sh
sh scripts/shell/cb_experiment.sh
sh scripts/shell/xdeepfm_experiment.sh
sh scripts/shell/fibinet_experiment.sh
python -m scripts.ensemble
