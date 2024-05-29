python scripts/covert_to_parquet.py
sh scripts/sampling_dataset.sh
sh scripts/lgb_experiment.sh
sh scripts/cb_experiment.sh
sh scripts/wdl_experiment.sh
sh scripts/xdeepfm_experiment.sh
sh scripts/fibinet_experiment.sh
python src/ensemble.py
