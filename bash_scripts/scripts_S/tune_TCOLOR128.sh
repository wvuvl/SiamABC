python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-24-21-24-32_Tracking_SiamABC_small_full/AEVT/trained_model_ckpt_17.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/Temple-color-128' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/Temple-color-128/TColor128.json' \
    --dataset 'TCOLOR128' \
    --num_trials 400 \
    --model_size 'S'
