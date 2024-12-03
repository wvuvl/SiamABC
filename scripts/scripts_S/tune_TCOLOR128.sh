python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC _cross_att/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2024-01-27-00-07-35_Tracking_SiamABC_cross_att_ssl_S/SiamABC/trained_model_ckpt_22.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/Temple-color-128' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/Temple-color-128/TColor128.json' \
    --dataset 'TCOLOR128' \
    --num_trials 400 \
    --model_size 'S'
