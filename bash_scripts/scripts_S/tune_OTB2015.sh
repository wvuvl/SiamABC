python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-24-21-24-32_Tracking_SiamABC_small_full/AEVT/trained_model_ckpt_17.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/OTB' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/OTB/OTB.json' \
    --dataset 'OTB2015' \
    --num_trials 100 \
    --model_size 'S' 