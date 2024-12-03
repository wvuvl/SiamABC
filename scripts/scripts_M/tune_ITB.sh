python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2024-02-22-21-26-44_Tracking_SiamABC/SiamABC/trained_model_ckpt_18.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/ITB' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/ITB/ITB.json' \
    --dataset 'ITB' \
    --num_trials 400 \
    --model_size 'M' \
    --trial_per_gpu 8 \
    --tta