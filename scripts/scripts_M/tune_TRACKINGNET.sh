python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC _cross_att/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2024-01-27-00-32-30_Tracking_SiamABC_cross_att_ssl_M/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/TrackingNet/TRAIN_11/frames' \
    --json_file '/luna_data/zaveri/code/SiamABC _cross_att/TrackingNet_TRAIN_11.json' \
    --dataset 'trackingnet' \
    --num_trials 400 \
    --model_size 'M' \
    --trial_per_gpu 8