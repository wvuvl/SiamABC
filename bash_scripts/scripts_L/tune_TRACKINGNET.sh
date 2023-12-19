python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-31-18-28-55_Tracking_SiamABC_resnet50_layer_4_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/TrackingNet/TRAIN_11/frames' \
    --json_file '/luna_data/zaveri/code/SiamABC/TrackingNet_TRAIN_11.json' \
    --dataset 'trackingnet' \
    --num_trials 100 \
    --model_size 'L' \
    --trial_per_gpu 8