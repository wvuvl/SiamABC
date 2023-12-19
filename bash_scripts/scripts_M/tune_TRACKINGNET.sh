python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/TrackingNet/TRAIN_11/frames' \
    --json_file '/new_local_storage/zaveri/code/SiamABC/TrackingNet_TRAIN_11.json' \
    --dataset 'trackingnet' \
    --num_trials 100 \
    --model_size 'M' \
    --trial_per_gpu 16