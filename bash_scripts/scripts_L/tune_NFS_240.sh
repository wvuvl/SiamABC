python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-31-18-28-55_Tracking_SiamABC_resnet50_layer_4_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/NFS/NFS240.json' \
    --dataset 'NFS240' \
    --num_trials 400 \
    --model_size 'L' \
    --trial_per_gpu 12
