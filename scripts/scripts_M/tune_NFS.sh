python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-02-22-21-26-44_Tracking_SiamABC/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/NFS/NFS30.json' \
    --dataset 'NFS30' \
    --num_trials 400 \
    --model_size 'M'