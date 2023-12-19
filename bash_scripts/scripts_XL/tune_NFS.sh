python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-29-21-02-32_Tracking_SiamABC_regmetX_full/AEVT/trained_model_ckpt_15.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/NFS/NFS30.json' \
    --dataset 'NFS30' \
    --num_trials 400 \
    --model_size 'XL'
