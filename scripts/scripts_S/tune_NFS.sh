python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-02-18-01-34-04_Tracking_SiamABC_S_mixed_polarized_att_w_corr_att_tn_trained/SiamABC/trained_model_ckpt_19.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/NFS/NFS30.json' \
    --dataset 'NFS30' \
    --num_trials 400 \
    --model_size 'S' \
    --trial_per_gpu 8