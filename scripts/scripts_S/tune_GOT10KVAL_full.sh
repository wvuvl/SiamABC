python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC _cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-01-27-00-07-35_Tracking_SiamABC_cross_att_ssl_S/SiamABC/trained_model_ckpt_22.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/val' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/GOT10KVAL.json' \
    --dataset 'GOT10KVAL' \
    --num_trials 400 \
    --model_size 'S' \
    --trial_per_gpu 8
