python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-26-15-01-39_Tracking_SiamABC_small_got10k/SiamABC/trained_model_ckpt_22.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/val' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/GOT10KVAL.json' \
    --dataset 'GOT10KVAL' \
    --num_trials 400 \
    --model_size 'S' \
    --trial_per_gpu 8
