python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-27-15-40-48_Tracking_SiamABC_resnet50_got10k/AEVT/trained_model_ckpt_23.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/val' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/GOT10KVAL.json' \
    --dataset 'GOT10KVAL' \
    --num_trials 400 \
    --model_size 'M' \
    --trial_per_gpu 16


# python3 tune_tpe.py \
#     --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
#     --weights_path '/luna_data/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt' \
#     --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/val' \
#     --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/GOT10KVAL.json' \
#     --dataset 'GOT10KVAL' \
#     --num_trials 100 \
#     --model_size 'XL'