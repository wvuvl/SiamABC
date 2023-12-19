python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-11-04-22-08-52_Tracking_SiamABC_resnet50_layer_4_L_got10k/AEVT/trained_model_ckpt_22.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/val' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/GOT10KVAL.json' \
    --dataset 'GOT10KVAL' \
    --num_trials 400 \
    --model_size 'L' \
    --trial_per_gpu 16


# python3 tune_tpe.py \
#     --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
#     --weights_path '/luna_data/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt' \
#     --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/val' \
#     --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/GOT10KVAL.json' \
#     --dataset 'GOT10KVAL' \
#     --num_trials 100 \
#     --model_size 'XL'