python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2024-02-22-21-26-44_Tracking_SiamABC/SiamABC/trained_model_ckpt_18.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/OTB' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/OTB/OTB.json' \
    --dataset 'OTB2015' \
    --num_trials 400 \
    --model_size 'M' \
    --trial_per_gpu 5 \
    --tta

# python3 tune_tpe.py \
#     --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
#     --weights_path '/luna_data/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/SiamABC/trained_model_ckpt_9.pt' \
#     --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/OTB' \
#     --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/OTB/OTB.json' \
#     --dataset 'OTB2015' \
#     --num_trials 100 \
#     --model_size 'XL'