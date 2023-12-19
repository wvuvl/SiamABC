python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/OTB' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/OTB/OTB.json' \
    --dataset 'OTB2015' \
    --num_trials 100 \
    --model_size 'M'

# python3 tune_tpe.py \
#     --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
#     --weights_path '/luna_data/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt' \
#     --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/OTB' \
#     --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/OTB/OTB.json' \
#     --dataset 'OTB2015' \
#     --num_trials 100 \
#     --model_size 'XL'