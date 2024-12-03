python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-02-18-01-34-04_Tracking_SiamABC_S_mixed_polarized_att_w_corr_att_tn_trained/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
    --dataset 'DTB70' \
    --num_trials 500 \
    --model_size 'S' \
    --trial_per_gpu 8 \
    --tta

python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-02-22-21-26-44_Tracking_SiamABC/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
    --dataset 'DTB70' \
    --num_trials 500 \
    --model_size 'M' \
    --trial_per_gpu 8 \
    --tta

# python3 tune_tpe.py \
#     --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
#     --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-31-18-28-55_Tracking_SiamABC_resnet50_layer_4_full/SiamABC/trained_model_ckpt_20.pt' \
#     --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
#     --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
#     --dataset 'DTB70' \
#     --num_trials 500 \
#     --model_size 'L' \
#     --trial_per_gpu 8



# python3 tune_tpe.py \
#     --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
#     --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-29-21-02-32_Tracking_SiamABC_regmetX_full/SiamABC/trained_model_ckpt_15.pt' \
#     --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
#     --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
#     --dataset 'DTB70' \
#     --num_trials 500 \
#     --model_size 'XL' \
#     --trial_per_gpu 16
