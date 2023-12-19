# python3 tune_tpe.py \
#     --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
#     --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-24-21-24-32_Tracking_SiamABC_small_full/AEVT/trained_model_ckpt_17.pt' \
#     --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
#     --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
#     --dataset 'DTB70' \
#     --num_trials 500 \
#     --model_size 'S' \
#     --trial_per_gpu 16

# python3 tune_tpe.py \
#     --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
#     --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-31-18-28-55_Tracking_SiamABC_resnet50_layer_4_full/AEVT/trained_model_ckpt_20.pt' \
#     --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
#     --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
#     --dataset 'DTB70' \
#     --num_trials 500 \
#     --model_size 'L' \
#     --trial_per_gpu 16

python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
    --dataset 'DTB70' \
    --num_trials 500 \
    --model_size 'M' \
    --trial_per_gpu 16

python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-29-21-02-32_Tracking_SiamABC_regmetX_full/AEVT/trained_model_ckpt_15.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/DTB70.json' \
    --dataset 'DTB70' \
    --num_trials 500 \
    --model_size 'XL' \
    --trial_per_gpu 16
