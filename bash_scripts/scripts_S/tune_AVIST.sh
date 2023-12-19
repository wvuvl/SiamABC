python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-11-01-22-11-46_Tracking_SiamABC_small_full_no_correspondence_loss/AEVT/trained_model_ckpt_15.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/sequences' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/AVIST.json' \
    --dataset 'AVIST' \
    --num_trials 400 \
    --model_size 'S' \
    --trial_per_gpu 16