python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-29-21-02-32_Tracking_SiamABC_regmetX_full/AEVT/trained_model_ckpt_15.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB/ITB.json' \
    --dataset 'ITB' \
    --num_trials 400 \
    --model_size 'XL' \
    --trial_per_gpu 16