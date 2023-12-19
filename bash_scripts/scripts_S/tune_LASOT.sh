python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-24-21-24-32_Tracking_SiamABC_small_full/AEVT/trained_model_ckpt_17.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LASOTTEST.json' \
    --dataset 'LASOT' \
    --num_trials 100 \
    --model_size 'S' \
    --trial_per_gpu 16