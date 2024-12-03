python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-02-22-21-26-44_Tracking_SiamABC/SiamABC/trained_model_ckpt_18.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LASOTTEST.json' \
    --dataset 'LASOT' \
    --num_trials 400 \
    --model_size 'M' \
    --trial_per_gpu 8 \
    --tta