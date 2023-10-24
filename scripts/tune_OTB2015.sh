python3 tune_tpe.py \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/OTB' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/OTB/OTB.json' \
    --dataset 'OTB2015' \
    --num_trials 1 \
    --model_size 'XL'