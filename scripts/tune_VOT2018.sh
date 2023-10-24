python3 tune_tpe.py \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt' \
    --base_path '/new_local_storage/zaveri/code/SiamABC/vot2018/VOT2018' \
    --json_file '/new_local_storage/zaveri/code/SiamABC/vot2018/VOT2018/VOT2018.json' \
    --dataset 'VOT2018' \
    --num_trials 1 \
    --model_size 'XL'