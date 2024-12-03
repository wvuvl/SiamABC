python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-02-18-01-34-04_Tracking_SiamABC_S_mixed_polarized_att_w_corr_att_tn_trained/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/code/SiamABC_cross_att/vot2018/VOT2018' \
    --json_file '/new_local_storage/zaveri/code/SiamABC_cross_att/vot2018/VOT2018/VOT2018.json' \
    --dataset 'VOT2018' \
    --num_trials 500 \
    --model_size 'S' \
    --tta