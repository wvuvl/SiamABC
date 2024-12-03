python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC _cross_att/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2024-01-27-00-32-30_Tracking_SiamABC_cross_att_ssl_M/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/code/SiamABC _cross_att/vot2018/VOT2018' \
    --json_file '/luna_data/zaveri/code/SiamABC _cross_att/vot2018/VOT2018/VOT2018.json' \
    --dataset 'VOT2018' \
    --num_trials 500 \
    --model_size 'M'