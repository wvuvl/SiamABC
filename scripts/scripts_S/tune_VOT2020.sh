python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2024-02-18-01-34-04_Tracking_SiamABC_S_mixed_polarized_att_w_corr_att_tn_trained/SiamABC/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/code/SiamABC_cross_att/vot2020_st/sequences' \
    --json_file '/luna_data/zaveri/code/SiamABC _cross_att/vot2018/VOT2018/VOT2018.json' \
    --dataset 'VOT2020' \
    --num_trials 500 \
    --model_size 'S' \
    --tta