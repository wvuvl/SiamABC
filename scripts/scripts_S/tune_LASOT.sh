python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC_cross_att/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2024-02-18-01-34-04_Tracking_SiamABC_S_mixed_polarized_att_w_corr_att_tn_trained/SiamABC/trained_model_ckpt_18.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/LaSOT/LASOTTEST.json' \
    --dataset 'LASOT' \
    --num_trials 400 \
    --model_size 'S' \
    --trial_per_gpu 8 \
    --tta