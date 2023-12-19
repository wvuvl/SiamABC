python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-11-01-22-11-46_Tracking_SiamABC_small_full_no_correspondence_loss/AEVT/trained_model_ckpt_15.pt' \
    --base_path '/luna_data/zaveri/code/SiamABC/vot2018/VOT2018' \
    --json_file '/luna_data/zaveri/code/SiamABC/vot2018/VOT2018/VOT2018.json' \
    --dataset 'VOT2018' \
    --num_trials 500 \
    --model_size 'S'