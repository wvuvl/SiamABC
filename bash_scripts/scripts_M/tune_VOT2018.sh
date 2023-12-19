python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/code/SiamABC/vot2018/VOT2018' \
    --json_file '/luna_data/zaveri/code/SiamABC/vot2018/VOT2018/VOT2018.json' \
    --dataset 'VOT2018' \
    --num_trials 500 \
    --model_size 'M'