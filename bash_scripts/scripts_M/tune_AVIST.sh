python3 tune_tpe.py \
    --config_path '/luna_data/zaveri/code/SiamABC/core/config' \
    --weights_path '/luna_data/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/luna_data/zaveri/SOTA_Tracking_datasets/avist/sequences' \
    --json_file '/luna_data/zaveri/SOTA_Tracking_datasets/avist/AVIST.json' \
    --dataset 'AVIST' \
    --num_trials 400 \
    --model_size 'M'\
    --trial_per_gpu 16