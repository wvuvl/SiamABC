python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-31-18-28-55_Tracking_SiamABC_resnet50_layer_4_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB/ITB.json' \
    --dataset 'ITB' \
    --num_trials 400 \
    --model_size 'L' \
    --trial_per_gpu 16