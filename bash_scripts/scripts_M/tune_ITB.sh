python3 tune_tpe.py \
    --config_path '/new_local_storage/zaveri/code/SiamABC/core/config' \
    --weights_path '/new_local_storage/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt' \
    --base_path '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB' \
    --json_file '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB/ITB.json' \
    --dataset 'ITB' \
    --num_trials 400 \
    --model_size 'M' \
    --trial_per_gpu 16