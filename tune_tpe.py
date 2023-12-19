import argparse
from AEVT_tracker import AEVTTracker
from core.utils.torch_stuff import load_from_lighting
from eval_SiamABC import auc_otb, eao_vot, auc_got10k, auc_lasot, auc_nfs, auc_uav123, auc_avist, auc_tcolor128, auc_dtb70, auc_trackingnet, auc_itb
import os

from pprint import pprint
import yaml 
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import RunConfig
from ray.tune import CLIReporter

from core.models.AEVT import AEVTNet

parser = argparse.ArgumentParser(description='parameters for SiamABC tracker')
parser.add_argument('--config_path', default='core/config', type=str, help='config path')
parser.add_argument('--config_name', default='AEVT_tracker', type=str, help='config name')
parser.add_argument('--model_size', default='S', type=str, help='model prefrence')
parser.add_argument('--weights_path', default='/new_local_storage/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt', type=str, help='weights path')
parser.add_argument('--cache_dir', default='./TPE_results', type=str, help='directory to store cache')
parser.add_argument('--gpu_nums', default=1, type=int, help='gpu numbers')
parser.add_argument('--trial_per_gpu', default=8, type=int, help='trail per gpu')
parser.add_argument('--align', default='True', type=str, help='align')
parser.add_argument('--num_trials', default='1', type=int)
parser.add_argument('--dynamic_update', action='store_true', default='False')

# data specific
parser.add_argument('--base_path', type=str, help='base path', default='/new_local_storage/zaveri/code/SiamABC/vot2018/VOT2018')
parser.add_argument('--json_file', type=str, help='json path', default='/new_local_storage/zaveri/code/SiamABC/vot2018/VOT2018/VOT2018.json')
parser.add_argument('--dataset', default='VOT2018', type=str, help='dataset')

args = parser.parse_args()

params = {}
params["N"] = 75
params["penalty_k"] = 0.001 #hp.quniform('penalty_k', 0.001, 0.2, 0.001)
params['lr']= 0.3 #hp.quniform('scale_lr', 0.3, 0.8, 0.001)
params['window_influence']= 0.15 #hp.quniform('window_influence', 0.15, 0.65, 0.001)
params["config_path"] = args.config_path #"core/config"
params ["config_name"] = args.config_name #"AEVT_tracker"
params["weights_path"] = args.weights_path #"/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/AEVT/models/small/trained_model_ckpt_20.pt"

params["base_path"] = args.base_path
params["json_file"] = args.json_file
print('tuning range: ')
pprint(params)    
    


def get_tracker(config_path, config_name, weights_path, penalty_k, window_influence, lr) -> AEVTTracker:
    with open(os.path.join(args.config_path,'tracker/siam_tracker.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    config["penalty_k"] = penalty_k
    config["window_influence"] = window_influence
    config["lr"] = lr
    model = AEVTNet(model_size=args.model_size)
    model = load_from_lighting(model, weights_path).cuda().eval()
    tracker = AEVTTracker(model=model, tracking_config=config)
    return tracker


main_dict = {}

def fitness(config):    
    
    config_path = params["config_path"] #"core/config"
    config_name = params ["config_name"] #"AEVT_tracker"
    weights_path = params["weights_path"] #"/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/AEVT/models/small/trained_model_ckpt_20.pt"
    
    base_path = params["base_path"]
    json_file = params["json_file"]
    
    penalty_k = config["penalty_k"]
    lr = config["lr"]
    window_influence = config["window_influence"]
    
    tracker = get_tracker(config_path=config_path, config_name=config_name, weights_path=weights_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr)
    
    print(args.dataset)
     
    
    # VOT 
    if args.dataset.startswith('VOT'):    
        
        eao = eao_vot(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file)
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, eao: {eao}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, eao: {eao}"] = eao
        train.report({"AUC": eao})

    # OTB
    if args.dataset.startswith('OTB'):
        auc = auc_otb(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
        
    # GOT10K
    if args.dataset.startswith('GOT'):
        auc = auc_got10k(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
        
    # LASOT
    if args.dataset.startswith('LASOT'):
        auc = auc_lasot(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
        
    # NFS
    if args.dataset.startswith('NFS'):
        auc = auc_nfs(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})

    # UAV123
    if args.dataset.startswith('UAV'):
        auc = auc_uav123(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})

    # AVIST
    if args.dataset.startswith('AVIST'):
        auc = auc_avist(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
        
        
    # TCOLOR128
    if args.dataset.lower().startswith('tcolor'):
        auc = auc_tcolor128(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
        
    # DTB70
    if args.dataset.startswith('DTB'):
        auc = auc_dtb70(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
    

    # trackingnet
    if args.dataset.lower().startswith('trackingnet'):
        auc = auc_trackingnet(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})
    
    # ITB
    if args.dataset.startswith('ITB'):
        auc = auc_itb(tracker=tracker, dataset_name=args.dataset, data_path=base_path, penalty_k=penalty_k, window_influence=window_influence, lr=lr, base_path=base_path, json_path=json_file )
        print(f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}")
        main_dict[f"penalty_k: {penalty_k}, window_influence: {window_influence}, lr:{lr}, AUC: {auc}"] = auc
        train.report({"AUC": auc})

    # train.report({"AUC": 1.0})


def main():
    
    # while True:
    #     params["penalty_k"] = 0.0620 #np.random.uniform(0.001, 0.2) #hp.quniform('penalty_k', 0.001, 0.2, 0.001)
    #     params['lr']= 0.7650 #np.random.uniform(0.3, 0.8) #hp.quniform('scale_lr', 0.3, 0.8, 0.001)
    #     params['window_influence']= 0.380 #np.random.uniform(0.15, 0.65) #hp.quniform('window_influence', 0.15, 0.65, 0.001)
    #     fitness(params)
    #     print(max(main_dict, key=main_dict.get))
   
    #     break
    
    # ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8,  object_store_memory=500000000)
    config = {
            "penalty_k": tune.quniform( 0.001, 0.2, 0.001),
            "lr": tune.quniform(0.3, 0.8, 0.001),
            "window_influence": tune.quniform( 0.15, 0.65, 0.001),
            # "N": tune.choice([35, 75, 90, 150])
        }
    
    # init_best_config = [{
    #         "penalty_k": 0.019,
    #         "lr": 0.745,
    #         "window_influence": 0.274,
    #     }]
    init_best_config = None
    
    algo = HyperOptSearch(metric="AUC", mode="max", points_to_evaluate=init_best_config)
    
    scheduler = AsyncHyperBandScheduler(
            metric='AUC',
            mode='max',
            max_t=500,
            grace_period=20
        )
    
    reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("AUC")
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(fitness),
            resources={"cpu": 1, "gpu": 1/args.trial_per_gpu}
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            scheduler=scheduler,
            num_samples=args.num_trials,
            # max_concurrent_trials=args.trial_per_gpu,  
                 
        ),
        param_space=config,
        run_config=RunConfig(progress_reporter=reporter)
    )
    
    # /home/ramzav/ray_results/fitness_2023-11-06_23-15-08 for SiamABC_M_got10k
    # # /home/ramzav/ray_results/fitness_2023-11-06_03-54-42 for SiamABC_XL_avist
    # /home/ramzav/ray_results/fitness_2023-11-08_01-03-22 for SiamABC_XL_uav123
    # tuner = tune.Tuner.restore('/home/ramzav/ray_results/fitness_2023-11-06_23-06-27', 
    #                            trainable=tune.with_resources(
    #                                 tune.with_parameters(fitness),
    #                                 resources={"cpu": 1, "gpu": 1/args.trial_per_gpu},
                                    
    #                                 ),
    #                            param_space=config,)
    
    results = tuner.fit()
    
    best_result = results.get_best_result("AUC", "max")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final AUC: {}".format(
        best_result.metrics["AUC"]))
    
    penalty_k = best_result.config["penalty_k"]
    lr = best_result.config["lr"]
    window_influence = best_result.config["window_influence"]
    AUC = best_result.metrics["AUC"]
    
    df = results.get_dataframe()
    df.to_csv(f"{args.dataset}_SiamABC_{args.model_size}_k_{penalty_k}_lr_{lr}_wi_{window_influence}_AUC_{AUC}_runs.csv")
    
    
    
if __name__ == "__main__":
    main()



