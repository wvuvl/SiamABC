import torch

def convert_model(checkpoint_path: str,save_path: str = None):
    checkpoint = torch.load(checkpoint_path)
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("module.")}
    
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("module.classifier") or key.startswith("module.predictor"): pass
        else:  new_state_dict[key] = state_dict[key]
            
    print(new_state_dict.keys())
    torch.save(new_state_dict, save_path)


if __name__ == '__main__':
    model_path = '/luna_data/zaveri/code/experiments/2024-02-29-01-15-49_Tracking_SiamABC_M_fast_mixed_att_at_neck/AEVT/trained_model_ckpt_19.pt'
    convert_model(model_path, '/luna_data/zaveri/code/wacv_code/SiamABC/assets/model_S_Small_v5.pt')