from typing import Dict, Any, Optional

import torch
import torch.nn as nn

import core.constants as constants



def calc_iou(reg_target: torch.Tensor, pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    target_area = (reg_target[..., 0] + reg_target[..., 2]) * (reg_target[..., 1] + reg_target[..., 3])
    pred_area = (pred[..., 0] + pred[..., 2]) * (pred[..., 1] + pred[..., 3])

    w_intersect = torch.min(pred[..., 0], reg_target[..., 0]) + torch.min(pred[..., 2], reg_target[..., 2])
    h_intersect = torch.min(pred[..., 3], reg_target[..., 3]) + torch.min(pred[..., 1], reg_target[..., 1])

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    return (area_intersect + smooth) / (area_union + smooth)


class BoxLoss(nn.Module):
    """
    BBOX Loss: optimizes IoU of bounding boxes
    Original implentation:
    losses = -torch.log(calc_iou(reg_target=target, pred=pred)) was computationally unstable
    those was replaced with: 1 - IoU
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        losses = 1 - calc_iou(target, pred)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.mean()




class SiamABCLoss(nn.Module):
    def __init__(self, coeffs: Dict[str, float]):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = BoxLoss()
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        self.coeffs = coeffs

    def _regression_loss(
        self, bbox_pred: torch.Tensor, reg_target: torch.Tensor, reg_weight: torch.Tensor
    ) -> torch.Tensor:
        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self.regression_loss(bbox_pred_flatten, reg_target_flatten)

        return loss.abs()

    def _weighted_cls_loss(self, pred: torch.Tensor, label: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
        if len(select.size()) == 0:
            return torch.Tensor([0])
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.classification_loss(pred, label)

    def _classification_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze()
        neg = label.data.eq(0).nonzero().squeeze()

        loss_pos = self._weighted_cls_loss(pred, label, pos)
        loss_neg = self._weighted_cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def forward(self, outputs: Dict[str, torch.Tensor], gt: Dict[str, Any]) -> Dict[str, Any]:
        
        regression_loss = self._regression_loss(
            bbox_pred=outputs[constants.TARGET_REGRESSION_LABEL_KEY],
            reg_target=gt[constants.TARGET_REGRESSION_LABEL_KEY],
            reg_weight=gt[constants.TARGET_REGRESSION_WEIGHT_KEY],
        )
        
        classification_loss = self._classification_loss(
            pred=outputs[constants.TARGET_CLASSIFICATION_KEY], label=gt[constants.TARGET_CLASSIFICATION_KEY]
        )
        
        presence = gt[constants.TARGET_VISIBILITY_KEY].squeeze(1)
        presnece_idx = torch.argwhere(presence==1).squeeze(1)
        absence_idx = torch.argwhere(presence==0).squeeze(1)
        
        # applying simsiam based on presense of the mask in the search image
        # symmetricizing
        p1_search, p2_search, z1_search, z2_search = outputs[constants.SIMSIAM_SEARCH_OUT_KEY]
        cos_sim_loss_search = 0.5 * (1 - (self.cos_sim_loss(p1_search[presnece_idx], z2_search[presnece_idx]).mean() + self.cos_sim_loss(p2_search[presnece_idx], z1_search[presnece_idx]).mean()) * 0.5)
        cos_dissim_loss_search = 0.25 * (1 + (self.cos_sim_loss(p1_search[absence_idx], z2_search[absence_idx]).mean() + self.cos_sim_loss(p2_search[absence_idx], z1_search[absence_idx]).mean()) * 0.5) if len(absence_idx)>0 else torch.tensor(0.,device=p1_search.device)
        
        p1_dynamic, p2_dynamic, z1_dynamic, z2_dynamic = outputs[constants.SIMSIAM_DYNAMIC_OUT_KEY]
        cos_sim_loss_dynamic = 0.5 * (1 - (self.cos_sim_loss(p1_dynamic[presnece_idx], z2_dynamic[presnece_idx]).mean() + self.cos_sim_loss(p2_dynamic[presnece_idx], z1_dynamic[presnece_idx]).mean()) * 0.5)
        cos_dissim_loss_dynamic = 0.25 * (1 + (self.cos_sim_loss(p1_dynamic[absence_idx], z2_dynamic[absence_idx]).mean() + self.cos_sim_loss(p2_dynamic[absence_idx], z1_dynamic[absence_idx]).mean()) * 0.5) if len(absence_idx)>0 else torch.tensor(0.,device=p1_dynamic.device)
        
        return {
            constants.TARGET_CLASSIFICATION_KEY: classification_loss * self.coeffs[constants.TARGET_CLASSIFICATION_KEY],
            constants.TARGET_REGRESSION_LABEL_KEY: regression_loss * self.coeffs[constants.TARGET_REGRESSION_LABEL_KEY],
            constants.SIMSIAM_SEARCH_OUT_KEY:  cos_sim_loss_search,
            constants.SIMSIAM_DYNAMIC_OUT_KEY: cos_sim_loss_dynamic,
            constants.SIMSIAM_NEGATIVE_OUT_KEY: cos_dissim_loss_search+cos_dissim_loss_dynamic * 0,
        }
