import torch
import math


def bbox_xyxy2xywh(bbox):
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])

def bbox_get_center_point(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def get_image_bounding_box(image_size):
    bbox = [0, 0, image_size[0]-1, image_size[1]-1]
    bbox = [bbox[0] + 0.5, bbox[1] + 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
    return bbox

def get_image_center_point(image_size):
    return bbox_get_center_point(get_image_bounding_box(image_size))


def bbox_scale_and_translate(bbox, scale, input_center, output_center):
    '''
        (i - input_center) * scale = o - output_center
        :return XYXY format
    '''
    x1, y1, x2, y2 = bbox
    ic_x, ic_y = input_center
    oc_x, oc_y = output_center
    s_x, s_y = scale
    o_x1 = oc_x + (x1 - ic_x) * s_x
    o_y1 = oc_y + (y1 - ic_y) * s_y
    o_x2 = oc_x + (x2 - ic_x) * s_x
    o_y2 = oc_y + (y2 - ic_y) * s_y
    return [o_x1, o_y1, o_x2, o_y2]

def bbox_is_valid(bbox):
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]

def bbox_get_intersection(bbox1, bbox2):
    """bbox: x1, y1, x2, y2"""
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    if inter_x2 - inter_x1 <= 0 or inter_y2 - inter_y1 <= 0:
        return (0, 0, 0, 0)
    return (inter_x1, inter_y1, inter_x2, inter_y2)

def bounding_box_is_intersect_with_image(bounding_box, image_size):
    """bbox: x1, y1, x2, y2"""
    image_bounding_box = get_image_bounding_box(image_size)
    return bbox_is_valid(bbox_get_intersection(image_bounding_box, bounding_box))
    
def get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor):
    scaling = scaling / torch.exp(torch.randn(2) * scaling_jitter_factor)
    bbox = bbox_xyxy2xywh(bbox)
    max_translate = (torch.tensor(bbox[2:4]) * scaling).sum() * 0.5 * translation_jitter_factor
    translate = (torch.rand(2) - 0.5) * max_translate
    return scaling, translate


def get_scaling_factor_from_area_factor(bbox, area_factor, output_size):
    bbox = bbox_xyxy2xywh(bbox)
    w, h = bbox[2: 4]
    w_z = w + (area_factor - 1) * ((w + h) * 0.5)
    h_z = h + (area_factor - 1) * ((w + h) * 0.5)
    scaling = math.sqrt((output_size[0] * output_size[1]) / (w_z * h_z))
    return torch.tensor((scaling, scaling), dtype=torch.float64)


def get_scaling_and_translation_parameters(bbox, area_factor, output_size):
    scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)

    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    source_center = torch.tensor(source_center)
    target_center = torch.tensor(target_center)
    return scaling, source_center, target_center


def prepare_SiamFC_curation_with_position_augmentation(bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    while True:    
        scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
        scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor,
                                                                       translation_jitter_factor)                                        
        source_center = bbox_get_center_point(bbox)
        target_center = get_image_center_point(output_size)
        target_center = (torch.tensor(target_center) - translate)
        output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)
        if bounding_box_is_intersect_with_image(output_bbox, output_size):
            break
        else :
            print(bbox)

    source_center = torch.tensor(source_center)
    output_bbox = torch.tensor(output_bbox)
    curation_parameter = torch.stack((scaling, source_center, target_center))

    return curation_parameter, output_bbox


def prepare_SiamFC_curation(bbox, area_factor, output_size):
    curation_scaling, curation_source_center_point, curation_target_center_point = get_scaling_and_translation_parameters(bbox, area_factor, output_size)
    output_bbox = bbox_scale_and_translate(bbox, curation_scaling, curation_source_center_point, curation_target_center_point)
    output_bbox = torch.tensor(output_bbox)

    curation_parameter = torch.stack((curation_scaling, curation_source_center_point, curation_target_center_point))

    return curation_parameter, output_bbox


def do_SiamFC_curation(image, output_size, curation_parameter, interpolation_mode, image_mean=None, out_img=None, out_image_mean=None):
    if image_mean is None:
        image_mean = get_image_mean(image, out_image_mean)
    else:
        if out_image_mean is not None:
            out_image_mean[:] = image_mean
    output_image, _ = torch_scale_and_translate_half_pixel_offset(image, output_size, curation_parameter[0], curation_parameter[1], curation_parameter[2], image_mean, interpolation_mode, out_img)
    return output_image, image_mean





def SiamTracker_training_prepare_SiamFC_curation(bbox, area_factor, output_size, scaling_jitter_factor,     ##有用
                                                 translation_jitter_factor):
    curation_parameter, bbox = \
        prepare_SiamFC_curation_with_position_augmentation(bbox, area_factor, output_size,
                                                           scaling_jitter_factor, translation_jitter_factor)
    bbox = bbox_restrict_in_image_boundary_(bbox, output_size)
    return bbox, curation_parameter



####
#### helper Functions
####

def box_xyxy_to_cxcywh(x: torch.Tensor):
    x0, y0, x1, y1 = x.unbind(-1)
    out = torch.empty_like(x)
    out[..., 0] = (x0 + x1) / 2
    out[..., 1] = (y0 + y1) / 2
    out[..., 2] = (x1 - x0)
    out[..., 3] = (y1 - y0)
    return out

def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    out = torch.empty_like(x)
    half_w = 0.5 * w
    half_h = 0.5 * h
    out[..., 0] = (x_c - half_w)
    out[..., 1] = (y_c - half_h)
    out[..., 2] = (x_c + half_w)
    out[..., 3] = (y_c + half_h)
    return out

def _adjust_bbox_size(bounding_box, min_wh):
    bounding_box = box_xyxy_to_cxcywh(bounding_box)
    torch.clamp_(bounding_box[2], min=min_wh[0])
    torch.clamp_(bounding_box[3], min=min_wh[1])
    return box_cxcywh_to_xyxy(bounding_box)

def get_image_mean_nchw(image, out=None):
    """
    Args:
        image(torch.Tensor): (n, c, h, w)
    """
    return torch.mean(image, dim=(2, 3), out=out)


def get_image_mean_chw(image, out=None):
    """
    Args:
        image(torch.Tensor): (c, h, w)
    """
    return torch.mean(image, dim=(1, 2), out=out)


def get_image_mean_hw(image, out=None):
    """
    Args:
        image(torch.Tensor): (h, w)
    """
    return torch.mean(image, out=out)


def get_image_mean(image, out=None):
    assert image.ndim in (2, 3, 4)
    if image.ndim == 2:
        return torch.mean(image, out=out)
    elif image.ndim == 3:
        return get_image_mean_chw(image, out)
    else:
        return get_image_mean_nchw(image, out)
    
def bbox_restrict_in_image_boundary_(bbox: torch.Tensor, image_size):
    torch.clamp_min_(bbox[..., :2], 0.5)
    torch.clamp_max_(bbox[..., 2], image_size[0] - 0.5)
    torch.clamp_max_(bbox[..., 3], image_size[1] - 0.5)
    return bbox

def bbox_is_valid_vectorized(bbox: torch.Tensor):
    validity = bbox[..., :2] < bbox[..., 2:]
    return torch.logical_and(validity[..., 0], validity[..., 1])

def bbox_scale_and_translate_vectorized(bbox, scale, input_center, output_center):
    """
    (i - input_center) * scale = o - output_center
    Args:
        bbox (torch.Tensor): (n, 4)
        scale (torch.Tensor): (n, 2)
        input_center (torch.Tensor): (n, 2)
        output_center (torch.Tensor): (n, 2)
    Returns:
        torch.Tensor: scaled torch tensor, (n, 4)
    """
    out_bbox = torch.empty_like(bbox)
    out_bbox[..., ::2] = bbox[..., ::2] - input_center[..., (0, )]
    out_bbox[..., ::2] *= scale[..., (0, )]
    out_bbox[..., ::2] += output_center[..., (0, )]

    out_bbox[..., 1::2] = bbox[..., 1::2] - input_center[..., (1, )]
    out_bbox[..., 1::2] *= scale[..., (1, )]
    out_bbox[..., 1::2] += output_center[..., (1, )]
    return out_bbox

def torch_scale_and_translate_half_pixel_offset(img, output_size, scale, input_center, output_center,
                                                background_color=None, mode='bilinear', output_img=None):
    """
    Args:
        img (torch.Tensor): (n, c, h, w) or (c, h, w)
        output_size (int, int): (2)
        scale (torch.Tensor): (n, 2) or (2)
        input_center (torch.Tensor): (n, 2) or (2)
        output_center (torch.Tensor): (n, 2) or (2)
        background_color (torch.Tensor | None): (n, c) or (n, 1) or (c)
        mode (str): interpolate algorithm
    Returns:
        (torch.Tensor, torch.Tensor): tuple containing:
            output_image(torch.Tensor): (n, c, h, w) or (c, h, w), curated image
            image_bbox (torch.Tensor): (n, 2) or (2)
    """
    if mode in ('bilinear', 'bicubic'):
        align_corners = True
    else:
        align_corners = None
    assert img.ndim in (3, 4)
    batch_mode = img.ndim == 4
    if not batch_mode:
        img = img.unsqueeze(0)
    if output_img is not None:
        if batch_mode:
            assert output_img.ndim == 4
        else:
            assert output_img.ndim in (3, 4)
            if output_img.ndim == 4:
                assert output_img.shape[0] == 1
            else:
                output_img = output_img.unsqueeze(0)
    n, c, h, w = img.shape
    if background_color is not None:
        if background_color.ndim == 1:
            if output_img is None:
                output_img = background_color.reshape(1, -1, 1, 1).repeat(
                    n, c // background_color.shape[0], output_size[1], output_size[0])
            else:
                output_img[:] = background_color.reshape(1, -1, 1, 1)
        elif background_color.ndim == 2:
            b_n, b_c = background_color.shape
            assert b_n == n
            if output_img is None:
                output_img = background_color.reshape(b_n, b_c, 1, 1).repeat(1, c // b_c, output_size[1], output_size[0])
            else:
                output_img[:] = background_color.reshape(b_n, b_c, 1, 1)
        else:
            raise RuntimeError(f"Incompatible background_color shape")
    else:
        if output_img is None:
            output_img = torch.zeros((n, c, output_size[1], output_size[0]), dtype=img.dtype, device=img.device)

    output_bbox = bbox_scale_and_translate_vectorized(
        torch.tensor((0, 0, w, h), dtype=torch.float64, device=scale.device), scale, input_center, output_center)
    bbox_restrict_in_image_boundary_(output_bbox, output_size)
    input_bbox = bbox_scale_and_translate_vectorized(output_bbox, 1 / scale, output_center, input_center)
    output_bbox = output_bbox.to(torch.int)
    input_bbox = input_bbox.to(torch.int)
    output_bbox_validity = bbox_is_valid_vectorized(output_bbox)

    assert output_bbox.ndim in (1, 2)

    if output_bbox.ndim == 2:
        assert output_bbox.shape[0] == n
        for i_n in range(n):
            if not output_bbox_validity[i_n]:
                continue
            output_img[i_n, :, output_bbox[i_n, 1]: output_bbox[i_n, 3] + 1, output_bbox[i_n, 0]: output_bbox[i_n, 2] + 1] = torch.nn.functional.interpolate(
                img[i_n: i_n + 1, :, input_bbox[i_n, 1]: input_bbox[i_n, 3] + 1, input_bbox[i_n, 0]: input_bbox[i_n, 2] + 1],
                (output_bbox[i_n, 3] - output_bbox[i_n, 1] + 1, output_bbox[i_n, 2] - output_bbox[i_n, 0] + 1),
                mode=mode,
                align_corners=align_corners)
    else:
        if output_bbox_validity:
            for i_n in range(n):
                output_img[i_n, :, output_bbox[1]: output_bbox[3] + 1, output_bbox[0]: output_bbox[2] + 1] = torch.nn.functional.interpolate(
                    img[i_n: i_n + 1, :, input_bbox[1]: input_bbox[3] + 1, input_bbox[0]: input_bbox[2] + 1],
                    (output_bbox[3] - output_bbox[1] + 1, output_bbox[2] - output_bbox[0] + 1),
                    mode=mode,
                    align_corners=align_corners)
    if not batch_mode:
        output_img = output_img.squeeze(0)
    return output_img, output_bbox