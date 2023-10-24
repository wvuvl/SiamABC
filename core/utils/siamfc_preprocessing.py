
import numpy as np
import math


def get_image_bounding_box(image_size):
    bbox = [0, 0, image_size[0]-1, image_size[1]-1]
    bbox = [bbox[0] + 0.5, bbox[1] + 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
    return bbox

def bbox_get_intersection(bbox1, bbox2):
    """bbox: x1, y1, x2, y2"""
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    if inter_x2 - inter_x1 <= 0 or inter_y2 - inter_y1 <= 0:
        return (0, 0, 0, 0)
    return (inter_x1, inter_y1, inter_x2, inter_y2)

def bbox_is_valid(bbox):
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]

def bounding_box_is_intersect_with_image(bounding_box, image_size):
    """bbox: x1, y1, x2, y2"""
    image_bounding_box = get_image_bounding_box(image_size)
    
    return bbox_is_valid(bbox_get_intersection(image_bounding_box, bounding_box))

def convert_xywh_to_xyxy(bbox: np.array) -> np.array:
    return [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]

def convert_xyxy_to_xywh(bbox: np.array) -> np.array:
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


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

def bbox_get_center_point(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def get_image_center_point(image_size):
    return bbox_get_center_point(get_image_bounding_box(image_size))
    
def get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor,rng):
    # bbox: x,y,w,h
    scaling = scaling / np.exp(np.random.randn(2) * scaling_jitter_factor)
    max_translate = (bbox[2:4] * scaling).sum() * 0.5 * translation_jitter_factor
    translate = (np.random.randn(2)- 0.5) * max_translate
    return scaling, translate


def get_scaling_factor_from_area_factor(bbox, area_factor, output_size):
    w, h = bbox[2: 4]
    w_z = w + (area_factor - 1) * ((w + h) * 0.5)
    h_z = h + (area_factor - 1) * ((w + h) * 0.5)
    scaling = math.sqrt((output_size[0] * output_size[1]) / (w_z * h_z))
    return scaling, scaling


def get_scaling_and_translation_parameters(bbox, area_factor, output_size):
    scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    return scaling, source_center, target_center


def prepare_SiamFC_curation_with_position_augmentation(bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor,rng):
    while True:    
        scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
        scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor,
                                                                       translation_jitter_factor,rng)                                        
        source_center = bbox_get_center_point(bbox)
        target_center = get_image_center_point(output_size)
        target_center = target_center - translate
        bbox = convert_xywh_to_xyxy(bbox)
        output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)
        if bounding_box_is_intersect_with_image(output_bbox, output_size):
            break

        
    output_bbox = convert_xyxy_to_xywh(output_bbox)
    curation_parameter =[scaling, source_center, target_center]

    return curation_parameter, output_bbox


def prepare_SiamFC_curation(bbox, area_factor, output_size):
    curation_scaling, curation_source_center_point, curation_target_center_point = get_scaling_and_translation_parameters(bbox, area_factor, output_size)
    bbox = convert_xywh_to_xyxy(bbox)
    output_bbox = bbox_scale_and_translate(bbox, curation_scaling, curation_source_center_point, curation_target_center_point)

    output_bbox = convert_xyxy_to_xywh(output_bbox)
    curation_parameter = (curation_scaling, curation_source_center_point, curation_target_center_point)

    return curation_parameter, output_bbox


# def do_SiamFC_curation(image, output_size, curation_parameter, interpolation_mode):
#     image_mean = np.mean(image, axis=(0, 1))

#     output_image, _ = torch_scale_and_translate_half_pixel_offset(image, output_size, curation_parameter[0], curation_parameter[1], curation_parameter[2], image_mean, interpolation_mode)
#     return output_image, image_mean