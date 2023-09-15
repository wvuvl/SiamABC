import random
from typing import Any
import torch
import torchvision.transforms as transforms
import kornia.augmentation as K

    

def normalize(mode: str = "imagenet"):

    if mode == "imagenet":
        return K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif mode == "mean":
        return K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        return None

def torch_resize(size):
    return transforms.Resize((size, size))

def numpy_to_tensor():
    return transforms.ToTensor()

# template_augmentation = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.2, 0.2)  # not strengthened
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.05),
#         normalize()
#     ]
# )

# search_augmentation = transforms.Compose([
#         # transforms.Resize((256, 256)),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.2, 0.2)  # not strengthened
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.05),
#         transforms.RandomApply([transforms.GaussianBlur((3,7), (.1, 2.))], p=0.5),
#         normalize()
#     ]
# )


color_augmentation = K.AugmentationSequential(
        K.AugmentationSequential(
                K.RandomContrast((0.5,2.),p=0.5),
                K.RandomBrightness((0.5,2.),p=0.5),
                K.RandomGamma((0.5,2.),(1.5,1.5),p=0.5),
                K.RandomHue((-0.5,0.5),p=0.5),
                K.RandomRGBShift(p=0.5),
                K.RandomEqualize(p=0.5),
                K.ColorJiggle(0.2, 0.2, 0.2, 0.2, p=0.8),
                random_apply=1
        ),
        K.RandomGrayscale(p=0.1)
)

tracking_aug = K.AugmentationSequential(
        K.RandomMotionBlur(kernel_size=(3,7), angle=35., direction=0.5,p=0.5),
        K.RandomMedianBlur(p=0.5),
        K.RandomBoxBlur(p=0.5),
        K.RandomGaussianBlur((3,7), (.1, 2.),p=0.5),
        random_apply=1 
)

K_normalize = normalize()


# augmentation = transforms.Compose(
#         [
            
#         #color aug
#         K.RandomGrayscale(p=0.05),
#         transforms.RandomChoice(
#             [
#                 K.RandomContrast(),
#                 K.RandomBrightness(),
#                 K.RandomGamma(),
#                 K.RandomHue(),
#                 K.RandomRGBShift(),
#                 K.RandomEqualize(),
#                 K.ColorJitter(),
#             ]
#         ),
        
#         #photometric
        # transforms.RandomChoice(
        #     [
        #     K.RandomMotionBlur(kernel_size=(3,7), angle=35., direction=0.5),
        #     K.RandomMedianBlur(),
        #     K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
        #     K.RandomBoxBlur(),
        #     ]
        # ),
#         K.RandomGaussianNoise(p=0.2),
        # transforms.RandomChoice(
        #     [
        #     K.RandomSnow(),
        #     K.RandomRain()   
        #     ]        
        # ),
#         normalize()
#     ]
# )
