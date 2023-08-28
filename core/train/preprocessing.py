import random
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

def numpy_to_tensor():
    return transforms.ToTensor()

def torch_resize(size):
    return transforms.Resize((size, size))



augmentation = K.AugmentationSequential(
    
        K.ColorJitter(p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
        normalize()
    
)


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
#         transforms.RandomChoice(
#             [
#             K.RandomMotionBlur(kernel_size=(3,7), angle=35., direction=0.5),
#             K.RandomMedianBlur(),
#             K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
#             K.RandomBoxBlur(),
#             ]
#         ),
#         K.RandomGaussianNoise(p=0.2),
#         transforms.RandomChoice(
#             [
#             K.RandomSnow(),
#             K.RandomRain()   
#             ]        
#         ),
#         normalize()
#     ]
# )
