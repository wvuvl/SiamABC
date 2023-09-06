import random
import torch
import torchvision.transforms as transforms

def normalize(mode: str = "imagenet"):
    if mode == "imagenet":
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif mode == "mean":
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        return None

def numpy_to_tensor():
    return transforms.ToTensor()


template_augmentation = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.ColorJitter(0.2, 0.2),
        # transforms.RandomGrayscale(p=0.05),
        normalize()
    ]
)

search_augmentation = transforms.Compose([
        # transforms.Resize((256, 256)),
        # transforms.ColorJitter(0.2, 0.2),
        # transforms.RandomGrayscale(p=0.05),
        normalize()
    ]
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
