"""Data augmentation pipelines for DR model training.

Uses Albumentations for efficient, GPU-friendly augmentations
designed specifically for fundus photography.
"""
from __future__ import annotations 

import albumentations as A 
from albumentations .pytorch import ToTensorV2 


def get_train_transforms (image_size :int =384 )->A .Compose :
    """Training augmentation pipeline.

    Augmentations are designed to simulate real-world variations in
    portable fundus camera images:
    - Rotation/flips: fundus orientation varies
    - Brightness/contrast: illumination differences across cameras
    - CoarseDropout: simulates dust/artifacts on lens
    - Blur: simulates focus issues in field conditions

    Args:
        image_size: Target image size after transforms.

    Returns:
        Albumentations Compose pipeline.
    """
    return A .Compose ([
    A .Resize (image_size ,image_size ),
    A .RandomRotate90 (p =0.5 ),
    A .HorizontalFlip (p =0.5 ),
    A .VerticalFlip (p =0.5 ),
    A .ShiftScaleRotate (
    shift_limit =0.1 ,
    scale_limit =0.15 ,
    rotate_limit =30 ,
    border_mode =0 ,
    p =0.5 ,
    ),
    A .OneOf ([
    A .RandomBrightnessContrast (
    brightness_limit =0.2 ,
    contrast_limit =0.2 ,
    p =1.0 ,
    ),
    A .HueSaturationValue (
    hue_shift_limit =10 ,
    sat_shift_limit =20 ,
    val_shift_limit =20 ,
    p =1.0 ,
    ),
    ],p =0.7 ),
    A .OneOf ([
    A .GaussianBlur (blur_limit =(3 ,5 ),p =1.0 ),
    A .MedianBlur (blur_limit =5 ,p =1.0 ),
    ],p =0.2 ),
    A .CoarseDropout (
    max_holes =8 ,
    max_height =int (image_size *0.05 ),
    max_width =int (image_size *0.05 ),
    fill_value =0 ,
    p =0.3 ,
    ),
    A .Normalize (
    mean =[0.485 ,0.456 ,0.406 ],
    std =[0.229 ,0.224 ,0.225 ],
    ),
    ToTensorV2 (),
    ])


def get_val_transforms (image_size :int =384 )->A .Compose :
    """Validation/inference transform pipeline.

    Minimal transforms: just resize and normalize.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return A .Compose ([
    A .Resize (image_size ,image_size ),
    A .Normalize (
    mean =[0.485 ,0.456 ,0.406 ],
    std =[0.229 ,0.224 ,0.225 ],
    ),
    ToTensorV2 (),
    ])
