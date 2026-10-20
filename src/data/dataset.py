"""PyTorch Dataset for APTOS 2019 Blindness Detection.

Handles loading, preprocessing, and augmentation of the APTOS 2019
dataset for training the DR grading model.
"""
from __future__ import annotations 

import os 
from pathlib import Path 

import cv2 
import numpy as np 
import pandas as pd 
import torch 
from sklearn .model_selection import StratifiedKFold 
from torch .utils .data import DataLoader ,Dataset ,WeightedRandomSampler 

from src .config import (
BATCH_SIZE ,
IMAGE_SIZE ,
NUM_WORKERS ,
RANDOM_SEED ,
RAW_DATA_DIR ,
VAL_SPLIT ,
)
from src .data .augmentation import get_train_transforms ,get_val_transforms 
from src .data .preprocessing import preprocess_fundus 


class APTOSDataset (Dataset ):
    """PyTorch Dataset for APTOS 2019 Diabetic Retinopathy images.

    Loads fundus images, applies preprocessing (circle crop, Ben Graham,
    CLAHE) and augmentation, and returns (image_tensor, label) pairs.

    Args:
        image_dir: Path to directory containing fundus images.
        labels_df: DataFrame with columns ['id_code', 'diagnosis'].
        transform: Albumentations transform pipeline.
        preprocess: Whether to apply fundus-specific preprocessing.
    """

    def __init__ (
    self ,
    image_dir :str |Path ,
    labels_df :pd .DataFrame ,
    transform =None ,
    preprocess :bool =True ,
    ):
        self .image_dir =Path (image_dir )
        self .labels_df =labels_df .reset_index (drop =True )
        self .transform =transform 
        self .preprocess =preprocess 

    def __len__ (self )->int :
        return len (self .labels_df )

    def __getitem__ (self ,idx :int )->tuple [torch .Tensor ,int ]:
        row =self .labels_df .iloc [idx ]
        image_id =row ["id_code"]
        label =int (row ["diagnosis"])


        image_path =None 
        for ext in [".png",".jpg",".jpeg"]:
            candidate =self .image_dir /f"{image_id}{ext}"
            if candidate .exists ():
                image_path =candidate 
                break 

        if image_path is None :
            raise FileNotFoundError (
            f"Image not found for id_code={image_id} in {self.image_dir}"
            )


        image =cv2 .imread (str (image_path ))
        if image is None :
            raise ValueError (f"Failed to load: {image_path}")


        if self .preprocess :
            image =preprocess_fundus (image ,size =IMAGE_SIZE )


        image =cv2 .cvtColor (image ,cv2 .COLOR_BGR2RGB )


        if self .transform :
            augmented =self .transform (image =image )
            image =augmented ["image"]
        else :

            image =image .astype (np .float32 )/255.0 
            mean =np .array ([0.485 ,0.456 ,0.406 ])
            std =np .array ([0.229 ,0.224 ,0.225 ])
            image =(image -mean )/std 
            image =torch .from_numpy (image .transpose (2 ,0 ,1 )).float ()

        return image ,label 


def get_class_weights (labels :np .ndarray )->torch .Tensor :
    """Compute inverse-frequency class weights for imbalanced dataset.

    Args:
        labels: Array of integer class labels.

    Returns:
        Tensor of class weights.
    """
    class_counts =np .bincount (labels ,minlength =5 )

    class_counts =np .maximum (class_counts ,1 )
    total =len (labels )
    weights =total /(len (class_counts )*class_counts )
    return torch .FloatTensor (weights )


def get_weighted_sampler (labels :np .ndarray )->WeightedRandomSampler :
    """Create a weighted random sampler for class-imbalanced training.

    Oversamples minority classes (severe/proliferative DR) to ensure
    the model sees balanced batches during training.

    Args:
        labels: Array of integer class labels.

    Returns:
        WeightedRandomSampler for use with DataLoader.
    """
    class_weights =get_class_weights (labels )
    sample_weights =class_weights [labels ]
    return WeightedRandomSampler (
    weights =sample_weights .tolist (),
    num_samples =len (labels ),
    replacement =True ,
    )


def get_data_loaders (
data_dir :str |Path |None =None ,
batch_size :int =BATCH_SIZE ,
image_size :int =IMAGE_SIZE ,
val_split :float =VAL_SPLIT ,
seed :int =RANDOM_SEED ,
num_workers :int =NUM_WORKERS ,
)->tuple [DataLoader ,DataLoader ]:
    """Create train and validation DataLoaders for APTOS 2019.

    Performs stratified splitting to maintain class distribution,
    applies weighted sampling for training, and uses appropriate
    augmentations for each split.

    Args:
        data_dir: Path to APTOS dataset directory (containing
                  train_images/ and train.csv).
        batch_size: Batch size for DataLoaders.
        image_size: Target image size.
        val_split: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if data_dir is None :
        data_dir =RAW_DATA_DIR /"aptos2019"

    data_dir =Path (data_dir )
    csv_path =data_dir /"train.csv"
    image_dir =data_dir /"train_images"

    if not csv_path .exists ():
        raise FileNotFoundError (
        f"APTOS train.csv not found at {csv_path}. "
        f"Download from: https://www.kaggle.com/c/aptos2019-blindness-detection"
        )


    df =pd .read_csv (csv_path )


    skf =StratifiedKFold (n_splits =int (1 /val_split ),shuffle =True ,random_state =seed )
    train_idx ,val_idx =next (skf .split (df ,df ["diagnosis"]))

    train_df =df .iloc [train_idx ]
    val_df =df .iloc [val_idx ]


    train_transform =get_train_transforms (image_size )
    val_transform =get_val_transforms (image_size )


    train_dataset =APTOSDataset (image_dir ,train_df ,transform =train_transform )
    val_dataset =APTOSDataset (image_dir ,val_df ,transform =val_transform )


    train_labels =train_df ["diagnosis"].values 
    sampler =get_weighted_sampler (train_labels )


    train_loader =DataLoader (
    train_dataset ,
    batch_size =batch_size ,
    sampler =sampler ,
    num_workers =num_workers ,
    pin_memory =True ,
    drop_last =True ,
    )

    val_loader =DataLoader (
    val_dataset ,
    batch_size =batch_size ,
    shuffle =False ,
    num_workers =num_workers ,
    pin_memory =True ,
    )

    return train_loader ,val_loader 
