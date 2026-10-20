"""Fundus image preprocessing pipeline for DR grading.

Implements Ben Graham's preprocessing technique (Kaggle DR winner),
CLAHE contrast enhancement, and circular cropping for robust
handling of variable-quality portable fundus camera images.
"""
from __future__ import annotations 

import cv2 
import numpy as np 
from PIL import Image 


def circle_crop (image :np .ndarray )->np .ndarray :
    """Crop the fundus image to its circular region of interest.

    Detects the circular boundary of the fundus photograph and removes
    the black border regions, which vary across camera models.

    Args:
        image: BGR image as numpy array (H, W, 3).

    Returns:
        Cropped BGR image containing only the circular ROI.
    """
    gray =cv2 .cvtColor (image ,cv2 .COLOR_BGR2GRAY )

    _ ,thresh =cv2 .threshold (gray ,10 ,255 ,cv2 .THRESH_BINARY )


    contours ,_ =cv2 .findContours (thresh ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )
    if not contours :
        return image 

    largest =max (contours ,key =cv2 .contourArea )
    x ,y ,w ,h =cv2 .boundingRect (largest )


    pad =5 
    x =max (0 ,x -pad )
    y =max (0 ,y -pad )
    w =min (image .shape [1 ]-x ,w +2 *pad )
    h =min (image .shape [0 ]-y ,h +2 *pad )

    return image [y :y +h ,x :x +w ]


def apply_clahe (
image :np .ndarray ,
clip_limit :float =2.0 ,
tile_grid_size :tuple [int ,int ]=(8 ,8 ),
)->np .ndarray :
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast in the fundus image, which is critical for
    detecting subtle lesions (microaneurysms, hard exudates) in images
    from portable/low-cost fundus cameras.

    Args:
        image: BGR image as numpy array.
        clip_limit: CLAHE clip limit for contrast limiting.
        tile_grid_size: Size of the grid for histogram equalization.

    Returns:
        Contrast-enhanced BGR image.
    """

    lab =cv2 .cvtColor (image ,cv2 .COLOR_BGR2LAB )
    l_channel ,a_channel ,b_channel =cv2 .split (lab )

    clahe =cv2 .createCLAHE (clipLimit =clip_limit ,tileGridSize =tile_grid_size )
    l_enhanced =clahe .apply (l_channel )

    lab_enhanced =cv2 .merge ([l_enhanced ,a_channel ,b_channel ])
    return cv2 .cvtColor (lab_enhanced ,cv2 .COLOR_LAB2BGR )


def ben_graham_preprocessing (
image :np .ndarray ,
sigma :int =30 ,
alpha :float =4.0 ,
beta :float =-4.0 ,
gamma :int =128 ,
)->np .ndarray :
    """Apply Ben Graham's preprocessing technique.

    This method, used by the winner of the Kaggle Diabetic Retinopathy
    competition, subtracts the local average color and rescales to enhance
    vessel and lesion visibility while normalizing illumination.

    Reference: https://www.kaggle.com/c/diabetic-retinopathy-detection

    Args:
        image: BGR image as numpy array.
        sigma: Gaussian blur sigma for local average estimation.
        alpha: Weight for the original image.
        beta: Weight for the Gaussian-blurred image (negative to subtract).
        gamma: Scalar added to the result.

    Returns:
        Preprocessed BGR image with enhanced contrast.
    """
    blurred =cv2 .GaussianBlur (image ,(0 ,0 ),sigma )
    enhanced =cv2 .addWeighted (image ,alpha ,blurred ,beta ,gamma )
    return enhanced 


def preprocess_fundus (
image :np .ndarray ,
size :int =384 ,
apply_ben_graham :bool =True ,
apply_clahe_enhancement :bool =True ,
)->np .ndarray :
    """Full preprocessing pipeline for a single fundus image.

    Pipeline:
        1. Circle crop (remove black borders)
        2. Ben Graham's color normalization (optional)
        3. CLAHE contrast enhancement (optional)
        4. Resize to target dimensions

    Args:
        image: BGR image as numpy array (from cv2.imread).
        size: Target output size (square).
        apply_ben_graham: Whether to apply Ben Graham's preprocessing.
        apply_clahe_enhancement: Whether to apply CLAHE.

    Returns:
        Preprocessed BGR image of shape (size, size, 3).
    """

    image =circle_crop (image )


    if apply_ben_graham :
        image =ben_graham_preprocessing (image )


    if apply_clahe_enhancement :
        image =apply_clahe (image )


    image =cv2 .resize (image ,(size ,size ),interpolation =cv2 .INTER_AREA )

    return image 


def load_and_preprocess (
image_path :str ,
size :int =384 ,
)->np .ndarray :
    """Load an image from disk and apply the full preprocessing pipeline.

    Args:
        image_path: Path to the fundus image file.
        size: Target output size.

    Returns:
        Preprocessed BGR image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    image =cv2 .imread (str (image_path ))
    if image is None :
        raise ValueError (f"Failed to load image: {image_path}")
    return preprocess_fundus (image ,size =size )


def preprocess_for_inference (
image :np .ndarray |str ,
size :int =384 ,
)->np .ndarray :
    """Preprocess a fundus image and convert to model-ready tensor format.

    Returns a float32 numpy array in CHW format with ImageNet normalization,
    ready to be wrapped in torch.from_numpy().

    Args:
        image: BGR image array or path to image file.
        size: Target size.

    Returns:
        Float32 array of shape (3, size, size) with ImageNet normalization.
    """
    if isinstance (image ,str ):
        image =load_and_preprocess (image ,size =size )
    else :
        image =preprocess_fundus (image ,size =size )


    image =cv2 .cvtColor (image ,cv2 .COLOR_BGR2RGB )


    image =image .astype (np .float32 )/255.0 
    mean =np .array ([0.485 ,0.456 ,0.406 ],dtype =np .float32 )
    std =np .array ([0.229 ,0.224 ,0.225 ],dtype =np .float32 )
    image =(image -mean )/std 


    image =np .transpose (image ,(2 ,0 ,1 ))

    return image 
