"""
Module: feature_extractor

Description:
This module combines the texture-based and image quality-based
descriptors used for face anti-spoofing detection.

It extracts Local Binary Pattern (LBP) features and image quality
features from a face image, then concatenates them into a single
fixed-length feature vector.

The resulting combined descriptor is used as input to the
presentation attack detection (PAD) classifier.
"""

import numpy as np
from src.antispoofing.texture_features import extract_lbp_features
from src.antispoofing.quality_features import extract_quality_features
from src.antispoofing.lft_features import extract_lft_features

def extract_pad_features(face_image):
    '''Extract combined anti-spoofing features from a given face image.'''

    lbp_features = extract_lbp_features(face_image)
    quality_features = extract_quality_features(face_image)
    lft_features = extract_lft_features(face_image)

    # DEBUG
    # print("LBP:", lbp_features.shape)
    # print("QUALITY:", quality_features.shape)
    # print("LFT:", lft_features.shape)

    # Combine LBP and quality features into a single feature vector
    combined_features = np.concatenate((
        lbp_features, 
        quality_features,
        lft_features
    ))

    #print("TOTAL:", combined_features.shape)

    return combined_features.astype(np.float32)