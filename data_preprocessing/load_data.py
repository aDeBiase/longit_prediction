import os
import re
import tempfile
import nibabel as nib
import numpy as np
import torch
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Any, Mapping, Hashable
from PIL import Image
import random
from scipy import ndimage
from monai.config.type_definitions import NdarrayOrTensor
import numpy as np
import SimpleITK as sitk
from monai.config import KeysCollection

from monai.utils import convert_to_tensor

random.seed(42)

from monai.transforms import (
    MapTransform,
    ConcatItemsd,
    Compose,
    LoadImaged,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ToTensord,
    ImageFilterd,
    Resize,
    MaskIntensityd,
    Spacingd,
    RandRotate90d,
    RandAxisFlipd,
    RandGaussianNoised,
    SpatialPadd,
    RandFlipd,
    RandAffined,
    Rand3DElasticd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstD,
)

def read_split_file(path):

    with open(path, 'r') as inf:
        dict_from_file = eval(inf.read())
        
    return dict_from_file


def dataset_prep(lists, data_folder, time_steps, outcomes, endpoint, planningCT):

    labels = list()
    dataset_= list()
    outcome_file = pd.read_excel(outcomes)
    list_out = outcome_file[endpoint].values
    transformed_out = torch.nn.functional.one_hot(torch.as_tensor(np.array(list_out)), num_classes=2).float()

    for patient in lists:

        dict_ = {}
        
        all_names = os.listdir(os.path.join(data_folder,patient))

        if planningCT:
            dict_['ct'] = os.path.join(data_folder,patient,patient+"_ct.nii.gz")

        fractions = [int(match.group(1)) for path in all_names for match in re.finditer(r'fr_(\d+)(?=_reg)', path)]
        fractions.sort()
        selected_fr = fractions[:(time_steps+1)]
        
        if time_steps>1: #if time_step
            for j in range(time_steps):
                index_ = 'ct_'+str(j+1)
                dict_[str(index_)] = os.path.join(data_folder,patient,patient+"_ct_fr_"+str(selected_fr[j])+"_reg.nii.gz")

        dict_['mask'] = os.path.join(data_folder,patient,patient+"_ct_GTVtot.nii.gz")
        dict_['ID'] = patient

        index = list(outcome_file["PatientID"]).index(patient)
        dict_['label'] = transformed_out[index]
        labels.append(list_out[index])
        
        dataset_.append(dict_)

    return dataset_ , labels

class Resized_mine(MapTransform):
    
    def __init__(self, keys, spatial_size):
        
        super().__init__(keys)
        self.spatial_size = spatial_size
        
    def resize(self, img):
        if np.shape(img)!=self.spatial_size:
            new_ = img[:self.spatial_size[0],:self.spatial_size[1],:self.spatial_size[2]] #self.resizer(img) 
            return new_
        else:
            return img
             
    def __call__(self, dictionary):
        
        dictionary = dict(dictionary)
        
        for key in self.keys:
            dictionary[key] = self.resize(dictionary[key]) 
        
        return dictionary


def initialize_transform(images_keys, model_type, level, window, image_size, ct_norm="z_norm", test=False, augmentation = False):
    #function to use!
    
    normalize_ct={
        "z_norm": NormalizeIntensityd(keys=images_keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(keys=images_keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
     
    max_ = 200 #level + window/2
    min_ = -200 #level - window/2
    
    sequence_prep=[
        LoadImaged(keys=images_keys),
        EnsureChannelFirstD(keys=images_keys),
        Spacingd(keys=images_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Resized_mine(keys=images_keys, spatial_size=(image_size,image_size,image_size)),
        SpatialPadd(keys=images_keys, spatial_size=(image_size,image_size,image_size)),
        ThresholdIntensityd(keys=images_keys, threshold=min_, above=True, cval=min_, allow_missing_keys=False),
        ThresholdIntensityd(keys=images_keys, threshold=max_, above=False, cval=max_, allow_missing_keys=False),
        normalize_ct[ct_norm],
        ]
    
    augment_prep = [
        RandFlipd(keys=images_keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=images_keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys=images_keys, spatial_axis=2, prob=0.5),
        RandAffined(
            keys=images_keys,
            rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),  # rotation degree: 7.5 degrees
            scale_range=(0.07, 0.07, 0.07),  # scale range [0.93–1.07]
            translate_range=(7, 7, 7),  # translation range: [0-7]
            prob=0.5
        ),
        Rand3DElasticd(
            keys=images_keys,
            sigma_range=(5, 7),
            magnitude_range=(100, 200),
            rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),  # rotation degree: 7.5 degrees
            scale_range=(0.07, 0.07, 0.07),  # scale range [0.93–1.07]
            translate_range=(7, 7, 7),  # translation range: 7
            prob=0.2
        ),
        RandScaleIntensityd(keys=images_keys, factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=images_keys, offsets=0.1, prob=0.5),
    ]
    
    channel_prep=[
            ConcatItemsd(keys=images_keys, name="input"),
            ToTensord(keys=["input"])
            ]
    
    if augmentation == True:
        print("Data augmentation applied to training set!")
        list_transform = sequence_prep+augment_prep+channel_prep
    else:
        list_transform = sequence_prep+channel_prep
        
    train_tranforms = Compose(list_transform)
    
    return train_tranforms


def initialize_transform_gtv(images_keys, model_type, level, window, image_size, ct_norm="z_norm", test=False, augmentation = False):
    #function to use!
    
    all_images_keys = images_keys + ("mask",)
    
    normalize_ct={
        "z_norm": NormalizeIntensityd(keys=images_keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(keys=images_keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
     
    max_ = 200 #level + window/2
    min_ = -200 #level - window/2
    
    sequence_prep=[
        LoadImaged(keys=all_images_keys),
        EnsureChannelFirstD(keys=all_images_keys),
        Spacingd(keys=all_images_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Resized_mine(keys=all_images_keys, spatial_size=(image_size,image_size,image_size)),
        SpatialPadd(keys=all_images_keys, spatial_size=(image_size,image_size,image_size)),
        ImageFilterd(keys = ["mask",], kernel='elliptical', kernel_size = 5),
        ThresholdIntensityd(keys=images_keys, threshold=min_, above=True, cval=min_, allow_missing_keys=False),
        ThresholdIntensityd(keys=images_keys, threshold=max_, above=False, cval=max_, allow_missing_keys=False),
        normalize_ct[ct_norm],
        ]
    
    augment_prep = [
        RandFlipd(keys=all_images_keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=all_images_keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys=all_images_keys, spatial_axis=2, prob=0.5),
        RandAffined(
            keys=all_images_keys,
            rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),  # rotation degree: 7.5 degrees
            scale_range=(0.07, 0.07, 0.07),  # scale range [0.93–1.07]
            translate_range=(7, 7, 7),  # translation range: [0-7]
            prob=0.5
        ),
        Rand3DElasticd(
            keys=all_images_keys,
            sigma_range=(5, 7),
            magnitude_range=(100, 200),
            rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),  # rotation degree: 7.5 degrees
            scale_range=(0.07, 0.07, 0.07),  # scale range [0.93–1.07]
            translate_range=(7, 7, 7),  # translation range: 7
            prob=0.2
        ),
        RandScaleIntensityd(keys=all_images_keys, factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=all_images_keys, offsets=0.1, prob=0.5),
    ]
    
    keys_masks = ["mask"]*len(images_keys)
    
    channel_prep=[
            ConcatItemsd(keys=images_keys, name="input"),
            ConcatItemsd(keys=keys_masks, name="mask"),
            ToTensord(keys=["input", "mask"])
            ]
    
    if augmentation == True:
        print("Data augmentation applied to training set!")
        list_transform = sequence_prep+augment_prep+channel_prep
    else:
        list_transform = sequence_prep+channel_prep
        
    train_tranforms = Compose(list_transform)
    
    return train_tranforms

'''
def initialize_transform_masked(images_keys, model_type, level, window, image_size, ct_norm="z_norm", test=False, augmentation = False):
    
    normalize_ct={
        "z_norm": NormalizeIntensityd(keys=images_keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False),
        "min_max_norm": NormalizeIntensityd(keys=images_keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False, allow_missing_keys=False)
    }
     
    max_ = 200 #level + window/2
    min_ = -200 #level - window/2

    mask_included = ("mask",) + images_keys
    
    if model_type!="DenseNet121":
        sequence_prep=[
            LoadImaged(keys=mask_included),
            Resized_mine(keys=mask_included, spatial_size=(image_size,image_size,image_size)),
            ThresholdIntensityd(keys=images_keys, threshold=min_, above=True, cval=min_, allow_missing_keys=False),
            ThresholdIntensityd(keys=images_keys, threshold=max_, above=False, cval=max_, allow_missing_keys=False),
            normalize_ct[ct_norm],
            Dilation(keys=["mask"], kernel_size=3),
            MaskIntensityd(keys=images_keys, mask_key = "mask", select_fn=lambda x: x>0), #CropForegroundd
            AddChanneld(keys=images_keys), 
            AddChanneld(keys=images_keys),
            ConcatItemsd(keys=images_keys, name="input"),
            ToTensord(keys=["input"])
        ]

    else:
        sequence_prep=[
            LoadImaged(keys=mask_included),
            Resized_mine(keys=mask_included, spatial_size=(image_size,image_size,image_size)),
            ThresholdIntensityd(keys=images_keys, threshold=min_, above=True, cval=min_, allow_missing_keys=False),
            ThresholdIntensityd(keys=images_keys, threshold=max_, above=False, cval=max_, allow_missing_keys=False),
            normalize_ct[ct_norm],
            Dilation(keys=["mask"], kernel_size=3),
            MaskIntensityd(keys=images_keys, mask_key = "mask", select_fn=lambda x: x>0),
            AddChanneld(keys=images_keys), 
            ConcatItemsd(keys=images_keys, name="input"),
            ToTensord(keys=["input"])
        ]
    
    train_tranforms = Compose(sequence_prep)
    
    return train_tranforms
'''