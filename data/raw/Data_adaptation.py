'''
# change the data created by Boris

nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart
├── dataset.json
├── imagesTr
│   ├── la_003_0000.nii.gz
│   ├── la_004_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── la_001_0000.nii.gz
│   ├── la_002_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── la_003.nii.gz
    ├── la_004.nii.gz
    ├── ...

'''
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import os

'''
Convention:

aorta : 1
heart_atrium_left : 2
heart_atrium_right : 3
heart_myocardiuma : 4
heart_ventricle_left : 5
heart_ventricle_right : 6
'''

# create the folder in task format

name_new_folder = '/user/jfierrou/home/dl-seg/data/raw/Task001_Heart'
if not os.path.exists(name_new_folder):
    os.mkdir(name_new_folder)
    os.mkdir(name_new_folder+'/imagesTr')
    os.mkdir(name_new_folder+'/labelsTr')

counter = 0
base_folder = '/user/jfierrou/home/dl-seg/data/raw/baseline_train'
for folder in tqdm(os.listdir(base_folder)):
#for folder in ['/user/jfierrou/home/dl-seg/data/raw/baseline_train/s0014/']:
    aorta = nib.load(base_folder+'/'+folder+'/segmentations/aorta.nii.gz')
    heart_atrium_left = nib.load(base_folder+'/'+folder+'/segmentations/heart_atrium_left.nii.gz')
    heart_atrium_right = nib.load(base_folder+'/'+folder+'/segmentations/heart_atrium_right.nii.gz')
    heart_myocardiuma = nib.load(base_folder+'/'+folder+'/segmentations/heart_myocardium.nii.gz')
    heart_ventricle_left = nib.load(base_folder+'/'+folder+'/segmentations/heart_ventricle_left.nii.gz')
    heart_ventricle_right = nib.load(base_folder+'/'+folder+'/segmentations/heart_ventricle_right.nii.gz')
    
    label_data = aorta.get_fdata() + 2*heart_atrium_left.get_fdata() + \
                    3*heart_atrium_right.get_fdata() + 4*heart_myocardiuma.get_fdata()  + \
                    5*heart_ventricle_left.get_fdata() + 6*heart_ventricle_right.get_fdata()

    #assert(np.max(label_data)<=6)
    if np.max(label_data != 0):
        training_image = nib.load(base_folder+'/'+folder+'/ct.nii.gz')
        string_1 = '/imagesTr/la_{:0>3d}_0000.nii.gz'.format(counter)
        nib.save(training_image,name_new_folder+string_1)

        label = nib.Nifti1Image(label_data, training_image.affine)
        string_2 = '/labelsTr/la_{:0>3d}.nii.gz'.format(counter)
        nib.save(label,name_new_folder+string_2)
        counter+=1


# generate labels

# create the json

