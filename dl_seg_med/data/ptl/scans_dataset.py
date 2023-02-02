from torch.utils.data import Dataset
import SimpleITK as sitk
import nibabel as nib
from collections import OrderedDict, defaultdict
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import itertools
import tqdm


class Scan():
    """
    Class representing a scan with corresponding segmentation masks
    
    Attributes:
        - input_data: A numpy array of N dimensions corresponding to the input image
        - segmentation_data: An ordered dict of segmentation images for each class
        - shape: shape of the scan
    """
    
    INPUT_IMAGE_FILENAME = "ct.nii.gz"
    SEGMENTATION_FOLDERNAME= "segmentations"
    DEFAULT_SEGMENTATION_CLASSES = [
        ("aorta", "aorta.nii.gz"),
        ("heart_atrium_left", "heart_atrium_left.nii.gz"),
        ("heart_atrium_right", "heart_atrium_right.nii.gz"),
        ("heart_myocardium", "heart_myocardium.nii.gz"),
        ("heart_ventricle_left", "heart_ventricle_left.nii.gz"),
        ("heart_ventricle_right", "heart_ventricle_right.nii.gz"),        
    ]
    
    def __init__(self, scan_dir, input_image_file_name=INPUT_IMAGE_FILENAME,
                 segementation_foldername=SEGMENTATION_FOLDERNAME,
                 default_segmentation_classes=DEFAULT_SEGMENTATION_CLASSES,
                 start_points=None, target_shape=None
                 ):        
        self.scan_dir = scan_dir
        if start_points:
            self.start_points=start_points
        else:
            self.start_points=(0,0,0)
        self.shape = nib.load(os.path.join(scan_dir,input_image_file_name)).shape
        if target_shape:
            self.target_shape=target_shape
        else:
            self.target_shape = self.shape
        self.input_data = self.load_data(os.path.join(scan_dir,input_image_file_name))
        
        self.segmentation_data = OrderedDict()
        self.segmentation_availability = OrderedDict()
        for c_class_name, c_lass_filename in default_segmentation_classes:
            c_class_filepath = os.path.join(self.scan_dir, segementation_foldername, c_lass_filename)
            loaded_data = self.load_data(c_class_filepath)
            assert loaded_data.shape == self.target_shape, f"Segmentation image having path: {c_class_filepath} has different shape from input image!"
            self.segmentation_data[c_class_name] = loaded_data
            self.segmentation_availability[c_class_name] = bool(loaded_data.max())
            
    
    def load_data(self, filename):
        data = nib.load(filename).get_fdata()
        data = data[
            self.start_points[0]:self.start_points[0]+self.target_shape[0],
            self.start_points[1]:self.start_points[1]+self.target_shape[1],
            self.start_points[2]:self.start_points[2]+self.target_shape[2]
            ]
        padding_settings = [(0,self.target_shape[i]-data.shape[i]) for i in range(len(data.shape))]        
        padded_data = np.pad(data, padding_settings)
        return padded_data

class ScansDataset(Dataset):
    """
    Attributes:
        - scans_topdir: top directory containing all scans
        - scans_paths: list of all scans paths
        - scans_shapes: dict of all scans shapes (key is scan idx && values is scan shape)
    """
    
    def __init__(self, scans_topdir):
        super().__init__()
        self.scans_topdir = scans_topdir
        self.scans_paths = list(
            filter(
                lambda x: os.path.isdir(x),
                [os.path.join( self.scans_topdir, c_scan_name ) for c_scan_name in os.listdir(self.scans_topdir)]
            )
        )
        self._init_dataset_shapes()
    
    def _init_dataset_shapes(self):
        self.scans_shapes = {}
        for c_scan_idx , c_scan_path in enumerate(self.scans_paths):
            self.scans_shapes[c_scan_idx] = nib.load(os.path.join(c_scan_path, Scan.INPUT_IMAGE_FILENAME)).shape
    
    def __len__(self):
        return len(self.scans_paths)
    
    def __getitem__(self, index):
        if type(index)==int:    
            return Scan(self.scans_paths[index])
        else:
            assert type(index)==tuple
            index, start_points, target_shape = index
            return Scan(self.scans_paths[index], start_points=start_points, target_shape=target_shape)

class UniqueShapeScanDataset(Dataset):
    
    def __init__(self, wrapped_dataset, target_shape=(128,128,128), stride=(50,50,50)) :
        super().__init__()
        self.wrapped_dataset = wrapped_dataset
        self.target_shape = target_shape
        self.stride = stride
        self._dataset_repartition()
    
    def _dataset_repartition(self):
        self.window_maps=[]
        self.total_size=0
        for c_scan_idx in range(len(self.wrapped_dataset)):
            c_scan_shape = self.wrapped_dataset.scans_shapes[c_scan_idx]
            
            window_start_settings_per_dim = defaultdict(list)
            padding_settings_per_dim = defaultdict(list)
            
            for _dim_idx, _dim_size in enumerate(self.target_shape):
                # case is lower
                if c_scan_shape[_dim_idx] <= _dim_size:
                    window_start_settings_per_dim[_dim_idx].append(0)
                    padding_settings_per_dim[_dim_idx].append(_dim_size-c_scan_shape[_dim_idx])
                else:
                    c_window_start_points = list(range(0, c_scan_shape[_dim_idx], self.stride[_dim_idx]))
                    window_start_settings_per_dim[_dim_idx].extend( c_window_start_points )
                    ### hard to explain. sorry!
                    padding_settings_per_dim[_dim_idx].extend( [(0,0)]*(len(c_window_start_points)-1) )
                    padding_settings_per_dim[_dim_idx].append( (0,_dim_size-(c_scan_shape[_dim_idx]-c_window_start_points[-1])) )
            
            c_scan_possible_start_points = list(itertools.product(*window_start_settings_per_dim.values()))
            c_scan_possible_padding_settings = list(itertools.product(*padding_settings_per_dim.values()))
            c_scan_ids = [c_scan_idx] * len(c_scan_possible_start_points)
            
            self.window_maps.extend( list(zip(c_scan_ids,c_scan_possible_start_points,c_scan_possible_padding_settings)) )
        
    def __len__(self):
        return len(self.window_maps)
    
    def __getitem__(self, index):
        
        scan_idx, start_pt, padd_settings = self.window_maps[index]        
        loaded_scan = self.wrapped_dataset.__getitem__((scan_idx, start_pt, self.target_shape))        
        return loaded_scan
                

                    
    
class ScanLearningDataset(Dataset):

    def __init__(self, scans_dataset, preprocessing=lambda x:x, patch_size=(128,128,128)):
        super().__init__()
        self.scans_dataset = scans_dataset
        self.patch_size = patch_size
        self.preprocessing=preprocessing
        
    def __len__(self):
        return len(self.scans_dataset)
    
    def __getitem__(self, index):
        scan_element = self.scans_dataset.__getitem__(index)
        X = np.expand_dims(scan_element.input_data,0)
        X = self.preprocessing(X)
        Y = np.stack( list(scan_element.segmentation_data.values()))
        background_class = ~Y.any(0)
        Y = np.concatenate((background_class[None], Y),axis= 0)
        return X,Y

if __name__ == "__main__":    
    sample_scan_path = "G:/DATA_SANDBOX/3IA-INHEART/Totalsegmentator_dataset/"
    s1 = ScanLearningDataset(UniqueShapeScanDataset(ScansDataset(sample_scan_path)))
    for idx in tqdm.tqdm(range(30)):
        el = s1[idx]