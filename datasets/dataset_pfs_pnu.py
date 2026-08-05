from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


from collections import OrderedDict
import threading
import queue
import time


class Dataset_PFS_PNU(Dataset) : 
    def __init__(self, csv_path = '', 
                 data_dir = '', 
                 shuffle = False, seed = 7, label_col = 'BCR') : 
        
        self.seed = seed
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.label_col = label_col
        
        slide_data = pd.read_csv(csv_path)  
        feat_names = self.feat_naming(slide_data)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
            np.random.shuffle(feat_names)
        
        self.slide_data = slide_data
        self.feat_names = feat_names
         
    
    def feat_naming(self, slide_data) : 
        feat_names = []

        for name in os.listdir(self.data_dir) : 
            try : 
                patientno = float(name.split('-')[1])
                if patientno in list(slide_data['Patient_no']) : 
                    feat_names.append(patientno)
            except : 
                patientno = float(name.split('-')[1][:-3])
                if patientno in list(slide_data['Patient_no']) : 
                    feat_names.append(patientno)
                
        return list(set(feat_names))
    
    def slide_sector_agg(self, patientno) : 
        slide_sector = []
        for word in os.listdir(self.data_dir) : 
            try : 
                if float(word.split('-')[1]) == patientno : 
                    slide_sector.append(word)
                else : 
                    continue
            except : 
                if float(word.split('-')[1][:-3]) == patientno : 
                    slide_sector.append(word)
                else : 
                    continue
        
        return slide_sector
    
    
    def __getitem__(self, idx) : 
        
        feat_file_name = self.feat_names[idx]
        feat_name = feat_file_name
        patientno = feat_name
        label = float(self.slide_data[self.slide_data['Patient_no'] == patientno][self.label_col])
        
        slide_sector = self.slide_sector_agg(patientno)
        
        path_features = []
        for feature_name in slide_sector : 
            with h5py.File(os.path.join(self.data_dir, feature_name), 'r') as file : 
                feature = file['features'][:]
            path_features.append(feature)
        
        path_features = torch.tensor(np.vstack(path_features))
        
        return (path_features, label, feat_name)
        
    def __len__(self) : 
        return len(self.feat_names)