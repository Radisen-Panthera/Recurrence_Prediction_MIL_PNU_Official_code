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
        
        ### We consider recur case only within 5 years
        #slide_data['label'] = 0 
        #slide_data.loc[(slide_data['BCR interval'] <= 60) & (slide_data['BCR'] == 1), 'label'] = 1
        #slide_data.loc[(slide_data['BCR interval'] >= 1825) , 'label'] = 0
                
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
        #feat_name = feat_file_name[:-3] 
        feat_name = feat_file_name
        #patientno = float(feat_name.split('-')[1])
        patientno = feat_name
        label = float(self.slide_data[self.slide_data['Patient_no'] == patientno][self.label_col])
        
        #slide_sector = [word for word in os.listdir(self.data_dir) if float(word.split('-')[1]) == patientno ]
        slide_sector = self.slide_sector_agg(patientno)
        
        path_features = []
        for feature_name in slide_sector : 
            with h5py.File(os.path.join(self.data_dir, feature_name), 'r') as file : 
                feature = file['features'][:]
            path_features.append(feature)
        
        #wsi_path = os.path.join(self.data_dir, feat_file_name)
        
        path_features = torch.tensor(np.vstack(path_features))
        
        return (path_features, label, feat_name)
        
    def __len__(self) : 
        return len(self.feat_names)


class Dataset_PFS_PNU_single_slide(Dataset) : 
    def __init__(self, csv_path = '', 
                 data_dir = '', 
                 shuffle = False, seed = 7, label_col = 'BCR', single_slide=None) : 
        
        self.seed = seed
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.label_col = label_col
        self.single_slide = single_slide
        
        slide_data = pd.read_csv(csv_path)
        
        ### We consider recur case only within 5 years
        #slide_data['label'] = 0 
        #slide_data.loc[(slide_data['BCR interval'] <= 60) & (slide_data['BCR'] == 1), 'label'] = 1
        #slide_data.loc[(slide_data['BCR interval'] >= 1825) , 'label'] = 0
                
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
            if name.split('.')[-1] != 'done' :        
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
        
        if not self.single_slide : 
            slide_sector = []
            for word in os.listdir(self.data_dir) : 
                if word.split('.')[-1] != 'done' :
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
        else : 
            slide_sector = []
            data_csv = pd.read_csv(self.single_slide)
            lists = [word+'.h5' for word in list(data_csv['slide'])]
            for word in lists : 
                if word.split('.')[-1] != 'done' :
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
        #feat_name = feat_file_name[:-3] 
        feat_name = feat_file_name
        #patientno = float(feat_name.split('-')[1])
        patientno = feat_name
        label = float(self.slide_data[self.slide_data['Patient_no'] == patientno][self.label_col])
        
        #slide_sector = [word for word in os.listdir(self.data_dir) if float(word.split('-')[1]) == patientno ]
        slide_sector = self.slide_sector_agg(patientno)
        
        path_features = []
        for feature_name in slide_sector : 
            with h5py.File(os.path.join(self.data_dir, feature_name), 'r') as file : 
                feature = file['features'][:]
            path_features.append(feature)
        
        #wsi_path = os.path.join(self.data_dir, feat_file_name)
        
        path_features = torch.tensor(np.vstack(path_features))
        
        return (path_features, label, feat_name)
        
    def __len__(self) : 
        return len(self.feat_names)



# 방법 1: LRU Cache + 압축 저장
class Dataset_PFS_PNU_LRU(Dataset):
    def __init__(self, csv_path='', data_dir='', shuffle=False, seed=7, 
                 label_col='BCR', cache_size=1000, compress_cache=True):
        
        self.seed = seed
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.label_col = label_col
        self.cache_size = cache_size
        self.compress_cache = compress_cache
        
        # LRU Cache 구현
        self.file_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        slide_data = pd.read_csv(csv_path)
        
        feat_names = self.feat_naming(slide_data)
        
        if shuffle:
            np.random.seed(seed)
            indices = np.arange(len(feat_names))
            np.random.shuffle(indices)
            feat_names = [feat_names[i] for i in indices]
        
        self.slide_data = slide_data
        self.feat_names = feat_names
        
        # 라벨 캐시 미리 구성
        self.label_cache = dict(zip(slide_data['Patient_no'], slide_data[self.label_col]))
    
    def feat_naming(self, slide_data):
        feat_names = []
        patient_nos = set(slide_data['Patient_no'])

        for name in os.listdir(self.data_dir):
            try:
                patientno = float(name.split('-')[1])
                if patientno in patient_nos:
                    feat_names.append(name)
            except (IndexError, ValueError):
                continue
        
        return feat_names
    
    def _compress_tensor(self, tensor):
        """텐서를 numpy로 변환하고 압축"""
        if self.compress_cache:
            return tensor.numpy().astype(np.float16)  # 메모리 절약을 위해 float16 사용
        return tensor
    
    def _decompress_tensor(self, data):
        """압축된 데이터를 텐서로 복원"""
        if self.compress_cache and isinstance(data, np.ndarray):
            return torch.from_numpy(data.astype(np.float32))
        return data
    
    def __getitem__(self, idx):
        feat_file_name = self.feat_names[idx]
        feat_name = feat_file_name[:-3]
        patientno = float(feat_name.split('-')[1])
        
        label = float(self.label_cache[patientno])
        
        # LRU Cache 확인
        if feat_file_name in self.file_cache:
            # 캐시 히트: 항목을 맨 뒤로 이동 (최근 사용)
            cached_data = self.file_cache.pop(feat_file_name)
            self.file_cache[feat_file_name] = cached_data
            path_features = self._decompress_tensor(cached_data)
            self.cache_hits += 1
        else:
            # 캐시 미스: 파일 로드
            wsi_path = os.path.join(self.data_dir, feat_file_name)
            path_features = torch.load(wsi_path)
            
            # 캐시 사이즈 관리 (LRU 방식)
            if len(self.file_cache) >= self.cache_size:
                # 가장 오래된 항목 제거
                self.file_cache.popitem(last=False)
            
            # 새 항목 추가
            compressed_data = self._compress_tensor(path_features)
            self.file_cache[feat_file_name] = compressed_data
            self.cache_misses += 1
        
        return (path_features, label, feat_name)
    
    def __len__(self):
        return len(self.feat_names)
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return f"Cache hit rate: {hit_rate:.2%} ({self.cache_hits}/{total})"


class Dataset_PFS_KBSMC(Dataset) : 
    def __init__(self, csv_path = '', 
                 data_dir = '', 
                 shuffle = False, seed = 7) : 
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            n_bins (int): Whether to print a summary of the dataset
            label_col (str): Dictionary with key, value pairs for converting str labels to int
        """
        
        self.seed = seed
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.csv_path = csv_path
        self.data_dir = data_dir
        
        try : 
            slide_data = pd.read_csv(csv_path, low_memory=False)
        except : 
            slide_data = pd.read_csv(csv_path, low_memory=False, encoding='cp949')
        slide_data = self.processing(slide_data)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        
        #slide_data['PFS_DAYS'] = slide_data[label_col] * 30
        
        ##### Definition of risk groups
        
        slide_data['label'] = 0 ### DEFAULT low-risk group
        slide_data['RFS'].astype(float) ## 기존 PFS에서 변경
        slide_data.loc[(slide_data['RFS'] < 1825) & (slide_data['Recur'] == 1), 'label'] = 1 #### Define high-risk group based on "5 years"
        
        slide_data.loc[(slide_data['RFS'] >= 1825) , 'Recur'] = 0 ### 전부 다 재발로 re-labeling
        
        self.slide_data = slide_data
    
    def processing(self, slide_data) : 
        unique_id_01 = [word for word in list(slide_data['SampleID']) if word.endswith("01")]
        slide_data = slide_data[slide_data['SampleID'].isin(unique_id_01)] ### 임상에서 중복 샘플 제거
        slide_data = slide_data[pd.to_numeric(slide_data['RFS'], errors='coerce').notna()] ### 결측치, NA 제거
        slide_data = slide_data[slide_data['Recur'].isin([0,1])] ### Unknown 케이스 제거
        #slide_data['Progression'] = 1-slide_data['Progression'] ## 방향 변경
        slide_data['RFS'] = slide_data['RFS'].astype(float) ### Type 변경
        slide_name_lists = [word[:-3] for word in os.listdir(self.data_dir)]
        
        ### 이미지와 임상 데이터가 같이 있는 샘플들만 filtering
        slide_new_names = []
        for name in slide_name_lists : 
            if '-' in name : 
                new_name = name.split('-')[0] + '_' + name.split('-')[1]
                slide_new_names.append(new_name)
            else : 
                slide_new_names.append(name)
        
        metadata_names = list(slide_data['tube label'])
        
        intersect_wsi = set(slide_new_names).intersection(set(metadata_names))        
        slide_data = slide_data[slide_data['tube label'].isin(intersect_wsi)]
        
        ### noisy case 제거
        noisy_case = ['1_053_10', '1_110_16']
        slide_data = slide_data[~slide_data['tube label'].isin(noisy_case)]
        
        return slide_data.reset_index(drop=True)
        
    def __getitem__(self, idx) : 
        case_id = self.slide_data['tube label'][idx]
        label = self.slide_data['label'][idx]
        event_time = self.slide_data['RFS'][idx]
        c = self.slide_data['Recur'][idx]
        slide_ids = self.slide_data['tube label'][idx]
        
        path_features = []
        if type(slide_ids) == str : 
            slide_ids_ = [slide_ids]
        else : 
            slide_ids_ = slide_ids

        for slide_id in slide_ids_:
            try : 
                wsi_path = os.path.join(self.data_dir, '{}.h5'.format(slide_id))
                with h5py.File(wsi_path, 'r') as file : 
                    wsi_bag = file['features'][:]
                path_features.append(torch.tensor(wsi_bag))
            except : 
                slide_id_new = slide_id.split('_')[0] + '-' + slide_id.split('_')[1] + '_' + slide_id.split('_')[2]
                wsi_path = os.path.join(self.data_dir, '{}.h5'.format(slide_id_new))
                with h5py.File(wsi_path, 'r') as file : 
                    wsi_bag = file['features'][:]
                path_features.append(torch.tensor(wsi_bag))
            path_features = torch.cat(path_features, dim=0)
        
        #### C-index 계산 시 필요한 정보들 산출
        return (path_features, label, event_time)
        
    def __len__(self) : 
        return len(self.slide_data)

'''
class Dataset_PFS_PNU(Dataset): 
    def __init__(self, csv_path='', 
                 data_dir='', 
                 shuffle=False, seed=7, label_col='BCR', cache_size=500): 
        
        self.seed = seed
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.label_col = label_col
        self.cache_size = cache_size
        
        # 캐시 초기화
        self.file_cache = {}
        self.label_cache = {}  # 라벨 조회도 캐시
        
        slide_data = pd.read_csv(csv_path)
        
        feat_names = self.feat_naming(slide_data)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
            np.random.shuffle(feat_names)
        
        self.slide_data = slide_data
        self.feat_names = feat_names
        
        # 라벨 캐시 미리 구성 (pandas 조회 속도 향상)
        for _, row in slide_data.iterrows():
            self.label_cache[row['Patient_no']] = row[self.label_col]
    
    def feat_naming(self, slide_data): 
        feat_names = []
        patient_nos = set(slide_data['Patient_no'])  # set으로 변환해서 조회 속도 향상

        for name in os.listdir(self.data_dir): 
            patientno = float(name.split('-')[1])
            if patientno in patient_nos: 
                feat_names.append(name)
        
        return feat_names
            
    def __getitem__(self, idx): 
        
        feat_file_name = self.feat_names[idx]
        feat_name = feat_file_name[:-3] 
        patientno = float(feat_name.split('-')[1])
        
        # 캐시된 라벨 조회 (pandas 조회보다 빠름)
        label = float(self.label_cache[patientno])
        
        # 캐시에서 먼저 확인
        if feat_file_name in self.file_cache:
            wsi_bag = self.file_cache[feat_file_name]
        else:
            wsi_path = os.path.join(self.data_dir, feat_file_name)
            wsi_bag = torch.load(wsi_path)
            #with h5py.File(wsi_path, 'r') as file: 
            #    wsi_bag = file['features'][:]
            
            # 캐시 사이즈 관리
            if len(self.file_cache) >= self.cache_size:
                # 가장 오래된 항목 제거 (간단한 FIFO)
                oldest_key = next(iter(self.file_cache))
                del self.file_cache[oldest_key]
            
            self.file_cache[feat_file_name] = wsi_bag
        
        # tensor 변환 최적화
        #path_features = torch.from_numpy(wsi_bag.astype(np.float32))
        path_features = wsi_bag
        
        return (path_features, label, feat_name)
        
    def __len__(self): 
        return len(self.feat_names)

'''