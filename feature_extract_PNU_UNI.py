import os
import openslide

import argparse
import math
from PIL import Image, ImageEnhance

from tqdm import tqdm

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance
from tqdm import tqdm

import timm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import cucim
import h5py

import pandas as pd

import time

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_IMG_SIZE = 224

DEVICE = torch.device("cuda:{}".format(args.gpu))

parser = argparse.ArgumentParser()
parser.add_argument('--root_h5', type=str, 
                    default='[YOUR COORDINATES DIR]')
parser.add_argument('--root_wsi', type=str, 
                    default='/mnt/fileserver_data/nfs/shared/Pathology/Prostate_Busan')
parser.add_argument('--output_dir', type=str, 
                    default='[YOUR FEATURES DIR]')
parser.add_argument('--model_dir', type=str, 
                    default='[YOUR MODEL DIR]')
parser.add_argument('--fps', action='store_true', default=False, help='Using FPS or Not')
parser.add_argument('--patch_size', type=int, 
                    default=512)
parser.add_argument('--gpu', type=int, 
                    default=0)

args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms

def kde_density_sampling(coords, num_samples, bandwidth=50.0, min_dist=50.0):
    """
    Sampling function based on KDE + a distance constraint.

    Args:
        coords (np.ndarray): coordinate array of shape (N, 2)
        num_samples (int): number of patches to sample
        bandwidth (float): KDE bandwidth value
        min_dist (float): minimum distance between selected points

    Returns:
        selected_coords (np.ndarray): final sampled coordinates of shape (num_samples, 2)
    """
    # Step 1: Fit the KDE model
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(coords)

    # Step 2: Estimate the log-density of each coordinate and convert to a probability
    log_dens = kde.score_samples(coords)
    prob = np.exp(log_dens)
    prob = prob / prob.sum()

    # Step 3: Probability-based initial sampling (draw a few extra as a margin)
    #oversample_factor = 3
    oversample_factor = 1 ### Not applied here either — remove as much bias as possible.
    num_candidate = num_samples * oversample_factor
    candidate_indices = np.random.choice(len(coords), size=num_candidate, p=prob)

    candidate_coords = coords[candidate_indices]

    # Step 4: Selection under the distance constraint
    selected_coords = []
    for pt in tqdm(candidate_coords):
        if len(selected_coords) == 0:
            selected_coords.append(pt)
        else:
            dists = distance.cdist([pt], selected_coords)
            if np.all(dists > min_dist):
                selected_coords.append(pt)
        if len(selected_coords) >= num_samples:
            break

    return np.array(selected_coords)

class PatchDataset(Dataset) : 
    def __init__(self, imglist, transform, slide_path, tile_path, patch_size=512, patch_level=0) :
        
        self.imglist = imglist
        self.transform = transform
        self.slide_path = slide_path
        self.tile_path = tile_path
        self.patch_size = patch_size
        self.patch_level = patch_level
         
        self.slide = openslide.OpenSlide(self.slide_path) 
        h5_file = h5py.File(self.tile_path, "r")
        self.tile_coords = h5_file["tile_coords"][:]
        print("Original # of patches {}".format(self.tile_coords.shape[0]))
        if args.fps : 
            print("Performing FPS")
            self.tile_coords = kde_density_sampling(self.tile_coords, int(self.tile_coords.shape[0]/10), bandwidth=self.patch_size, min_dist=self.patch_size) ### FPS Sampling
        print("After FPS # of patches {}".format(self.tile_coords.shape[0]))
    
    def load_tile(self, coord) : 

        img = self.slide.read_region(
                location = coord,
                size = (self.patch_size,self.patch_size),
                level = self.patch_level
            )
        
        rgb_img = img.convert('RGB')
        enhancer = ImageEnhance.Contrast(rgb_img)
        rgb_contrast = enhancer.enhance(1.5)     
                   
        return self.transform(rgb_contrast)  
         
    def __getitem__(self, idx) :
        coord = self.tile_coords[idx]
        img = self.load_tile(coord)
         
        return {"img" : img, "coord": coord}
    
    def __len__(self) : 
        return self.tile_coords.shape[0] 


def img_list_pnu(args) : 
    imgList = []
    states = ["busan_set{}".format(i) for i in range(1, 16)]
    
    for state in states :
        for slides in os.listdir(os.path.join(args.root_wsi, state)) : 
            if slides.endswith(".svs") : 
                final_directory = os.path.join(args.root_wsi,state,slides)
                imgList.append(final_directory)
    
    return imgList

def feature_extracting(args, imgList, transform, model) : 
    start = time.perf_counter()
    
    for name in imgList: 
        slide_name = name[:-4]
        #slide_name = name.split("/")[-1][:-4]
        
        tile_path = f'{args.root_h5}/{slide_name}.h5'
        slide_path = os.path.join(args.root_wsi, name)

        # Define the dataset and loader
        dataset = PatchDataset(imglist=imgList, transform=transform, slide_path=slide_path, tile_path=tile_path, patch_size=args.patch_size)
        loader = DataLoader(dataset=dataset, batch_size=16, pin_memory=True, shuffle=False)
        
        # Output file path
        h5_file_path = os.path.join(args.output_dir, f'{slide_name}.h5')
        
        if slide_name+'.h5' not in os.listdir(args.output_dir) : 
            with h5py.File(h5_file_path, 'w') as h5_file:
                total_patches = len(dataset)  # total number of patches
                feat_dataset = h5_file.create_dataset(
                    "features",
                    shape=(total_patches, 1024), ### UNI feature size 1024
                    dtype=np.float32
                )
                coord_dataset = h5_file.create_dataset(
                    "coords",
                    shape=(total_patches, 2),
                    dtype=np.float32
                )
                
                # Process batch by batch, managing the running index
                start_idx = 0
                for i, batch in enumerate(tqdm(loader, desc=f"Processing {slide_name}", dynamic_ncols=True)):
                    patch, coord = batch['img'], batch['coord']  
                    
                    with torch.inference_mode():
                        features = model(patch.to(DEVICE)).squeeze()  # (B, 1024)
                    
                    batch_size = patch.shape[0]
                    end_idx = start_idx + batch_size

                    # Save data batch by batch
                    feat_dataset[start_idx:end_idx] = features.cpu().numpy()
                    coord_dataset[start_idx:end_idx] = coord.cpu().numpy()
                    
                    start_idx = end_idx
            
            print("FEAT SIZE {}".format(features.shape))
            
        else : 
            print("PASSING {} ".format(slide_name))
    
    print("Feature extraction completed!")
    end = time.perf_counter()

    elapsed = end - start  # seconds (float)
    print(f"took {elapsed:.6f} s")


def main() : 
    #### UNI Model Loading
    model = timm.create_model("vit_large_patch16_224",
                    init_values=1e-5, 
                    num_classes=0, 
                    dynamic_img_size=True)
    model.load_state_dict(torch.load(args.nodel_dir))
    model = model.to(DEVICE)
    
    transform = get_eval_transforms(mean=IMAGENET_MEAN,
                                            std=IMAGENET_STD,
                                            target_img_size = TARGET_IMG_SIZE)

    ##### Imaging name list
    imgList = img_list_pnu(args)
    
    feature_extracting(args, imgList, transform, model) 

if __name__ == "__main__":
    main()