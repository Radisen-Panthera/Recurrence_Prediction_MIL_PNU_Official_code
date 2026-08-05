"""Training script for the PNU prostate biochemical-recurrence (BCR) classifier.

Trains a Multiple-Instance-Learning model (ABMIL by default, or TransMIL when
``--model_large`` is set) on per-patient WSI feature bags to predict binary
recurrence. Each epoch runs a train pass and an evaluation pass (optionally with
Monte-Carlo dropout for uncertainty), logs AUC/accuracy/F1 to TensorBoard,
writes per-epoch prediction CSVs, and checkpoints the model weights.
"""
import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

import warnings
warnings.filterwarnings("ignore")

import math
import pdb
import pickle
import re

import h5py
from scipy import stats
from sklearn.preprocessing import StandardScaler

#### Newly defined library
from datasets.dataset_pfs_pnu import *
from models.model_double import *
from utils.focal_loss import FocalLoss, calculate_class_weights

from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"  # 인피니밴드 사용 안 하도록 (네트워크 문제 우회)
os.environ["NCCL_P2P_DISABLE"] = "1"  # P2P 비활성화로 deadlock 우회
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

def get_split_loader(dataset, training = False, batch_size=1):
    """
        return either the validation loader or training loader 
    """
    kwargs = {'num_workers': 4} 
    
    if training : 
        loader = DataLoader(dataset, batch_size=batch_size, sampler = RandomSampler(dataset), **kwargs)   
    else : 
        loader = DataLoader(dataset, batch_size=batch_size, sampler = SequentialSampler(dataset),  **kwargs)
    return loader

def l1_reg_all(model, reg_type=None):
    """Sum of absolute values of all model parameters (L1 penalty term)."""
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

def l2_reg_all(model):
    """Sum of squared values of all model parameters (L2 penalty term)."""
    l2_reg = None
    for W in model.parameters():
        if l2_reg is None:
            l2_reg = torch.sum(W ** 2)
        else:
            l2_reg = l2_reg + torch.sum(W ** 2)
    return l2_reg


def enable_dropout(model):
    """Enable all Dropout layers in the model"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_inference(model, data_wsi, n_samples=10):
    """Inference by MC Dropout
    
    Args:
        model: model
        data_wsi: Input data
        n_samples: # of sampling
    
    Returns:
        mean logits, mean prob, pred val, attention scores, uncertainty
    """
    model.eval()
    enable_dropout(model)  # Dropout activation
    
    all_logits = []
    all_probs = []
    all_attention_scores = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            binary_logits, binary_probs, _, attention_scores = model(x=data_wsi)
            all_logits.append(binary_logits)
            all_probs.append(binary_probs)
            all_attention_scores.append(attention_scores)
    
    # calculate mean
    mean_logits = torch.stack(all_logits).mean(dim=0)
    mean_probs = torch.stack(all_probs).mean(dim=0)
    mean_pred = torch.argmax(mean_logits, dim=1)
    
    # calculate uncertainty (variance of probability)
    prob_std = torch.stack(all_probs).std(dim=0)
    uncertainty = prob_std.mean().item()
    
    # mean of attention scores 
    mean_attention_scores = {}
    for key in all_attention_scores[0].keys():
        mean_attention_scores[key] = torch.stack([att[key] for att in all_attention_scores]).mean(dim=0)
    
    return mean_logits, mean_probs, mean_pred, mean_attention_scores, uncertainty

### Training settingst
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--train_clinical',   type=str, default='./folds/train_fold0.csv', help='clinical information from TCGA dataset')
parser.add_argument('--train_img',   type=str, default='/mnt/fileserver/Pathology/PNU/embedded_features_512_updated', help='Data directory to WSI features (TCGA)')
parser.add_argument('--test_clinical',   type=str, default='./folds/val_fold0.csv', help='clinical information from KBSMC dataset')
parser.add_argument('--test_img',   type=str, default='/mnt/fileserver/Pathology/PNU/embedded_features_512_updated', help='Data directory to WSI features (KBSMC)')
parser.add_argument('--gpu',   type=int, default=0, help='Which GPU would be used')
parser.add_argument('--gamma',   type=float, default=1.0, help='power of focal loss penalty to majority')
parser.add_argument('--dropout',   type=float, default=0.25, help='Dropout ratio') ######## majority 클래스에 fit되는 것을 방지하기 위해 적은 값을 사용
parser.add_argument('--attn_branch',   type=int, default=2, help='# of attention branches') 
parser.add_argument('--mc_dropout', action='store_true', default=False, help='Use or not with test mc dropout')
parser.add_argument('--mc_samples',   type=int, default=10, help='# of samples for MC dropouts')
parser.add_argument('--lambda_reg',   type=float, default=5e-7, help='regularized term weights')
parser.add_argument('--epoch',   type=int, default=100, help='# of epochs')
parser.add_argument('--gc',   type=int, default=16, help='gradient accumulation')
parser.add_argument('--lr',   type=float, default=5e-6, help='learning rate')
parser.add_argument('--wd',   type=float, default=5e-7, help='weight decay')
parser.add_argument('--writer_dir',   type=str, default='./tensorboard_log_fold0_double_reproductivity', help='tensorboard logging directory')
parser.add_argument('--result_df_path',   type=str, default='results_dataframe_fold0_double_reproductivity', help='Predictions for each train/test datase')
parser.add_argument('--weights_dir',   type=str, default='weights_saving_fold0_double_reproductivity', help='model weights directory per each epochs')
parser.add_argument('--layer_norm', action='store_true', default=False, help='Use layer normalization for aggregation features')
parser.add_argument('--seed', 	type=int, default=7, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--single_slide', action='store_true', default=False, help='Single Slide Selection')
parser.add_argument('--single_slide_dir',   type=str, default='/home/yscho/code_cloud_MIL_train/patient_representative_top1_train.csv', help='selection reference csv files')

args = parser.parse_args()

if not os.path.isdir("./{}".format(args.result_df_path)) : 
    os.mkdir("./{}".format(args.result_df_path))

if not os.path.isdir(args.writer_dir) : 
    os.mkdir(args.writer_dir)

if not os.path.isdir("./{}".format(args.weights_dir)) :
    os.mkdir("./{}".format(args.weights_dir))

def set_seed(seed):
    """
    Fix random seed for reproducible results
    
    Args:
        seed (int): Random seed value
    """
    import random
    import numpy as np
    import torch
    
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
    print(f"Random seed set to {seed}")

def train(args, train_dataset, test_dataset, model, device, binary_loss, optimizer) :
    """Run the full train/eval loop over ``args.epoch`` epochs.

    Logs metrics to TensorBoard, dumps per-epoch prediction CSVs, and saves a
    checkpoint after every epoch."""

    writer = SummaryWriter(args.writer_dir, flush_secs=15)
    
    train_loader = get_split_loader(train_dataset, training=True, batch_size=1)
    test_loader = get_split_loader(test_dataset, training=False, batch_size=1)
    0
    best_loss = 100.00
    
    print(f"MC DROPOUT {args.mc_dropout}")
    print(f"LayerNorm {args.layer_norm}")
    
    patience = 0

    for epoch in range(args.epoch):
        
        ### Early Stop 조건

        ################# TRAIN STEP ######################
        
        train_loss = 0.0
        train_epoch_logits = []      # (2,) 벡터들을 저장
        train_epoch_probs = []       # (2,) 확률들을 저장
        train_epoch_predictions = []
        train_epoch_targets = []
        
        model.train()
        
        for batch_idx, (data_WSI, c, name) in enumerate(tqdm(train_loader)):

            data_WSI = data_WSI.to(device)
            c = c.type(torch.LongTensor).to(device)  # LongTensor로 변경!

            # Forward pass
            binary_logits, binary_probs, binary_pred, attention_scores = model(x=data_WSI)

            # Loss 계산 (CrossEntropy)
            loss = binary_loss(binary_logits, c)  # 2-class CrossEntropy
            loss_value = loss.item()
            
            loss_reg = l2_reg_all(model) * args.lambda_reg ## L2 regularization
            train_loss += loss_value + loss_reg

            loss = loss / args.gc + loss_reg
            loss.backward()
            
            if c.item() == 1 : 
                print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value + loss_reg, c.item()))

            if (batch_idx + 1) % 10 == 0:
                print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value + loss_reg, c.item()))

            if (batch_idx + 1) % args.gc == 0: 
                optimizer.step()
                optimizer.zero_grad()
            
            # 예측값들 저장
            with torch.no_grad():
                # 2-class의 경우 class 1 (재발)의 확률을 AUC 계산에 사용
                prob_class1 = binary_probs[0, 1].cpu().item()  # class 1의 확률
                pred = binary_pred.cpu().item()
                target = c.cpu().item()
                
                train_epoch_logits.append(binary_logits.squeeze().cpu().numpy())  # (2,) 배열
                train_epoch_probs.append(prob_class1)  # class 1 확률만 저장 (AUC용)
                train_epoch_predictions.append(pred)
                train_epoch_targets.append(target)

        # Epoch 종료 후 성능 평가
        train_loss /= len(train_loader)
        
        train_epoch_targets = np.array(train_epoch_targets)
        train_epoch_probs = np.array(train_epoch_probs)  # class 1의 확률들
        train_epoch_predictions = np.array(train_epoch_predictions)
        
        pd.DataFrame({"predictions": train_epoch_predictions, "GT" : train_epoch_targets}).to_csv("./{}/df_train_{}.csv".format(args.result_df_path, epoch))
        
        # AUC 계산 (class 1의 확률 사용)
        if len(np.unique(train_epoch_targets)) > 1:
            auc_score = roc_auc_score(train_epoch_targets, train_epoch_probs)
        else:
            auc_score = 0.5
        
        accuracy = accuracy_score(train_epoch_targets, train_epoch_predictions)
        f1 = f1_score(train_epoch_targets, train_epoch_predictions, zero_division=0)
        
        print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, '
            f'AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
        
        # TensorBoard 로깅
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/AUC', auc_score, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)
        writer.add_scalar('Train/F1', f1, epoch)

        ################# TEST STEP ######################
        
        test_loss = 0.
        test_epoch_logits = []
        test_epoch_probs = []
        test_epoch_predictions = []
        test_epoch_targets = []
        test_epoch_uncertainties = []
        
        for batch_idx, (data_WSI, c, name) in enumerate(tqdm(test_loader)):
            data_WSI = data_WSI.to(device)
            c = c.type(torch.LongTensor).to(device)
            
            if args.mc_dropout : 
                # MC Dropout 
                mean_logits, mean_probs, mean_pred, mean_attention_scores, uncertainty = mc_dropout_inference(
                    model, data_WSI, n_samples=args.mc_samples
                )
                
                loss = binary_loss(mean_logits, c)
            
            else : 
                model.eval()
                with torch.no_grad() : 
                    binary_logits, binary_probs, binary_pred, attention_scores = model(x=data_WSI)
                    results_dict = model(x=data_WSI)
                loss = binary_loss(binary_logits, c)
                
            loss_value = loss.item()
            
            loss_reg = l2_reg_all(model) * args.lambda_reg
            test_loss += loss_value + loss_reg
            
            if c.item() == 1 : 
                print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value, c.item()))
            
            if (batch_idx + 1) % 10 == 0:
                print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value, c.item()))
                
            if args.mc_dropout : 
                prob_class1 = mean_probs[0, 1].cpu().item()
                pred = mean_pred.cpu().item()
                target = c.cpu().item()
                
                test_epoch_logits.append(mean_logits.squeeze().cpu().numpy())
                test_epoch_probs.append(prob_class1)
                test_epoch_predictions.append(pred)
                test_epoch_targets.append(target)
                test_epoch_uncertainties.append(uncertainty)
            else : 
                with torch.no_grad():
                    prob_class1 = binary_probs[0, 1].cpu().item()  # class 1의 확률
                    pred = binary_pred.cpu().item()
                    target = c.cpu().item()
                    
                    test_epoch_logits.append(binary_logits.squeeze().cpu().numpy())  # (2,) 배열
                    test_epoch_probs.append(prob_class1)  # class 1 확률만 저장 (AUC용)
                    test_epoch_predictions.append(pred)
                    test_epoch_targets.append(target)
            
        test_loss /= len(test_loader)
        
        test_epoch_targets = np.array(test_epoch_targets)
        test_epoch_probs = np.array(test_epoch_probs)
        test_epoch_predictions = np.array(test_epoch_predictions)
        
        if args.mc_dropout : 
            test_epoch_uncertainties = np.array(test_epoch_uncertainties)
            
            pd.DataFrame({
                "predictions": test_epoch_predictions, 
                "GT": test_epoch_targets,
                "uncertainty": test_epoch_uncertainties
            }).to_csv("./{}/df_test_mc_{}.csv".format(args.result_df_path, epoch))
        else : 
            pd.DataFrame({
                "predictions": test_epoch_predictions, 
                "GT": test_epoch_targets
            }).to_csv("./{}/df_test_{}.csv".format(args.result_df_path, epoch))
            

        if len(np.unique(test_epoch_targets)) > 1:
            auc_score = roc_auc_score(test_epoch_targets, test_epoch_probs)
        else:
            auc_score = 0.5
        
        accuracy = accuracy_score(test_epoch_targets, test_epoch_predictions)
        f1 = f1_score(test_epoch_targets, test_epoch_predictions, zero_division=0)
        mean_uncertainty = np.mean(test_epoch_uncertainties)
        
        print(f'Epoch: {epoch}, test_loss: {test_loss:.4f}, '
            f'AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, '
            f'Mean Uncertainty: {mean_uncertainty:.4f}')
        
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/AUC', auc_score, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/F1', f1, epoch)
        writer.add_scalar('Test/Mean_Uncertainty', mean_uncertainty, epoch)
        
        torch.save(model.state_dict(), "/home/yscho/code_cloud_MIL_train/{}/checkpoint_{}.pth".format(args.weights_dir, epoch))

        #### Early stop을 위한 patient 진행

    writer.close()

def main(args) :
    """Set up datasets, model, loss and optimizer, then launch training."""

    set_seed(args.seed)
    
    print("########### ATTENTION BRANCH ########### {}".format(args.attn_branch))
    
    device = "cuda:{}".format(args.gpu)
    #device = torch.device("cuda") ### Multi-GPU 연산 고려
    
    train_dataset = Dataset_PFS_PNU(csv_path = args.train_clinical,
                        data_dir = args.train_img,
                        label_col = 'BCR')

    test_dataset = Dataset_PFS_PNU(csv_path = args.test_clinical,
                        data_dir = args.test_img,
                        label_col = 'BCR')
        
    #class_weights = [0.5824742268041238, 3.53125]  ### 반대가 되면 된다
    #class_weights = [0.5623115577889447, 4.512096774193548]
    #class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    #focal_loss = FocalLoss(
    #    alpha=class_weights,  # 클래스별 가중치
    #    gamma=args.gamma,           # focusing parameter (높을수록 어려운 샘플에 집중)
    #    reduction='mean'
    #).to(device)
    
    binary_loss = nn.CrossEntropyLoss()
    
    if args.dropout > 0.0 : 
        model = ABMIL(n_classes=2, mode='binary', dropout=args.dropout, layer_norm = args.layer_norm, attention_branch=args.attn_branch)
    else : 
        model = ABMIL(n_classes=2, mode='binary', dropout=0.0, layer_norm = args.layer_norm, attention_branch=args.attn_branch)
    
    ####### 중단된 이유로 이어서 진행!!

    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs")
    #    model = nn.DataParallel(model, device_ids=[0, 1, 2]).to(device)  # 명시적으로 device_ids 지정
    
    model = model.to(device)
    print(model)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    train(args, train_dataset, test_dataset, model, device, binary_loss, optimizer)
        
if __name__ == "__main__":
	main(args)
    
    
    
    
    
    
