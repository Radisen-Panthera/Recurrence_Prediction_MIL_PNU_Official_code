import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance
        
        Args:
            alpha (float or Tensor): 클래스별 가중치 
                - None: 가중치 없음
                - float: alpha for class 1, (1-alpha) for class 0
                - Tensor: [weight_class0, weight_class1]
            gamma (float): focusing parameter (default: 2.0)
            reduction (str): 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - 2-class logits
            targets: (N,) - class indices (0 or 1)
        """
        # CrossEntropy 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Softmax 확률 계산
        p = torch.exp(-ce_loss)  # p = softmax의 정답 클래스 확률
        
        # Focal weight 계산: (1-p)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Alpha 가중치 적용
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # alpha for class 1, (1-alpha) for class 0
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # alpha is tensor [weight_class0, weight_class1]
                alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        # Focal loss 계산
        focal_loss = focal_weight * ce_loss
        
        # Reduction 적용
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_weights(dataloader):
    from tqdm.notebook import tqdm
    """
    데이터셋에서 클래스 분포를 계산하여 가중치 결정
    """
    class_counts = [0, 0]  # [class_0_count, class_1_count]
    total_samples = 0
    
    for batch_idx, (_, _, _, c) in enumerate(tqdm(dataloader)):
        # c: 0=재발, 1=재발안함
        for label in c:
            class_counts[int(label.item())] += 1
            total_samples += 1
    
    # 클래스별 가중치 계산 (inverse frequency)
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = total_samples / (len(class_counts) * count)
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    print(f"Class distribution: Class 0 (재발): {class_counts[0]}, Class 1 (재발안함): {class_counts[1]}")
    print(f"Class weights: {class_weights}")
    
    return torch.tensor(class_weights, dtype=torch.float)
