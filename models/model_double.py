import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = 0.0, n_classes = 1, model_size_wsi='small'):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        if model_size_wsi == 'small' : 
            self.attention_a = [
                nn.Linear(L, D),
                nn.Tanh()]
            
            self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        else : 
            self.attention_a = nn.Sequential(
                nn.Linear(L, D),
                nn.ReLU(),
                nn.Linear(D, D),
                nn.Tanh()
            )
            self.attention_b = nn.Sequential(
                nn.Linear(L, D),
                nn.ReLU(),
                nn.Linear(D, D),
                nn.Sigmoid()
            )

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # (N, D)
        b = self.attention_b(x)  # (N, D)
        A = a.mul(b)  # (N, D)
        A = self.attention_c(A)  # (N, n_classes)
        return A, x


class ABMIL(nn.Module):
    def __init__(self, n_classes=4,
                 model_size_wsi: str='small', dropout=0.0, mode='PFS', layer_norm=False, feat_size=1024, instance_norm=False, attention_branch=1): ### Dropout을 주자
        super(ABMIL, self).__init__()
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [feat_size, feat_size], "big": [feat_size, 768, feat_size]}
        self.mode = mode
        self.layer_norm = layer_norm
        self.instance_norm = instance_norm
        self.attention_branch = attention_branch

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        if model_size_wsi == 'small' : 
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        else : 
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Linear(size[1], size[2]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        self.instancenorm = nn.InstanceNorm1d(feat_size, affine=False)

        ### Path Transformer + Attention Head
        self.path_attention_head = Attn_Net_Gated(L=feat_size, D=feat_size, dropout=dropout, n_classes=attention_branch, model_size_wsi=model_size_wsi)
            
        ### Layer normalization 추가 (slide-representation 정규화를 하자)
        self.norm = nn.LayerNorm(feat_size)
        if model_size_wsi == 'small' : 
            self.path_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        else : 
            self.path_rho = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Classifier
        self.classifier = nn.Linear(feat_size, n_classes)
        
        self.binary_classifier = nn.Linear(feat_size, 2)  # 2개 출력
    
    def forward(self, x):
        # x: (B, N, d). 보통 B=1 (MIL)
        B, N, d = x.size(0), x.size(1), x.size(2)
        assert B == 1, "Current implementation assumes batch_size=1 for MIL."

        x_reshaped = x.view(-1, x.size(-1))  # (N, d)
        # 정규화: InstanceNorm1d 대신 LayerNorm 권장
        if self.instance_norm:
            # self.instancenorm는 InstanceNorm1d라면 비활성화하고, 아래처럼 교체 권장:
            # x_reshaped = self.instancenorm(x_reshaped)
            pass

        h_tokens = self.wsi_net(x_reshaped)  # (N, D=feat_size)
        if self.instance_norm:
            pass  # 동일

        # Attention
        # A_path: (N, K), h_tokens: (N, D)
        A_path, _ = self.path_attention_head(h_tokens)
        K = A_path.size(1)  # attention_branch

        # 각 브랜치(=클래스)별로 instance-softmax
        A_path = F.softmax(A_path, dim=0)  # (N, K)
        # 클래스별 풀링: (K, D)
        H = torch.matmul(A_path.transpose(1, 0), h_tokens)  # (K, D)

        if self.layer_norm:
            # LayerNorm은 (D,)에 적용하므로 각 행에 적용
            H = torch.stack([self.norm(H[k]) for k in range(K)], dim=0)  # (K, D)

        # path_rho 적용: rho는 (D->D)로 가정
        H = self.path_rho(H)  # (K, D)

        attention_scores = {'path': A_path}  # (N, K)

        if self.mode == 'binary':
            # 기존 binary_classifier를 재활용하여
            # "클래스 k의 풀링 H[k]"와 "클래스 k의 가중치 W[k]"만 곱해서 대각선 성분으로 로짓을 만듭니다.
            W = self.binary_classifier.weight  # (2, D)
            b = self.binary_classifier.bias    # (2,)
            # logits_k = <W_k, H_k> + b_k
            logits = (H * W).sum(dim=1) + b    # (K,)  (K는 2여야 함)
            logits = logits.unsqueeze(0)       # (1, 2)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)  # (1,)
            return logits, probs, pred, attention_scores

        else:
            # 멀티클래스: classifier.weight (n_classes, D), bias (n_classes)
            # 각 클래스 로짓은 "자기 클래스 풀링 H[class]와 자기 가중치"만 쓰는 동일한 아이디어
            # 단, attention_branch==n_classes일 때만 “대각선” 방식이 깔끔합니다.
            assert K == self.n_classes, "Set attention_branch == n_classes for class-specific pooling."
            W = self.classifier.weight  # (C, D)
            b = self.classifier.bias    # (C,)
            logits = (H * W).sum(dim=1) + b  # (C,)
            logits = logits.unsqueeze(0)     # (1, C)

            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, attention_scores

        
    '''
    
    #####################################################
    # ys ver                                             
    # label 0 - label 1의 평균 pooling으로 넘어감          
    #####################################################
    
    def forward(self, x):
        # x: (batch_size, N, d) -> (1, N, 2048)
        batch_size = x.size(0)
        
        # WSI feature extraction for each patch
        # x를 (batch_size * N, d)로 reshape하여 FC layer에 통과
        x_reshaped = x.view(-1, x.size(-1))  # (N, 2048)
        if self.instance_norm : 
            x_reshaped = self.instancenorm(x_reshaped)
        
        h_path_bag = self.wsi_net(x_reshaped)  # (N, 2048)
        
        if self.instance_norm : 
            h_path_bag = self.instancenorm(h_path_bag)
        
        # Attention mechanism
        A_path, h_path = self.path_attention_head(h_path_bag)  # A_path: (N, 1), h_path: (N, 2048)
        
        # Attention weights normalization and aggregation
        #A_path = F.softmax(A_path, dim=0)  # (N, 1)
        #h_path = torch.mm(A_path.transpose(1, 0), h_path)  # (1, 2048)
        
        # 수정된 코드:
        if self.attention_branch == 1:
            A_path = F.softmax(A_path, dim=0)  # (N, 1)
            h_path = torch.mm(A_path.transpose(1, 0), h_path)  # (1, 2048)
        else:  # attention_branch == 2
            A_path = F.softmax(A_path, dim=0)  # (N, 2)
            # 각 클래스별로 attention aggregation
            h_path_0 = torch.mm(A_path[:, 0:1].transpose(1, 0), h_path)  # (1, 2048)
            h_path_1 = torch.mm(A_path[:, 1:2].transpose(1, 0), h_path)  # (1, 2048)
            h_path = (h_path_0 + h_path_1) / 2  # 평균 또는 다른 방식으로 결합
        
        if self.layer_norm : 
            h_path = self.norm(h_path) ### Layer normalization
                        
        h_path = self.path_rho(h_path).squeeze(0)  # (2048,)

        # Final feature representation
        h = h_path
        
        attention_scores = {'path': A_path}
        
        if self.mode == 'binary' : 
            binary_logits = self.binary_classifier(h).unsqueeze(0)  # (1, 2)
            binary_probs = F.softmax(binary_logits, dim=1)  # (1, 2)
            binary_pred = torch.argmax(binary_logits, dim=1)  # (1,) - 0 or 1
            
            return binary_logits, binary_probs, binary_pred, attention_scores
        
        else : 
            logits = self.classifier(h).unsqueeze(0)  # (1, n_classes)
            Y_hat = torch.topk(logits, 1, dim=1)[1]  # 예측 label
            
            hazards = torch.sigmoid(logits)  # (1, n_classes)
            S = torch.cumprod(1 - hazards, dim=1)  # (1, n_classes)
            return hazards, S, Y_hat, attention_scores
    '''