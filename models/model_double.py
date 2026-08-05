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
                 model_size_wsi: str='small', dropout=0.0, mode='PFS', layer_norm=False, feat_size=1024, instance_norm=False, attention_branch=1): ### apply dropout
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
            
        ### Add layer normalization (normalize the slide representation)
        self.norm = nn.LayerNorm(feat_size)
        if model_size_wsi == 'small' : 
            self.path_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        else : 
            self.path_rho = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Classifier
        self.classifier = nn.Linear(feat_size, n_classes)
        
        self.binary_classifier = nn.Linear(feat_size, 2)  # 2 outputs
    
    def forward(self, x):
        # x: (B, N, d). Typically B=1 (MIL)
        B, N, d = x.size(0), x.size(1), x.size(2)
        assert B == 1, "Current implementation assumes batch_size=1 for MIL."

        x_reshaped = x.view(-1, x.size(-1))  # (N, d)
        # Normalization: LayerNorm recommended over InstanceNorm1d
        if self.instance_norm:
            # If self.instancenorm is InstanceNorm1d, disable it and replace as below:
            # x_reshaped = self.instancenorm(x_reshaped)
            pass

        h_tokens = self.wsi_net(x_reshaped)  # (N, D=feat_size)
        if self.instance_norm:
            pass  # same as above

        # Attention
        # A_path: (N, K), h_tokens: (N, D)
        A_path, _ = self.path_attention_head(h_tokens)
        K = A_path.size(1)  # attention_branch

        # instance-softmax over each branch (= class)
        A_path = F.softmax(A_path, dim=0)  # (N, K)
        # per-class pooling: (K, D)
        H = torch.matmul(A_path.transpose(1, 0), h_tokens)  # (K, D)

        if self.layer_norm:
            # LayerNorm is applied over (D,), so apply it per row
            H = torch.stack([self.norm(H[k]) for k in range(K)], dim=0)  # (K, D)

        # Apply path_rho: rho is assumed to map (D -> D)
        H = self.path_rho(H)  # (K, D)

        attention_scores = {'path': A_path}  # (N, K)

        if self.mode == 'binary':
            W = self.binary_classifier.weight  # (2, D)
            b = self.binary_classifier.bias    # (2,)
            # logits_k = <W_k, H_k> + b_k
            logits = (H * W).sum(dim=1) + b    # (K,)  (K must be 2)
            logits = logits.unsqueeze(0)       # (1, 2)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)  # (1,)
            return logits, probs, pred, attention_scores

        else:
            assert K == self.n_classes, "Set attention_branch == n_classes for class-specific pooling."
            W = self.classifier.weight  # (C, D)
            b = self.classifier.bias    # (C,)
            logits = (H * W).sum(dim=1) + b  # (C,)
            logits = logits.unsqueeze(0)     # (1, C)

            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, attention_scores