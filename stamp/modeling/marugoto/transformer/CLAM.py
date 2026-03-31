"""
Implementation of Clustering-constrained Attention Multiple Instance Learning (CLAM)
Based on: https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        if dropout:
            self.module.append(nn.Dropout(0.25))
            
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)
        
    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
            
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()]
            
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
        
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_SB(nn.Module):
    """Single-branch CLAM.

    Args:
        gate: whether to use gated attention network
        num_classes: number of classes
        input_dim: input feature dimension
        size_arg: config for network size ('small' or 'big')
        dropout: dropout probability
        k_sample: number of positive/neg patches to sample for instance-level training
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
    """

    def __init__(self,
        gate: bool = True,
        num_classes: int = 2,
        input_dim: int = 1024,
        size_arg: str = "small",
        dropout: float = 0.,
        k_sample: int = 8,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = False,
    ):
        super().__init__()
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], num_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = num_classes
        self.subtyping = subtyping
        
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
        
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
        
    # Instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
            
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
        
    # Instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
            
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        p_targets = self.create_negative_targets(self.k_sample, device)
        
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
        
    def forward(self, x, lens=None, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        Forward pass for CLAM_SB (Single Branch)
        
        Args:
            x: Input features, can be 3D [batch, instances, features] or 
               4D [batch, instances, patches, features]
            lens: Length of each bag (optional)
            label: Labels for instance-level training (optional)
            instance_eval: Whether to evaluate instance-level classifiers
            return_features: Whether to return features
            attention_only: Whether to return only attention weights
        """
        # Process input if it has 4 dimensions (class and patch tokens)
        if x.dim() == 4:
            b, n, num_patches, d = x.shape
            
            class_tokens = x[:, :, 0]  # (batch, seq_len, feat_dim)
            # Skip register tokens (indices 1-4) and average remaining patch tokens
            patch_tokens = x[:, :, 5:]
            patch_tokens_mean = patch_tokens.mean(dim=-2)

            x = torch.cat([class_tokens, patch_tokens_mean], dim=-1)
        elif x.dim() == 3:
            b, n, d = x.shape
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")

        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.n_classes).float().to(x.device)

        all_instance_loss = 0
        all_instance_preds = []
        all_instance_targets = []

        for i in range(batch_size):
            bag_feats = x[i]  # [n, d]

            A, h = self.attention_net(bag_feats)  # A: [n, 1], h: [n, d]
            A = F.softmax(A, dim=0)

            if attention_only:
                continue

            if instance_eval and self.instance_loss_fn is not None and label is not None:
                bag_label = label[i] if label.dim() == 1 else label[i, 0]

                if self.n_classes == 2:
                    if bag_label.item() == 1:
                        instance_loss, preds, targets = self.inst_eval(A, h, self.instance_classifiers[0])
                        all_instance_preds.extend(preds.cpu().numpy())
                        all_instance_targets.extend(targets.cpu().numpy())
                    else:
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, self.instance_classifiers[0])
                            all_instance_preds.extend(preds.cpu().numpy())
                            all_instance_targets.extend(targets.cpu().numpy())
                        else:
                            instance_loss = 0
                    all_instance_loss += instance_loss
                else:
                    inst_labels = F.one_hot(bag_label.long(), num_classes=self.n_classes).squeeze()

                    bag_instance_loss = 0
                    for c in range(self.n_classes):
                        inst_label = inst_labels[c].item()
                        classifier = self.instance_classifiers[c]

                        if inst_label == 1:
                            instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                            all_instance_preds.extend(preds.cpu().numpy())
                            all_instance_targets.extend(targets.cpu().numpy())
                        else:
                            if self.subtyping:
                                instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                                all_instance_preds.extend(preds.cpu().numpy())
                                all_instance_targets.extend(targets.cpu().numpy())
                            else:
                                continue

                        bag_instance_loss += instance_loss

                    if self.subtyping:
                        bag_instance_loss /= self.n_classes

                    all_instance_loss += bag_instance_loss

            M = torch.mm(A.transpose(0, 1), h)  # [1, d]
            logits[i] = self.classifiers(M).squeeze()

        if attention_only:
            return A

        if instance_eval and self.instance_loss_fn is not None and label is not None:
            all_instance_loss /= batch_size
            results_dict = {
                'instance_loss': all_instance_loss,
                'inst_labels': np.array(all_instance_targets),
                'inst_preds': np.array(all_instance_preds)
            }
        else:
            results_dict = {}

        self.logits = logits
        self.results_dict = results_dict

        return logits


class CLAM_MB(CLAM_SB):
    """Multi-branch CLAM — one attention branch per class."""

    def __init__(self,
        gate: bool = True,
        num_classes: int = 2,
        input_dim: int = 1024,
        size_arg: str = "small",
        dropout: float = 0.,
        k_sample: int = 8,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = True
    ):
        nn.Module.__init__(self)
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = num_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(num_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = num_classes
        self.subtyping = subtyping

    def forward(self, x, lens, label=None, instance_eval=False, return_features=False, attention_only=False):
        """Forward pass for CLAM_MB (Multi Branch)."""

        if x.dim() == 4:
            b, n, num_patches, d = x.shape

            class_tokens = x[:, :, 0]
            # Skip register tokens (indices 1-4) and average remaining patch tokens
            patch_tokens = x[:, :, 5:]
            
            # Compute mean over patch tokens
            patch_tokens_mean = patch_tokens.mean(dim=-2)
            
            # Concatenate class tokens and mean of patch tokens
            x = torch.cat([class_tokens, patch_tokens_mean], dim=-1)
        elif x.dim() == 3:
            b, n, d = x.shape
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")
            
        # Process each bag in the batch separately
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.n_classes).float().to(x.device)
        
        # For storing instance-level results
        all_instance_loss = 0
        all_instance_preds = []
        all_instance_targets = []
        
        for i in range(batch_size):
            # Get features for this bag
            bag_feats = x[i]  # Shape: [n, d]
            
            # Apply attention network
            A, h = self.attention_net(bag_feats)  # A: [n, n_classes], h: [n, d]
            A = F.softmax(A, dim=0)  # softmax over n
            
            if attention_only:
                continue
            
            # Handle instance-level evaluation if requested
            if instance_eval and self.instance_loss_fn is not None and label is not None:
                # Get the label for this bag
                bag_label = label[i] if label.dim() == 1 else label[i, 0]
                
                # Convert to one-hot encoding
                inst_labels = F.one_hot(bag_label.long(), num_classes=self.n_classes).squeeze()
                
                # Evaluate instances for each class
                bag_instance_loss = 0
                for c in range(self.n_classes):
                    inst_label = inst_labels[c].item()
                    classifier = self.instance_classifiers[c]
                    
                    if inst_label == 1:  # in-the-class
                        instance_loss, preds, targets = self.inst_eval(A[:, c], h, classifier)
                        all_instance_preds.extend(preds.cpu().numpy())
                        all_instance_targets.extend(targets.cpu().numpy())
                    else:  # out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A[:, c], h, classifier)
                            all_instance_preds.extend(preds.cpu().numpy())
                            all_instance_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                            
                    bag_instance_loss += instance_loss
                    
                if self.subtyping:
                    bag_instance_loss /= self.n_classes
                    
                all_instance_loss += bag_instance_loss
            
            # For each class
            for c in range(self.n_classes):
                # Get attention weights for this class
                A_c = A[:, c]  # [n]
                
                # Compute weighted average of instance features
                M_c = torch.mm(A_c.unsqueeze(0), h)  # [1, d]
                
                # Get logit for this class
                logits[i, c] = self.classifiers[c](M_c).squeeze()
        
        if attention_only:
            return A
        
        # Compute final instance-level results
        if instance_eval and self.instance_loss_fn is not None and label is not None:
            # Average instance loss across all bags
            all_instance_loss /= batch_size
            
            # Create results dictionary
            results_dict = {
                'instance_loss': all_instance_loss, 
                'inst_labels': np.array(all_instance_targets),
                'inst_preds': np.array(all_instance_preds)
            }
        else:
            results_dict = {}
        
        if return_features:
            # We don't have a single M to return, so we'll skip this for now
            pass
        
        # Store additional outputs for later access
        self.logits = logits
        self.results_dict = results_dict
        
        return logits