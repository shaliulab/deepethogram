import os

import numpy as np
import torch
from torch import nn


class NLLLossCNN(nn.Module):
    """ A simple wrapper around Pytorch's NLL loss. Appropriate for models with a softmax activation function.
    Adds:
        optional label smoothing
        set loss to zero if label = ignore_index (when images have been added at beginning or end of a video, for
            example)
    """
    def __init__(self, alpha=0.1, weight=None, ignore_index=-1):
        super().__init__()

        self.alpha = alpha
        self.should_smooth = self.alpha != 0.0

        # self.nll = nn.NLLLoss(weight=weight, reduction='none')

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        if weight is None:
            weight = torch.from_numpy(np.array([1.0])).float()
        self.weight = weight

    def forward(self, outputs, label):
        # make sure labels are one-hot
        if outputs.shape != label.shape:
            # see if it's just a batch issue
            if (1, *label.shape) == outputs.shape:
                label = label.unsqueeze(0)

        assert outputs.shape == label.shape, 'Outputs shape must match labels! {}, {}'.format(outputs.shape, label.shape)
        # N, K, T = outputs.shape
        label = label.float()

        # figure out which index to ignore before smoothing
        mask = 1 - (label == self.ignore_index).to(torch.float).to(outputs.device)

        if self.should_smooth:
            K = label.shape[1]
            # when does label smoothing help? paper
            label = label * (1 - self.alpha) + self.alpha / K
        # take the log softmax to get log(p) from outputs
        outputs = self.log_softmax(outputs)

        if self.weight.device != label.device:
            self.weight = self.weight.to(label.device)
        # 1 if not ignore_index, else 0

        # negative log likelihood
        # -y * log(p)
        loss = -((label * mask * outputs * self.weight).sum(dim=1))
        loss = loss.mean()

        if loss < 0:
            print('negative loss')
            import pdb
            pdb.set_trace()

        return loss


class BCELossCustom(nn.Module):
    """Simple wrapper around nn.BCEWithLogitsLoss. Adds masking if label = ignore_index, and support for sequence
    inputs of shape N,K,T
    """
    def __init__(self, pos_weight=None, ignore_index=-1):
        super().__init__()
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(weight=None,
                                                      reduction='none', pos_weight=pos_weight)
        self.ignore_index = ignore_index

    def forward(self, outputs, label):
        # make sure labels are one-hot
        if outputs.shape != label.shape:
            # see if it's just a batch issue
            if (1, *label.shape) == outputs.shape:
                label = label.unsqueeze(0)
        assert outputs.shape == label.shape, 'Outputs shape must match labels! {}, {}'.format(outputs.shape,
                                                                                              label.shape)

        if outputs.ndim == 3:
            sequence = True
        else:
            sequence = False

        if sequence:
            N, K, T = outputs.shape
        else:
            N, K = outputs.shape

        label = label.float()

        if sequence:
            # change from N x K x T -> N x T x K
            outputs, label = outputs.permute(0, 2, 1).contiguous(), label.permute(0, 2, 1).contiguous()
            # change from N x T x K -> N*T x K
            outputs, label = outputs.view(-1, K), label.view(-1, K)

        # figure out which index to ignore before smoothing
        mask = 1 - (label == self.ignore_index).to(torch.float).to(outputs.device)

        bceloss = self.bcewithlogitsloss(outputs, label)
        # mask the sequences outside the range of the current movie
        # e.g. if your sequence is 30 frames long, and you start on the first frame, it contains 15 bogus frames
        # and 15 real ones
        # sum across classes
        loss = (bceloss * mask).sum(dim=1)

        if sequence:
            loss_over_time = loss.view(N, T)
            # sum across time, mean across batch
            loss = loss_over_time.sum(dim=1).mean()
        else:
            # mean across batch
            loss = loss.mean()

        if loss < 0:
            print('negative loss')
            import pdb
            pdb.set_trace()

        return loss
    
class BinaryFocalLoss(nn.Module):
    """Simple wrapper around nn.BCEWithLogitsLoss. Adds masking if label = ignore_index, and support for sequence
    inputs of shape N,K,T
    
    References:
    - https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
    - https://arxiv.org/pdf/1708.02002.pdf
    - https://amaarora.github.io/2020/06/29/FocalLoss.html
    """
    def __init__(self, pos_weight=None, ignore_index=-1, gamma:float = 0):
        super().__init__()
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(weight=None,
                                                      reduction='none', pos_weight=pos_weight)
        self.ignore_index = ignore_index
        self.gamma = gamma
        # self.alpha = alpha

    def forward(self, outputs, label):
        # make sure labels are one-hot
        if outputs.shape != label.shape:
            # see if it's just a batch issue
            if (1, *label.shape) == outputs.shape:
                label = label.unsqueeze(0)
        assert outputs.shape == label.shape, 'Outputs shape must match labels! {}, {}'.format(outputs.shape,
                                                                                              label.shape)

        if outputs.ndim == 3:
            sequence = True
        else:
            sequence = False

        if sequence:
            N, K, T = outputs.shape
        else:
            N, K = outputs.shape

        label = label.float()

        if sequence:
            # change from N x K x T -> N x T x K
            outputs, label = outputs.permute(0, 2, 1).contiguous(), label.permute(0, 2, 1).contiguous()
            # change from N x T x K -> N*T x K
            outputs, label = outputs.view(-1, K), label.view(-1, K)

        # figure out which index to ignore before smoothing
        mask = 1 - (label == self.ignore_index).to(torch.float).to(outputs.device)

        bceloss = self.bcewithlogitsloss(outputs, label)
        # mask the sequences outside the range of the current movie
        # e.g. if your sequence is 30 frames long, and you start on the first frame, it contains 15 bogus frames
        # and 15 real ones
        # sum across classes
        prob = torch.sigmoid(outputs)
        
        # focal loss
        # if y==1, weight = (1-P)**gamma
        # if y==0, weight = (P)**gamma
        # we also mask here, in case of ignore_index
        # https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/focal.html
        weight = torch.pow(1-prob, self.gamma)*label*mask + torch.pow(prob, self.gamma)*(1-label)*mask
        
        # sum over classes
        loss = (bceloss*weight).sum(dim=1)

        if sequence:
            loss_over_time = loss.view(N, T)
            # sum across time
            loss = loss_over_time.sum(dim=1)
        # mean across batch
        loss = loss.mean()

        if loss < 0:
            print('negative loss')
            import pdb
            pdb.set_trace()

        return loss
        
class ClassificationLoss(nn.Module):
    def __init__(self, data_criterion: nn.Module, regularization_criterion: nn.Module):
        super().__init__()
        self.data_criterion = data_criterion
        self.regularization_criterion = regularization_criterion
        
    def forward(self, outputs, label, model):
        data_loss = self.data_criterion(outputs, label)
        reg_loss = self.regularization_criterion(model)
        
        loss = data_loss + reg_loss
        
        loss_dict = {'data_loss': data_loss.detach(), 
                  'reg_loss': reg_loss.detach()}
        
        return loss, loss_dict