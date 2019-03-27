from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class CremiLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(CremiLoss, self).__init__(size_average, reduce)

    def soft_cremi_score(self, input, target, distance):
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            dflat = distance[index].view(-1)

            adgt = (iflat * (1-tflat) * dflat).sum() / (iflat * (1-tflat)).sum()

            loss += adgt

        return loss / float(input.size()[0])  

    def forward(self, input, target, distance):
        #_assert_no_grad(target)
        return self.soft_cremi_score(input, target, distance)      

class DiceLoss(nn.Module):

    def __init__(self, size_average=True, reduce=True, smooth=100.0):
        super(DiceLoss, self).__init__(size_average, reduce)
        self.smooth = smooth
        self.reduce = reduce

    def dice_loss(self, input, target):
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / 
                    ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def dice_loss_batch(self, input, target):

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        loss = 1 - ((2. * intersection + self.smooth) / 
               ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))
        return loss

    def forward(self, input, target):
        #_assert_no_grad(target)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        if self.reduce:
            loss = self.dice_loss(input, target)
        else:    
            loss = self.dice_loss_batch(input, target)
        return loss

class WeightedMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, input, target, weight):
        s1 = torch.prod(torch.tensor(input.size()[2:]).float())
        s2 = input.size()[0]
        norm_term = (s1 * s2).cuda()
        return torch.sum(weight * (input - target) ** 2) / norm_term

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        return self.weighted_mse_loss(input, target, weight)  


# define a customized loss function for future development
class WeightedBCELoss(nn.Module):

    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight, self.size_average,
                                      self.reduce)

# Weighted binary cross entropy + Dice loss
class BCLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True, smooth=10.0):
        super(BCLoss, self).__init__(size_average, reduce)
        self.smooth = smooth

    def dice_loss(self, input, target):
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        """
        Weighted binary classification loss + Dice coefficient loss
        """
        loss1 = F.binary_cross_entropy(input, target, weight, self.size_average,
                                       self.reduce)
        loss2 = self.dice_loss(input, target)
        return loss1, loss2   

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True, gamma=2):
        super().__init__()
        self.gamma = gamma

    def focal_loss(self, input, target, weight):
        eps = 1e-7
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            wflat = weight[index].view(-1)

            iflat = iflat.clamp(eps, 1.0 - eps)
            fc_loss_pos = -1 * tflat * torch.log(iflat) * ((1 - iflat) ** self.gamma)
            fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat ** self.gamma)
            fc_loss = fc_loss_pos + fc_loss_neg
            fc_loss = fc_loss * wflat # weighted focal loss

            loss += fc_loss.mean()
        
        return loss / float(input.size()[0])  

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        """
        Weighted Focal Loss
        """
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        loss = self.focal_loss(input, target, weight)
        return loss   


# Focal Loss + Dice Loss
class BCLoss_focal(nn.Module):
    def __init__(self, size_average=True, reduce=True, smooth=10.0, gamma=2):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma

    def dice_loss(self, input, target):
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def focal_loss(self, input, target, weight):
        eps = 1e-7
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            wflat = weight[index].view(-1)

            iflat = iflat.clamp(eps, 1.0 - eps)
            fc_loss_pos = -1 * tflat * torch.log(iflat) * (1 - iflat) ** self.gamma
            fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat) ** self.gamma
            fc_loss = fc_loss_pos + fc_loss_neg
            fc_loss = fc_loss * wflat # weighted focal loss

            loss += fc_loss.mean()
        
        return loss / float(input.size()[0])  

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        """
        Weighted binary classification loss + Dice coefficient loss
        """
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        loss1 = self.focal_loss(input, target, weight)
        loss2 = self.dice_loss(input, target)
        return loss1, loss2

# Focal Loss
class FocalLossMul(nn.Module):
    def __init__(self, size_average=True, reduce=True, gamma=2):
        super().__init__(size_average, reduce)
        self.gamma = gamma

    def focal_loss(self, input, target, weight):
        eps = 1e-6
        loss = 0.

        for index in range(input.size()[0]):
            sample_loss = 0.
            for channel in range(input.size()[1]):
                iflat = input[index, channel].view(-1)
                tflat = target[index, channel].view(-1)
                wflat = weight[index].view(-1) # use the same weight matrix for all channels 

                iflat = iflat.clamp(eps, 1.0 - eps)
                fc_loss_pos = -1 * tflat * torch.log(iflat) * ((1 - iflat) ** self.gamma)
                fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat ** self.gamma)
                fc_loss = fc_loss_pos + fc_loss_neg
                fc_loss = fc_loss * wflat # weighted focal loss

                sample_loss += fc_loss.mean()

            loss += sample_loss
        
        return loss / float(input.size()[0])  

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        """
        Weighted Focal Loss
        """
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        loss = self.focal_loss(input, target, weight)
        return loss         