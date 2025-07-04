import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from slowfast_modules import head_helper, resnet_helper, stem_helper  # noqa
from slowfast_modules import weight_init_helper as init_helper
from slowfast_modules.checkpoint import load_checkpoint
from fvcore.common.config import CfgNode
import yaml
import pdb

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        beta_inv,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]
    

class FuseBiCat(nn.Module):
    """
    Fuses the information from each pathway to other pathway through concatenation. 
    Given the tensors from Slow pathway and Fast pathway, fuse information in bidirectional, 
    then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        beta_inv,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseBiCat, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)
        self.conv_s2f = nn.Conv3d(
            dim_in * beta_inv,
            dim_in,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.bn2 = norm_module(
            num_features=dim_in,
            eps=eps,
            momentum=bn_mmt
        )
        self.alpha = alpha

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        
        fuse1 = self.conv_f2s(x_f)
        fuse1 = self.bn(fuse1)
        fuse1 = self.relu(fuse1)
        x_s_fuse = torch.cat([x_s, fuse1], 1)
        fuse2 = self.conv_s2f(x_s)
        fuse2 = self.bn2(fuse2)
        fuse2 = self.relu(fuse2)
        fuse2 = nn.functional.interpolate(fuse2, x_f.shape[2:])
        x_f_fuse = torch.cat([x_f, fuse2], 1)
        return [x_s_fuse, x_f_fuse]


class FuseBiAdd(nn.Module):
    """
    Fuses the information from each pathway to other pathway through addition. 
    Given the tensors from Slow pathway and Fast pathway, fuse information in bidirectional, 
    then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        beta_inv,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseBiAdd, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * beta_inv,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * beta_inv,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)
        self.conv_s2f = nn.Conv3d(
            dim_in * beta_inv,
            dim_in,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.bn2 = norm_module(
            num_features=dim_in,
            eps=eps,
            momentum=bn_mmt
        )
        self.alpha = alpha
        self.weight_s = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.weight_f = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        
        fuse1 = self.conv_f2s(x_f)
        fuse1 = self.bn(fuse1)
        fuse1 = self.relu(fuse1)
        x_s_fuse = x_s + self.weight_s * fuse1
        fuse2 = self.conv_s2f(x_s)
        fuse2 = self.bn2(fuse2)
        fuse2 = self.relu(fuse2)
        fuse2 = nn.functional.interpolate(fuse2, x_f.shape[2:])
        x_f_fuse = x_f + self.weight_f * fuse2
        return [x_s_fuse, x_f_fuse]
    

class FuseBiAdd2(nn.Module):
    """
    Fuses the information from each pathway to other pathway through addition. 
    Given the tensors from Slow pathway and Fast pathway, fuse information in bidirectional, 
    then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        beta_inv,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseBiAdd2, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * beta_inv,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * beta_inv,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)
        self.conv_s2f = nn.Conv3d(
            dim_in * beta_inv,
            dim_in,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.bn2 = norm_module(
            num_features=dim_in,
            eps=eps,
            momentum=bn_mmt
        )
        self.alpha = alpha
        self.weight_s = nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight_f = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        
        fuse1 = self.conv_f2s(x_f)
        fuse1 = self.bn(fuse1)
        fuse1 = self.relu(fuse1)
        x_s_fuse = self.weight_s * x_s + (1 - self.weight_s) * fuse1
        fuse2 = self.conv_s2f(x_s)
        fuse2 = self.bn2(fuse2)
        fuse2 = self.relu(fuse2)
        fuse2 = nn.functional.interpolate(fuse2, x_f.shape[2:])
        x_f_fuse = self.weight_f * x_f + (1 - self.weight_f) * fuse2
        return [x_s_fuse, x_f_fuse]