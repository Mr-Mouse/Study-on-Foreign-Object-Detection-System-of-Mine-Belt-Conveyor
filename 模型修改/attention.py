"""
File: attention.py
Author: CUMT-Muzihao
Date: 2024/04/25
Description:抗遮挡改进模块
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("EMA", "SA", "SAI")


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SA(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super(SA, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = in_channels
        self.num_heads = num_heads

        # 线性变换层，用于将输入映射到查询、键、值空间
        self.qkv = nn.Linear(self.in_channels, self.embed_dim * 3)
        self.attn_dropout = nn.Dropout(0.1)
        # 用于将注意力加权后的值再次映射到原始空间
        self.proj = nn.Linear(self.embed_dim, self.in_channels)

    def forward(self, x):
        # 将二维特征图展平为一维序列
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)

        # 将输入张量映射为查询、键、值
        qkv = self.qkv(x_flat)
        q, k, v = torch.split(qkv, self.embed_dim, dim=-1)

        # 将查询、键、值张量重塑为多头
        q = q.view(b, h * w, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 对值进行加权并重塑为原始形状
        x_att = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, h * w, self.embed_dim)
        # 映射回原始空间
        x_att = self.proj(x_att.transpose(1, 2).contiguous().view(b * h * w, self.embed_dim))
        x_att = x_att.view(b, self.in_channels, h, w)

        return x_att


class SAI(nn.Module):
    def __init__(self, in_channels, num_heads=1, poolingsize=2):
        super(SAI, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = in_channels
        self.num_heads = num_heads

        # 线性变换层，用于将输入映射到查询、键、值空间
        self.qkv = nn.Linear(self.in_channels, self.embed_dim * 3)
        self.attn_dropout = nn.Dropout(0.1)
        # 用于将注意力加权后的值再次映射到原始空间
        self.proj = nn.Linear(self.embed_dim, self.in_channels)

        self.avgpool = nn.AvgPool2d(kernel_size=poolingsize, stride=poolingsize)
        self.upsample = nn.Upsample(scale_factor=poolingsize, mode='nearest')
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 将二维特征图展平为一维序列
        x_pool = self.avgpool(x)
        b, c, h, w = x_pool.size()
        x_flat = x_pool.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)

        # 将输入张量映射为查询、键、值
        qkv = self.qkv(x_flat)
        q, k, v = torch.split(qkv, self.embed_dim, dim=-1)

        # 将查询、键、值张量重塑为多头
        q = q.view(b, h * w, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 对值进行加权并重塑为原始形状
        x_att = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, h * w, self.embed_dim)
        # 映射回原始空间
        x_att = self.proj(x_att.transpose(1, 2).contiguous().view(b * h * w, self.embed_dim))
        x_att = x_att.view(b, self.in_channels, h, w)
        x_att = F.interpolate(self.upsample(x_att), size=x.size()[2:], mode='bilinear', align_corners=False) + x

        return self.act(self.bn(x_att))
