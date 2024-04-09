###########################################################
#
# model.py - Vision Transformer model architecture definition
#
# Actually a CNN + ViT hybrid
#
# Luke Sheneman
# sheneman@uidaho.edu
# April 2024
#
# Research Computing and Data Services (RCDS)
# Insitite for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
##########################################################

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import time
import math
import os, sys


#
# Some basic configurations for scaling our transformer
#
NUM_LAYERS	= 32    # number of sequential transformer blocks
NUM_HEADS	= 8     # number of self-attention heads per transformer block
EMBED_DIM	= 1024  # number of embedding dimensions
PATCH_SIZE	= 8     # width and height of patches
IMG_SIZE	= 512   # width and height of original image
LATENT_DIM	= 64    # the size after CNN maxpooling


#
# ConvEncoder()
#
# A module defining the convolutional encoder.  We apply this to our
# input images to extract a 64x64x256 latent feature map
#
# Our vision transformer works off the feature map, not pixel space
#
class ConvEncoder(nn.Module):
	def __init__(self, activation=F.leaky_relu):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(64)   # really helps stabilize gradients during training!
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.activation = activation    # leaky ReLU
		self.dropout = nn.Dropout(0.2)  # help prevent model overfitting

	def forward(self, x):

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.activation(x)
		x = self.pool(x)  # 256x256x64

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.activation(x)
		x = self.pool(x)  # 128x128x128

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.activation(x)
		x = self.pool(x)  # 64x64x256

		x = self.dropout(x)

		return x





#
# PatchEmbedding()
#
# A module for defining our continuous patch embedding layer using a simple Conv2d projection
#
class PatchEmbedding(nn.Module):
	def __init__(self, latent_dim=LATENT_DIM, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM):
		super().__init__()
		self.img_size = latent_dim   # latent_dim=64 is the reduced spatial size of our image after ConvEncoder().  
		self.patch_size = patch_size # e.g 8 = 8x8 patch
		self.num_patches = (latent_dim // patch_size) ** 2   # number of 8x8 patches in a 64x64 feature map = 64
		self.proj = nn.Conv2d(256, embed_dim, kernel_size=patch_size, stride=patch_size)  # project patch to embedding 
		self.dropout = nn.Dropout(0.1)  # mitigate overfitting

	def forward(self, x):
		x = self.proj(x)       # (batch_size, embed_dim, h, w)
		x = x.flatten(2)       # (batch_size, embed_dim, num_patches)
		x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
		x = self.dropout(x)

		return x





#
# Attention()
#
# Manually implement self-attention mechanism!
#
class Attention(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super().__init__()
		self.embed_dim = embed_dim                      # dimensionality of our embedding vector
		self.num_heads = num_heads			# num heads per layer
		self.head_dim = embed_dim // num_heads          # how much of the embedding each head will consider
		self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # a matrix representing the query, key, and value vectors

		self.projection = nn.Linear(embed_dim, embed_dim)  # adding an projection layer after weighted value vectors

	def forward(self, x):   # DO THE MATH!
		batch_size, num_patches, _ = x.size()
		qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
		q, k, v = qkv.permute(2, 0, 3, 1, 4)  # extract query, key, value vectors

		attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
		attn = F.softmax(attn, dim=-1)
		x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, self.embed_dim)

		x = self.projection(x)  # this is an extra capacity-adding transformation

		return x



#
# MLP()
#
# Implement a simple 3-layer neural network with an expanding hidden layer
# this is happens in the transformer block after the self-attention.
#
# Expansion layer has opportunity to extract additional nuance
#
class MLP(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.fc1 = nn.Linear(embed_dim, embed_dim*4)   # our expanding hidden layer is 4X our input layer
		self.fc2 = nn.Linear(embed_dim*4, embed_dim)   # contract back down to embed_dim
		self.act = nn.GELU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)

		return x



#
# TransformerEncoder()
#
# Define the transformer block here.  All the transformer stuff in order.
# Don't forget the residual (skip) connections defined in the forward pass
#
class TransformerEncoder(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super().__init__()
		self.attention = Attention(embed_dim, num_heads)
		self.mlp = MLP(embed_dim)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.dropout1 = nn.Dropout(0.1)
		self.dropout2 = nn.Dropout(0.1)

	def forward(self, x):
		x = x + self.dropout1(self.attention(self.norm1(x)))   # uses self attension and skip connections around it
		x = x + self.dropout2(self.mlp(self.norm2(x)))         # used normalization and MLP and skip connection too

		return x




#
# VisionTransformer()
#
# Define our overall CNN+ViT hybrid
#
class VisionTransformer(nn.Module):
	def __init__(self, img_size=512, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=8, device="cuda"):
		super().__init__()
		self.conv_encoder = ConvEncoder()   # our convolutional encoder
		self.patch_embedding = PatchEmbedding(latent_dim=64, patch_size=patch_size, embed_dim=embed_dim)
		self.num_patches = self.patch_embedding.num_patches
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # the CLS token is important 
		self.layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)])
		self.norm = nn.LayerNorm(embed_dim)
		self.classifier = nn.Linear(embed_dim, num_classes)
		self.pe_x = self._generate_positional_encodings(embed_dim, self.num_patches + 1).to(device)
		self.dropout = nn.Dropout(0.1)  # mitigate overfitting

	# we use 1D sinusoidal positional encoding
	def _generate_positional_encodings(self, embed_dim, num_positions):
		position_ids = torch.arange(num_positions).unsqueeze(-1)
		dim_t = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim // 2)
		temp_encodings = position_ids / (10000 ** (2 * dim_t.unsqueeze(0)))
		pe_x = torch.zeros((1, num_positions, embed_dim))
		pe_x[:, :, 0::2] = torch.sin(temp_encodings)
		pe_x[:, :, 1::2] = torch.cos(temp_encodings)

		return pe_x

	def forward(self, x):
		x = self.conv_encoder(x)
		x = self.patch_embedding(x)
		batch_size, _, _ = x.size()
		cls_token = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_token, x), dim=1)
		x = x + self.pe_x
		for layer in self.layers:
			x = layer(x)
		x = self.norm(x)[:, 0]  # extract the CLS token embedding
		x = self.dropout(x)
		x = self.classifier(x)  # a simple linear layer that maps CLS embedding to CAT or DOG

		return x

