###########################################################
#
# ViT - Vision Transformer 
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
import os, sys, shutil

import model

print(model.IMG_SIZE)

SEED            = int(time.time())
SEED            = 42
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 10000
BATCH_SIZE	= 24

DATA_DIR  = "data2"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

MODEL_DIR = "models"
TEMP_PATH = os.path.join(MODEL_DIR, "tmp.pt")
BEST_PATH = os.path.join(MODEL_DIR, "best.pt")

TRAIN_MORE = True


os.makedirs(MODEL_DIR, exist_ok=True)



# Define the transformations to be applied to the images
transform = transforms.Compose([

	transforms.Resize((model.IMG_SIZE, model.IMG_SIZE)),  # Resize the image 
	#transforms.CenterCrop(model.IMG_SIZE),

	# Data Augmentations
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(degrees=10),
	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
	transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
	transforms.RandomPerspective(distortion_scale=0.3, p=0.5),

	transforms.ToTensor(),  # Convert the image to a tensor
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
])


val_transform = transforms.Compose([
	transforms.Resize((model.IMG_SIZE, model.IMG_SIZE)),  # Resize the image 
	#transforms.CenterCrop(model.IMG_SIZE),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)



#
# Set seed and device (GPU or CPU)
#
torch.manual_seed(SEED)
if torch.cuda.is_available():
	device = "cuda"
else:
	device = "cpu"
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("DEVICE=", device)
print("\n")


# instatiate the ViT model
vit = model.VisionTransformer(num_classes=len(train_dataset.classes), device=device)

if(TRAIN_MORE == True):
	print("Loading previous weights.  Continue training using weights: ", BEST_PATH)
	state_dict = torch.load(BEST_PATH)
	vit.load_state_dict(state_dict)
	vit.train()

vit.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit.parameters(), lr=LEARNING_RATE, weight_decay=1e-2) 


best_score = 100.0
for epoch in range(NUM_EPOCHS):
	train_loss = 0.0
	val_loss   = 0.0
	train_acc  = 0.0
	val_acc    = 0.0

	# Training
	vit.train()
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()

		outputs = vit(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item() * images.size(0)
		train_acc += (outputs.max(1)[1] == labels).sum().item()

	# Validation
	vit.eval()
	with torch.no_grad():
		for images, labels in val_loader:
			images, labels = images.to(device), labels.to(device)

			outputs = vit(images)
			loss = criterion(outputs, labels)

			val_loss += loss.item() * images.size(0)
			val_acc += (outputs.max(1)[1] == labels).sum().item()

	train_loss /= len(train_dataset)
	train_acc  /= len(train_dataset)
	val_loss   /= len(val_dataset)
	val_acc    /= len(val_dataset)


	# Save the model weights every epoch
	#torch.save(vit.state_dict(), "last.pt")

	#score = 0.5 * train_acc + 0.5 * val_acc
	score = val_loss
	print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Score: {score:.4f}')

	# Save the best model weights
	if score < best_score:
		print("  ** BEST MODEL ACHIEVED **")
		best_score = score
		torch.save(vit.state_dict(), TEMP_PATH)
		shutil.move(TEMP_PATH, BEST_PATH)
		


    
