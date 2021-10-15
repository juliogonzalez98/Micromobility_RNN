import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
import random
import numpy as np

from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from torchvision.models import resnet50
from target_transforms import Compose as TargetCompose
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def resume_model(opt, model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(opt.resume_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Model Restored from Epoch {}".format(checkpoint['epoch']))
	start_epoch = checkpoint['epoch'] + 1
	return start_epoch


def get_loaders(opt):
	""" Make dataloaders for train and validation sets
	"""
	# train loader
	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	if opt.no_mean_norm and not opt.std_norm:
		norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not opt.std_norm:
		norm_method = Normalize(opt.mean, [1, 1, 1])
	else:
		norm_method = Normalize(opt.mean, opt.std)
	spatial_transform = Compose([
		# crop_method,
		Scale((opt.sample_size, opt.sample_size)),
		# RandomHorizontalFlip(),
		ToTensor(opt.norm_value), norm_method
	])
	temporal_transform = TemporalRandomCrop(16)
	target_transform = ClassLabel()
	training_data = get_training_set(opt, spatial_transform,
									 temporal_transform, target_transform)
	train_loader = torch.utils.data.DataLoader(
		training_data,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		pin_memory=True)

	# validation loader
	spatial_transform = Compose([
		Scale((opt.sample_size, opt.sample_size)),
		# CenterCrop(opt.sample_size),
		ToTensor(opt.norm_value), norm_method
	])
	target_transform = ClassLabel()
	temporal_transform = LoopPadding(16)
	validation_data = get_validation_set(
		opt, spatial_transform, temporal_transform, target_transform)
	val_loader = torch.utils.data.DataLoader(
		validation_data,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=True)
	return train_loader, val_loader


def Resnet_Model(opt):
	imgsize = 150

	transform = transforms.Compose(
		[transforms.Resize(int(imgsize)),  # Resize the short side of the image to 150 keeping aspect ratio
		 transforms.CenterCrop(int(imgsize)),  # Crop a square in the center of the image
		 transforms.ToTensor(),
		 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	# CUDA for PyTorch
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

	batch_size = 8

	classes = ('BikeBi', 'BikeU', 'Crosswalk', 'Road', 'Sidewalk')
	classes_bi = ('Bike', 'Other')

	pretrained_model = resnet50(pretrained=True)
	pretrained_model.eval()
	pretrained_model.to(device)

	ct = 0
	for child in pretrained_model.children():
		ct += 1
		if ct <= 5:
			for params in child.parameters():
				params.requires_grad = False

	# model
	model = nn.Sequential(
		pretrained_model,
		nn.Flatten(),
		nn.Dropout(p=0.5),
		# nn.Linear(int((imgsize/37.5))*int((imgsize/37.5))* 512, 256),
		nn.Linear(1000, 256),
		nn.ReLU(),
		nn.Dropout(p=0.5),
		nn.Linear(256, 5),
		nn.LogSoftmax(dim=1)
	)

	model.to(device)

	PATH = '../../final_model_ResNet3.pth'

	model.load_state_dict(torch.load(PATH))
	model.eval()
	model.to(device)

	validation_dir = '../../ModelTrain/processed_datalab/test/'
	testset = ImageFolder(root=validation_dir, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size), shuffle=True, num_workers=1)

	return model#, testloader

def main_worker():

	ResnetModel = False

	opt = parse_opts()
	print(opt)

	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	# defining model
	model =  generate_model(opt, device)

	# get data loaders
	_, val_loader = get_loaders(opt)


	# scheduler = lr_scheduler.ReduceLROnPlateau(
	# 	optimizer, 'min', patience=opt.lr_patience)
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-5)

	# resume model
	if opt.resume_path:
		start_epoch = resume_model(opt, model, optimizer)
	start_epoch = 1

	#Test with the previous Resnet based model
	if (ResnetModel):
		model = Resnet_Model(opt)


	# start testing
	val_epoch(model, val_loader, criterion, device)


if __name__ == "__main__":
	main_worker()