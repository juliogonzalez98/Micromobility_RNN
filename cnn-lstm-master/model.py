import torch
from torch import nn

from models import cnnlstm, cnnlstm_attention, cnnlstm_attention3

def generate_model(opt, device):
	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif opt.model == 'cnnlstm_attention3':
		print("Using Attention Model")
		model = cnnlstm_attention3.CNNLSTM_ATTENTION(num_classes=opt.n_classes)
	elif opt.model == 'cnnlstm_attention':
		print("Using Attention Model")
		model = cnnlstm_attention.CNNLSTM_ATTENTION(num_classes=opt.n_classes)
	return model.to(device)