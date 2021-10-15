import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
import time
from utils import AverageMeter, calculate_accuracy
from sklearn.metrics import confusion_matrix, classification_report


def val_epoch(model, data_loader, criterion, device):
    model.eval()
    classes = ('Sidewalk', 'BikeU', 'BikeBi', 'Crosswalk', 'Road')
    y_pred = []
    y_true = []

    t = time.time()
    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)

            trues=[]
            '''
            for j in range(targets.shape[0]):
                y_true.append(classes[targets[j]])
                trues.append(classes[targets[j]])
            '''
            outputs = model(data)

            predicted = []
            '''
            for j in range(targets.shape[0]):
                y_pred.append(classes[torch.topk(outputs[j],1)[1][0]])
                predicted.append(classes[torch.topk(outputs[j],1)[1][0]])
            
            print(trues)
            print(predicted)
            print('')
            '''
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    '''
    time_elapsed = time.time() - t
    print(" ")
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("")
    print("--------------------------------------------------------------------------")
    print("")
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred, labels=['Sidewalk', 'BikeU', 'BikeBi', 'Crosswalk', 'Road']))
    '''
    return losses.avg, accuracies.avg
