import torch
import torch.nn as nn
import os
from models import cnnlstm
from opts import parse_opts
from mean import get_mean, get_std
import cv2
from PIL import Image
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from torchvision.models import resnet50
import torchvision.transforms as transforms
import time


def extract_label(pred_array, top_n=1, ResNetModel=False):
    pred_max = torch.topk(pred_array, top_n)[1]
    # print(pred_array)
    if (ResNetModel):
        label_list = ('BikeBi', 'BikeU', 'Crosswalk', 'Road', 'Sidewalk')
    else:
        label_list = ('Sidewalk', 'BikeU', 'BikeBi', 'Crosswalk', 'Road')
    out_list = []
    out_list.append(label_list[pred_max[0]])
    return out_list


def resnet_model(opt):
    imgsize = 150

    # CUDA for PyTorch
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

    batch_size = 8

    classes = ('BikeBi', 'BikeU', 'Crosswalk', 'Road', 'Sidewalk')

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

    return model  # , testloader


opt = parse_opts()
print(opt)
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

target_transform = ClassLabel()
temporal_transform = LoopPadding(16)

if not torch.cuda.is_available():
    raise Exception("You should enable GPU in the runtime menu.")
device = torch.device("cuda:0")

# defining model
model = cnnlstm.CNNLSTM(num_classes=5)
model.to(device)
model.eval()

checkpoint = torch.load(opt.resume_path)
model.load_state_dict(checkpoint['state_dict'])

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    norm_method
])
spatial_transform = Compose([
    # crop_method,
    Scale((opt.sample_size, opt.sample_size)),
    # RandomHorizontalFlip(),
    ToTensor(opt.norm_value), norm_method
])

transform_resnet = transforms.Compose(
    [transforms.Resize(int(150)),  # Resize the short side of the image to 150 keeping aspect ratio
     transforms.CenterCrop(int(150)),  # Crop a square in the center of the image
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ResnetModel = opt.resnet_model
# Test with the previous Resnet based model
if (ResnetModel):
    print('Resnet Model Selected')
    model = resnet_model(opt)


files = []
for r, d, f in os.walk(opt.test_video_path):
    for file in f:
        if '.mp4' in file:
            files.append(r+"/"+file)

t = time.time()
NumberOfFrames = 0
for path in files:
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()

    #print("\nAnalyzing --> " + str(path))

    if opt.group_of_frames == 'True':

        while success:
            count = 0
            frames = []
            while count < 16:
                # image = cv2.resize(image, (opt.sample_size,opt.sample_size), interpolation=cv2.INTER_AREA)
                image = spatial_transform(Image.fromarray(image)).to(device)
                image = image.unsqueeze(0)  # add a batch dimension
                frames.append(image)
                if success:
                    image2 = image
                success, image = vidcap.read()
                count += 1
                while (not success) and (count < 16):
                    frames.append(image2)
                    count += 1

            clip = torch.stack(frames, 1)
            outputs = model(clip)
            '''
            for frame in outputs:
                print('Predicted group of frames (16 frames) --> ' + str(extract_label((frame))[0]))
            '''

    # To predict just 1 frame

    else:
        count = 0
        frames = []

        if not ResnetModel and success:
            image = spatial_transform(Image.fromarray(image)).to(device)
            image = image.unsqueeze(0)  # add a batch dimension
            while count < 16:
                frames.append(image)
                count += 1
            success, image = vidcap.read()

        while success:
            if (ResnetModel):
                # image = cv2.resize(image, (opt.sample_size,opt.sample_size), interpolation=cv2.INTER_AREA)
                image = transform_resnet(Image.fromarray(image)).to(device)
                image = image.unsqueeze(0)  # add a batch dimension
                outputs = model(image)
                NumberOfFrames+=1
                success, image = vidcap.read()

            else:
                while count < 16:
                    # image = cv2.resize(image, (opt.sample_size,opt.sample_size), interpolation=cv2.INTER_AREA)
                    image = spatial_transform(Image.fromarray(image)).to(device)
                    image = image.unsqueeze(0)  # add a batch dimension
                    frames.append(image)
                    success, image = vidcap.read()
                    count += 1
                    if not success:
                        image2 = frames[15]
                    while (not success) and (count < 16):
                        frames.append(image2)
                clip = torch.stack(frames, 1)
                outputs = model(clip)
                count = count - 1
                del frames[0]
            '''
            for frame in outputs:
                print('Predicted frame --> ' + str(extract_label((frame), ResNetModel=ResnetModel)[0]))
            '''

time_elapsed = time.time() - t

print(" ")
print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Total Number Of Frames: ' + str(NumberOfFrames))
print(" ")

# show info
print('FINISHED')
