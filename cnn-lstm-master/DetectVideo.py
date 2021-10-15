import torch
import torch.nn as nn
import os
from models import cnnlstm
from opts import parse_opts
from mean import get_mean, get_std
import cv2
from model import generate_model
from PIL import Image
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2

import torchvision.transforms as transforms
import time
from sklearn.metrics import confusion_matrix, classification_report


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

    pretrained_model = mobilenet_v2(pretrained=True)
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
    PATH = '../../final_model_MobileNet4.pth'
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
model = generate_model(opt, device)
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

PrevTag = 'None'
vid_number = 0
for path in files:
    vid_number+=1
    scale=1
    vidcap = cv2.VideoCapture(path)
    success, image_orig = vidcap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) / scale), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale))
    outvid = cv2.VideoWriter('./data/DetectVideos/output' + str(vid_number) + '.mp4', fourcc, fps, size)
    #print("\nAnalyzing --> " + str(path))


    # To predict just 1 frame
    count = 0
    frames = []

    if not ResnetModel and success:
        image = spatial_transform(Image.fromarray(image_orig)).to(device)
        image = image.unsqueeze(0)  # add a batch dimension
        while count < 16:
            frames.append(image)
            count += 1
        success, image_orig = vidcap.read()

    while success:
        if (ResnetModel):
            # image = cv2.resize(image, (opt.sample_size,opt.sample_size), interpolation=cv2.INTER_AREA)
            image = transform_resnet(Image.fromarray(image_orig)).to(device)
            image = image.unsqueeze(0)  # add a batch dimension
            outputs = model(image)
            success, image_orig = vidcap.read()

        else:
            while count < 16:
                # image = cv2.resize(image, (opt.sample_size,opt.sample_size), interpolation=cv2.INTER_AREA)
                image = spatial_transform(Image.fromarray(image_orig)).to(device)
                image = image.unsqueeze(0)  # add a batch dimension
                frames.append(image)
                success, image_orig = vidcap.read()
                count += 1
                if not success:
                    image2 = frames[15]
                while (not success) and (count < 16):
                    frames.append(image2)
            clip = torch.stack(frames, 1)

            outputs = model(clip)
            count = count - 1

            del frames[0]

        cv2.rectangle(image_orig, (35, 15), (195, 65), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(image_orig, (40, 20), (190, 60), (255, 255, 255), cv2.FILLED)
        cv2.putText(image_orig, str(extract_label((outputs[0]), ResNetModel=ResnetModel)[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_8)

        '''
        if str(extract_label((outputs[0]), ResNetModel=ResnetModel)[0]) != PrevTag:
            PrevTag = str(extract_label((outputs[0]), ResNetModel=ResnetModel)[0])
            print('Predicted frame (change) --> ' + str(extract_label((outputs[0]), ResNetModel=ResnetModel)[0]) + ' at time: ' + str(vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000))
        '''
        if success:
            outvid.write(cv2.resize(image_orig, size, interpolation=cv2.INTER_AREA))

        '''
        for frame in outputs:
            print('Predicted frame --> ' + str(extract_label((frame), ResNetModel=ResnetModel)[0]))
        '''

vidcap.release()
outvid.release()

cv2.destroyAllWindows()

time_elapsed = time.time() - t
print(" ")
print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))