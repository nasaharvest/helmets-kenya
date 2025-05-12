# Importing Libraries
import os
import cv2
import torch
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import matplotlib.pyplot as plt
from PIL import Image
CLASSES = ["background", "banana", "maize", "rice", "soybean", "sugarcane", "sunflower", "tobacco", "wheat"]

# DATACLASS Definition

class Data(Dataset):
    """
    Args:
        images_dir (str): path to images folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, images_dir, classes=None, augmentation=None, preprocessing=None,):
        self.ids = os.listdir(images_dir)
        self.images_fps = sorted([os.path.join(images_dir, image_id) for image_id in self.ids if image_id.split(".")[-2].split("_")[-1]!="mask" and image_id.split(".")[-1]!="csv"])
        self.masks_fps = sorted([os.path.join(images_dir, image_id) for image_id in self.ids  if image_id.split(".")[-2].split("_")[-1]=="mask"])
        print(f"{len(self.images_fps)=} and {len(self.masks_fps)=}")
        self.img_size = (512,512)
        # convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image=Image.open(self.images_fps[i])
        image=image.resize(self.img_size)
        mask=Image.open(self.masks_fps[i])
        mask=mask.resize(self.img_size)
        mask=np.array(mask)
        image=np.array(image)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.masks_fps)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def preprocess(img,**kwargs):
    img = img.astype(float)
    img = (255 * (img - np.min(img[:])) / (np.max(img[:]) - np.min(img[:]) + 0.1)).astype(float)
    img = (img + 0.5) / 256
    gamma = -1/np.nanmean(np.log(img))
    img = img**(gamma)
    print(f"{np.min(img)=} and {np.max(img)=}")
    return img

# Model Definition
ENCODER = 'xception'
ENCODER_WEIGHTS="imagenet"
CLASSES = CLASSES
ACTIVATION = 'softmax2d'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading Data
DATA_DIR = '[ENTER DATA DIRECTORY]'
MODEL_DIR = "[ENTER MODEL DIRECTORY]" + ENCODER + "/"

if(not(os.path.isdir(MODEL_DIR))):
    os.mkdir(MODEL_DIR)
valid_dir = os.path.join(DATA_DIR, 'valid')
#test_dir = os.path.join(DATA_DIR, 'test')
preprocessing_fn = preprocess
val_dataset = Data(valid_dir, classes = CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
#test_dataset = Data(test_dir, classes = CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
best_model = torch.load(MODEL_DIR + 'best_model.pth')
loss = smp_utils.losses.DiceLoss()
# metrics = [smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[1,2,3,4,5,6,7,8]),
#            smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,2,3,4,5,6,7,8]),
#            smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,3,4,5,6,7,8]),
#            smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,4,5,6,7,8]),
#            smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,3,5,6,7,8]),
#            smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[1,2,3,4,5,6,7,8]),
#            smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,2,3,4,5,6,7,8]),
#            smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,3,4,5,6,7,8]),
#            smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,4,5,6,7,8]),
#            smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,3,5,6,7,8]),
#            ]
metrics = [smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[1,2,3,4,5,6,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,2,3,4,5,6,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,3,4,5,6,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,4,5,6,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,3,5,6,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,3,4,6,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,3,4,5,7,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,3,4,5,6,8]),
        smp.utils.metrics.IoU(threshold=0.5,ignore_channels=[0,1,2,3,4,5,6,7]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[1,2,3,4,5,6,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,2,3,4,5,6,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,3,4,5,6,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,4,5,6,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,3,5,6,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,3,4,6,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,3,4,5,7,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,3,4,5,6,8]),
        smp.utils.metrics.AveragePrecision(threshold=0.5,ignore_channels=[0,1,2,3,4,5,6,7]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[1,2,3,4,5,6,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,2,3,4,5,6,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,3,4,5,6,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,4,5,6,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,5,6,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,4,6,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,4,5,7,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,4,5,6,8]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,4,5,6,7])
        ]
'''
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[1,2,3,4,5,6]),
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[0,2,3,4,5,6]),
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[0,1,3,4,5,6]),
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[0,1,2,4,5,6]),
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[0,1,2,3,5,6]),
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[0,1,2,3,4,6]),
        smp.utils.metrics.Accuracy(threshold=0.5,ignore_channels=[0,1,2,3,4,5]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[1,2,3,4,5,6]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[0,2,3,4,5,6]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[0,1,3,4,5,6]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[0,1,2,4,5,6]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[0,1,2,3,5,6]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[0,1,2,3,4,6]),
        smp.utils.metrics.Fscore(threshold=0.5,ignore_channels=[0,1,2,3,4,5]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[1,2,3,4,5,6]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,2,3,4,5,6]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,3,4,5,6]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,4,5,6]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,5,6]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,4,6]),
        smp.utils.metrics.Recall(threshold=0.5,ignore_channels=[0,1,2,3,4,5]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[1,2,3,4,5,6]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[0,2,3,4,5,6]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[0,1,3,4,5,6]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[0,1,2,4,5,6]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[0,1,2,3,5,6]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[0,1,2,3,4,6]),
        smp.utils.metrics.Precision(threshold=0.5,ignore_channels=[0,1,2,3,4,5])]
'''
met_names = ["IOU", "Average_Precision","Recall"]
valid_epoch = smp.utils.train.ValidEpoch(
    best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
for num in range(len(valid_epoch.metrics)):
    print(valid_epoch.metrics[num])
    setattr(valid_epoch.metrics[num],"__name__", CLASSES[num%9] + "_" + met_names[num//9])

val_logs = test_epoch.run(val_loader)
print(val_logs)
print("NewData(10ep) Crop Type Segmentation")
'''
test_logs = test_epoch.run(test_loader)
print(test_logs)
'''