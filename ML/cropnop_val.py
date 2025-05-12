import cv2
import numpy as np
import pandas as pd
import torch
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import os
from torchmetrics import ROC
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda import is_available
import matplotlib.pyplot as plt




device = ("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
  def __init__(self, img_dir, transform=None, target_transform=None):
    self.img_dir = img_dir
    self.class_dirs = sorted(os.listdir(img_dir))
    self.class_map = {class_name:num for num,class_name in enumerate(self.class_dirs)}
    self.transform = transform
    self.target_transform = target_transform
    self.VALID_EXT = ["png","jpg","jpeg"]
    self.img_labels = [[self.img_dir + "/" + class_dir + "/" + item, class_dir]
                       for class_dir in self.class_dirs
                       for item in os.listdir(self.img_dir + "/" + class_dir) if item.split(".")[-1].lower() in self.VALID_EXT]
    self.img_order = [i for i in range(len(self.img_labels))]
    random.shuffle(self.img_order)

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path, label = self.img_labels[self.img_order[idx]]
    image = plt.imread(img_path)
    image = cv2.resize(image,(300,300))/255
    label = self.class_map[label]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return self.to_tensor(image), float(label)

  def to_tensor(self, x):
    return x.transpose(2,0,1).astype('float32')

CLASSES = ['crop','not_Crop']
DATA_DIR ='[Enter data directory]'
model = torch.load("[Enter model directory]")
val_dir = os.path.join(DATA_DIR,'test')
val_dataset = CustomDataset(val_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    list1 = []
    list2 = []
    list3 = []
    loop = tqdm(val_loader,total = len(val_loader), leave = True)
    for images,labels in loop:
        images = images.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(images)
        #Sloss=criterion(outputs.squeeze(),labels.type(torch.cuda.LongTensor))
        for i in range(0,len(outputs)):
            if outputs[i] <= 0:
                list1.append(0)
                list3.append(outputs[i])
                list2.append(labels[i].cpu())
            else:
                list1.append(1)
                list3.append(outputs[i])
                list2.append(labels[i].cpu())
    
print(accuracy_score(np.array(list1),np.array(list2)))
print(classification_report(np.array(list1),np.array(list2))) 
print(confusion_matrix(np.array(list1),np.array(list2)))


roc=ROC(task="binary")
fpr,tpr,thresholds=roc(torch.tensor(list3),torch.tensor(list2))

plt.plot(np.array(fpr),np.array(tpr), linestyle="-",label="ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("[Enter Image location]")



best_thresh = thresholds[0]
min_diff = abs((1-fpr[0])-tpr[0])
for i in range(len(thresholds)):
  if(abs((1-fpr[0])-tpr[0])<min_diff):
    best_thresh = thresholds[i]
print(best_thresh)

with torch.no_grad():
    list1=[]
    list2=[]
    list3=[]
    loop=tqdm(val_loader,total=len(val_loader),leave=True)
    for images,labels in loop:
        images=images.to(device)
        labels = labels.type(torch.LongTensor)
        labels=labels.to(device)
        outputs=model(images)
        for i in range(0,len(outputs)):
            if outputs[i]<=best_thresh:
                list1.append(0)
                list3.append(outputs[i])
                list2.append(labels[i].cpu())
            else:
                list1.append(1)
                list3.append(outputs[i])
                list2.append(labels[i].cpu())
    
print(accuracy_score(np.array(list1),np.array(list2)))
print(classification_report(np.array(list1),np.array(list2))) 
print(confusion_matrix(np.array(list1),np.array(list2)))
