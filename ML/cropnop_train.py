# Importing Libraries
import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import BCEWithLogitsLoss

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

CLASSES=['crop','not_Crop']
DATA_DIR='/home/btokas/data/CropOrNop/'
MODEL_DIR='./models/QAQC_squeezeNet1_1/'

if(not(os.path.isdir(MODEL_DIR))):
  os.makedirs(MODEL_DIR)

train_dir=os.path.join(DATA_DIR,'train')
val_dir=os.path.join(DATA_DIR,'test')
train_dataset=CustomDataset(train_dir)
val_dataset=CustomDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    yhat = model(x)
    model.train()
    loss = loss_fn(yhat,y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss
  return train_step

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.squeezenet1_1(pretrained=True)
for params in model.parameters():
  params.requires_grad_ = False
model.classifier = torch.nn.Sequential(nn.Flatten(),nn.Linear(165888,1))
model = model.to(device)
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model
optimizer = torch.optim.Adam(model.classifier[-1].parameters())

train_step = make_train_step(model, optimizer, loss_fn)
from tqdm import tqdm
losses = []
val_losses = []
epoch_train_losses = []
epoch_test_losses = []
n_epochs = 10
early_stopping_tolerance = 3
early_stopping_threshold = 0.03
for epoch in range(n_epochs):
  epoch_loss = 0
  for i ,data in tqdm(enumerate(train_loader), total = len(train_loader)): #iterate ove batches
    x_batch , y_batch = data
    x_batch = x_batch.to(device) #move to gpu
    y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
    y_batch = y_batch.to(device) #move to gpu
    loss = train_step(x_batch, y_batch)
    epoch_loss += loss/len(train_loader)
    losses.append(loss)
  epoch_train_losses.append(epoch_loss)
  print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))
  #validation doesnt requires gradient
  with torch.no_grad():
    cum_loss = 0
    for x_batch, y_batch in val_loader:
      x_batch = x_batch.to(device)
      y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to(device)
      #model to eval mode
      model.eval()
      yhat = model(x_batch)
      print(f"{yhat=} and {y_batch=}")
      val_loss = loss_fn(yhat,y_batch)
      cum_loss += loss/len(val_loader)
      val_losses.append(val_loss.item())
    epoch_test_losses.append(cum_loss)
    print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))
    best_loss = min(epoch_test_losses)
    #save best model
    if cum_loss <= best_loss:
      best_model_wts = model.state_dict()
      torch.save(model, MODEL_DIR + 'best_model.pth')
      print(f'Model saved at {MODEL_DIR} !')
    #early stopping
    early_stopping_counter = 0
    if cum_loss > best_loss:
      early_stopping_counter +=1
    if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
      print("/nTerminating: early stopping")
      break #terminate training

#load best model
model.load_state_dict(best_model_wts)

def inference(val_data):
  idx = torch.randint(1, len(val_data), (1,))
  sample = torch.unsqueeze(torch.tensor(val_data[idx][0]), dim=0).to(device)
  if torch.sigmoid(model(sample)) < 0.5:
    print("Prediction : Crop")
  else:
    print("Prediction : Nop")
  plt.imshow(val_dataset[idx][0].permute(1, 2, 0))

inference(val_dataset)