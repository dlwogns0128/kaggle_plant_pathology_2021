import os
import random
import copy
import time
import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision import utils
import albumentations as A
import albumentations.pytorch
from apex import amp, optimizers
import torchmetrics
from efficientnet_pytorch import EfficientNet

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

APEX=True

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def train_transformation():
    tsfm = A.Compose([
        #A.Resize(224,224), 
        A.Resize(512, 512, interpolation=cv2.INTER_AREA),

        A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
        A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1,
        ),

        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),    
        A.pytorch.transforms.ToTensorV2()
    ])
    return tsfm

def test_transformation():
    tsfm = A.Compose([
        #A.Resize(224,224),
        A.Resize(512, 512, interpolation=cv2.INTER_AREA),
        
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),    
        A.pytorch.transforms.ToTensorV2()
    ])
    return tsfm

class PlantPathologyDataset(Dataset):
    def __init__(self, root_path, X, y, transform=None):
        self.root_path = root_path

        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):

        img_path = self.X[index]
        label = self.y[index]
        image = cv2.imread(os.path.join(self.root_path,img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmentated = self.transform(image=image)
            image = augmentated['image']
        
        return image, label   

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class PlantModel(nn.Module):
    def __init__(self, model_name, num_classes=6, pretrained=True):
        super(PlantModel, self).__init__()
        
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
            
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        self.FC = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.FC(x)
        return x
    

def train(epoch, model, dataloader, criterion, optimizer, device):
    
    running_loss = 0.0
    num_inputs = 0
    
    model.train()
    
    metric = torchmetrics.F1(num_classes=6, threshold=0.5, average='samples')
    metric = metric.to(device)
    
    pbar = tqdm(dataloader)
    for idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        score = metric(torch.sigmoid(outputs), labels)
        
        if APEX:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        num_inputs += inputs.size(0)
        running_loss += loss.item()*inputs.size(0)
        
        pbar.set_description("[{:02d} epoch][Train] Loss: {:.6f} F1 Score: {:.5f}".format(epoch, running_loss/num_inputs, metric.compute()))
        
    epoch_loss = running_loss / num_inputs
    epoch_score = metric.compute().item()
    
    return epoch_loss, epoch_score

def validation(epoch, model, dataloader, criterion, device):
    
    running_loss = 0.0
    num_inputs = 0
    
    model.eval()
    
    metric = torchmetrics.F1(num_classes=6, threshold=0.5, average='samples')
    metric = metric.to(device)
    
    pbar = tqdm(dataloader)
    for idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        score = metric(torch.sigmoid(outputs), labels)
        
        num_inputs += inputs.size(0)
        running_loss += loss.item()*inputs.size(0)
        
        pbar.set_description("[{:02d} epoch][Valid] Loss: {:.6f} F1 Score: {:.5f}".format(epoch, running_loss/num_inputs, metric.compute()))
        
    epoch_loss = running_loss / num_inputs
    epoch_score = metric.compute().item()
    
    return epoch_loss, epoch_score
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", help="root path plant-pathology-2021-fgvc8 saved")
    parser.add_argument("--save_path", default='./')
    parser.add_argument("--model", help="type model name(ex. efficientnet-b0)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    log = logging.getLogger('train_log')
    log.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(os.path.join(args.save_path,'train.log') , mode= "w")
    formatter = logging.Formatter('[%(asctime)s] : %(message)s')
    fileHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    
    log.info(vars(args))
    
    SEED = args.seed
    seed_everything(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_path = args.data_root_path
    image_path = os.path.join(root_path,'train_images')
    # read data file
    data_df = pd.read_csv(os.path.join(root_path,'mod_train.csv'))
        
    # 5-fold 
    KFold = 5    
    
    for num_fold in range(KFold):
        log.info("[{}FOLD] start".format(num_fold))
        
        if not os.path.exists(os.path.join(args.save_path, str(num_fold))):
            os.makedirs(os.path.join(args.save_path, str(num_fold)))
        
        X_train = data_df[data_df['fold']!=num_fold]['image'].to_numpy()
        X_test = data_df[data_df['fold']==num_fold]['image'].to_numpy()
        y_train = data_df[data_df['fold']!=num_fold][["healthy", "scab", "frog_eye_leaf_spot", "complex","rust","powdery_mildew"]].to_numpy()
        y_test = data_df[data_df['fold']==num_fold][["healthy", "scab", "frog_eye_leaf_spot", "complex","rust","powdery_mildew"]].to_numpy()
    
        train_dataset = PlantPathologyDataset(
            root_path = image_path, 
            X=X_train, y=y_train, 
            transform=train_transformation())

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, pin_memory=True)

        val_dataset = PlantPathologyDataset(
            root_path = image_path, 
            X=X_test, y=y_test, 
            transform=test_transformation())

        val_loader = DataLoader(
            val_dataset, 
            batch_size=16, 
            num_workers=args.num_workers, 
            shuffle=False, pin_memory=True)

        datasets = {'train': train_dataset,
                    'valid': val_dataset}

        dataloaders = {'train': train_loader,
                       'valid': val_loader}

        # Load model
        model = PlantModel(args.model, num_classes=6)

        # OPTIMIZER
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
        # LEARNING RATE SCHEDULER
        scheduler = CosineLRScheduler(optimizer, t_initial=15, decay_rate=0.5, lr_min=1e-7)

        criterion = nn.MultiLabelSoftMarginLoss() #FocalLoss(logits=True)

        #
        since = time.time()
        model = model.to(device)
        
        ## nvidia-apex amp
        if APEX:
            opt_level = 'O1'
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)           
        
        best_loss = 100.
        best_score = 0.
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(args.epochs):
            log.info('')
            log.info('Epoch {}/{}'.format(epoch, args.epochs-1))
            log.info('-' * 20)  
            
            train_loss, train_score = train(epoch, model, train_loader, criterion, optimizer, device)
            log.info('Train Loss: {:.6f}\t F1 Score: {:.5f}'.format(train_loss, train_score))
            
            val_loss, val_score = validation(epoch, model, val_loader, criterion, device)
            log.info('Validation Loss: {:.6f}\t F1 Score: {:.5f}'.format(val_loss, val_score))
            
            # CosineLRScheduler
            scheduler.step(epoch)
            
            # Save model
            if val_loss < best_loss:
                best_loss = val_loss
                best_score = val_score
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(args.save_path, str(num_fold), '{:02d}_epoch_model.pth'.format(epoch)))
                log.info('Saved {} epoch model.'.format(epoch))
            elif epoch%10 == 0:
                model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_wts, os.path.join(args.save_path, str(num_fold), '{:02d}_epoch_model.pth'.format(epoch)))
                log.info('Saved {} epoch model.'.format(epoch))
            
        time_elapsed = time.time() - since
        log.info('Training Complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed % 60))
        log.info('Best Val Loss: {:.6f} Val F1: {:.5f}'.format(best_loss, best_score))

        torch.save(best_model_wts, os.path.join(args.save_path,'{}_best_model_{}.pth'.format(str(num_fold),str(best_loss).replace('.','_'))))
