import os 
import copy
import random
import time
import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
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
        
# TRAIN
def train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, device, args, log, num_fold):
    
    if not os.path.exists(os.path.join(args.save_path,num_fold)):
        os.makedirs(os.path.join(args.save_path,num_fold))
        
    since = time.time()
    model = model.to(device)
    #amp
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.
    best_acc = 0.
    
    for epoch in range(args.epochs):
        log.info('')
        log.info('Epoch {}/{}'.format(epoch, args.epochs-1))
        log.info('-' * 20)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            num_inputs = 0
            metric = torchmetrics.F1(num_classes=6, threshold=0.5, average='micro')
            metric2 = torchmetrics.F1(num_classes=6, threshold=0.6, average='micro')
            metric = metric.to(device)
            metric2 = metric2.to(device)
            running_acc = 0.0
            
            pbar = tqdm(dataloaders[phase])
            for idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the grads
                optimizer.zero_grad()

                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = metric(torch.sigmoid(outputs), labels)
                acc2 = metric2(torch.sigmoid(outputs), labels)

                if phase == 'train':

                    #loss.backward()
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.item()*inputs.size(0)
                num_inputs += inputs.size(0)

                pbar.set_description("[{}/{}][{}] Loss: {:.6f}  F1 Score: {:.4f} {:.4f}".format(epoch, args.epochs, phase, running_loss/num_inputs, metric.compute(), metric2.compute()))
            
            if phase == 'valid':
                scheduler.step(running_loss)
                
            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = metric.compute()
            epoch_acc2 = metric2.compute()
            
            log.info('{} Loss: {:.6f}\t F1 Score: {:.4f} {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc2))
            
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(args.save_path, num_fold,'{:02d}_epoch_model.pth'.format(epoch)))
                log.info('Save {} epoch model.'.format(epoch))
            elif phase == 'valid' and epoch % 10 == 0:
                # save model manually
                model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_wts, os.path.join(args.save_path, num_fold, '{:02d}_epoch_model.pth'.format(epoch)))
                log.info('Save {} epoch model.'.format(epoch))
        
    
    time_elapsed = time.time() - since
    log.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    log.info('Best val Loss: {:.6f} val F1: {:.4f}'.format(best_loss, best_acc))
    
    #model.load_state_dict(best_model_wts)
    
    return best_model_wts, best_loss

def train_transformation():
    tsfm = A.Compose([
        #A.Resize(224,224), 
        A.Resize(512, 512),
        A.CenterCrop(height=224, width=224),

        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),            
        ], p=1),
        A.RandomGamma(p=0.25),

        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),    
        A.pytorch.transforms.ToTensor()
    ])
    return tsfm

def test_transformation():
    tsfm = A.Compose([
        #A.Resize(224,224),
        A.Resize(512, 512),
        A.CenterCrop(height=224, width=224),
        
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),    
        A.pytorch.transforms.ToTensor()
    ])
    return tsfm
    
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
    # read data file
    train_df = pd.read_csv(os.path.join(root_path,'train.csv'), index_col='image')
    train_files = os.listdir(os.path.join(root_path,'train_images'))
    
    # remove duplicate data
    with open('./duplicates.csv', 'r') as file:
        duplicates = [x.strip().split(',') for x in file.readlines()]
    init_len = len(train_df)
    for row in duplicates:
        unique_labels = train_df.loc[row].drop_duplicates().values
        if len(unique_labels) == 1:
            train_df = train_df.drop(row[1:], axis=0)
        else:
            train_df = train_df.drop(row, axis=0)

    log.info(f'Dropping {init_len - len(train_df)} duplicate samples.')
    
    # Multi-label Binarize
    train_df['labels'] = [x.split(' ') for x in train_df['labels']]
    
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(train_df['labels'].values)
    
    new_train_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_train_df.insert(0,'image', train_df.index)
    
    # split train / valid dataset
    X,Y = new_train_df['image'].to_numpy(), new_train_df[["healthy", "scab", "frog_eye_leaf_spot", "complex","rust","powdery_mildew"]].to_numpy(dtype=np.float32)
    
    # 5-fold 
    #msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    msss = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    num_fold = 1
    image_path = os.path.join(root_path,'train_images')
    
    for train_index, test_index in msss.split(X, Y):
        log.info("[{}FOLD] start".format(num_fold))
        log.info("TRAIN:{}\tTEST:{}".format(train_index,test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]        

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
            batch_size=32, 
            num_workers=args.num_workers, 
            shuffle=False, pin_memory=True)

        datasets = {'train': train_dataset,
                    'valid': val_dataset}

        dataloaders = {'train': train_loader,
                       'valid': val_loader}

        # LOAD PRETRAINED ViT MODEL
        model = EfficientNet.from_pretrained(args.model, num_classes=6)

        # OPTIMIZER
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.001)
        # LEARNING RATE SCHEDULER
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)

        criterion = FocalLoss(logits=True)


        best_model_wts, best_loss = train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, device, args, log, str(num_fold))
        torch.save(best_model_wts, os.path.join(args.save_path,'{}_best_model_{}.pth'.format(num_fold,str(best_loss).replace('.','_'))))
        num_fold +=1 