import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch
from efficientnet_pytorch import EfficientNet


import numpy as np
import pandas as pd
import cv2
import os


def test_transformation():
    tsfm = A.Compose([
        A.Resize(224,224, interpolation=cv2.INTER_AREA),
        
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),    
        A.pytorch.transforms.ToTensorV2()
    ])
    return tsfm

class PlantPathologyTestDataset(Dataset):
    def __init__(self, root_path, X, transform=None):
        self.root_path = root_path
        self.X = X
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        img_path = self.X[index]
        image = cv2.imread(os.path.join(self.root_path,img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmentated = self.transform(image=image)
            image = augmentated['image']
        
        return image, img_path
    
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
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_root_path = "/data/kaggle/plant-pathology-2021-fgvc8/test_images/"
    test_data = os.listdir(test_root_path)
    
    #Validation test
    #data_df = pd.read_csv(os.path.join('./mod_train.csv'))
    #test_root_path = "/data/kaggle/plant-pathology-2021-fgvc8/train_images/"
    #test_data = data_df[data_df['fold']==0]['image'].to_numpy()
    #test_data = test_data[:10]
    
    test_transform = test_transformation()
    
    test_dataset = PlantPathologyTestDataset(test_root_path, test_data, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=4)
    
    model = PlantModel("efficientnet-b3", num_classes=6, pretrained=False)
    model.load_state_dict(torch.load('/data/kaggle/plant-pathology-2021-fgvc8/trained_model/210515_efficientnet_b3_seed_1110_512/1_best_model_0_06420736894945336.pth'))
    
    model = model.to(device)
    model.eval() #..... 이거였다...
    
    classes = ["healthy", "scab", "frog_eye_leaf_spot", "complex","rust","powdery_mildew"]
    result = {'image':[], 'labels':[]}
    outputs = []
    labels = []
    
    for inputs, img_path in test_dataloader:
        inputs = inputs.to(device)
        
        output = model(inputs)
        output = output.squeeze()
        output = output.detach().cpu()
        output = torch.sigmoid(output).numpy()
        outputs.append(output)
        preds = output > 0.5
        #To avoid '' preds
        #preds[np.argmax(output)] = True
        result['image'].append(img_path[0])

        labels.append(preds)

    for label in labels:
        processed = ' '.join([classes[i] for i,value in enumerate(label) if value==True])

        result['labels'].append(processed)

    sub = pd.DataFrame(result)
    print(sub)
    print(outputs)
        
    
    
    