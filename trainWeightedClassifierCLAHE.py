import torch
import os
import numpy as np
import random
import copy
import argparse
import torch.nn as nn

from sklearn import metrics

from torchvision import models
from torch.utils.data import DataLoader
#Import the WeightedRandomSampler:
from torch.utils.data.sampler import WeightedRandomSampler
#Import cv2 and albumentations for preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset

random.seed(0)
np.random.seed(0)
#Should use torch.manual_seed: https://pytorch.org/vision/stable/transforms.html
torch.manual_seed(0)

argument_parser = argparse.ArgumentParser(description="")


# Hardware and arguments for training the model:
argument_parser.add_argument("--device_id", type=int, default=0, help="")
argument_parser.add_argument("-d", "--data_path", required=True, type=str)
argument_parser.add_argument("-o", "--output_path", type=str, default="output")
argument_parser.add_argument("-e", "--epochs", type=int, default=100)

args = argument_parser.parse_args()

#Device:
torch.cuda.set_device(args.device_id)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:',DEVICE)

#Define dataset class (necessary when applying albumentations transformations):
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(
            self, 
            filepaths, 
            augmentation=None, 
    ):
        self.filepaths = filepaths
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image_path = self.filepaths[i]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #Check the class:
        if os.path.normpath(image_path).split(os.sep)[-2]=='0':
            label = 0
        elif os.path.normpath(image_path).split(os.sep)[-2]=='1':
            label = 1
        else:
            print('Something is wrong with the classes...')
        # apply augmentations
        if self.augmentation:
            image = self.augmentation(image=image)['image']    
        return image, label
        
    def __len__(self):
        return len(self.filepaths)



def train(model, dataloaders, optimizer, criterion, n_epochs): 

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch_idx in range(n_epochs):

        for phase, dataloader in dataloaders.items():
            
            if phase == "TRAIN":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_acc = 0.0

            with torch.set_grad_enabled(phase == "TRAIN"):

                for i, (inputs, y_true) in enumerate(dataloader):
                    inputs = inputs.to(DEVICE)
                    y_true = y_true.to(DEVICE)

                    y_pred = model(inputs)
                    loss = criterion(y_pred, y_true)

                    if phase == "TRAIN":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    y_true = y_true.detach().cpu().numpy()
                    #Predict the most probable class:
                    y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
                    
                    running_loss += loss.item()
                    running_acc += metrics.accuracy_score(y_true, y_pred) 
                    
            mean_loss = running_loss / len(dataloader)
            mean_acc = running_acc / len(dataloader)
            
            if phase == "VALID" and mean_acc > best_acc:
                best_acc = mean_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, epoch_idx, mean_loss, mean_acc))
    #The best model on the validation set 
    #after all epochs (total epochs) is saved:
    print("Best val Acc: %.4f" % best_acc)
    model.load_state_dict(best_model_wts)
    return model


def train_model(output_path, data_dir,n_epochs):
    if not os.path.exists( output_path ):
        os.makedirs( output_path )

    model_save_path = os.path.join(output_path, "DiseaseResnet152_200epochsCLAHE_2outputs.pt")

    print('Loading the model...')
    model = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V2)
    #From the pytorch documentation:
    '''
     The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR, 
     followed by a central crop of crop_size=[224]. 
     Finally the values are first rescaled to [0.0, 1.0] and then 
     normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    '''
    #Define customized forward function to include Identity layers
    def _forward_impl_(self, x):
    # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.identity_0(x)  # added identity


        x = self.layer2(x)
        x = self.identity_1(x)  # added identity


        x = self.layer3(x)
        x = self.identity_2(x)  # added identity


        x = self.layer4(x)
        x = self.last_conv(x)  # added identity

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    #Binary classification, deciding to use two outputs for easier use of TCAV and CRP:
    n_classes = 2
    model.fc = nn.Linear(model.fc.in_features,n_classes)
    #Add Identity layers for exploring more of the model after training:
    model.identity_0 = nn.Identity()
    model.identity_1 = nn.Identity()
    model.identity_2 = nn.Identity()
    model.last_conv = nn.Identity()
    model._forward_impl = _forward_impl_.__get__(model)
    #Or from Steven
    #model._forward_impl = types.MethodType(_forward_impl_, model)
    model = model.to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Define transform using CLAHE from the albumentations library:
    #CLAHE applies a limit value that aims to reduce noise enhancement in the image
    #Clip limit set to 2 (as in this endoscopy analysis study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9025080/)
    transform_train_clahe = albu.Compose([
        #Always apply CLAHE
        albu.CLAHE(clip_limit=2.0,p=1.0),
        albu.Resize(232,232),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.augmentations.geometric.rotate.Rotate(limit=180,p=0.5),
        albu.ColorJitter (brightness=1, contrast=(0.8,1.2), saturation=1, hue=0.1, p=0.5),
        albu.Perspective(p=0.5),
        albu.AdvancedBlur(blur_limit=(7,13)),
        albu.augmentations.crops.transforms.RandomResizedCrop(232,232,scale = (0.9, 1.0),p=0.5),
        #Normalize to ImageNet:
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])
    transform_val_clahe = albu.Compose([
        albu.CLAHE(clip_limit=2.0,p=1.0),
        albu.Resize(232,232),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])

    #Create a customized dataset
    #See this link: https://albumentations.ai/docs/examples/pytorch_classification/
    train_folder = os.path.join(data_dir, "cropped-imagenet-train")
    small_list = [os.path.join(train_folder, str(class_id)) for class_id in range(n_classes)]
    print('Small list training:', small_list)
    train_filepath = []
    for _list in small_list:
        all_files = os.listdir(_list)
        print('Number of files:',len(all_files))
        all_paths = []
        #For each image in the class folder
        for _img in all_files:
            single_path = os.path.join(_list,_img)
            all_paths.append(single_path)
        #Add the full image path to image_list:
        train_filepath += all_paths
    print('Length of training files:',len(train_filepath))

    #Repeat for validation folder:
    valid_folder = os.path.join(data_dir, "cropped-imagenet-valid")
    small_listVal = [os.path.join(valid_folder, str(class_id)) for class_id in range(n_classes)]
    print('Small list validation:', small_listVal)
    valid_filepath = []
    for _list in small_listVal:
        all_files = os.listdir(_list)
        print('Number of files:',len(all_files))
        all_paths = []
        #For each image in the class folder
        for _img in all_files:
            single_path = os.path.join(_list,_img)
            all_paths.append(single_path)
        #Add the full image path to image_list:
        valid_filepath += all_paths
    print('Length of validation files:',len(valid_filepath))

    train_dataset = Dataset(train_filepath, augmentation = transform_train_clahe)
    valid_dataset = Dataset(valid_filepath, augmentation = transform_val_clahe)
    
    disease_observations = len(os.listdir(os.path.join(data_dir,'cropped-imagenet-train/1')))
    normal_observations = len(os.listdir(os.path.join(data_dir,'cropped-imagenet-train/0')))
    #The length of sample_weights must equal the total number of obs in training dataset:
    class_weights = [1./normal_observations, 1./disease_observations]
    targets = [0,1]
    sample_weights = []
    for _t in targets:
        #Get X number of class weights, where X is number of obs for that given class
        sample_weigths_targetList = [class_weights[_t] for i in range([normal_observations,disease_observations][_t])]
        print('Weights for class',_t)
        print(sample_weigths_targetList[0])
        sample_weights +=  sample_weigths_targetList
    
    sample_weights = np.array(sample_weights)
    class_weights = torch.from_numpy(sample_weights)
    my_sampler = WeightedRandomSampler(class_weights,len(class_weights))    
    print('Length of class weights:',len(class_weights))
    print('Looking at the class weights:',class_weights)

    #Add the weighted sampler (cannot use shuffle at the same time):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, sampler=my_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print('Starting to train the model...')

    model = train(
        model=model,
        n_epochs=n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders={
            "TRAIN": train_loader,
            "VALID": valid_loader
        })
    
    #Save best model (based on validation set)
    torch.save(model.state_dict(), model_save_path)
    
if __name__ == "__main__":

    train_model(
        output_path = args.output_path,
        data_dir = args.data_path,
        n_epochs = args.epochs)
    
