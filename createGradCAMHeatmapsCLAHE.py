import os
import random
import argparse
import torch
import copy

import numpy as np
import matplotlib.pyplot as plt
from torch import functional
import torch.nn as nn

from sklearn import metrics

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
#And import albumentations for CLAHE preprocessing:
import cv2 as cv
#From pytorch-grad-cam library https://github.com/jacobgil/pytorch-grad-cam/tree/master:
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

#For image augmentation:
import albumentations as albu
import cv2 as cv
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset
from crp.helper import get_layer_names

random.seed(0)
np.random.seed(0)
#Should use torch.manual_seed: https://pytorch.org/vision/stable/transforms.html
torch.manual_seed(0)

#First test on the test part of the DRDetection dataset:
test_path = './data/cropped-representative-concept-test'
print('This is the test path:',test_path)
#Define the device:
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#=========================================
# Helper functions and Datasets
#=========================================
#Define dataset class:
#Want to get the name of the image file
#To ensure that the name of the heatmap file corresponds to the image that the heatmap represents
#https://discuss.pytorch.org/t/print-an-image-file-name/87667
#Define dataset class (necessary when applying albumentations transformations):
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            filepaths, 
            augmentation=None, 
            #preprocessing=None,
    ):
        self.filepaths = filepaths
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image_path = self.filepaths[i]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #print('Checking the class:')
        #print(os.path.normpath(image_path).split(os.sep)[-2])
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
        return image, label, image_path
        
    def __len__(self):
        return len(self.filepaths)


def test_model(model, dataloader, gradcam_object):
    all_predicted = []
    all_true = []
    #Make sure the params are freezed:
    model.eval()
    running_acc = 0.0
    with torch.set_grad_enabled(False):
        for param in model.parameters():
            param.requires_grad = False
        for i, (inputs, y_true) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            y_true = y_true.to(DEVICE)
            y_pred = model(inputs)
            y_true = y_true.detach().cpu().numpy()
            #Predict most probable class:
            y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
            print('Y_pred:',y_pred)
            print('True value:', y_true)
            
            running_acc += metrics.accuracy_score(y_true, y_pred) 
            all_predicted.append(y_pred)
            all_true.append(y_true)

    mean_acc = running_acc / len(dataloader)
    print('Overall accuracy test set:',mean_acc)
    #Flatten the list
    all_predicted = [a.squeeze() for a in all_predicted]
    all_true = [a.squeeze() for a in all_true]
    print('Predicted values:')
    print(all_predicted)
    return all_predicted, all_true

def create_heatmaps(model, dataloader, cam_object):
    for i, (inputs, y_true,filepath) in enumerate(dataloader): #Use the customized dataset to get filename
    
        print('Round number',str(i+1))
        inputs = inputs.to(DEVICE)
        y_true = y_true.to(DEVICE)
        y_pred = model(inputs)
        y_true = y_true.detach().cpu().numpy()
        #Predict the most probable class:
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        print('Y_pred:',y_pred)
        print('True value:', y_true)
        for j in range(len(y_pred)):
            if y_pred[j] == y_true[j]:
                saliency = cam_object(input_tensor=inputs[j].unsqueeze(0).float(), targets=None)
                print('True label:', y_true[j])
                print('Shape of saliency map:', saliency.shape)
                #Import the original image: 
                image_path = filepath[0]
                original_img = cv.imread(image_path)
                #NOT converting to RGB -> The image will be in BGR format
                #Re-size to 232,232 to match the preprocessed input tensor
                original_img = cv.resize(original_img,(232,232),3)
                print('Looking at image:',test_filepath[j+i])
                
                #Preprocess to avoid error when showing heatmap on original image (a little down on the page):
                #https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
                original_img = np.array(original_img, np.float32)
                #Scale to lie between 0 and 1
                original_img *= (1.0/original_img.max())
                print('The original image:',original_img.shape)
                #Select the heatmap for the image:
                grayscale_cam = saliency[0, :]
                #If image is in RGB format, use_rgb = True. 
                #For us, the image is in BGR, so use_rgb = False
                #See lines 48-52 in source code: 
                #https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py
                visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=False) 
                folder_name = 'gradCAMheatmaps_CroppedRepresentativetest_Layer42Conv3'
                filepath = list(filepath)
                image_name = filepath[0].split('/')[-1]
                class_name = filepath[0].split('/')[-2]
                save_image_path = folder_name + '/' +  class_name + '/' + image_name[:-4] +'.jpg'
                print(save_image_path)
                cv.imwrite(save_image_path, visualization)

#Add all filepaths for the test dataset to a list
n_classes = 2
small_list = [os.path.join(test_path, str(class_id)) for class_id in range(n_classes)]
print('Small list testing:', small_list)
test_filepath = []
for _list in small_list:
    all_files = os.listdir(_list)
    print('Number of files:',len(all_files))
    all_paths = []
    #For each image in the class folder    
    for _img in all_files:
        single_path = os.path.join(_list,_img)
        all_paths.append(single_path)
        #Add the full image path to image_list:
    test_filepath += all_paths
print('Length of test files:',len(test_filepath))    

#Define the dataloader:
transform_test_clahe = albu.Compose([
        albu.CLAHE(clip_limit=2.0,p=1.0),
        albu.Resize(232,232),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])
#Create a customized dataset
#See this link: https://albumentations.ai/docs/examples/pytorch_classification/
n_classes = 2

small_list = [os.path.join(test_path, str(class_id)) for class_id in range(n_classes)]
print('Small list test:', small_list)
test_filepath = []
for _list in small_list:
    all_files = os.listdir(_list)
    print('Number of files:',len(all_files))
    all_paths = []
    #For each image in the class folder
    for _img in all_files:
        single_path = os.path.join(_list,_img)
        all_paths.append(single_path)
    #Add the full image path to image_list:
    test_filepath += all_paths
print('Length of test files:',len(test_filepath))

test_dataset = Dataset(test_filepath,augmentation = transform_test_clahe)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)
#Load the trained model:
model = models.resnet152()
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
#Since binary classification as a multiclass problem, we have two outputs (one for each class):
model.fc = nn.Linear(model.fc.in_features,n_classes)
#Add Identity layers for exploring more of the model after training:
model.identity_0 = nn.Identity()
model.identity_1 = nn.Identity()
model.identity_2 = nn.Identity()
model.last_conv = nn.Identity()
model._forward_impl = _forward_impl_.__get__(model)

#Load the checkpoints for the saved model:
print('Loading in the weights for the trained model...')
chkpoint_path = 'output/DiseaseResnet152_200epochsCLAHE_2outputs.pt'
chkpoint = torch.load(chkpoint_path, map_location = 'cpu')
model.load_state_dict(chkpoint)
model.to(DEVICE)

#Create the Grad-CAM object:
# Construct the CAM object once, and then re-use it on many images:
print('Creating heatmaps for the following layer of the Resnet152 model:',model.last_conv)

cam = GradCAM(model=model,
              target_layers=[model.last_conv],
              use_cuda=False)


if __name__ == "__main__":
    lol = create_heatmaps(model, test_loader,cam)
    