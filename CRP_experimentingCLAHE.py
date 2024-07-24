#Libraries for pytorch model and data preprocessing
import os
import random
import torch

import numpy as np
import torch.nn as nn

from torchvision import  models, transforms
from torch.utils.data import DataLoader
#And import albumentations for CLAHE preprocessing:
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2 as cv
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset

#Testing CRP (concept relevance propagation) following the 
#tips and tutorials in their GitHub repo
#https://github.com/rachtibat/zennit-crp/tree/master
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
#To plot the CRP heatmap:
from crp.image import imgify

from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer
from zennit.core import Composite


#=========================================
# Helper functions and Datasets
#=========================================
#Define dataset class:
#Want to get the name of the image file
#To ensure that the name of the heatmap file corresponds to the image that the heatmap represents
#https://discuss.pytorch.org/t/print-an-image-file-name/87667
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

random.seed(0)
np.random.seed(0)
#Should use torch.manual_seed: https://pytorch.org/vision/stable/transforms.html
torch.manual_seed(0)

#Define the device:
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
print('Device:',DEVICE)


test_path = './data/cropped-representative-concept-test'
#Define the transformation and dataloader:
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

print('Loading the model...')
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
model.eval()

layer_names = get_layer_names(model,[torch.nn.Identity]) #Get the identity layers
print('Last identity layer:',layer_names[-1])

#### CRP Stuff ####
cc = ChannelConcept()
#Define a zennit composite (the CRP builds upon zennit):
composite = EpsilonPlusFlat([ResNetCanonizer()])
#About the mask map from the GitHub tutorial:
#Per default, mask_map is set to ChannelConcept.mask. That's why, we will omit the parameter from now on.
#https://github.com/rachtibat/zennit-crp/blob/master/tutorials/attributions.ipynb

def create_crpAttributions(model, dataloader, composite, layernames):
    model.eval()
    for i, (inputs, y_true, filepath) in enumerate(dataloader): #When using the customized dataset above
        print('Round number',str(i+1))
        inputs = inputs.to(DEVICE)
        inputs.requires_grad=True #required when using zennit
        y_true = y_true.to(DEVICE)
        y_pred = model(inputs)
        y_true = y_true.detach().cpu().numpy()
        #Predict the most probable class:
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        print('Y_pred:',y_pred)
        for j in range(len(y_pred)):
            if (y_pred[j] == y_true[j]) and y_true[j]==1: #<- Only want to look at class 1
                print('True label (should be 1):',y_true)
                print('Creating the attribution object...')
                print('Looking at:',layernames[-1])
                attribution = CondAttribution(model, no_param_grad=True)
                #Inspect 5 (or 6) concepts at a time for easier plotting
                #All concept ids for instruments, class 0:
                #1730,1063,682,519,889, 385,915,1747,1860,1187
                #All concept ids for instruments, class 1:
                #1730,682,1063,889,519, 385,1187,1860,1134,915
                #All concept ids for polyps, class 0:
                #421,114,274,1270,2021,1772, 287,224,210,158,2015,747
                #All concept ids for polyps, class 1:
                #224,421,287,616,747, 2015,2021,1267,1460
                concept_ids = [421,114,274,1270,2021,1772]
                #Looking at the corresponding conditional heatmaps for each of these six channels
                new_conditions = [{layernames[-1]: [id], 'y': [y_pred[j]]} for id in concept_ids]
                heatmap, _, _, _ = attribution(inputs, new_conditions, composite)
                print('Filepath from the dataloader:',filepath)
                folder_name = 'Doctor-CRP-heatmaps/Polyps/1'
                #Convert the filepath to a list:
                filepath = list(filepath)
                image_name = filepath[0].split('/')[-1]
                print('Max heatmap value:',torch.max(heatmap).item())
                print('Min heatmap value:',torch.min(heatmap).item())
                save_string = folder_name + '/FirstFiveChannels_' + image_name[:-4]+ 'PredictedClass'+str(int(y_pred[j]==True)) + '_Polyp.jpg'
                #Blur the heatmap to avoid checkerboard effect:
                heatmap = transforms.functional.gaussian_blur(heatmap,19)
                pil_heatmap_image = imgify(heatmap, symmetric=True, grid = (1, len(concept_ids)))
                pil_heatmap_image.convert('RGB').save(save_string)
                
#To see how important the identified channel IDs (according to CAV) are
#for the individual model predictions on images. 
#See also lines 19 to 21 in the attributions tutorial notebook for CRP:
# https://github.com/rachtibat/zennit-crp/blob/master/tutorials/attributions.ipynb
def inspect_relevanceScores(model, dataloader, composite, layernames, channelconcept):
    model.eval()
    for i, (inputs, y_true, filepath) in enumerate(dataloader): #When using the customized dataset above
        print('Round number',str(i+1))
        inputs = inputs.to(DEVICE)
        inputs.requires_grad=True #required when using zennit
        y_true = y_true.to(DEVICE)
        y_pred = model(inputs)
        y_true = y_true.detach().cpu().numpy()
        #Predict the most probable class:
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        print('Y_pred:',y_pred)
        for j in range(len(y_pred)):
            if (y_pred[j] == y_true[j]) and y_true[j]==0 and ('ckcx9ogfz001v3b5yrrw3tifa.jpg' in filepath[0]): #<- Only want to look at class 1
                print('True label (should be 0):',y_true)
                print('Looking at:',filepath)
                print('Creating the attribution object...')
                print('Looking at:',layernames[-1])
                attribution = CondAttribution(model, no_param_grad=True)
                attr = attribution(inputs, conditions = [{'y':[y_pred[j]]}], composite = composite, 
                    record_layer=layernames)
                rel_c = channelconcept.attribute(attr.relevances[layernames[-1]], abs_norm=True)
                print('Shape of rel_c:',rel_c.shape)
                print('First part of rel_c:',rel_c[0])
                print(rel_c[0].shape)
                #The six most relevant concepts and their % contributions to final model prediction:
                rel_values, concept_ids = torch.topk(rel_c[0], 6)
                print('Top 6 channels for instrument concept, class 0:')
                print(concept_ids, rel_values*100)
                #The top 5 instrument concept IDs for class 0: 1730,1063,682,519,889
                print('Contributions to the prediction in percentage for the channels with highest mean CAV weights:')
                print('1730:',(rel_c[0][1730])*100)
                print('1063:',(rel_c[0][1063])*100)
                print('682:',(rel_c[0][682])*100)
                print('519:',(rel_c[0][519])*100)
                print('889:',(rel_c[0][889])*100)
                #Inspect 5 (or 6) concepts at a time for easier plotting
                #All concept ids for instruments, class 0:
                #1730,1063,682,519,889, 385,915,1747,1860,1187
                #All concept ids for instruments, class 1:
                #1730,682,1063,889,519, 385,1187,1860,1134,915
                #All concept ids for polyps, class 0:
                #421,114,274,1270,2021,1772, 287,224,210,158,2015,747
                #All concept ids for polyps, class 1:
                #224,421,287,616,747, 2015,2021,1267,1460


if __name__ == "__main__":
    #create_crpAttributions(model, test_loader, composite, layer_names)
    inspect_relevanceScores(model, test_loader, composite, layer_names, cc)
    print('Hello world')
