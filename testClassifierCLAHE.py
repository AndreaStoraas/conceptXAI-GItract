import torch
import os
import numpy as np
import torch.nn as nn

from sklearn import metrics

from torchvision import models
from torch.utils.data import DataLoader

#For image augmentation:
import albumentations as albu
import cv2 as cv
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset


#Device:
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



def test_model(model, dataloader): 
    all_predicted = []
    all_true = []
    #Make sure the params are freezed:
    model.eval()
    running_acc = 0.0
    with torch.set_grad_enabled(False):
        for param in model.parameters():
            param.requires_grad = False
        for i, (inputs,y_true) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            y_true = y_true.to(DEVICE)

            y_pred = model(inputs)
            y_true = y_true.detach().cpu().numpy()
            #Predict the most probable class:
            y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
            print('Predicted value:',y_pred)
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


test_data_dir = './data/cropped-representative-concept-test'
#Define the transformation and dataloader:
transform_test_clahe = albu.Compose([
        albu.CLAHE(clip_limit=2.0,p=1.0),
        albu.Resize(232,232),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])

#Create a customized dataset
#See this link: https://albumentations.ai/docs/examples/pytorch_classification/
n_classes = 2

small_list = [os.path.join(test_data_dir, str(class_id)) for class_id in range(n_classes)]
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

if __name__ == "__main__":
    predictions, y_true = test_model(model,test_loader)
    #Again, flatten the list
    predictions = [item.tolist() for item in predictions]
    y_true = [item.tolist() for item in y_true]
    print('Number of predictions:',len(predictions))
    #Calculating performance metrics (macro = unweighted mean across all samples):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true,predictions, average='macro')
    mcc = metrics.matthews_corrcoef(y_true, predictions)
    print('Showing results on:',test_data_dir)
    print('Precision:',precision)
    print('Recall:',recall)
    print('F1 score:',fscore)
    print('Support:',support)
    print('Balanced accuracy',metrics.balanced_accuracy_score(y_true,predictions))
    print('MCC:', mcc)
    print('Overall accuracy:',metrics.accuracy_score(y_true,predictions))
    print('Results for each class separately:')
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true,predictions)
    print('Precision separate:',precision)
    print('Recall separate:',recall)
    print('F1 score separate:',fscore)
    print('Support separate:',support)
    #Plot all predictions in one single confusion matrix:
    print(metrics.confusion_matrix(y_true, predictions))
    print(metrics.classification_report(y_true, predictions))
    
    
