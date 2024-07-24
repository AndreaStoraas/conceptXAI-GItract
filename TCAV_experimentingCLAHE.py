import os
import torch

import numpy as np
import torch.nn as nn
from scipy.stats import ttest_ind

from torchvision import  models
from torch.utils.data import DataLoader

from captum.concept import TCAV, Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str
import matplotlib.pyplot as plt
import glob
#And import albumentations for CLAHE preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2

#Try some monkey patching for getting the layer code to work:
from captum.concept import CAV
from captum.concept._core import tcav
from captum._utils.av import AV
from typing import Any, List,Dict, cast, Optional, Tuple, Union
#from sklearn.linear_model import SGDClassifier
import json
from collections import defaultdict
from captum.concept._core.tcav import LabelledDataset


#Since the GPUs are not working well here (memory issues), I use cpu:
DEVICE = torch.device("cpu")
n_classes = 2
#Create dataloaders for the positive and the negative examples
#Some help is provided here: https://captum.ai/api/concept.html#classifier
#NB! Check out this tutorial!
# https://captum.ai/tutorials/TCAV_Image 

#Use same transformations as what the model is tested on
def transform_val(img):
    transform_test_clahe = albu.Compose([
        albu.CLAHE(clip_limit=2.0,p=1.0),
        albu.Resize(232,232),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])
    return transform_test_clahe(image = img)['image']

def get_tensor_from_filename(filename):
    #img = Image.open(filename).convert("RGB")
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return transform_val(img)
    
def load_image_tensors(class_name, root_path='ConceptFoldersDiaretDB/BalancedConcepts', transform=True):
    path = os.path.join(root_path, class_name)
    #Since the images have four(!) different formats:
    filenames = glob.glob(path + '/*.png')
    filenames2 = glob.glob(path + '/*.jpg')
    filenames3 = glob.glob(path + '/*.jpeg')
    filenames4 = glob.glob(path + '/*.JPG')
    filenames5 = glob.glob(path + '/*.JPEG')
    filenames = filenames + filenames2 + filenames3 + filenames4 + filenames5
    print('Number of images that are loaded:',len(filenames))
    tensors = []
    for filename in filenames:
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensors.append(transform_val(img) if transform else img)
    return tensors

def assemble_concept(name, id, concepts_path="GastroConcepts"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=name, data_iter=concept_iter)

#Want to print the accuracy of the classification models for the CAVs:
#Modified from tcav.train_cav:
# https://github.com/pytorch/captum/blob/master/captum/concept/_core/tcav.py
def my_train_cav(
    model_id,
    concepts,
    layers,
    classifier,
    save_path,
    classifier_kwargs,
):
    concepts_key = concepts_to_str(concepts)
    cavs: Dict[str, Dict[str, CAV]] = defaultdict()
    cavs[concepts_key] = defaultdict()
    layers = [layers] if isinstance(layers, str) else layers
    for layer in layers:

        # Create data loader to initialize the trainer.
        datasets = [
            AV.load(save_path, model_id, concept.identifier, layer)
            for concept in concepts
        ]

        labels = [concept.id for concept in concepts]

        labelled_dataset = LabelledDataset(cast(List[AV.AVDataset], datasets), labels)

        def batch_collate(batch):
            inputs, labels = zip(*batch)
            return torch.cat(inputs), torch.cat(labels)

        dataloader = DataLoader(labelled_dataset, collate_fn=batch_collate)

        classifier_stats_dict = classifier.train_and_eval(
            dataloader, **classifier_kwargs
        )
        classifier_stats_dict = (
            {} if classifier_stats_dict is None else classifier_stats_dict
        )
        weights = classifier.weights()
        assert (
            weights is not None and len(weights) > 0
        ), "Model weights connot be None or empty"

        classes = classifier.classes()
        assert (
            classes is not None and len(classes) > 0
        ), "Classes cannot be None or empty"

        classes = (
            cast(torch.Tensor, classes).detach().numpy()
            if isinstance(classes, torch.Tensor)
            else classes
        )
        cavs[concepts_key][layer] = CAV(
            concepts,
            layer,
            {"weights": weights, "classes": classes, **classifier_stats_dict},
            save_path,
            model_id,
        )
        # Saving cavs on the disk
        cavs[concepts_key][layer].save()
        #Andrea added March 20:
        print('Classifier stats dict:')
        print(classifier_stats_dict)
        accuracy_dict = {}
        #Must convert the accuracy from tensor to a number
        accuracy_dict['accuracy']=classifier_stats_dict['accs'].cpu().numpy().tolist()
        layername = layer.strip('.')
        accuracy_filename = 'Accuracies/'+model_id+ '/'+str(concepts_key)+layername+'.txt'
        with open(accuracy_filename,'w') as file:
            file.write(json.dumps(accuracy_dict))
        file.close()
        #End Andrea code
        
    return cavs

tcav.train_cav = my_train_cav

#Just give the entire image:
instrument_concept = assemble_concept("PositiveExamples", 0, concepts_path="GastroConcepts/Instruments")
random_0_concept = assemble_concept("NegativeExamples1",1,concepts_path="GastroConcepts/Instruments")
random_1_concept = assemble_concept("NegativeExamples2",2,concepts_path="GastroConcepts/Instruments")
random_2_concept = assemble_concept("NegativeExamples3",3,concepts_path="GastroConcepts/Instruments")
random_3_concept = assemble_concept("NegativeExamples4",4,concepts_path="GastroConcepts/Instruments")
random_4_concept = assemble_concept("NegativeExamples5",5,concepts_path="GastroConcepts/Instruments")
random_5_concept = assemble_concept("NegativeExamples6",6,concepts_path="GastroConcepts/Instruments")
random_6_concept = assemble_concept("NegativeExamples7",7,concepts_path="GastroConcepts/Instruments")
random_7_concept = assemble_concept("NegativeExamples8",8,concepts_path="GastroConcepts/Instruments")
random_8_concept = assemble_concept("NegativeExamples9",9,concepts_path="GastroConcepts/Instruments")
random_9_concept = assemble_concept("NegativeExamples10",10,concepts_path="GastroConcepts/Instruments")
random_10_concept = assemble_concept("NegativeExamples11",11,concepts_path="GastroConcepts/Instruments")
random_11_concept = assemble_concept("NegativeExamples12",12,concepts_path="GastroConcepts/Instruments")
random_12_concept = assemble_concept("NegativeExamples13",13,concepts_path="GastroConcepts/Instruments")
random_13_concept = assemble_concept("NegativeExamples14",14,concepts_path="GastroConcepts/Instruments")
random_14_concept = assemble_concept("NegativeExamples15",15,concepts_path="GastroConcepts/Instruments")
random_15_concept = assemble_concept("NegativeExamples16",16,concepts_path="GastroConcepts/Instruments")
random_16_concept = assemble_concept("NegativeExamples17",17,concepts_path="GastroConcepts/Instruments")
random_17_concept = assemble_concept("NegativeExamples18",18,concepts_path="GastroConcepts/Instruments")
random_18_concept = assemble_concept("NegativeExamples19",19,concepts_path="GastroConcepts/Instruments")
random_19_concept = assemble_concept("NegativeExamples20",20,concepts_path="GastroConcepts/Instruments")

#Only test concept vs random:
random_concepts = [random_2_concept,random_3_concept,random_4_concept, random_5_concept,random_6_concept,
random_7_concept, random_8_concept,random_9_concept,random_10_concept,random_11_concept,random_12_concept, random_13_concept,random_14_concept,
random_15_concept, random_16_concept,random_17_concept,random_18_concept,random_19_concept]
experimental_sets = [[instrument_concept, random_0_concept],[instrument_concept,random_1_concept]]
experimental_sets.extend([[instrument_concept, random_concept] for random_concept in random_concepts])
print('List of experimental concepts:')
print(experimental_sets)

#Now, let's define a convenience function for assembling the experiments together 
#as lists of Concept objects, creating and running the TCAV:
def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])
    print('Score list:',score_list)
    return score_list
#And a function to look at p-values
#We label concept populations as overlapping if p-value > 0.05 otherwise disjoint.
def get_pval(scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=False):
    
    P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)
    
    if print_ret:
        print('P1[mean, std]: ', format_float(np.mean(P1)), format_float(np.std(P1)))
        print('P2[mean, std]: ', format_float(np.mean(P2)), format_float(np.std(P2)))

    _, pval = ttest_ind(P1, P2)

    if print_ret:
        print("p-values:", format_float(pval))

    if pval < alpha:    # alpha value is 0.05 or 5%
        relation = "Disjoint"
        if print_ret:
            print("Disjoint")
    else:
        relation = "Overlap"
        if print_ret:
            print("Overlap")
        
    return P1, P2, format_float(pval), relation

print('Loading the model...')
#Using a binary classifier with two outputs (one value for each class)
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

#Warning when using the default classifier, since all data must be in memory at the same time
# https://captum.ai/api/_modules/captum/concept/_utils/classifier.html
# NB! Remember to create new model_id name for each time!!!
mytcav = TCAV(model = model, layers = ['last_conv'],model_id = 'Resnet152_200epochsClass0_CroppedInstruments_2outputs') 

#Load the selected test images for class 0/1:
test_imgs = load_image_tensors('0',root_path='data/cropped-representative-concept-test', transform=False)
test_tensors = torch.stack([transform_val(img) for img in test_imgs])

scores = mytcav.interpret(inputs=test_tensors.to(DEVICE),
                                        experimental_sets=experimental_sets,
                                        target=0 #The target class (no disease) 
                                        )
print('Finished with interpretation!')

########## Code for plotting ################
#Boxplot for significance testing:
n=20 #Since 20 sets of positive + negative examples
def show_boxplotsAlone(layer,layerstring, metric='sign_count'):
    print('Analyzing:',layerstring)
    def format_label_text(experimental_sets):
        concept_id_list = [exp.name if i == 0 else \
                             exp.name.split('/')[-1][:-1] for i, exp in enumerate(experimental_sets[0])]
        return concept_id_list

    n_plots = 1 #Plot NV vs negative concepts + negative vs negative

    fig, ax = plt.subplots(1, n_plots, figsize = (25, 7 * 1))
    fs = 18
    
    esl = experimental_sets
    P1, P2, pval, relation = get_pval(scores, esl, layer, metric,print_ret=True)
    ax.set_ylim([0, 1])
    #Andrea: added if/else for compatibility with densenet:
    if len(layerstring)>1:
        ax.set_title(layerstring + "-" + metric + " (pval=" + str(pval) + " - " + relation + ")", fontsize=fs)
    else:
        ax.set_title(layer + "-" + metric + " (pval=" + str(pval) + " - " + relation + ")", fontsize=fs)
    ax.boxplot([P1, P2], showfliers=True)

    ax.set_xticklabels(format_label_text(esl), fontsize=fs)

    plt.show()
    print('Saving boxplot to file:')
    plt.savefig('Instruments_LastConv_BoxplotClass0CroppedImages_Resnet152_200epochs.png', bbox_inches = 'tight')

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

#Plot the boxplot for the last identity layer:
show_boxplotsAlone(layer = 'last_conv',layerstring = 'Resnet152-CroppedLastConv-Class0')