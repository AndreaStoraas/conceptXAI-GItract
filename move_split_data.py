import os
import shutil
import random
import numpy as np


lower_giFolder = './data/hyperkvasir-labeled-images/lower-gi-tract'
upper_giFolder = './data/hyperkvasir-labeled-images/upper-gi-tract'
disease_folder = './data/Disease'
normal_folder = './data/NoDisease'

lower_gi_diseaseFolders = ['pathological-findings/hemorrhoids','pathological-findings/polyps',
                           'pathological-findings/ulcerative-colitis-grade-0-1',
                           'pathological-findings/ulcerative-colitis-grade-1',
                           'pathological-findings/ulcerative-colitis-grade-1-2',
                           'pathological-findings/ulcerative-colitis-grade-2',
                           'pathological-findings/ulcerative-colitis-grade-2-3',
                           'pathological-findings/ulcerative-colitis-grade-3',
                           'therapeutic-interventions/dyed-lifted-polyps']

upper_gi_diseaseFolders = ['barretts','barretts-short-segment','esophagitis-a',
                          'esophagitis-b-d']

lower_gi_normalFolders = ['anatomical-landmarks/cecum','anatomical-landmarks/ileum','anatomical-landmarks/retroflex-rectum',
                          'quality-of-mucosal-views/bbps-0-1','quality-of-mucosal-views/bbps-2-3']

upper_gi_normalFolders = ['pylorus', 'retroflex-stomach','z-line']

'''
#Move all disease images to the Disease folder:
print('Moving the disease images for the lower GI tract...')
for _folder in lower_gi_diseaseFolders:
    print('Name of the folder:',_folder)
    folder_path = os.path.join(lower_giFolder,_folder)
    for _image in os.listdir(folder_path):
        source_path = os.path.join(folder_path,_image)
        shutil.copy(source_path,disease_folder)
    print('Number of images in disease-folder',len(os.listdir(disease_folder)))
    print('Number of images in original-folder',len(os.listdir(folder_path)))

print('Moving the disease images for the upper GI tract...')
for _folder in upper_gi_diseaseFolders:
    print('Name of the folder:',_folder)
    folder_path = os.path.join(upper_giFolder,'pathological-findings',_folder)
    for _image in os.listdir(folder_path):
        source_path = os.path.join(folder_path,_image)
        shutil.copy(source_path,disease_folder)
    print('Number of images in disease-folder',len(os.listdir(disease_folder)))
    print('Number of images in original-folder',len(os.listdir(folder_path)))


#Move all normal images to the NoDisease folder:
print('Moving normal images for the lower GI tract...')
for _folder in lower_gi_normalFolders:
    print('Name of the folder:',_folder)
    folder_path = os.path.join(lower_giFolder,_folder)
    for _image in os.listdir(folder_path):
        source_path = os.path.join(folder_path, _image)
        shutil.copy(source_path,normal_folder)
    print('Number of images in normal folder:',len(os.listdir(normal_folder)))
    print('Number of images in original-folder:',len(os.listdir(folder_path)))

print('Moving normal images for the upper GI tract...')
for _folder in upper_gi_normalFolders:
    print('Name of the folder:',_folder)
    folder_path = os.path.join(upper_giFolder,'anatomical-landmarks',_folder)
    for _image in os.listdir(folder_path):
        source_path = os.path.join(folder_path, _image)
        shutil.copy(source_path,normal_folder)
    print('Number of images in normal folder:',len(os.listdir(normal_folder)))
    print('Number of images in original-folder:',len(os.listdir(folder_path)))
'''

random.seed(0)
np.random.seed(0)

#Split into train, validation and test
#70, 20, 10%
train_disease = './data/train/1'
train_normal = './data/train/0'
valid_disease = './data/valid/1'
valid_normal = './data/valid/0'
test_disease = './data/test/1'
test_normal = './data/test/0'

num_disease_train = int(len(os.listdir(disease_folder))*0.7)
num_disease_valid = int(len(os.listdir(disease_folder))*0.2)

#print('Total number of disease images:',len(os.listdir(disease_folder)))
#print('Number of train disease images:',num_disease_train)
#print('Number of valid disease images:',num_disease_valid)

num_normal_train = int(len(os.listdir(normal_folder))*0.7)
num_normal_valid = int(len(os.listdir(normal_folder))*0.2)

#print('Total number of normal images:',len(os.listdir(normal_folder)))
#print('Number of train normal images:',num_normal_train)
#print('Number of valid normal images:',num_normal_valid)
disease_files = os.listdir(disease_folder)
normal_files = os.listdir(normal_folder)

def splitTrainValTest():
    #Start with the disease images:
    #Convert to set for easier substraction
    trainFilesDisease = set(random.sample(disease_files,num_disease_train))
    validTestFilesDisease = set(disease_files) - trainFilesDisease
    validFilesDisease = set(random.sample(validTestFilesDisease,num_disease_valid))
    testFilesDisease = validTestFilesDisease - validFilesDisease
    #Convert back to list again:
    trainFilesDisease = list(trainFilesDisease)
    validFilesDisease = list(validFilesDisease)
    testFilesDisease = list(testFilesDisease)
    #Check that no duplicates in the train and validation/test files:
    for tFile in trainFilesDisease:
        if tFile in validFilesDisease:
            print('Same image in valid and training list! Something went wrong here!')
        elif tFile in testFilesDisease:
            print('Same image in TEST and training list! Something went wrong here!')
    #Check that no duplicates in validation and test files:
    for vFile in validFilesDisease:
        if vFile in testFilesDisease:
            print('Same image in valid and testing list! Something went wrong!')
           
    #Next, move the files to their respective folders:
    #For the training data:
    for tFile in trainFilesDisease:
        source_path = os.path.join(disease_folder,tFile)
        target_path = os.path.join(train_disease,tFile)
        shutil.copy(source_path,target_path)
    #For the validation data:
    for vFile in validFilesDisease:
        source_path = os.path.join(disease_folder,vFile)
        target_path = os.path.join(valid_disease,vFile)
        shutil.copy(source_path,target_path)
    #For the testing data:
    for tFile in testFilesDisease:
        source_path = os.path.join(disease_folder,tFile)
        target_path = os.path.join(test_disease,tFile)
        shutil.copy(source_path,target_path)
    ####### Repeat for the normal images #############
    #Convert to set for easier substraction
    trainFilesNormal = set(random.sample(normal_files,num_normal_train))
    validTestFilesNormal = set(normal_files) - trainFilesNormal
    validFilesNormal = set(random.sample(validTestFilesNormal,num_normal_valid))
    testFilesNormal = validTestFilesNormal - validFilesNormal
    #Convert back to list again:
    trainFilesNormal = list(trainFilesNormal)
    validFilesNormal = list(validFilesNormal)
    testFilesNormal = list(testFilesNormal)
    #Check that no duplicates in the train and validation/test files:
    for tFile in trainFilesNormal:
        if tFile in validFilesNormal:
            print('Same image in valid and training list! Something went wrong here!')
        elif tFile in testFilesNormal:
            print('Same image in TEST and training list! Something went wrong here!')
    #Check that no duplicates in validation and test files:
    for vFile in validFilesNormal:
        if vFile in testFilesNormal:
            print('Same image in valid and testing list! Something went wrong!')
           
    #Next, move the files to their respective folders:
    #For the training data:
    for tFile in trainFilesNormal:
        source_path = os.path.join(normal_folder,tFile)
        target_path = os.path.join(train_normal,tFile)
        shutil.copy(source_path,target_path)
    #For the validation data:
    for vFile in validFilesNormal:
        source_path = os.path.join(normal_folder,vFile)
        target_path = os.path.join(valid_normal,vFile)
        shutil.copy(source_path,target_path)
    #For the testing data:
    for tFile in testFilesNormal:
        source_path = os.path.join(normal_folder,tFile)
        target_path = os.path.join(test_normal,tFile)
        shutil.copy(source_path,target_path)

#splitTrainValTest()

print('Number of disease training files:',len(os.listdir(train_disease)))
print('Number of disease validation files:',len(os.listdir(valid_disease)))
print('Number of disease test files:',len(os.listdir(test_disease)))
print('Number of normal training files:',len(os.listdir(train_normal)))
print('Number of normal validation files:',len(os.listdir(valid_normal)))
print('Number of normal test files:',len(os.listdir(test_normal)))

'''
#Pick 10 000 images from Imagenet folder at random and move to random-imagenet-images folder
imagenet_sourceFolder = './data/ILSVRC/Data/CLS-LOC/test' #This folder was deleted afterwards to save storage space
imagenet_targetFolder = './data/random-imagenet-images'

imagenet_images = os.listdir(imagenet_sourceFolder)
num_images = 10000
def pick_random_imagenet():
    selected_images = set(random.sample(imagenet_images, num_images))
    selected_images = list(selected_images)
    for _image in selected_images:
        source_path = os.path.join(imagenet_sourceFolder,_image)
        target_path = os.path.join(imagenet_targetFolder,_image)
        shutil.copy(source_path,target_path)

#pick_random_imagenet()
print('Number of Imagenet images in target folder:',len(os.listdir(imagenet_targetFolder)))
'''
#Want to pick 700, 200 and 100 random images from imagenet and 
#move them to train, valid and test folders for the negative class
target_folder_train = 'data/imagenet-train/0'
target_folder_valid = 'data/imagenet-valid/0'
target_folder_test = 'data/imagenet-test/0'
imagenet_folder = 'data/random-imagenet-images'
imagenet_images = os.listdir(imagenet_folder)
#Since original splitting is 70-20-10 for train, validation and test:
num_train = 700
num_valid = 200
num_test = 100
total_images = 1000

def move_imagenet_images():
    selected_total_images = set(random.sample(imagenet_images,total_images))
    selected_total_images = list(selected_total_images)
    selected_train_images = set(random.sample(selected_total_images,num_train))
    valid_test_images = set(selected_total_images) - selected_train_images
    ###Continue here with selecting/subtracting valid and test files!
    selected_valid_images = set(random.sample(valid_test_images,num_valid))
    selected_test_images = valid_test_images - selected_valid_images
    selected_train_images = list(selected_train_images)
    selected_valid_images = list(selected_valid_images)
    selected_test_images = list(selected_test_images)
    print('Number of selected training images:',len(selected_train_images))
    print('Number of selected validation images:',len(selected_valid_images))
    print('Number of selected test images:',len(selected_test_images))
    #Check that test images are not in train/valid set:
    for _file in selected_test_images:
        if (_file in selected_train_images) or (_file in selected_valid_images):
            print('Test image also in train/validation set!')
    for _file in selected_valid_images:
        if (_file in selected_train_images):
            print('Validation image also in training set!')
    for tFile in selected_train_images:
        source_path = os.path.join(imagenet_folder,tFile)
        target_path = os.path.join(target_folder_train,tFile)
        shutil.copy(source_path,target_path)
    for vFile in selected_valid_images:
        source_path = os.path.join(imagenet_folder,vFile)
        target_path = os.path.join(target_folder_valid,vFile)
        shutil.copy(source_path,target_path)
    for tFile in selected_test_images:
        source_path = os.path.join(imagenet_folder,tFile)
        target_path = os.path.join(target_folder_test,tFile)
        shutil.copy(source_path,target_path)

#move_imagenet_images()

print('Number of normal training files after adding Imagenet images:',len(os.listdir(target_folder_train)))
print('Number of normal validation files:',len(os.listdir(target_folder_valid)))
print('Number of normal test files:',len(os.listdir(target_folder_test)))

final_train_normal = './data/cropped-imagenet-train/0'
final_train_disease = './data/cropped-imagenet-train/1'
final_valid_normal = './data/cropped-imagenet-valid/0'
final_valid_disease = './data/cropped-imagenet-valid/1'
final_test_normal = './data/cropped-imagenet-test/0'
final_test_disease = './data/cropped-imagenet-test/1'

#print('Final number of normal training files:', len(os.listdir(final_train_normal)))
#print('Final number of disease training files:', len(os.listdir(final_train_disease)))
#print('Final number of normal validation files:', len(os.listdir(final_valid_normal)))
#print('Final number of disease validation files:', len(os.listdir(final_valid_disease)))
#print('Final number of normal test files:', len(os.listdir(final_test_normal)))
#print('Final number of disease test files:', len(os.listdir(final_test_disease)))
