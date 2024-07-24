import os
import shutil
import copy
import random
import numpy as np

#Pick representative test images 
#with segmentation masks for polyps/instruments.
polyps_and_moreFolder = './data/SegmentedImages/Kvasir-SEG/images'
only_instrumentsFolder = './GastroConcepts/SortedImages/InstrumentConcepts/Instruments'
polyps_instrumentsFolder = './GastroConcepts/SortedImages/PolypsConcept/polypsInstruments'
representative_testFolder = './data/representative-concept-test'
#Get the kvasir-SEG images with only polyps (no instruments):
only_polyps = []
for _image in os.listdir(polyps_and_moreFolder):
    if _image not in os.listdir(polyps_instrumentsFolder):
        only_polyps.append(_image)

print('Number of segmented polyps images in Kvasir-SEG:', len(os.listdir(polyps_and_moreFolder)))
print('Number of segmented polyps images with instruments in Kvasir-SEG or Kvasir-instrument:', len(os.listdir(polyps_instrumentsFolder)))
print('Number of polyps without instruments in Kvasir-SEG:', len(only_polyps))

#Only pick 50 images of disease/no disease as in DR concept project
#Due to memory limitations
random.seed(0)
np.random.seed(0)


polyps_instruments_files = os.listdir(polyps_instrumentsFolder)
only_instruments_files = os.listdir(only_instrumentsFolder)

#For the disease images:
#Pick 25 images with only polyps:
selected_polyps = set(random.sample(only_polyps,25))
#And 25 images with polyps and instruments:
selected_polyps_instruments = set(random.sample(polyps_instruments_files,25))
selected_polyps = list(selected_polyps)
selected_polyps_instruments = list(selected_polyps_instruments)
#After manual inspection of selected images, some are not obvious disease images
#Pick 5 more images
new_selected_polyps = set(random.sample(only_polyps,3))
new_selected_polyps_instruments = set(random.sample(polyps_instruments_files,2))
new_selected_polyps = list(new_selected_polyps)
new_selected_polyps_instruments = list(new_selected_polyps_instruments)
#Copy the images to folder 1 (disease) for the representative concept test images:
for _image in new_selected_polyps:
    source_path = os.path.join(polyps_and_moreFolder,_image)
    target_path = os.path.join(representative_testFolder,'1',_image)
    #shutil.copy(source_path,target_path)
for _image in new_selected_polyps_instruments:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(representative_testFolder,'1',_image)
    #shutil.copy(source_path,target_path)

#For the no disease images:
#Pick 50 images with only instruments (no polyps):
selected_instruments = set(random.sample(only_instruments_files,50))
selected_instruments = list(selected_instruments)
#After manual inspection, identify 7 images that are not necessarily healthy (looks like hemorrhoids)
#Pick some more instrument files:
new_selected_instruments = set(random.sample(only_instruments_files,7))
new_selected_instruments = list(new_selected_instruments)
#Again after manual check, 3 more images were removed since not obviously healthy 
#Add 4 since one image was already present in the folder -> gets 50 images in total
new_selected_instruments2 = set(random.sample(only_instruments_files,4))
new_selected_instruments2 = list(new_selected_instruments2)
#Copy the images to folder 0 (no disease) for the representative concept test images:
for _image in new_selected_instruments2:
    source_path = os.path.join(only_instrumentsFolder,_image)
    target_path = os.path.join(representative_testFolder,'0',_image)
    #shutil.copy(source_path,target_path)

print('Number of representative disease test images:',len(os.listdir(os.path.join(representative_testFolder,'1'))))
print('Number of representative no disease test images:',len(os.listdir(os.path.join(representative_testFolder,'0'))))

######## Polyps concept ########
#Select 45 positive examples for the polyps concept
#From the hyperkvasir polyps images,
#the polyps + instruments images IF NOT in folder 1 for representative test set
#the only polyps images IF NOT in folder 1 for representative test set
hyperKvasir_polypFolder = './data/hyperkvasir-labeled-images/lower-gi-tract/pathological-findings/polyps'
representative_testDiseaseImages = os.listdir(os.path.join(representative_testFolder,'1'))

#Pick 15 hyperkvasir polyps
all_hyperKvasir_polyps = os.listdir(hyperKvasir_polypFolder)
selected_hyperkvasir_polyps = set(random.sample(all_hyperKvasir_polyps, 15))
selected_hyperkvasir_polyps = list(selected_hyperkvasir_polyps)
#Pick 20 polyps + instruments AND check that they are NOT in folder 1
selected_polyps_instruments_concepts = []
while len(selected_polyps_instruments_concepts)<20:
    polyp_instrument_image = set(random.sample(polyps_instruments_files, 1))
    #Check that the image is NOT in the representative test set for disease:
    if polyp_instrument_image not in representative_testDiseaseImages:
        polyp_instrument_image = list(polyp_instrument_image)
        selected_polyps_instruments_concepts += polyp_instrument_image
#Pick 15 polyps AND check that they are NOT in folder 1
selected_polyps_concepts = [] 
while len(selected_polyps_concepts)<10:
    polyp_image = set(random.sample(only_polyps, 1))
    #Check that the image is NOT in the representative test set for disease:
    if polyp_image not in representative_testDiseaseImages:
        polyp_image = list(polyp_image)
        selected_polyps_concepts += polyp_image

print('Length of the HyperKvasir positive polyps images:',len(selected_hyperkvasir_polyps))
print('Length of the polyps + instruments selected images:',len(selected_polyps_instruments_concepts))
print('Length of the only polyps images from Kvasir-SEG:',len(selected_polyps_concepts))

#Get 8 more polyps + instruments images since some of the previous 
# image did not obviously show polyps
selected_polyps_instruments_conceptsNew = []
while len(selected_polyps_instruments_conceptsNew)<8:
    polyp_instrument_image = set(random.sample(polyps_instruments_files, 1))
    #Check that the image is NOT in the representative test set for disease:
    if polyp_instrument_image not in representative_testDiseaseImages:
        polyp_instrument_image = list(polyp_instrument_image)
        selected_polyps_instruments_conceptsNew += polyp_instrument_image
#Add 3 more polyps + instrument images for same reason:
selected_polyps_instruments_conceptsNew2 = []
while len(selected_polyps_instruments_conceptsNew2)!=3:
    polyp_instrument_image = set(random.sample(polyps_instruments_files, 1))
    #Check that the image is NOT in the representative test set for disease:
    if polyp_instrument_image not in representative_testDiseaseImages:
        polyp_instrument_image = list(polyp_instrument_image)
        selected_polyps_instruments_conceptsNew2 += polyp_instrument_image      
#Add 2 more polyps + instrument images for same reason:
selected_polyps_instruments_conceptsNew3 = []
while len(selected_polyps_instruments_conceptsNew3)!=2:
    polyp_instrument_image = set(random.sample(polyps_instruments_files, 1))
    #Check that the image is NOT in the representative test set for disease:
    if polyp_instrument_image not in representative_testDiseaseImages:
        polyp_instrument_image = list(polyp_instrument_image)
        selected_polyps_instruments_conceptsNew3 += polyp_instrument_image      

#The final polyps + instrument images for same reason:
selected_polyps_instruments_conceptsNew4 = []
while len(selected_polyps_instruments_conceptsNew4)!=1:
    polyp_instrument_image = set(random.sample(polyps_instruments_files, 1))
    #Check that the image is NOT in the representative test set for disease:
    if polyp_instrument_image not in representative_testDiseaseImages:
        polyp_instrument_image = list(polyp_instrument_image)
        selected_polyps_instruments_conceptsNew4 += polyp_instrument_image

polyp_positive_exampleFolder = './GastroConcepts/Polyps/PositiveExamples'
#Copy the selected images over to the positive polyp example folder:
for _image in selected_hyperkvasir_polyps:
    source_path = os.path.join(hyperKvasir_polypFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_polyps_instruments_concepts:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_polyps_concepts:
    source_path = os.path.join(polyps_and_moreFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_polyps_instruments_conceptsNew:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_polyps_instruments_conceptsNew2:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_polyps_instruments_conceptsNew3:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_polyps_instruments_conceptsNew4:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(polyp_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)
#print('Number of positive example images for the polyps concepts:',len(os.listdir(polyp_positive_exampleFolder)))

#For each of the 20 negative example folders, select 45 negative images
#From no disease folder
# the only instruments folder IF NOT in folder 0 for representative test set
hyperKvasir_noDiseaseFolder = './data/NoDisease'
hyperKvasir_noDiseaseImages = os.listdir(hyperKvasir_noDiseaseFolder)
representative_testNoDiseaseImages = os.listdir(os.path.join(representative_testFolder,'0'))

large_negative_polyp_concepts = []
for i in range(20):
    negative_polyp_concepts = []
    #First, pick 25 images from the Hyperkvasir noDisease folder:
    while len(negative_polyp_concepts)<25:
        negative_polyp_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
        #print(str(negative_polyp_image).strip('}{'))
        negative_polyp_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(negative_polyp_image).strip("'}{"))
        #To avoid that the same image is selected twice:
        if negative_polyp_image not in negative_polyp_concepts:
            negative_polyp_concepts.append(negative_polyp_image)
    #Next, pick 20 instrument images from only instruments 
    while len(negative_polyp_concepts)<45:
        negative_polyp_image = set(random.sample(only_instruments_files,1))
        #Check that the images are NOT in the representative test set:
        if negative_polyp_image not in representative_testNoDiseaseImages:
            negative_polyp_image = str(only_instrumentsFolder+ '/'+str(negative_polyp_image).strip("'}{"))
            #To avoid that the same image is selected twice:
            if negative_polyp_image not in negative_polyp_concepts:
                negative_polyp_concepts.append(negative_polyp_image)
    large_negative_polyp_concepts.append(negative_polyp_concepts)

#print('Length of the large negative concepts list:',len(large_negative_polyp_concepts))
#print('Length of the last element of the large negative concepts list:', len(large_negative_polyp_concepts[-1]))

polyp_negative_example_generalFolder = './GastroConcepts/Polyps/NegativeExamples'
'''
#Copy the selected files to each of the 20 negative example sets:
for i in range(20):
    selected_negative_examples = large_negative_polyp_concepts[i]
    print('Number of selected negative images before copy:',len(selected_negative_examples))
    for _imagePath in selected_negative_examples:
        image_name = _imagePath.split('/')[-1]
        target_path = os.path.join(polyp_negative_example_generalFolder + str(i + 1),image_name)
        #shutil.copy(_imagePath, target_path)
    print('Number of negative example images:',len(os.listdir(polyp_negative_example_generalFolder+str(i+1))))
'''
#Need to add some extra images to some of the negative folders
#since they seem to include polyps:
#Add 1 extra image to each of these folders:
extra_negative_instrument_folders = ['NegativeExamples2','NegativeExamples6','NegativeExamples8',
                                     'NegativeExamples9','NegativeExamples10','NegativeExamples11',
                                     'NegativeExamples13','NegativeExamples17','NegativeExamples18']
#Add 1 extra image to each of these folders EXCEPT folder 16, where 2 images are added:
extra_negative_clean_folders = ['NegativeExamples10','NegativeExamples16','NegativeExamples19']

#print('Selecting extra images with instruments, no polyps...')
for _neg_instrument_folder in extra_negative_instrument_folders:
    neg_images = []
    while len(neg_images)!=1:
       negative_polyp_image = set(random.sample(only_instruments_files,1)) 
       #Check that the images are NOT in the representative test set:
       if negative_polyp_image not in representative_testNoDiseaseImages:
            negative_polyp_image = str(only_instrumentsFolder+ '/'+str(negative_polyp_image).strip("'}{"))
            #To avoid that the same image is selected twice:
            if negative_polyp_image not in negative_polyp_concepts:
                neg_images.append(negative_polyp_image)
    #Copy the image to the negative image folder:
    #print('Length of negative instrument images for folder', _neg_instrument_folder,len(neg_images))
    for _image in neg_images:
        image_name = _image.split('/')[-1]
        target_path = os.path.join('./GastroConcepts/Polyps/',_neg_instrument_folder,image_name)
        #shutil.copy(_image, target_path)
#print('Selecting extra clean images from hyperkvasir no disease images...')
#Repeat for the clean polyp images:
for _neg_clean_folder in extra_negative_clean_folders:
    if _neg_clean_folder == 'NegativeExamples16':
        num_images = 2
    else:
        num_images = 1
    neg_images = []
    while len(neg_images)!=num_images:
       negative_polyp_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
       negative_polyp_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(negative_polyp_image).strip("'}{"))
        #To avoid that the same image is selected twice:
       if negative_polyp_image not in negative_polyp_concepts:
            neg_images.append(negative_polyp_image)
    #print('Length of negative clean images for folder', _neg_clean_folder,len(neg_images))
    #Copy the image(s) to the negative folder:
    for _image in neg_images:
        image_name = _image.split('/')[-1]
        target_path = os.path.join('./GastroConcepts/Polyps/',_neg_clean_folder,image_name)
        #shutil.copy(_image, target_path)

###### Instrument concepts ########
#Select 45 positive examples for the instrument concept
#20 instruments + polyps, 25 only instruments
selected_polyps_instruments_concepts = []
#Pick 21 instead of 20 since one of the 45 images are not copied over to the folder, 
#resulting in one image too little...
while len(selected_polyps_instruments_concepts)!=21:
    polyp_instrument_image = set(random.sample(polyps_instruments_files, 1))
    #Check that the image is NOT in the representative test set for disease:
    if (polyp_instrument_image not in representative_testDiseaseImages) and (polyp_instrument_image not in selected_polyps_instruments_concepts):
        polyp_instrument_image = list(polyp_instrument_image)
        selected_polyps_instruments_concepts += polyp_instrument_image

selected_only_instruments_concepts = []
while len(selected_only_instruments_concepts)!=25:
    only_instrument_image = set(random.sample(only_instruments_files,1))
    #Check that the images are NOT in the representative test set:
    if (only_instrument_image not in representative_testNoDiseaseImages) and (only_instrument_image not in selected_only_instruments_concepts):
        only_instrument_image = list(only_instrument_image)
        selected_only_instruments_concepts += only_instrument_image

instrument_positive_exampleFolder = './GastroConcepts/Instruments/PositiveExamples'
#Copy the selected images over to the positive polyp example folder:
print('Copying selected positive example images to the positive instrument folder...')
for _image in selected_polyps_instruments_concepts:
    source_path = os.path.join(polyps_instrumentsFolder,_image)
    target_path = os.path.join(instrument_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)

for _image in selected_only_instruments_concepts:
    source_path = os.path.join(only_instrumentsFolder,_image)
    target_path = os.path.join(instrument_positive_exampleFolder,_image)
    #shutil.copy(source_path,target_path)
print('Number of positive example images:',len(os.listdir(instrument_positive_exampleFolder)))

#Select 45 negative images for each of the 20 negative image folders:
large_negative_instrument_concepts = []
for i in range(20):
    negative_instrument_concepts = []
    #First, pick 25 images from the Hyperkvasir noDisease folder:
    while len(negative_instrument_concepts)!=25:
        negative_instrument_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
        negative_instrument_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(negative_instrument_image).strip("'}{"))
        #To avoid that the same image is selected twice:
        if negative_instrument_image not in negative_instrument_concepts:
            negative_instrument_concepts.append(negative_instrument_image)
    #Next, pick 10 images from the hyperKvasir polyps folder:
    while len(negative_instrument_concepts)!=35:
        negative_instrument_image = set(random.sample(all_hyperKvasir_polyps,1))
        negative_instrument_image = str(hyperKvasir_polypFolder+ '/'+str(negative_instrument_image).strip("'}{"))
        if negative_instrument_image not in negative_instrument_concepts:
            negative_instrument_concepts.append(negative_instrument_image)
    #Then, pick 10 images from the only polyps folder
    while len(negative_instrument_concepts)!=45:
        polyp_image = set(random.sample(only_polyps, 1))
        #Check that the image is NOT in the representative test set for disease:
        if polyp_image not in representative_testDiseaseImages:
            polyp_image = str(polyps_and_moreFolder+ '/'+str(polyp_image).strip("'}{"))
            negative_instrument_concepts.append(polyp_image)
    large_negative_instrument_concepts.append(negative_instrument_concepts)

print('Number of smaller image lists the large instrument concept list:',len(large_negative_instrument_concepts))
print('Length of the first list:',len(large_negative_instrument_concepts[0]))

#For each of the 20 sets with negative example images, 
#copy the images to the corresponding negative folders:
instrument_negative_example_generalFolder = './GastroConcepts/Instruments/NegativeExamples'
'''
for i in range(20):
    image_list = large_negative_instrument_concepts[i]
    for _imagePath in image_list:
        image_name = _imagePath.split('/')[-1]
        target_path = os.path.join(instrument_negative_example_generalFolder + str(i + 1),image_name)
        #shutil.copy(_imagePath, target_path)
    print('Number of negative example images for folder:',str(i+1),':',len(os.listdir(instrument_negative_example_generalFolder+str(i+1))))
'''
#For each folder, must pick additional images:
large_negative_instrument_concepts2 = []
num_NoDisease_images = [5,7,3,10,7,6,2,9,6,5,5,7,6,7,4,5,7,6,4,1]
num_PolypHyperkvasir_images = [0,0,1,0,2,2,0,1,1,0,1,1,2,0,2,0,0,0,1,0]
num_OnlyPolyp_images = [0,1,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0]
for i in range(20):
    negative_concept_images = []
    #Pick corresponding No Disease images:
    while len(negative_concept_images)!=num_NoDisease_images[i]:
        negative_instrument_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
        negative_instrument_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(negative_instrument_image).strip("'}{"))
        #To avoid that the same image is selected twice:
        if negative_instrument_image not in negative_concept_images:
            negative_concept_images.append(negative_instrument_image)
    while len(negative_concept_images)!=(num_NoDisease_images[i]+num_PolypHyperkvasir_images[i]):
        negative_instrument_image = set(random.sample(all_hyperKvasir_polyps,1))
        negative_instrument_image = str(hyperKvasir_polypFolder+ '/'+str(negative_instrument_image).strip("'}{"))
        if negative_instrument_image not in negative_concept_images:
            negative_concept_images.append(negative_instrument_image)
    while len(negative_concept_images)!=(num_NoDisease_images[i]+num_PolypHyperkvasir_images[i]+num_OnlyPolyp_images[i]):
        #print('Round',i)
        #print(num_NoDisease_images[i]+num_PolypHyperkvasir_images[i]+num_OnlyPolyp_images[i])
        negative_instrument_image = set(random.sample(only_polyps,1))
        #Check that the image is NOT in the representative test set for disease:
        if negative_instrument_image not in representative_testDiseaseImages:
            negative_instrument_image = str(polyps_and_moreFolder+ '/'+str(negative_instrument_image).strip("'}{"))
            negative_concept_images.append(negative_instrument_image)
    large_negative_instrument_concepts2.append(negative_concept_images)

#Copy the selected images to the corresponding negative example folders:
for i in range(20):
    negative_examples_folder = instrument_negative_example_generalFolder+str(i+1)
    image_list = large_negative_instrument_concepts2[i]
    for _imagePath in image_list:
        image_name = _imagePath.split('/')[-1]
        target_path = os.path.join(negative_examples_folder,image_name) 
        #shutil.copy(_imagePath, target_path)
    print('Number of images in negative folder',str(i+1),':',len(os.listdir(negative_examples_folder)))

#Add 1 extra image to each of these folders:
#Except folders 3 (two extra), 5 and 10 and 12(three extra):
extra_negative_noDisease_folders = ['NegativeExamples1','NegativeExamples3','NegativeExamples5',
                                    'NegativeExamples6','NegativeExamples9','NegativeExamples10',
                                    'NegativeExamples12','NegativeExamples13','NegativeExamples14',
                                    'NegativeExamples15','NegativeExamples17','NegativeExamples18']
#Add 1 extra image to each of these folders:
extra_negative_HyperKvasirPolyp_folders = ['NegativeExamples13','NegativeExamples15']

print('Adding extra no disease images to the negative instrument folders...')
#Pick extra NoDisease images and move to the negative examples folder:
for _neg_noDisease_folder in extra_negative_noDisease_folders:
    if _neg_noDisease_folder == 'NegativeExamples3':
        num_images = 2
    elif _neg_noDisease_folder in ['NegativeExamples5','NegativeExamples10','NegativeExamples12']:
        num_images = 3
    else:
        num_images = 1
    neg_images = []
    while len(neg_images)!=num_images:
       negative_noDisease_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
       negative_noDisease_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(negative_noDisease_image).strip("'}{"))
        #To avoid that the same image is selected twice:
       if negative_noDisease_image not in neg_images:
            neg_images.append(negative_noDisease_image)
    #Copy the image(s) to the negative folder:
    for _image in neg_images:
        image_name = _image.split('/')[-1]
        target_path = os.path.join('./GastroConcepts/Instruments/',_neg_noDisease_folder,image_name)
        #shutil.copy(_image, target_path)

print('Adding extra polyp images to the negative instrument folders...')
#Pick extra hyperkvasir polyp images and move to the negative examples folders
for _neg_Kvasirpolyp_folder in extra_negative_HyperKvasirPolyp_folders:
    neg_images = []
    while len(neg_images)!=1:
       negative_kvasirPolyp_image = set(random.sample(all_hyperKvasir_polyps,1))
       negative_kvasirPolyp_image = str(hyperKvasir_polypFolder+ '/'+str(negative_kvasirPolyp_image).strip("'}{"))
        #To avoid that the same image is selected twice:
       if negative_kvasirPolyp_image not in neg_images:
            neg_images.append(negative_kvasirPolyp_image)
    #Copy the image(s) to the negative folder:
    for _image in neg_images:
        image_name = _image.split('/')[-1]
        target_path = os.path.join('./GastroConcepts/Instruments/',_neg_Kvasirpolyp_folder,image_name)
        #shutil.copy(_image, target_path)


#Add 1 extra image to each of these folders:
#Except folders 3 (two extra):
extra_negative_noDisease_folders2 = ['NegativeExamples3','NegativeExamples5','NegativeExamples10',
                                    'NegativeExamples12','NegativeExamples18']

for _folder in extra_negative_noDisease_folders2:
    if _folder == 'NegativeExamples3':
        num_images = 2
    else:
        num_images = 1
    neg_images = []
    while len(neg_images)!=num_images:
        neg_noDisease_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
        neg_noDisease_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(neg_noDisease_image).strip("'}{"))
        #To avoid that the same image is selected twice:
        if negative_noDisease_image not in neg_images:
            neg_images.append(neg_noDisease_image)
    #Copy the image(s) to the negative folder:
    for _image in neg_images:
        image_name = _image.split('/')[-1]
        target_path = os.path.join('./GastroConcepts/Instruments/',_folder,image_name)
        #shutil.copy(_image, target_path)

#Final image for Negative instrument folder 18:
neg_final_image = []
while len(neg_final_image)!=1:
    neg_image = set(random.sample(hyperKvasir_noDiseaseImages,1))
    neg_image = str(hyperKvasir_noDiseaseFolder+ '/'+str(neg_image).strip("'}{"))
    neg_final_image.append(neg_image)
image_name = neg_final_image[0].split('/')[-1]
target_path = os.path.join('./GastroConcepts/Instruments/NegativeExamples18',image_name)
#shutil.copy(neg_final_image[0],target_path)

for i in range(20):
    print('Number of images in folder',str(i+1),':',len(os.listdir(instrument_negative_example_generalFolder+str(i+1))))

