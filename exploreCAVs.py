import os
import torch

from collections import Counter

#Loading a CAV
#cavs (dict): A dictionary of CAV objects indexed by concept ids and
#layer names. It gives access to the weights of each concept
#in a given layer and model statistics such as accuracies
#that resulted in trained concept weights.

#Repeat for each combination of class (1/0) and concept (polyp/instrument)
#The model_id name defined when calculating the CAVs in TCAV_experimentingCLAHE.py:
cav_folder_path = './cav/Resnet152_200epochsClass1_CroppedInstruments_2outputs'
cav_files = os.listdir(cav_folder_path)
#Collect the 10 indexes with highest mean values 
#for each of the 20 sets w/positive and negative examples
large_mean_indexes = []
for _file in cav_files:
    print('Name of file:',_file)
    data_path = os.path.join(cav_folder_path,_file)
    cav_data = torch.load(data_path)
    print('Check that index 0 is for the positive examples:')
    print(cav_data['concept_names'])
    cav = cav_data['stats']['weights'][0]
    #Reshape the CAV back to output dimensions of the last conv. layer:
    #reshaped_cav = torch.reshape(cav,(1024,15,15)) #For identity_2
    reshaped_cav = torch.reshape(cav,(2048,8,8)) #For last_conv and layer4.2.conv3
    #Get the mean values for each of the 32 channels (want the highest weights)
    mean_weights = reshaped_cav.mean((1,2))
    #Get the max values for each of the 32 channels
    max_weights = torch.amax(reshaped_cav, dim=(1, 2))

    highest_weights, channel_ids = torch.topk(mean_weights, 10)
    highest_weights_max, channel_ids_max = torch.topk(max_weights, 10)
    print('The channels indexes with heighest mean weights:',channel_ids)
    print('The channels indexes with heighest max weights:',channel_ids_max)
    #Convert from tensor to a "normal" list
    channel_ids = channel_ids.tolist()
    large_mean_indexes += channel_ids

print('Length of large list with top 10 channels from each file:',len(large_mean_indexes))
#Find the channel indexes that occur most frequently
#https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
#The documentation: https://docs.python.org/3/library/collections.html
occurrence_count = Counter(large_mean_indexes)
frequent_indexes = occurrence_count.most_common(10)
print('Top 10 most frequent channel indexes across the 20 example sets and corresponding counts:')
print(frequent_indexes)
