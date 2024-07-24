import os
import numpy as np
import cv2 as cv

#The source test images (before cropping):
test_path = 'data/representative-concept-test'
class0 = os.path.join(test_path,'0')
class1 = os.path.join(test_path,'1')

class0_images = os.listdir(class0)
class1_images = os.listdir(class1)
#Target folder for the cropped images:
cropped_test_path = 'data/cropped-representative-concept-test'
cropped_class0 = os.path.join(cropped_test_path,'0')
cropped_class1 = os.path.join(cropped_test_path,'1')


#Use the following suggestion to crop away black edges in the images:
#https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
for i in range(len(class0_images)): #<- Repeat for class 1 images afterward
    image_name = class0_images[i]
    print('Name of image:',image_name)
    image_path = os.path.join(class0,image_name)
    img = cv.imread(image_path)
    print('Shape of original image:',img.shape) #width, height (and channels)
    #Following the example here: 
    #https://stackoverflow.com/questions/58398300/find-all-coordinates-of-black-grey-pixels-in-image-using-python
    #Convert to grayscale and set threshold
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #Set threshold to 15
    thresh = 15
    #Find all coordinates with pixels below the threshold:
    coords = np.column_stack(np.where(gray < thresh))
    #Find the pixel coordinates on the image edges where the pixels are black
    #For the x axis, the largest value that is still smaller than the image center and 
    #the smallest value that is larger than the image center are used to define the new image width
    #In addition, the pixel must lie on the edge of the image (y=0 or y=height for x, and x=0 or x=width for y)
    #The same logic applies to the y axis. 
    min_x,  min_y, max_x, max_y = 0, 0, img.shape[0],img.shape[1]
    for coordinate_set in coords:
        if (coordinate_set[0]> min_x) and (coordinate_set[0]<int(img.shape[0]/2)): 
            if (coordinate_set[1]==0) or (coordinate_set[1]==img.shape[1]):
                min_x = coordinate_set[0]
        elif (coordinate_set[0]<max_x) and (coordinate_set[0]>int(img.shape[0]/2)):
            if (coordinate_set[1]==0) or (coordinate_set[1]==img.shape[1]):
                max_x = coordinate_set[0]
        if (coordinate_set[1]> min_y) and (coordinate_set[1]<int(img.shape[1]/2)):
            if (coordinate_set[0]==0) or (coordinate_set[0]==img.shape[0]):
                min_y = coordinate_set[1]
        elif (coordinate_set[1]<max_y) and (coordinate_set[1]>int(img.shape[1]/2)):
            if (coordinate_set[0]==0) or (coordinate_set[0]==img.shape[0]):
                max_y = coordinate_set[1]
    new_width = False
    new_height = False
    if ((max_x - min_x)<(img.shape[0]/2.5)):
        #If the width is too small, use original width:
        print('Too narrow cropping! Using original image width instead...')
        new_width = img.shape[0]
    if ((max_y - min_y)<(img.shape[1]/2.5)):
        #If the height is too small, use original height:
        print('Too much cropping! Using original image height instead...')
        new_height = img.shape[1]
    if new_width and new_height:
        print('No cropping!')
        crop = img[:,:]
    elif new_width and not new_height:
        print('Too narrow crop! Use original width...')
        crop = img[:,min_y:max_y]
    elif not new_width and new_height:
        print('Too low crop! Use original height...')
        crop = img[min_x:max_x,:]
    else:
        print('Cropping both sides...')
        #Crop the image based on the contours:
        crop = img[min_x:max_x,min_y:max_y]
    print('Shape of cropped image:',crop.shape)
    save_path = os.path.join(cropped_class0,image_name)
    cv.imwrite(save_path,crop)