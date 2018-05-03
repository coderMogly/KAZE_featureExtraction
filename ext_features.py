import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
import pickle
import os
import copy
#loading images from a given folder storing them as a list and returning them as a list of matrixes
def load_images(folder):
	Images = [];
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename))
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if img is not None:
			Images.append(img)
			#print(img.shape)
	return Images


#extracting the features using KAZE algorithm as a replacement to SIFT features.
#makers of KAZE claim that it outperforms SIFT.
#The following function exracts KAZE features from an image and returns keypoints and corresponding feature vectors. 
def extract_KAZE(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	alg = cv2.KAZE_create()
	kps = alg.detect(image)
	kps = sorted(kps, key=lambda x: -x.response)
	kps, dsc = alg.compute(image, kps)
	#print(dsc.shape)
	kps = kps[:40]	#limiting our search to only top 40 features
	dsc = dsc[:40,:]
	return kps, dsc



#Asking the directory in which template images are there
template_image_dir = input("Please enter the template image directory name:")
#template_image_dir = "./temp_images"


#loading template images in a list
temp_images = load_images(template_image_dir)

keypoints = []
descriptor = [] 

#extracting the desciptors out of images
for im in temp_images:
	kps, dsc = extract_KAZE(im)
	#print(kps)
	#print(dsc)
	di = zip(kps, dsc)
	for kp, ds in di: #this is coloring the region around detected keypoint. 
		temp_img = copy.deepcopy(im) 
		for i in range(-2,2):
			for j in range(-2,2):
				if np.int(kp.pt[1]+i) > temp_img.shape[0]:
					print("out of range")
					continue

				if np.int(kp.pt[0]+j) > temp_img.shape[1]:
					print("out of range")
					continue
				temp_img[np.int(kp.pt[1]+i), np.int(kp.pt[0]+j), :] = (0,0,255)  #setting keypoint in red color. 
		cv2.imshow("feature position", temp_img)
		cv2.waitKey(1)
		is_good = np.int(input("Do you want to keep this feature vector(1:keep , 0:discard)")) # asking if the keypoint displayed needs to be selected
		if (is_good != 0):
			#print("adding")
			keypoints.append(kp)
			descriptor.append(ds)




pkl_filename = "temp_descriptors.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(descriptor, file)

pkl_filename = "temp_keypoints.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(keypoints, file)

