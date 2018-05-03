import numpy as np
import cv2
from matplotlib import pyplot as plt
from six.moves import xrange
import pickle 

img1 = cv2.imread('template.jpg',0)          # queryImage
img2 = cv2.imread('test.jpg',0) # trainImage


#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
ka = cv2.KAZE_create()

# find the keypoints and descriptors with kaze 
kp1, des1 = ka.detectAndCompute(img1,None)

#temp_descriptors_pkl = "temp_descriptors.pkl"
#with open(temp_descriptors_pkl, "rb") as fp:
#	des = pickle.load(fp)

#des1 = np.zeros((len(des), 64), dtype = np.float32)
#for ii in range(0,len(des)):
#	des1[ii,:] = des[ii] 

#temp_keypoints_pkl = "temp_keypoints.pkl"
#with open(temp_keypoints_pkl, "rb") as fp:
#	kp1 = pickle.load(fp)


kp2, des2 = ka.detectAndCompute(img2,None)

#print(des1.shape)
#print(len(kp1))
#print(des1.shape)
#print(des2.shape)


# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
#matchesMask = [[0,0] for i in xrange(len(matches))]

good_match = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good_match.append(m)

#draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)

#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

#for m,n in matches:
#	print("query index m "+str(m.queryIdx))
#	print("query index n "+str(n.queryIdx))
#	print("train index m "+str(m.trainIdx))
#	print("query index n "+str(n.trainIdx))
#	print("distance m "+str(m.distance))
#	print("distance n "+str(n.distance))


#print(des1.shape)
#print(des2.shape)
#plt.imshow(img3,),plt.show()

#for m in good_match:
#	print(kp2[m.queryIdx].octave)

#print(len(good_match))

#test_mt = good_match[0]

#print(str(test_mt.queryIdx)+ ":test   train:"+ str(test_mt.trainIdx) + "  distance:" + str(test_mt.distance))

#print(kp1[test_mt.trainIdx].angle)

good_match = sorted(good_match, key=lambda match: match.distance)

#for m in good_match:
#	print(m.queryIdx)

#print(len(kp2))

test_mt = good_match[0]

#img3 = img2[np.int (kp2[test_mt.queryIdx].pt[1]-kp1[test_mt.trainIdx].pt[1]):np.int(kp2[test_mt.queryIdx].pt[1]-kp1[test_mt.trainIdx].pt[1] + img1.shape[0]),np.int(kp2[test_mt.queryIdx].pt[0]-kp1[test_mt.trainIdx].pt[0]):np.int(kp2[test_mt.queryIdx].pt[0]-kp1[test_mt.trainIdx].pt[0] + img1.shape[1])]

#cv2.imshow("cropped image", img3)
#cv2.waitKey(0)
#print(img1.shape)
#print(img3.shape)
imc =0
confidence_mat = np.zeros(img2.shape , dtype = np.float32)
#print(confidence_mat.shape)
for m in good_match:
#	print(imc)
	imc = imc + 1 
	test_mt = m
	
	start_row = np.int(kp2[test_mt.trainIdx].pt[1]-kp1[test_mt.queryIdx].pt[1])
	end_row = np.int(kp2[test_mt.trainIdx].pt[1]-kp1[test_mt.queryIdx].pt[1] + img1.shape[0])
	start_col = np.int(kp2[test_mt.trainIdx].pt[0]-kp1[test_mt.queryIdx].pt[0])
	end_col = np.int(kp2[test_mt.trainIdx].pt[0]-kp1[test_mt.queryIdx].pt[0] + img1.shape[1])
	
	if(start_row<=0 or start_row>=img2.shape[0]):
		continue
	if(end_row<=0 or end_row>=img2.shape[0]):
		continue
	if(start_col<=0 or start_col>=img2.shape[1]):
		continue
	if(end_col<=0 or end_col>=img2.shape[1]):
		continue

	#img3 = img2[start_row:end_row, start_col:end_col]

	#img4 = img2[71:385,384:884]
	#cv2.imshow("cropped image", img3)
	#print(str(kp2[test_mt.queryIdx].pt[1]) + "        " + str(kp2[test_mt.queryIdx].pt[0]))
	#print(str(kp1[test_mt.trainIdx].pt[1]) + "        " + str(kp1[test_mt.trainIdx].pt[0]))	
	#cv2.waitKey(0)
	print(str(start_col) + "  "+ str(start_row) + "  " +str(end_col)+ "  "+str(end_row))
	for i in range(0,img2.shape[0]):
		for j in range(0,img2.shape[1]):
			confidence_mat[i][j] = confidence_mat[i][j] - 1


	for i in range(start_row,end_row):
		for j in range(start_col,end_col):
			confidence_mat[i][j] = confidence_mat[i][j] + 2;
	
	#print(confidence_mat)
	#cv2.imshow("hand", img4)
	#cv2.waitKey(0)
#print(confidence_mat.shape)
max_val = confidence_mat.max()
min_val = confidence_mat.min()
for a in range(0,confidence_mat.shape[0]):
	for b in range(0,confidence_mat.shape[1]):
		confidence_mat[a][b] = (confidence_mat[a][b] - min_val)/(max_val - min_val)
		#print(confidence_mat[a][b])
		#cv2.waitKey(0)


pkl_filename = "confimat.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(confidence_mat, file)


#print(confidence_mat)
cv2.imshow("blah", confidence_mat)
#cv2.imshow("blah blah", confidence_mat*255)
cv2.waitKey(0)

#print(img1.shape)
#print(img3.shape)