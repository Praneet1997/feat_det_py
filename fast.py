import cv2
import sys
import numpy as np
import time


def main():
	
	cap = cv2.VideoCapture(int(sys.argv[1]))
	cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
	fast = cv2.FastFeatureDetector_create()
	#sift=cv2.xfeatures2d.SIFT_create()
	
	while True:
		

		ret,frame = cap.read()
		left = frame[:,:,0]
		right = frame[:,:,1]
		
		img2 = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
		#kp = fast.detect(left, None)
		fast.setNonmaxSuppression(1)
		kp = fast.detect(left,None)
		img2 = cv2.drawKeypoints(left, kp, None, color=(255,0,0))
		print(len(kp))
		"""kp=sift.detect(left, None)
		
		a=0
		for kpt in kp:
			a=a+1
			x, y = kpt.pt
			#print (kpt.pt)
			cv2.circle(img2, (int(x), int(y)), 2, [0,255,255])"""

		cv2.imshow('left',left)
		cv2.imshow('keypoints',img2)
		key = cv2.waitKey(1)
		if key == 27:
			break


main()















"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('simple.jpg',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# Print all default params
print "Threshold: ", fast.getInt('threshold')
print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
print "neighborhood: ", fast.getInt('type')
print "Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)"""