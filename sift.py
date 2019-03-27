import cv2
import sys
import numpy as np
import time


def main():
	
	cap = cv2.VideoCapture(int(sys.argv[1]))
	cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
	#fast = cv2.FastFeatureDetector()
	sift=cv2.xfeatures2d.SIFT_create()
	
	while True:
		

		ret,frame = cap.read()
		left = frame[:,:,0]
		right = frame[:,:,1]
		
		img2 = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
		#kp = fast.detect(left,None)
		#img2 = cv2.drawKeypoints(left, kp, color=(255,0,0))
		kp=sift.detect(left, None)
		print(len(kp))
		a=0
		for kpt in kp:
			a=a+1
			x, y = kpt.pt
			#print (kpt.pt)
			cv2.circle(img2, (int(x), int(y)), 2, [0,255,255])

		cv2.imshow('left',left)
		cv2.imshow('keypoints',img2)
		key = cv2.waitKey(1)
		if key == 27:
			break


main()