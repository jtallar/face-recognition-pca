
import numpy as np
import argparse
import cv2
import os
 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
faces = []

for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		faces.append(cv2.resize(cv2.cvtColor(image[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY), (50,50)))

for i in range(0, len(faces)):
	cv2.imshow("Face x", faces[i])
	cv2.waitKey(0)

import library as l

dir = './data'

# recognize the ohm image
ohm_img = l.get_ohm_image(faces[0], dir)
print("Ohm img, ", ohm_img)

# searches for the index of the matching face
i = l.face_space_distance(ohm_img, dir)
print(i)

# gets the corresponding path given the index
path = l.get_matching_path(i, dir)
print(path)