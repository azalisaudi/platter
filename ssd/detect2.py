import cv2 as cv
import os.path
from imutils import paths
import numpy as np

LABELS = ["null","plate"]     

# load the class labels from disk
#rows = open("synset_words.txt").read().strip().split("\n")
#classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

cvNet = cv.dnn.readNetFromTensorflow("model/frozen_inference_graph.pb", "model/graph.pbtxt")

def detect_plate(pix_file):
	img = cv.imread(pix_file)
	rows = img.shape[0]
	cols = img.shape[1]
	cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
	cvOut = cvNet.forward()

	idx = np.argsort(cvOut[0])[::-1][0]
	print("idx>> ", idx)
	print(cvOut)
	#text = "Label: {}, {:.2f}%".format(idx, cvOut[0][idx] * 100)
	#print(text)

	flag = 0
	for detection in cvOut[0,0,:,:]:
		print("detection>> ", detection)
		score = float(detection[2])
		if score > 0.5:
			left = detection[3] * cols
			top = detection[4] * rows
			right = detection[5] * cols
			bottom = detection[6] * rows
			print("{} {:.2f} left:{:.0f} top:{:.0f} right:{:.0f} bottom:{:.0f}".format(pix_file,detection[2] * 100,left,top,right,bottom))
			flag = 1
#		else:
#			print("FAILED: {}, score: {:.2f}".format(pix_file, detection[2] * 100))
	return flag

# grab the paths to the input images
imagePaths = sorted(list(paths.list_images("../Picture/")))

x, y = (1001, 2928)
counter = 0.
total = 0.
#for i in range(x, y):
for pix in imagePaths:
#for i in range(1001, 1005):
	#pix = "/home/azali/Numbur/Picture/IMG_" + str(i) + ".JPG"
	if os.path.isfile(pix):
		total = total + 1.
		found = detect_plate(pix)
		if found == 1:
			counter = counter + 1.

print("Accuracy: {:.2f}".format( (counter / total) * 100))

