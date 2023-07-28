from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import scipy
from scipy.signal import argrelextrema
import cv2
import time
import datetime

# OPENCV IMAGE PROCESSING
THRESHOLD_VALUE = 135
MINIMUM_SURFACE_FILTER = 100
MAXIMUM_SURFACE_FILTER = 300
# Surface - Weight model
SURFACIQUE_WEIGHT = 1.93
Y_INTERCEPT = 30.52
# Logs
LOG_FILE_NAME = "logs.txt"
IMAGE_FOLDER = "img"
MAX_CAPTURE_SAVE = 30 # Save the latest 30 pictures 
captureSaveCount = 1 # current count for file name and log matching

# Thresholding. Takes as input a single channel image
# Returns a mask, a binary image.
def threshold(singleChannelImage, threshold):

	gThresh = singleChannelImage[:,:]
	#Threshold
	gThresh = np.where(gThresh > threshold, 0, 1)

	foreground = np.where(gThresh > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
	mask = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
	return mask

# Takes as input a binary image (a mask), and remove all objects whose surface area is smaller than minObjectSurface
# Returns a binary image of the same size as the input image
def getDenoisedContours(binaryImage, minObjectSurface, maxObjectSurface):
	# Find contours
	contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) <= 0:
		return []

	# Order contours based on their surface
	orderedContours = np.array(sorted(contours, key=cv2.contourArea, reverse=True))

	# Get the list of contour's area
	surfaces = np.array(list((map(cv2.contourArea, orderedContours))))

	# Filter contours based on min & max accepted surface of objects
	# Make sure to use an ARRAY inside the brackets, and not a LIST
	higherSurfacesFilterArray = (surfaces >= minObjectSurface)
	lowerSurfacesFilterArray = (surfaces <= maxObjectSurface)
	filterArray = higherSurfacesFilterArray & lowerSurfacesFilterArray
	filteredContours = orderedContours[filterArray]
	print("Number of contour after filtering: " + str(len(filteredContours)))
	return filteredContours


app = Flask(__name__)

@app.route('/upload_image', methods = ['POST'])
def receive_image():
	global captureSaveCount
	#=====IMAGE RECEPTION=====
	print(request.headers)
	if not request.files:
		return "No image received"
	file = request.files['image']


	# convert string of image data to uint8
	nparr = np.fromstring(file.read(), np.uint8)
	# decode image
	image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	#=====IMAGE PROCESSING=====
	# Record time
	startTime = time.time()

	# Convert the image to hsv
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Image shape: " + str(hsv.shape))
	valueChannel = hsv[:,:,2]

	#Calc the histogram
	hist = cv2.calcHist([valueChannel], [0], None, [256], [0, 256])
	localMinIndices = argrelextrema(hist, np.less)
	minValues = hist[localMinIndices]
	minValues = np.delete(minValues, np.where(minValues <=  100.0))# Remove minimums with too few pixel occurence
	print("Locals min: " + str(minValues))
	#indexOfLowestValueMinimum = np.argmin(minValues, axis=0)# Index of the minimun which has the lowest value in the histogram
	lowestMinValue = np.min(minValues)
	#lowestMinValue = minValues[1]
	thresholdVals = np.where(hist == lowestMinValue)
	print("thresholds: " + str(thresholdVals))
	thresholdVal = thresholdVals[0][0]
	print("lowestMin value: " + str(lowestMinValue))

	# NEXT STEP: ADJUST THRESHOLD VALUE BASED ON THE AVG BRIGHNESS OF THE IMAGE / HISTOGRAM DECILE

	# Threshold the value chanel
	larvaeThresh = threshold(valueChannel, thresholdVal)

	# Remove noise, and get contours of only relevant objects (which surface is larger than MINIMUM_SURFACE_FILTER)
	cleanedLarvaeContours = getDenoisedContours(larvaeThresh, MINIMUM_SURFACE_FILTER, MAXIMUM_SURFACE_FILTER)

	# Get timestamp
	# ct stores current time
	ct = datetime.datetime.now()
	
	if len(cleanedLarvaeContours) > 0:
		# Display cleaned contours in Green
		cv2.drawContours(image, cleanedLarvaeContours, -1, (0, 255, 0), 2)
		surfaceAreaList = list(map(lambda x: cv2.contourArea(x), cleanedLarvaeContours))
		print("surfaceAreaList")
		print(surfaceAreaList)
		avgSurface = np.mean(surfaceAreaList)
		print("Average surface (px): " + str(avgSurface))

		# MISSING PIXEL^2 TO MM^2 CONVERSION

		# Get weight from contours
		weightList = list(map(lambda x: SURFACIQUE_WEIGHT*x + Y_INTERCEPT, surfaceAreaList))
		avgWeight = np.mean(weightList)
		print("weightList")
		print(weightList)
		print("Average weight: " + str(avgWeight) + " g")

		#Output logs in file
		logsFile = open(LOG_FILE_NAME, "a")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-surface_list: " + str(surfaceAreaList) + "\n")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-weight_list: " + str(weightList) + "\n")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-average_weight: " + str(avgWeight) + "\n\n")

	else:
		print("Empty contour list")

	# Display image
	cv2.imwrite(IMAGE_FOLDER + "/" + str(captureSaveCount) + "-contoured_img.jpg", image)

	captureSaveCount = captureSaveCount + 1
	if captureSaveCount > MAX_CAPTURE_SAVE:
		captureSaveCount = 1
	
	endTime = time.time()
	print("Processing time: " + str(endTime-startTime))
	
	return "yo"

@app.route('/', methods = ['GET'])
def sayHi():
	return "Helloow"