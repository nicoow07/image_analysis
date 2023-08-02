from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import scipy
from scipy.signal import argrelextrema
import cv2
import csv
import json
import time
import datetime

# OPENCV IMAGE PROCESSING
THRESHOLD_VALUE = 135
MIN_THRESHOLD_ITERATION_SPACE = 20
MINIMUM_SURFACE_FILTER = 300
MAXIMUM_SURFACE_FILTER = 900
# Surface - Weight model
SURFACIQUE_WEIGHT = 1.93
Y_INTERCEPT = 30.52
# Logs
LOG_FILE_NAME = "logs"
IMAGE_FOLDER = "img"
MAX_CAPTURE_SAVE = 100 # Save the latest pictures 
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

	#Calc the histogram.
	hist = cv2.calcHist([valueChannel], [0], None, [256], [0, 256])
	# Compute a single threshold value based on the histogram
	localMinIndices = argrelextrema(hist, np.less)[0]

	# Remove some local min if too close together
	localMinIndicesCpy = localMinIndices.copy()
	prevMinValue = localMinIndices[0]
	for minVal in localMinIndicesCpy[1:-1]:
		if minVal - prevMinValue < MIN_THRESHOLD_ITERATION_SPACE:
			localMinIndices = np.delete(localMinIndices,np.where(localMinIndices == minVal))
		else:# Only update previous value if it is left in the array
			prevMinValue = minVal

	print("Local min after filtering: " + str(localMinIndices))

	# BEST THRESHOLD ASSESSMENT
	# Assess the number of objects contoured with all minima of histogram as thresholds
	# Stop iterating when the number of objects starts to decline. Assumption: it will only decline from there on
	objectsDetectedByThreshold = list()
	ind = 0
	lastNbrObjectsDetected = 0
	currentNbrObjectsDetected = 0
	while ind < len(localMinIndices) and lastNbrObjectsDetected <= currentNbrObjectsDetected:
		thresholdVal = localMinIndices[ind]
		# Threshold the value chanel
		larvaeThresh = threshold(valueChannel, thresholdVal)

		# Remove noise, and get contours of only relevant objects (which surface is larger than MINIMUM_SURFACE_FILTER)
		cleanedLarvaeContours = getDenoisedContours(larvaeThresh, MINIMUM_SURFACE_FILTER, MAXIMUM_SURFACE_FILTER)

		# update last nbr of detected objects
		lastNbrObjectsDetected = currentNbrObjectsDetected
		# Get current number of detected objects
		currentNbrObjectsDetected = len(cleanedLarvaeContours)
		objectsDetectedByThreshold.append(currentNbrObjectsDetected)
		ind = ind + 1

	print("Objects for each thresholds: " + str(objectsDetectedByThreshold))
	bestThreshold = 0
	if len(objectsDetectedByThreshold) > 0:
		bestThreshold = localMinIndices[np.argmax(objectsDetectedByThreshold)]
	print("Best threshold: " + str(bestThreshold))

	#get again contour with the best threshold
	# Threshold the value chanel
	larvaeThresh = threshold(valueChannel, bestThreshold)

	# Remove noise, and get contours of only relevant objects (which surface is larger than MINIMUM_SURFACE_FILTER)
	cleanedLarvaeContours = getDenoisedContours(larvaeThresh, MINIMUM_SURFACE_FILTER, MAXIMUM_SURFACE_FILTER)


	# Get timestamp for pict names & logs
	ct = datetime.datetime.now()
	
	if len(cleanedLarvaeContours) > 0:
		# Display cleaned contours in Green
		cv2.drawContours(image, cleanedLarvaeContours, -1, (0, 255, 0), 2)
		surfaceAreaList = list(map(lambda x: cv2.contourArea(x), cleanedLarvaeContours))
		#print("surfaceAreaList")
		#print(surfaceAreaList)
		avgSurface = np.mean(surfaceAreaList)
		print("Average surface (px): " + str(avgSurface))

		# MISSING PIXEL^2 TO MM^2 CONVERSION

		# Get weight from contours
		weightList = list(map(lambda x: SURFACIQUE_WEIGHT*x + Y_INTERCEPT, surfaceAreaList))
		avgWeight = np.mean(weightList)
		#print("weightList")
		#print(weightList)
		#print("Average weight: " + str(avgWeight) + " g")

		#Output logs in file
		logsFile = open(LOG_FILE_NAME + ".txt", "a")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-average_surface: " + str(avgSurface) + "\n")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-average_weight: " + str(avgWeight) + "\n\n")

		# Output logs in CSV
		outputCSVFile = open(LOG_FILE_NAME + ".csv", 'a')
		outputWriter = csv.writer(outputCSVFile)
		row = [str(captureSaveCount), str(ct), str(avgSurface), str(avgWeight)]
		outputWriter.writerow(row)

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