from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import scipy
from scipy.signal import argrelextrema
import cv2
import csv
import json
import time
import sys
import os
import datetime

# OPENCV IMAGE PROCESSING
MIN_THRESHOLD_ITERATION_SPACE = 20
MINIMUM_SURFACE_FILTER = 300
MAXIMUM_SURFACE_FILTER = 1000
MAX_ROUGHNESS_VALUE = 1.18
MIN_LARVAE_TO_DETECT=1
# Surface - Weight model
CALIBRATION = "benchmark"
BENCHMARK_METRIC_LENGTH = 40
AREA_WEIGHT = 1.93
Y_INTERCEPT = 30.52
# Logs
LOG_FILE_NAME = "logs"
IMAGE_FOLDER = "img"
MAX_CAPTURE_SAVE = 100 # Save the latest pictures 

# Global variables to keep between several POST request received
captureSaveCount = 1 # current count for file name and log matching
# SESSION variables
captureSessionCount = 0 # First value to be sent by the image acq app is 1
currSession_recognisedLarvaeCount = 0 # The total number of larvae detected for the current session
currSession_totalSurface = 0 # The cumulative surface of recognised larvae for the current session. In px^2
currSession_totalWeight = 0 # The cumulative weight of recognised larvae for the current session. In g

# Thresholding. Takes as input a single channel image
# Returns a mask, a binary image.
def threshold(singleChannelImage, threshold):

	gThresh = singleChannelImage[:,:]
	#Threshold
	gThresh = np.where(gThresh > threshold, 0, 1)

	foreground = np.where(gThresh > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
	mask = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
	return mask

#Takes two HSV points for low and high range of the accepted colors. All points in this range of color are output with value 255 (white), the others are output with value 0 (black) 
def hsvThreshold(hsv, lowHSVRange, highHSVRange):
	object_color_range = [lowHSVRange, highHSVRange]
	thresh = cv2.inRange(hsv, object_color_range[0], object_color_range[1])
	return thresh


# Takes as input a binary image (a mask), and remove all objects whose surface area is smaller than minObjectSurface, and larger than maxObjectSurface
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

# Takes the original image and the hsv image as input
# Returns the nbr of pixels corresponding to the benchmark on the picture (for instance a red strip of paper of 5cm)
# Returns -1 of no benchmark found
def get_benchmark_length(image, hsv):
	# Filter RED color because the benchmark is a red paper strip
	low_red_thresh = hsvThreshold(hsv, np.array([150, 60, 100]), np.array([180, 255, 255]))
	high_red_thresh = hsvThreshold(hsv, np.array([0, 60, 100]), np.array([30, 255, 255]))
	thresh = cv2.bitwise_or(low_red_thresh, high_red_thresh)

	# Find contours in the binary image
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	

	# If no benchmark found on the image, return -1
	if len(contours) <= 0:
		cv2.putText(image, "No benchmark found", (20, 20 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		return -1
	# Filter the contours to only keep the largest one
	largest_contour = max(contours, key=cv2.contourArea)


	# Draw contour and add "benchmark" text
	rect = cv2.minAreaRect(largest_contour)

	# Box object is simply an array of the coordinates of the 4 vertices of the rectangle
	box = np.int0(cv2.boxPoints(rect))
	cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
	# These are the width and height of the rotated bounding rect
	[w,h] = rect[1]
	[x,y] = box[0]
	cv2.imwrite(IMAGE_FOLDER + "/" + str(captureSaveCount) + "-thresh_benchmark.jpg", thresh)

	length = round(max(w,h),2)
	cv2.putText(image, "Benchmark: "+str(length)+" px = "+str(BENCHMARK_METRIC_LENGTH)+" mm", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
	
	return length

def loadConfig(configFileName):
	print("Loading config file: " + configFileName)

	global MIN_THRESHOLD_ITERATION_SPACE
	global MINIMUM_SURFACE_FILTER
	global MAXIMUM_SURFACE_FILTER
	global MAX_ROUGHNESS_VALUE
	global MIN_LARVAE_TO_DETECT
	global CALIBRATION
	global BENCHMARK_METRIC_LENGTH
	global AREA_WEIGHT
	global Y_INTERCEPT
	global LOG_FILE_NAME
	global IMAGE_FOLDER
	global MAX_CAPTURE_SAVE

	# Load config file
	try:
		with open(configFileName) as f:
			config = json.load(f)

			# OPENCV IMAGE PROCESSING
			MIN_THRESHOLD_ITERATION_SPACE = int(config['img_processing']['histogram_min_iteration_span'])
			MINIMUM_SURFACE_FILTER = int(config['img_processing']['minimum_surface_filter'])
			MAXIMUM_SURFACE_FILTER = int(config['img_processing']['maximum_surface_filter'])
			MAX_ROUGHNESS_VALUE = float(config['img_processing']['max_roughness_value'])
			MIN_LARVAE_TO_DETECT = int(config['img_processing']['min_larvae_to_detect'])
			# Surface - Weight model
			CALIBRATION = str(config['weight_model']['calibration'])
			BENCHMARK_METRIC_LENGTH = int(config['weight_model']['benchmark_metric_length'])
			AREA_WEIGHT = float(config['weight_model']['area_weight'])
			Y_INTERCEPT = float(config['weight_model']['y_intercept'])
			# Logs
			LOG_FILE_NAME = str(config['logs']['log_file_name'])
			IMAGE_FOLDER = str(config['logs']['image_folder'])
			MAX_CAPTURE_SAVE = int(config['logs']['max_capture_save'])
			print("Config loaded successfully.")
			print(str(config))
	except FileNotFoundError as e:
		print("Config file not found.")
		sys.exit(1)
	except OSError as e:
		print(e.errno)
		sys.exit(2)

# Get config file name from the environment variable CONFIG_FILE
conf_file_name = os.environ.get('CONFIG_FILE')
loadConfig(conf_file_name)

app = Flask(__name__)

@app.route('/new_capture_session', methods = ['POST'])
def newCaptureSession():
	#=====CAPTURE SESSION COUNT RECEPTION=====
	request_capture_session_count = request.form.get('capture_session_count')

	if request_capture_session_count is None:
		print("capture_session_count is None")
		return "capture_session_count not found in the request."
	print("New capture session num: " + request_capture_session_count)

	# Update & initialise session variables
	global captureSessionCount
	global currSession_recognisedLarvaeCount
	global currSession_totalSurface
	global currSession_totalWeight
	captureSessionCount = request_capture_session_count
	currSession_totalWeight = 0
	currSession_totalSurface = 0
	currSession_recognisedLarvaeCount = 0

	return "Session " + str(captureSessionCount) + " opened."

@app.route('/close_capture_session', methods = ['POST','GET'])
def closeCaptureSession():
	global captureSaveCount
	global captureSessionCount
	global currSession_recognisedLarvaeCount
	global currSession_totalSurface
	global currSession_totalWeight

	# Save session variables in log file
	outputCSVFile = open(LOG_FILE_NAME + "_sessions.csv", 'a')
	outputWriter = csv.writer(outputCSVFile)

	#Gather the data to write into the file
	ct = datetime.datetime.now()
	sessionAvgSurface = 0
	sessionAvgWeight = 0
	if currSession_recognisedLarvaeCount > 0:
		sessionAvgSurface = currSession_totalSurface / currSession_recognisedLarvaeCount
		sessionAvgWeight = currSession_totalWeight / currSession_recognisedLarvaeCount

	# Print session info
	print("-----Session " + str(captureSessionCount) + " closed-----")
	print("Session total larvae count: " + str(currSession_recognisedLarvaeCount))
	print("Session average surface: " + str(sessionAvgSurface) + " px^2")
	print("Session average weight: " + str(sessionAvgWeight) + " g")

	row = [str(captureSessionCount), str(captureSaveCount), str(ct), str(currSession_recognisedLarvaeCount), str(sessionAvgSurface), str(sessionAvgWeight)]
	outputWriter.writerow(row)

	return "Session " + str(captureSessionCount) + " closed."

@app.route('/upload_image', methods = ['POST'])
def process_image():
	global captureSaveCount
	global captureSessionCount
	global currSession_recognisedLarvaeCount
	global currSession_totalSurface
	global currSession_totalWeight

	#=====IMAGE RECEPTION=====
	if not request.files:
		print("No image received")
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
	print("Image No " + str(captureSaveCount) + ", shape: " + str(hsv.shape))
	valueChannel = hsv[:,:,2]

	# Benchmark recognition
	benchmark_length_px = -2 # -2 means calibration is without benchmark. To differentiate from -1 which means "Benchmark not found"
	if CALIBRATION == "benchmark":
		# Get the length of the benchmark in pixels
		benchmark_length_px = get_benchmark_length(image, hsv)
		# To avoid division by 0
		if benchmark_length_px == 0.0:
			benchmark_length_px = -1
		print("Benchmark length: " + str(benchmark_length_px) + " px")

		# For benchmark use case, convert min&max_surface_filter from mm^2 to pixels^2 thanks to the benchmark recognised above
		pxPerMm = benchmark_length_px / BENCHMARK_METRIC_LENGTH
		global MINIMUM_SURFACE_FILTER
		global MAXIMUM_SURFACE_FILTER
		MINIMUM_SURFACE_FILTER = MINIMUM_SURFACE_FILTER * pxPerMm**2
		MAXIMUM_SURFACE_FILTER = MAXIMUM_SURFACE_FILTER * pxPerMm**2

	# Threshold the value chanel of HSV image with best threshold hue value
	backgroundHSVLowRange = np.array([12*255.0/360.0, 0.18*255, 0.22*255])
	backgroundHSVHighRange = np.array([32*255.0/360.0, 0.48*255, 0.5*255])
	larvaeThresh = hsvThreshold(hsv, backgroundHSVLowRange, backgroundHSVHighRange)
	larvaeThresh = 255 - larvaeThresh
	#larvaeThresh = threshold(valueChannel, bestThreshold)

	# Get again contour with the best threshold
	cleanedLarvaeContours = getDenoisedContours(larvaeThresh, MINIMUM_SURFACE_FILTER, MAXIMUM_SURFACE_FILTER)

	cv2.imwrite(IMAGE_FOLDER + "/" + str(captureSaveCount) + "-larvae_thresh.jpg", larvaeThresh)
	
	# Filter contours based on Roughness. Draw removed contours in red
	nbrRecognisedLarvae = len(cleanedLarvaeContours)
	if nbrRecognisedLarvae > 0:
		contourRoughnessList = []
		for contour in cleanedLarvaeContours:
			perimeter = cv2.arcLength(contour, True)
			cvxHull = cv2.convexHull(contour)
			cvxHullPerimeter = cv2.arcLength(cvxHull, True)
			roughness = round(perimeter / cvxHullPerimeter, 2)
			contourRoughnessList.append(roughness)
			# Display roughness next to the contour concerned
			if roughness > MAX_ROUGHNESS_VALUE:
				(x, y, w, h) = cv2.boundingRect(contour)
				cv2.putText(image, "Rg: " + str(roughness), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				cv2.drawContours(image, cleanedLarvaeContours, -1, (0, 0, 255), 2)

		contourRoughnessList = np.array(contourRoughnessList)

		# Remove all contours with a roughness above the maximum = filter value
		cleanedLarvaeContours = cleanedLarvaeContours[contourRoughnessList <= MAX_ROUGHNESS_VALUE]

	# Draw accepted contours in green
	# Calculate average surface and weight
	# Update number of recognised larvae after filtering based on contour's roughness
	nbrRecognisedLarvae = len(cleanedLarvaeContours)
	if nbrRecognisedLarvae >= MIN_LARVAE_TO_DETECT:
		# Display cleaned contours in Green
		cv2.drawContours(image, cleanedLarvaeContours, -1, (0, 255, 0), 2)
		pxSurfaceAreaList = list(map(lambda x: cv2.contourArea(x), cleanedLarvaeContours))		

		# Compute weights
		weightList = np.array(0)
		# If benchmark is used, convert surfaces in px^2 into mm^2
		if CALIBRATION == "benchmark":
			#  pxSurfaceAreaList elements are in px^2. mmPerPx is in mm / px. x*mmPerPx**2 is in mm^2
			# AREA_WEIGHT is in mg / mm^2. Y_INTERCEPT is in mg
			mmPerPx = BENCHMARK_METRIC_LENGTH / benchmark_length_px
			weightList = list(map(lambda x: AREA_WEIGHT*(x*mmPerPx**2) + Y_INTERCEPT, pxSurfaceAreaList))

			# Draw
			for i in range(0, nbrRecognisedLarvae):
				larva_contour = cleanedLarvaeContours[i]
				surfacePx = pxSurfaceAreaList[i]
				weight = weightList[i]
				(xc, yc, wc, hc) = cv2.boundingRect(larva_contour)
				#cv2.putText(image, "Surface: " + str(round(surfacePx, 2)) + " px^2, ", (xc, yc ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				cv2.putText(image, "Surface: " + str(round(surfacePx*mmPerPx**2, 2)) + " mm^2", (xc, yc-13 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				cv2.putText(image, "Weight: " + str(round(weight, 2)) + " g", (xc, yc-26 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


		# Otherwise, without calibration, the AREA_WEIGHT already includes the conversion from px to mm
		else:
			weightList = list(map(lambda x: AREA_WEIGHT*x + Y_INTERCEPT, pxSurfaceAreaList))

		# Get avg weight
		avgWeight = np.mean(weightList)

		# Display info on the image; avg weight, number of larvae detected
		cv2.putText(image, "Average surface " + str(round(np.mean(pxSurfaceAreaList), 2)) + " px^2", (20, 20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
		cv2.putText(image, "Average weight " + ("(Calib=benchmark): " if (CALIBRATION == "benchmark") else "(Calibration=direct): ") + str(round(avgWeight, 2)) + " g", (20, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
		cv2.putText(image, "Number of larvae recognised: " + str(nbrRecognisedLarvae), (20, 60 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

		# Print info
		avgSurface = np.mean(pxSurfaceAreaList)
		print("Recognised larvae surface list: " + str(pxSurfaceAreaList))
		print("Recognised larvae weight list: " + str(weightList))
		print(str(nbrRecognisedLarvae) + " larvae recognised")
		print("Average surface (px^2): " + str(round(avgSurface,2)))
		print("Average weight (px^2): " + str(round(avgWeight, 2)))

		# Add metrics from this capture to the session
		currSession_recognisedLarvaeCount = currSession_recognisedLarvaeCount + nbrRecognisedLarvae
		currSession_totalSurface = currSession_totalSurface + sum(pxSurfaceAreaList)
		currSession_totalWeight = currSession_totalWeight + sum(weightList)

		#Output logs in text file
		# Get timestamp for pict names & logs
		ct = datetime.datetime.now()
		logsFile = open(LOG_FILE_NAME + ".txt", "a")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-average_surface: " + str(avgSurface) + "\n")
		logsFile.write(str(ct) + "-" + str(captureSaveCount) + "-average_weight: " + str(avgWeight) + "\n\n")

		# Output logs in CSV, with Benchmark length recognised if selected in config
		outputCSVFile = open(LOG_FILE_NAME + ".csv", 'a')
		outputWriter = csv.writer(outputCSVFile)
		if CALIBRATION == "benchmark":
			mmPerPx = BENCHMARK_METRIC_LENGTH / benchmark_length_px
			row = [str(captureSaveCount), str(ct), str(nbrRecognisedLarvae), str(avgSurface), str(avgSurface*mmPerPx**2), str(avgWeight), str(benchmark_length_px)]
		else:
			row = [str(captureSaveCount), str(ct), str(nbrRecognisedLarvae), str(avgSurface), 0, str(avgWeight)]
		outputWriter.writerow(row)

		# Save image with contoured larvae
		cv2.imwrite(IMAGE_FOLDER + "/" + str(captureSaveCount) + "-contoured_larvae.jpg", image)

	else:
		print(str(nbrRecognisedLarvae) + " larvae detected is too few. Minimum is: " + str(MIN_LARVAE_TO_DETECT))

	captureSaveCount = captureSaveCount + 1
	if captureSaveCount > MAX_CAPTURE_SAVE:
		captureSaveCount = 1
	
	endTime = time.time()
	print("Processing time: " + str(endTime-startTime) + "\n")
	
	return "200Ok"
