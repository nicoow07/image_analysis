# DOCUMENTATION FOR THE WEBSERVER CALCULATING AVERAGE WEIGHT FROM STREAMED IMAGES
This python script is a webserver used to receive images from the data acquisition hardware (A smartphone as of August 2023) taking pictures of the harvested larvae. The images are processed to extract the average weight of the harvest as a performance metric of the grow-out process.

## Input
The webserver is listening POST requests on the route /upload_image for iamges of free falling larvae. The image have to be in the "files" section of the form-data request. A code in Java sending an image with the proper request format is available in the file "image_POST_request_example.java".

The route '/new_capture_session' is a POST type, taking a multiform data as input. The field "capture_session_count" determines the new captures session ID. All images analysed during the same capture session will have their data compiled and output once in the 'logs_sessions.csv' file. You will find the total count, average surface & weight of all larvae analysed during the session.

The route '/close_capture_session' is a GET type. When called, the data gathered during the capture session are saved in the 'logs_sessions.csv' when this route is called. Any further images will not be taken into account in the calculation. But will still be logged in logs.csv and logs.txt

## Parameters
The parameters are set in the configuration file, given by the value of the environment variable CONFIG_FILE.
To change the config file in use, execute this command:
`export CONFIG_FILE="config/new_config_file.json"`

The parameters of this script are:

### OpenCV Image processing
**img_processing.histogram_min_iteration_span** = The minimum value between two threshold values tested to get the best value.
Threshold is a value between 0 and 255. A value of 20 is recommended to avoid unecessary processing, while having accurate best threshold value determination.

**img_processing.minimum_surface_filter** = The objects detected are filtered based on their surface in pixels^2. This is the lower bound. All objects with a surface lower than this parameter are excluded from the surface and average weight calculation. If calibration is "benchmark", then this value is in mm^2
**img_processing.maximum_surface_filter** = This is the upper bound of the filter.

**img_processing.max_roughness_value** = The maximum allowed roughness value for a shape to be included in the calculation. The roughness is defined as: perimeter of the contour of the shape / perimeter of the convex hull of the shape.

**img_processing.min_larvae_to_detect** = The minimum number of larvae to be detected so that the measure is added to the csv file, and the image saved

**img_processing.max_larvae_elongation** = The maximum elongation of a larvae to be accepted. The elongation is the division between the height and the width or the width / height. Elongation is always >= 1

### Surface - Weight model
To convert the surface of larvae into a weight, a linear model is used.
Such as: weight = surface * SURFACIQUE_WEIGHT + Y_INTERCEPT

**weight_model.calibration** = Values can be either "benchmark", or "direct".
1. If "benchmark" is selected, then the script will search for a red strip of exactly 5cm in the picture, and use this to output a weight in grams. 
In this case, the two following parameters "area_weight" and "y_intercept" describe the linear model defined as such:
weight (g) = surface (mm^2) * AREA_WEIGHT (mm^2 / g) + Y_INTERCEPT (g)

2. If "direct" is selected, then the script will use the two following parameters "area_weight" and "y_intercept" as decribing the linear model defined as:
weight (g) = surface (pixels^2) * AREA_WEIGHT (pixels^2 / g) + Y_INTERCEPT (g)

**weight_model.benchmark_metric_length** = The length of the benchmark in millimeters.

**weight_model.area_weight** = The value allowing the script to convert a larvae's surface into a weight. It's dimension is in mg / mm^2

**weight_model.y_intercept** = The absciss from the linear model when its y value is 0. 

### Logs
**logs.log_file_name** = The filename of the .md file recording all measures output processed by the webserver.
If no object are detected, no log is added to the files.
**logs.image_folder** = The folder name where all images are saved temporarily.
**logs.max_capture_save** = The maximum number of pictures to be saved. The script will overwrite the oldest pictures to save space.

## Output
### Files
**logs_sessions.csv**: Each row is a summary of the data analysed during a capture session.
Each column mean: Session ID, Image ID, Timestamp, Total recognised larvae, Avg surface, Avg weight, Surface List.
Surface list is in px^2 if calibration is "direct", or in mm^2 for "benchmark" calibration


## Image processing algorithm
## Processing speed