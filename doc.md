# DOCUMENTATION FOR THE WEBSERVER CALCULATING AVERAGE WEIGHT FROM STREAMED IMAGES
This python script is a webserver used to receive images from the data acquisition hardware (A smartphone as of August 2023) taking pictures of the harvested larvae. The images are processed to extract the average weight of the harvest as a performance metric of the grow-out process.

## Input
The webserver is listening POST requests on the route /upload_image for iamges of free falling larvae. The image have to be in the "files" section of the form-data request. A code in Java sending an image with the proper request format is available in the file "image_POST_request_example.java".

## Parameters
The parameters are set in the configuration file, by default names "config.json", or given as first argument when launching the script.
The parameters of this script are:

### OpenCV Image processing
**img_processing.histogram_min_iteration_span** = The minimum value between two threshold values tested to get the best value.
Threshold is a value between 0 and 255. A value of 20 is recommended to avoid unecessary processing, while having accurate best threshold value determination.

**img_processing.minimum_surface_filter** = The objects detected are filtered based on their surface in pixels^2. This is the lower bound. All objects with a surface lower than this parameter are excluded from the surface and average weight calculation.
**img_processing.maximum_surface_filter** = This is the upper bound of the filter.

**img_processing.max_roughness_value** = The maximum allowed roughness value for a shape to be included in the calculation. The roughness is defined as: perimeter of the contour of the shape / perimeter of the convex hull of the shape.

### Surface - Weight model
To convert the surface of larvae into a weight, a linear model is used.
Such as: weight = surface * SURFACIQUE_WEIGHT + Y_INTERCEPT

**weight_model.area_weight** = The value allowing the script to convert a larvae's surface into a weight. It's dimension is in mm^2 / g

**weight_model.y_intercept** = The absciss from the linear model when its y value is 0. 

### Logs
**logs.log_file_name** = The filename of the .md file recording all measures output processed by the webserver.
If no object are detected, no log is added to the files.
**logs.image_folder** = The folder name where all images are saved temporarily.
**logs.max_capture_save** = The maximum number of pictures to be saved. The script will overwrite the oldest pictures to save space.

## Output
## Image processing algorithm
## Processing speed