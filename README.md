# FootballTeamDetection_YOLOnOpenCV
Implementation of YOLOv3 with OpenCV for color based classification of objects. (Demonstrated for football team detection)

## Requirements
* imutils
* NumPy
* OpenCV

## Demo Result
For input [video](https://www.youtube.com/watch?v=m6OelxDt1kk) ~1:20-~1:30 snapshots of the results are as shown below:<br/>
![Result Image 1](https://github.com/VenissaCarolQuadros/FootballTeamDetection_YOLOnOpenCV/blob/main/result_images/result_1.JPG)<br/>
[[https://github.com/VenissaCarolQuadros/FootballTeamDetection_YOLOnOpenCV/result_images/result_2.JPG]]<br/>
[[https://github.com/VenissaCarolQuadros/FootballTeamDetection_YOLOnOpenCV/result_images/result_3.JPG]]<br/>


## How to use?
* Clone repository
```
git clone https://github.com/VenissaCarolQuadros/FootballTeamDetection_YOLOnOpenCV`
```

* Install requirements if not installed (use a virtual environment if desired)
```
pip install -r requirements.txt
```

* Replace yolov3.weights in yolo-coco with the file downloaded from the original [source](https://pjreddie.com/media/files/yolov3.weights)

* Place input video in location videos/Test.mp4.

*NOTE*: The make_label.py script has boundaries set for this [video](https://www.youtube.com/watch?v=m6OelxDt1kk) ~1:20-~1:30. Use this video as Test.mp4 to replicate demo results.
Alternatively, refer [Further Instructions](#further-instructions) to set the script up for other videos

* In your python code import Detect() from run.py using
```python
from run import Detect
```
* Run object detection using
```python
d=Detect(yolo_path="yolo-coco", video_path="videos/Test.mp4", output_path="output/output.avi", confidence=0.5, threshold=0.3)
```
The above values are set by default and will yield equivalent results even when used as follows
```python
d=Detect()
```
However, one or more of the default paramters can be changed by explicit mention as shown above.

yolo_path -> Path to directory containing coco.names, yolov3.cfg and yolov3.weights<br/>
video_path -> Path to input video<br/>
output_path -> Path to output video<br/>
confidence -> Minimum probability to filter weak detections <br/>
threshold -> Non-maxima suppression threshold i.e. boxes overlapping with a ratio greater than 30% are suppressed for threshold=0.3<br/>

## Further Instructions
* To use the script for general object detection purposes uncomment lines 82 to 93 and comment lines 93 to 110 in run.py

* To detect other list of items replace list ['sports ball', 'person'] in line 96 of run.py by your desired list of items. Make appropriate changes in if conditional statement at line 103 also.

* To use the colour based team detection for other teams (or videos) replace the dict provided in labeler/boundaries.json by the required values. This dict represent the boundary values within which the jersey colour of a particular team or a category  are most likely to be present.
The format for this dict is <br/>
*{"label1": [[B_lower_limit, G_lower_limit, R_lower_limit], [B_upper_limit, G_upper_limit, R_upper_limit], [B_label_colour, G_label_colour, R_label_colour]], "label2": [[B_lower_limit, G_lower_limit, R_lower_limit], [B_upper_limit, G_upper_limit, R_upper_limit], [B_label_colour, G_label_colour, R_label_colour]], ... }*
