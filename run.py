import numpy as np
import argparse
import imutils
import time
import cv2
import os
from labeler import make_label

class Detect():
    def __init__(self, yolo_path="yolo-coco", video_path="videos/Test.mp4", output_path="output/output.avi", confidence=0.5, threshold=0.3):
        self.yolo_path = yolo_path
        self.video_path = video_path
        self.output_path = output_path
        self.confidence = confidence
        self.threshold = threshold
    
    def run(self):
        # Creating the directory if it doesn't already exist
        directory=("/").join(self.output_path.split("/")[:-1])
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # Loading the model
        labelsPath = os.path.sep.join([self.yolo_path, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        weightsPath = os.path.sep.join([self.yolo_path, "yolov3.weights"])
        configPath = os.path.sep.join([self.yolo_path, "yolov3.cfg"])
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        np.random.seed(2)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

        # Reading the video and go!
        vs = cv2.VideoCapture(self.video_path)
        writer = None
        (W, H) = (None, None)

        while True:
            (grabbed, frame) = vs.read()

            if not grabbed:
                break

            if W is None or H is None:
                (H, W) = frame.shape[:2]


            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
              for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                op_confidence = scores[classID]

                if op_confidence > self.confidence:
                  box = detection[0:4] * np.array([W, H, W, H])
                  (centerX, centerY, width, height) = box.astype("int")

                  x = int(centerX - (width / 2))
                  y = int(centerY - (height / 2))

                  boxes.append([x, y, int(width), int(height)])
                  confidences.append(float(op_confidence))
                  classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

            try:
                for i in idxs.flatten():

                    """

                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    label = LABELS[classIDs[i]]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    """

                    # Limiting labels to only what we need
                    if LABELS[classIDs[i]] in ['sports ball', 'person']:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        color = [int(c) for c in COLORS[classIDs[i]]]
                        
                        # Since we need to go beyond just the generic model
                        if (LABELS[classIDs[i]]=="person"):
                            label, c= make_label.refined_labeling(frame, max(0,x), min(x + w, W), max(0,y), min(y + h, H) )
                            color=c
                        else:
                            label = LABELS[classIDs[i]]

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        
            except:
                # Usually occurs when there are no detections in a frame. Worry only when you see too many of these for a short video
                print("Exception")
                pass
            
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)

        writer.release()
        vs.release()