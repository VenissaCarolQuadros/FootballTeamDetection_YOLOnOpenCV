import cv2
import numpy as np
from datetime import datetime
import json

def refined_labeling(image, x1, x2, y1,y2):
    with open("labeler/boundaries.json", "r") as jsonfile:
        boundaries = json.load(jsonfile)
    output={category: 0 for category in boundaries }
    img=image[y1:y2, x1:x2]
    for category in boundaries:
        lower = np.array(boundaries[category][0], dtype = "uint8")
        upper = np.array(boundaries[category][1], dtype = "uint8")
        mask = cv2.inRange(img,lower, upper)
        output[category]=np.count_nonzero(mask)/(mask.shape[0]+mask.shape[1])
    label=max(output, key=output.get)
    return label, list(boundaries[label][2])

def print_full_images(image):
    cv2.imwrite("labeler/full_im/"+ datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")+".jpg", image)
    return
    
def print_images(image, x1, x2, y1,y2):
    cv2.imwrite("labeler/images/"+ datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")+".jpg", image[y1:y2, x1:x2])
    return