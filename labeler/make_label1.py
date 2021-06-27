import cv2
import numpy as np
from datetime import datetime
import json
import pytesseract
#from pytesseract import Output
import re

def refined_labeling(image, x1, x2, y1,y2):
    with open("labeler/boundaries.json", "r") as jsonfile:
        boundaries = json.load(jsonfile)
    output={category: 0 for category in boundaries }
    img=image[y1:y2, x1:x2]
    #Trying really hard to use OCR to read the jerseys but the predictions are miserably failing. Someone get this working!!!!
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_string(gray, lang='eng')
    ocr_result = replace_chars(ocr_result)
    """

    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """
    for category in boundaries:
        lower = np.array(boundaries[category][0], dtype = "uint8")
        upper = np.array(boundaries[category][1], dtype = "uint8")
        mask = cv2.inRange(img,lower, upper)
        output[category]=np.count_nonzero(mask)/(mask.shape[0]+mask.shape[1])
    label=max(output, key=output.get)
    return label+str(ocr_result), list(boundaries[label][2])

def replace_chars(text):
    list_of_numbers = re.findall(r'\d+', text)
    result_number = ''.join(list_of_numbers)
    return result_number
    
    
def print_full_images(image):
    cv2.imwrite("labeler/full_im/"+ datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")+".jpg", image)
    return
    
def print_images(image, x1, x2, y1,y2):
    cv2.imwrite("labeler/images/"+ datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")+".jpg", image[y1:y2, x1:x2])
    return
    