#OCR detection-----------------------------------
import pytesseract
import cv2
from google.colab.patches import cv2_imshow
from pytesseract import Output

ocr_img = cv2.imread('/content/n4.jpeg')                       #load image file
custom = r'output --oem 3  --psm 6'                            #default configuration for OCR engine (has options to choose lang) 

def characterBox(img):                                         #function to draw a bbox over characters only to avoid junk values
  d = pytesseract.image_to_data(img, output_type=Output.DICT)  #convert image to dictionary file format
  #print(d.keys())
  n_boxes = len(d['text'])
  for i in range(n_boxes):
      if int(d['conf'][i]) > 60:
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])  #fetch x,y,width & height
          img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)       #draw BBOX
          return img                                                                 #return image

filtered_img = characterBox(ocr_img)                              #call function
text = pytesseract.image_to_string(filtered_img, config=custom)   #predict characters using pytesseract-OCR

with open('/content/file.txt', mode = 'w') as f:                   #open a new text file in write mode
    f.write(text)                                                  #write predicted value to a file                      
    cv2_imshow(ocr_img)                                            #show input image
    print('License Plate Number: ',text)                           #print predicted value       
